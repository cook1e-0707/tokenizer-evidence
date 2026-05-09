from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.natural_evidence_v1.common import read_jsonl, read_yaml, resolve_repo_path, write_json


TRAINER_SCHEMA = "natural_evidence_bucket_lora_trainer_review_v1"
TARGET_MODE = "natural_transcript_bucket_mass"
REQUIRED_CLAIM_STATUS = "NO_PAPER_CLAIM"
SUPPORTED_ARMS = {"qwen_protected", "qwen_task_only_lora"}
SUPPORTED_CONDITIONS = {"diagnostic_high_risk"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or run natural_evidence_v1 natural-response bucket-mass "
            "LoRA training. Without --start-training this is a CPU-only review."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--contract-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--arm", required=True, choices=sorted(SUPPORTED_ARMS))
    parser.add_argument("--payload-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--prompt-split-id", required=True)
    parser.add_argument("--budget-cap", required=True)
    parser.add_argument("--condition", required=True, choices=sorted(SUPPORTED_CONDITIONS))
    parser.add_argument("--paper-claim-status", required=True)
    parser.add_argument("--query-budgets", required=True)
    parser.add_argument("--eval-owner-probes", required=True, type=int)
    parser.add_argument("--organic-null-prompts", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lambda-task", type=float, default=1.0)
    parser.add_argument("--lambda-bucket", type=float, default=1.0)
    parser.add_argument("--start-training", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


def _parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    return [int(item) for item in items]


def _payload_ids(config: Mapping[str, Any]) -> set[str]:
    payloads = config.get("payloads", [])
    if not isinstance(payloads, list):
        return set()
    return {
        str(payload.get("payload_id", ""))
        for payload in payloads
        if isinstance(payload, dict) and str(payload.get("payload_id", ""))
    }


def _surface_hits(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in ("FIELD=", "SECTION=", "TOPIC="):
        if pattern in text:
            hits.append(pattern)
    for pattern in ("OWNER", "PAYLOAD", "CERT", "EVIDENCE", "CARRIER"):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(pattern)}(?![A-Za-z0-9_])", text):
            hits.append(pattern)
    lowered = text.lower()
    for phrase in ("carrier block", "structured evidence block"):
        if phrase in lowered:
            hits.append(phrase)
    return hits


def _token_ids(values: object) -> list[int]:
    if not isinstance(values, list):
        return []
    output: list[int] = []
    for value in values:
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            continue
    return output


def _validate_position(position: Mapping[str, Any], row_index: int, position_index: int) -> list[str]:
    errors: list[str] = []
    target_bucket = str(position.get("target_bucket", ""))
    bucket_to_token_ids = position.get("bucket_to_token_ids", {})
    if not isinstance(bucket_to_token_ids, dict) or not bucket_to_token_ids:
        errors.append(f"row {row_index} position {position_index} missing bucket_to_token_ids")
        return errors
    target_bucket_token_ids = _token_ids(position.get("target_bucket_token_ids", []))
    candidate_token_ids = set(_token_ids(position.get("candidate_token_ids", [])))
    bucket_token_ids = set(_token_ids(bucket_to_token_ids.get(target_bucket, [])))
    if not target_bucket_token_ids:
        errors.append(f"row {row_index} position {position_index} missing target bucket token ids")
    if target_bucket not in {str(key) for key in bucket_to_token_ids}:
        errors.append(f"row {row_index} position {position_index} target bucket absent from bucket map")
    if target_bucket_token_ids and bucket_token_ids and not set(target_bucket_token_ids).issubset(bucket_token_ids):
        errors.append(f"row {row_index} position {position_index} target ids are not in target bucket")
    all_bucket_token_ids: set[int] = set()
    for token_ids in bucket_to_token_ids.values():
        all_bucket_token_ids.update(_token_ids(token_ids))
    if candidate_token_ids and not set(target_bucket_token_ids).issubset(candidate_token_ids):
        errors.append(f"row {row_index} position {position_index} target ids are not in candidate ids")
    if candidate_token_ids and all_bucket_token_ids and not all_bucket_token_ids.issubset(candidate_token_ids):
        errors.append(f"row {row_index} position {position_index} bucket ids are not all in candidate ids")
    try:
        token_index = int(position.get("token_index"))
    except (TypeError, ValueError):
        errors.append(f"row {row_index} position {position_index} missing integer token_index")
    else:
        if token_index < 0:
            errors.append(f"row {row_index} position {position_index} has negative token_index")
    compatible_bucket_ids = position.get("compatible_bucket_ids")
    if isinstance(compatible_bucket_ids, list) and compatible_bucket_ids:
        compatible_bucket_ids = [str(value) for value in compatible_bucket_ids]
        try:
            target_digit = int(position.get("target_digit"))
            target_radix = int(position.get("target_radix"))
        except (TypeError, ValueError):
            errors.append(f"row {row_index} position {position_index} missing variable_radix target digit/radix")
        else:
            if target_radix != len(compatible_bucket_ids):
                errors.append(f"row {row_index} position {position_index} variable_radix target_radix mismatch")
            if target_digit < 0 or target_digit >= len(compatible_bucket_ids):
                errors.append(f"row {row_index} position {position_index} variable_radix target_digit out of range")
            elif str(target_bucket) != compatible_bucket_ids[target_digit]:
                errors.append(f"row {row_index} position {position_index} variable_radix target bucket mismatch")
        extra_buckets = {str(key) for key in bucket_to_token_ids} - set(compatible_bucket_ids)
        if extra_buckets:
            errors.append(f"row {row_index} position {position_index} variable_radix bucket map includes incompatible buckets")
    return errors


def review_training_contract(
    *,
    config: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    query_budgets = _parse_int_list(args.query_budgets)
    diagnostic_scale = dict(config.get("diagnostic_high_risk_pilot_scale", {}))
    expected_model = str(dict(config.get("models", {})).get("qwen", {}).get("model_name", ""))
    expected_tokenizer = str(dict(config.get("models", {})).get("qwen", {}).get("tokenizer_name", ""))
    expected_budgets = [int(value) for value in diagnostic_scale.get("query_budgets", [])]

    if args.paper_claim_status != REQUIRED_CLAIM_STATUS:
        errors.append("paper claim status must be NO_PAPER_CLAIM")
    if args.model_name != expected_model:
        errors.append(f"model-name must be {expected_model}")
    if args.tokenizer_name != expected_tokenizer:
        errors.append(f"tokenizer-name must be {expected_tokenizer}")
    if args.payload_id not in _payload_ids(config):
        errors.append(f"payload-id {args.payload_id!r} is not configured")
    if query_budgets != expected_budgets:
        errors.append(f"query budgets must match diagnostic scale: {expected_budgets}")
    if args.eval_owner_probes < int(diagnostic_scale.get("eval_owner_probes", 2048)):
        errors.append("eval-owner-probes is below diagnostic minimum")
    if args.organic_null_prompts < int(diagnostic_scale.get("organic_null_prompts", 2048)):
        errors.append("organic-null-prompts is below diagnostic minimum")
    if args.batch_size <= 0:
        errors.append("batch-size must be positive")
    if args.max_steps < 0:
        errors.append("max-steps must be non-negative")
    if args.epochs <= 0:
        errors.append("epochs must be positive")
    if args.learning_rate <= 0.0:
        errors.append("learning-rate must be positive")
    if args.arm == "qwen_protected" and args.lambda_bucket <= 0.0:
        errors.append("protected bucket-mass arm requires positive lambda-bucket")
    if args.arm == "qwen_task_only_lora" and args.lambda_bucket != 0.0:
        errors.append("task-only LoRA arm requires lambda-bucket=0")
    if args.lambda_task < 0.0:
        errors.append("lambda-task must be non-negative")

    contract_schema = str(contract.get("schema_name", ""))
    allowed_contract_schemas = {
        "natural_evidence_train_contract_v1",
        "natural_evidence_variable_radix_train_contract_v1",
    }
    if contract_schema not in allowed_contract_schemas:
        errors.append("contract schema must be natural_evidence_train_contract_v1 or natural_evidence_variable_radix_train_contract_v1")
    if str(contract.get("payload_id", "")) != args.payload_id:
        errors.append("contract payload_id does not match requested payload")
    encoding_mode = str(contract.get("encoding_mode", "fixed_radix"))
    if contract_schema == "natural_evidence_variable_radix_train_contract_v1" and encoding_mode != "variable_radix":
        errors.append("variable-radix contract must set encoding_mode=variable_radix")
    if encoding_mode == "variable_radix" and not bool(contract.get("variable_radix_min_positions_satisfied", True)):
        errors.append("variable-radix contract does not satisfy requested minimum positions")
    claim_control = dict(contract.get("claim_control", {}))
    if claim_control.get("contains_field_value_outputs", True):
        errors.append("contract claims FIELD=value outputs are present")
    if claim_control.get("contains_structured_evidence_blocks", True):
        errors.append("contract claims structured evidence blocks are present")

    evidence_rows = 0
    total_positions = 0
    bucket_histogram: dict[str, int] = {}
    for row_index, row in enumerate(train_rows):
        if str(row.get("schema_name", "")) != "natural_evidence_train_example_v1":
            errors.append(f"row {row_index} has wrong schema")
            continue
        text_for_scan = "\n".join(
            str(row.get(key, ""))
            for key in ("prompt", "user_probe", "response_text")
            if str(row.get(key, ""))
        )
        hits = _surface_hits(text_for_scan)
        if hits:
            errors.append(f"row {row_index} natural text contains forbidden surface patterns: {hits}")
        positions = row.get("eligible_positions", [])
        if not isinstance(positions, list):
            errors.append(f"row {row_index} eligible_positions is not a list")
            continue
        if positions:
            evidence_rows += 1
            total_positions += len(positions)
        for position_index, position in enumerate(positions):
            if not isinstance(position, dict):
                errors.append(f"row {row_index} position {position_index} is not an object")
                continue
            errors.extend(_validate_position(position, row_index, position_index))
            target_bucket = str(position.get("target_bucket", ""))
            if target_bucket:
                bucket_histogram[target_bucket] = bucket_histogram.get(target_bucket, 0) + 1

    if not train_rows:
        errors.append("training JSONL has no rows")
    if evidence_rows == 0:
        errors.append("training JSONL has no evidence rows")
    if total_positions == 0:
        errors.append("training JSONL has no eligible evidence positions")
    if args.arm == "qwen_protected" and total_positions == 0:
        errors.append("protected arm requires eligible evidence positions")
    if args.arm == "qwen_task_only_lora" and total_positions:
        warnings.append("task-only arm uses the natural transcript rows but disables bucket loss")

    return {
        "schema_name": TRAINER_SCHEMA,
        "status": "PASS_PREFLIGHT_DRY_RUN_NOT_TRAINED" if not errors else "FAIL_PREFLIGHT",
        "errors": errors,
        "warnings": warnings,
        "paper_claim_allowed": False,
        "training_started": False,
        "gpu_required_for_training": True,
        "trainer_review_status": "PRESENT_REVIEWED_DRY_RUN_READY" if not errors else "PRESENT_REVIEW_FAILED",
        "result_claim": "trainer_contract_review_not_payload_recovery",
        "model": args.model_name,
        "tokenizer": args.tokenizer_name,
        "arm": args.arm,
        "condition": args.condition,
        "payload_id": args.payload_id,
        "seed": args.seed,
        "prompt_split_id": args.prompt_split_id,
        "budget_cap": args.budget_cap,
        "encoding_mode": encoding_mode,
        "query_budgets": query_budgets,
        "eval_owner_probes": args.eval_owner_probes,
        "organic_null_prompts": args.organic_null_prompts,
        "example_count": len(train_rows),
        "evidence_example_count": evidence_rows,
        "task_only_example_count": len(train_rows) - evidence_rows,
        "total_eligible_positions": total_positions,
        "target_bucket_histogram": bucket_histogram,
        "losses": {
            "target_mode": TARGET_MODE,
            "task_loss": args.lambda_task > 0.0,
            "bucket_mass_loss": args.arm == "qwen_protected" and args.lambda_bucket > 0.0,
            "kl_reference_loss": False,
        },
        "safety": {
            "uses_old_compiled_train_script": False,
            "contains_structured_evidence_blocks": False,
            "contains_field_value_outputs": False,
            "requires_explicit_start_training_flag": True,
        },
        "variable_radix": {
            "enabled": encoding_mode == "variable_radix",
            "contract_schema": contract_schema,
            "frame_policy": str(contract.get("variable_radix_frame_policy", "")),
            "frame_count": int(contract.get("variable_radix_frame_count", 0) or 0),
            "min_positions_satisfied": bool(contract.get("variable_radix_min_positions_satisfied", True)),
            "used_positions": int(contract.get("variable_radix_used_positions", total_positions) or 0),
            "unused_tail_positions": int(contract.get("variable_radix_unused_tail_positions", 0) or 0),
            "production_train_path": (
                "variable_radix_bucket_mass_loss_ready"
                if encoding_mode == "variable_radix" and not errors
                else "not_variable_radix_or_failed_review"
            ),
        },
    }


def _encode_no_special(tokenizer: object, text: str) -> list[int]:
    try:
        return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
    except TypeError:
        return [int(token_id) for token_id in tokenizer.encode(text)]


def _build_batch(
    *,
    torch_module: object,
    tokenizer: object,
    rows: Sequence[Mapping[str, Any]],
    max_length: int,
    device: object,
    include_task_loss: bool,
) -> tuple[object, object, object, list[dict[str, Any]]]:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise RuntimeError("tokenizer.pad_token_id must be set before training")
    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    evidence_specs: list[dict[str, Any]] = []
    max_width = 0
    for batch_row_index, row in enumerate(rows):
        prompt_ids = _encode_no_special(tokenizer, str(row.get("prompt", "")))
        response_ids = _encode_no_special(tokenizer, str(row.get("response_text", "")))
        if not response_ids:
            raise RuntimeError(f"row {batch_row_index} has an empty tokenized response")
        input_ids = [*prompt_ids, *response_ids]
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        labels = list(input_ids)
        for index in range(min(len(prompt_ids), len(labels))):
            labels[index] = -100
        if not include_task_loss:
            labels = [-100 for _ in labels]
        for position in row.get("eligible_positions", []):
            if not isinstance(position, dict):
                continue
            token_index = int(position.get("token_index", -1))
            prediction_index = len(prompt_ids) + token_index - 1
            target_index = len(prompt_ids) + token_index
            if prediction_index < 0 or target_index >= len(input_ids):
                continue
            candidate_token_ids = _token_ids(position.get("candidate_token_ids", []))
            target_bucket_token_ids = _token_ids(position.get("target_bucket_token_ids", []))
            if not candidate_token_ids or not target_bucket_token_ids:
                continue
            evidence_specs.append(
                {
                    "batch_row_index": batch_row_index,
                    "prediction_index": prediction_index,
                    "candidate_token_ids": candidate_token_ids,
                    "target_bucket_token_ids": target_bucket_token_ids,
                }
            )
        input_rows.append(input_ids)
        attention_rows.append([1] * len(input_ids))
        label_rows.append(labels)
        max_width = max(max_width, len(input_ids))
    for input_ids, attention, labels in zip(input_rows, attention_rows, label_rows, strict=True):
        pad_width = max_width - len(input_ids)
        if pad_width <= 0:
            continue
        input_ids.extend([int(pad_token_id)] * pad_width)
        attention.extend([0] * pad_width)
        labels.extend([-100] * pad_width)
    return (
        torch_module.tensor(input_rows, dtype=torch_module.long, device=device),
        torch_module.tensor(attention_rows, dtype=torch_module.long, device=device),
        torch_module.tensor(label_rows, dtype=torch_module.long, device=device),
        evidence_specs,
    )


def _bucket_mass_loss(
    *,
    torch_module: object,
    logits: object,
    evidence_specs: Sequence[Mapping[str, Any]],
) -> object:
    losses: list[object] = []
    for spec in evidence_specs:
        row_logits = logits[int(spec["batch_row_index"]), int(spec["prediction_index"]), :]
        candidate_token_ids = [int(token_id) for token_id in spec["candidate_token_ids"]]
        target_bucket_token_ids = [int(token_id) for token_id in spec["target_bucket_token_ids"]]
        token_id_to_position = {
            token_id: index
            for index, token_id in enumerate(candidate_token_ids)
        }
        target_positions = [
            token_id_to_position[token_id]
            for token_id in target_bucket_token_ids
            if token_id in token_id_to_position
        ]
        if not target_positions:
            continue
        allowed_logits = row_logits[candidate_token_ids]
        allowed_log_probs = torch_module.log_softmax(allowed_logits, dim=0)
        target_log_mass = torch_module.logsumexp(allowed_log_probs[target_positions], dim=0)
        losses.append(-target_log_mass)
    if not losses:
        return logits.sum() * 0.0
    return torch_module.stack(losses).mean()


def _run_training(args: argparse.Namespace, train_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("GPU training requires torch and transformers") from error
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        raise RuntimeError("LoRA training requires peft") from error

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("require-cuda was set but torch.cuda.is_available() is false")
    if args.max_steps <= 0:
        raise RuntimeError("start-training requires --max-steps > 0")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        ),
    )
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rows = list(train_rows)
    metrics_path = output_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    step = 0
    final_loss = 0.0
    while step < args.max_steps:
        random.shuffle(rows)
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            input_ids, attention_mask, labels, evidence_specs = _build_batch(
                torch_module=torch,
                tokenizer=tokenizer,
                rows=batch_rows,
                max_length=args.max_length,
                device=device,
                include_task_loss=args.lambda_task > 0.0,
            )
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            task_loss = outputs.loss if args.lambda_task > 0.0 else outputs.logits.sum() * 0.0
            bucket_loss = (
                _bucket_mass_loss(torch_module=torch, logits=outputs.logits, evidence_specs=evidence_specs)
                if args.arm == "qwen_protected"
                else outputs.logits.sum() * 0.0
            )
            loss = task_loss * float(args.lambda_task) + bucket_loss * float(args.lambda_bucket)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            final_loss = float(loss.detach().cpu().item())
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": final_loss,
                            "task_loss": float(task_loss.detach().cpu().item()),
                            "bucket_loss": float(bucket_loss.detach().cpu().item()),
                            "evidence_positions_in_batch": len(evidence_specs),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            if step >= args.max_steps:
                break
    checkpoint_dir = output_dir / "checkpoints" / "natural_bucket_lora_last"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return {
        "training_started": True,
        "training_completed": True,
        "steps": step,
        "final_loss": final_loss,
        "checkpoint_dir": str(checkpoint_dir),
        "metrics_path": str(metrics_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    output_dir = resolve_repo_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_yaml(resolve_repo_path(args.config, root))
    train_rows = read_jsonl(resolve_repo_path(args.train_jsonl, root))
    contract = json.loads(resolve_repo_path(args.contract_json, root).read_text(encoding="utf-8"))
    summary = review_training_contract(
        config=config,
        train_rows=train_rows,
        contract=contract,
        args=args,
    )
    summary["output_dir"] = str(output_dir)
    summary_path = output_dir / "natural_bucket_lora_trainer_review.json"
    if summary["errors"]:
        write_json(summary_path, summary)
        print(json.dumps(summary, sort_keys=True))
        return 1
    if not args.start_training:
        write_json(summary_path, summary)
        print(json.dumps(summary, sort_keys=True))
        return 0
    training_result = _run_training(args, train_rows, output_dir)
    summary.update(training_result)
    summary["status"] = "TRAINING_COMPLETED_DIAGNOSTIC_HIGH_RISK_NOT_PAPER_CLAIM"
    summary["result_claim"] = "training_completed_not_payload_recovery"
    write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
