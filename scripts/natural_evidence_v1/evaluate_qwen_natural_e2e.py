from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import inspect
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.natural_evidence_v1.build_bucket_bank import _bucketize
from scripts.natural_evidence_v1.common import (
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    write_json,
)
from scripts.natural_evidence_v1.evaluate_diagnostic_e2e import (
    _append_csv,
    _append_jsonl,
    _decode_observation_group,
    _generate_outputs,
    _load_model,
    _prepare_decode_rows,
    _release_model,
    _token_ids,
    _decode_token,
)


SCHEMA_NAME = "natural_evidence_qwen_natural_e2e_eval_v1"
REQUIRED_CLAIM_STATUS = "NO_PAPER_CLAIM"
REQUIRED_CONDITION = "diagnostic_high_risk"
REQUIRED_ARMS = (
    "qwen_protected",
    "qwen_raw",
    "qwen_task_only_lora",
    "wrong_key",
    "wrong_payload",
)
TRAINED_ARMS = ("qwen_protected", "qwen_task_only_lora")


def _decoder_dependency_errors() -> list[str]:
    try:
        signature = inspect.signature(_decode_observation_group)
    except (TypeError, ValueError) as exc:
        return [f"unable to inspect imported decoder helper signature: {exc}"]
    if "decoder_mode" not in signature.parameters:
        return [
            "imported evaluate_diagnostic_e2e._decode_observation_group lacks "
            "decoder_mode; sync scripts/natural_evidence_v1/evaluate_diagnostic_e2e.py"
        ]
    return []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or run the Qwen natural-output variable-radix five-arm "
            "E2E evaluation wrapper. Without --start-eval this writes a dry-run "
            "review only; it does not load a model, generate outputs, or train."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--payload-ids", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--query-budgets", required=True)
    parser.add_argument("--eval-owner-probes", type=int, required=True)
    parser.add_argument("--organic-null-prompts", type=int, required=True)
    parser.add_argument("--prompt-split-id", required=True)
    parser.add_argument("--budget-cap", required=True)
    parser.add_argument("--condition", default=REQUIRED_CONDITION)
    parser.add_argument("--paper-claim-status", default=REQUIRED_CLAIM_STATUS)
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--training-job-id", default="")
    parser.add_argument("--wrong-key-count", type=int, default=0)
    parser.add_argument("--max-prompts", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--start-eval", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _payload_text_by_id(config: Mapping[str, Any]) -> dict[str, str]:
    payloads = config.get("payloads", [])
    output: dict[str, str] = {}
    if isinstance(payloads, list):
        for payload in payloads:
            if isinstance(payload, dict) and payload.get("payload_id"):
                payload_id = str(payload["payload_id"])
                output[payload_id] = str(payload.get("payload_text", payload_id))
    return output


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


def _model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, dict) or not isinstance(models.get("qwen"), dict):
        raise ValueError("missing qwen model config")
    return dict(models["qwen"])


def _diagnostic_scale(config: Mapping[str, Any]) -> dict[str, Any]:
    scale = config.get("diagnostic_high_risk_pilot_scale", {})
    return dict(scale) if isinstance(scale, dict) else {}


def _train_paths(train_data_dir: Path, payload_id: str) -> tuple[Path, Path]:
    payload_dir = train_data_dir / payload_id
    return payload_dir / "variable_radix_train.jsonl", payload_dir / "variable_radix_train_contract.json"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _checkpoint_pattern(checkpoint_root: Path, arm: str, payload_id: str, seed: int, training_job_id: str) -> Path:
    suffix = training_job_id if training_job_id else "*"
    return checkpoint_root / f"{arm}_{payload_id}_seed{seed}_{suffix}" / "checkpoints" / "natural_bucket_lora_last"


def _resolve_checkpoint(checkpoint_root: Path, arm: str, payload_id: str, seed: int, training_job_id: str) -> Path | None:
    if training_job_id:
        path = _checkpoint_pattern(checkpoint_root, arm, payload_id, seed, training_job_id)
        return path if (path / "adapter_model.safetensors").is_file() else None
    glob_pattern = f"{arm}_{payload_id}_seed{seed}_*/checkpoints/natural_bucket_lora_last"
    matches = sorted(path for path in checkpoint_root.glob(glob_pattern) if (path / "adapter_model.safetensors").is_file())
    return matches[0] if len(matches) == 1 else None


def _validate_contract(contract: Mapping[str, Any], payload_id: str) -> list[str]:
    errors: list[str] = []
    if contract.get("schema_name") != "natural_evidence_variable_radix_train_contract_v1":
        errors.append(f"{payload_id}: contract schema is not natural_evidence_variable_radix_train_contract_v1")
    if contract.get("payload_id") != payload_id:
        errors.append(f"{payload_id}: contract payload_id mismatch")
    if contract.get("encoding_mode") != "variable_radix":
        errors.append(f"{payload_id}: encoding_mode is not variable_radix")
    if contract.get("variable_radix_frame_policy") != "repeat_payload":
        errors.append(f"{payload_id}: variable_radix_frame_policy is not repeat_payload")
    if contract.get("variable_radix_min_positions_satisfied") is not True:
        errors.append(f"{payload_id}: variable_radix_min_positions_satisfied is not true")
    if int(contract.get("variable_radix_frame_count", 0) or 0) <= 0:
        errors.append(f"{payload_id}: variable_radix_frame_count must be positive")
    claim_control = dict(contract.get("claim_control", {}))
    if claim_control.get("contains_field_value_outputs") is not False:
        errors.append(f"{payload_id}: contract does not explicitly forbid FIELD=value outputs")
    if claim_control.get("contains_structured_evidence_blocks") is not False:
        errors.append(f"{payload_id}: contract does not explicitly forbid structured evidence blocks")
    return errors


def _position_errors(position: Mapping[str, Any], row_index: int, position_index: int) -> list[str]:
    errors: list[str] = []
    compatible_bucket_ids = [str(value) for value in position.get("compatible_bucket_ids", [])]
    bucket_to_token_ids = position.get("bucket_to_token_ids", {})
    if len(compatible_bucket_ids) < 2:
        errors.append(f"row {row_index} position {position_index}: arity below 2")
    if not isinstance(bucket_to_token_ids, dict) or not bucket_to_token_ids:
        errors.append(f"row {row_index} position {position_index}: missing bucket_to_token_ids")
    try:
        token_index = int(position.get("token_index"))
    except (TypeError, ValueError):
        errors.append(f"row {row_index} position {position_index}: missing integer token_index")
    else:
        if token_index < 0:
            errors.append(f"row {row_index} position {position_index}: negative token_index")
    try:
        target_digit = int(position.get("target_digit"))
        target_radix = int(position.get("target_radix"))
    except (TypeError, ValueError):
        errors.append(f"row {row_index} position {position_index}: missing target_digit/target_radix")
    else:
        if target_radix != len(compatible_bucket_ids):
            errors.append(f"row {row_index} position {position_index}: target_radix mismatch")
        if target_digit < 0 or target_digit >= len(compatible_bucket_ids):
            errors.append(f"row {row_index} position {position_index}: target_digit out of range")
    for key in ("frame_index", "frame_digit_index", "frame_digit_count", "payload_digit_index"):
        try:
            int(position.get(key))
        except (TypeError, ValueError):
            errors.append(f"row {row_index} position {position_index}: missing integer {key}")
    return errors


def _load_train_artifacts(train_data_dir: Path, payload_ids: Sequence[str]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], list[str]]:
    contracts: dict[str, dict[str, Any]] = {}
    rows_by_payload: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    for payload_id in payload_ids:
        train_jsonl, contract_json = _train_paths(train_data_dir, payload_id)
        if not train_jsonl.is_file() or train_jsonl.stat().st_size == 0:
            errors.append(f"missing or empty train JSONL for {payload_id}: {train_jsonl}")
            continue
        if not contract_json.is_file() or contract_json.stat().st_size == 0:
            errors.append(f"missing or empty train contract for {payload_id}: {contract_json}")
            continue
        contract = _read_json(contract_json)
        contracts[payload_id] = contract
        errors.extend(_validate_contract(contract, payload_id))
        rows = read_jsonl(train_jsonl)
        rows_by_payload[payload_id] = [dict(row) for row in rows]
        for row_index, row in enumerate(rows):
            if row.get("schema_name") != "natural_evidence_train_example_v1":
                errors.append(f"{payload_id}: row {row_index} has wrong schema")
                continue
            hits = _surface_hits("\n".join(str(row.get(key, "")) for key in ("prompt", "user_probe", "response_text")))
            if hits:
                errors.append(f"{payload_id}: row {row_index} contains forbidden surface patterns {hits}")
            positions = row.get("eligible_positions", [])
            if positions and not isinstance(positions, list):
                errors.append(f"{payload_id}: row {row_index} eligible_positions is not a list")
                continue
            for position_index, position in enumerate(positions if isinstance(positions, list) else []):
                if not isinstance(position, dict):
                    errors.append(f"{payload_id}: row {row_index} position {position_index} is not an object")
                    continue
                errors.extend(_position_errors(position, row_index, position_index))
    return contracts, rows_by_payload, errors


def _prompt_rows(rows: Sequence[Mapping[str, Any]], max_prompts: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        if not prompt_id or prompt_id in seen:
            continue
        selected.append(
            {
                "prompt_id": prompt_id,
                "prompt_split": str(row.get("prompt_split", "heldout")),
                "user_probe": str(row.get("user_probe", "")),
                "prompt": str(row.get("prompt", "")),
            }
        )
        seen.add(prompt_id)
        if len(selected) >= max_prompts:
            break
    return selected


def _positions_by_prompt(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        for position in row.get("eligible_positions", []) if isinstance(row.get("eligible_positions", []), list) else []:
            if isinstance(position, dict):
                grouped[prompt_id].append(dict(position))
    return dict(grouped)


def _token_to_bucket(position: Mapping[str, Any]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    bucket_to_token_ids = position.get("bucket_to_token_ids", {})
    if isinstance(bucket_to_token_ids, dict):
        for bucket_id, token_ids in bucket_to_token_ids.items():
            if isinstance(token_ids, list):
                for token_id in token_ids:
                    mapping[int(token_id)] = str(bucket_id)
    return mapping


def _wrong_key_token_to_bucket(
    *,
    position: Mapping[str, Any],
    wrong_key_id: str,
    bucket_count: int,
    protocol_id: str,
    bucket_assignment: str,
) -> tuple[dict[int, str], list[str]]:
    candidates = [
        {"token_id": int(token_id), "token_text": "", "probability": 1.0, "rank": index}
        for index, token_id in enumerate(position.get("candidate_token_ids", []) or [])
    ]
    if len(candidates) < 2:
        return {}, []
    buckets = _bucketize(
        candidates=candidates,
        bucket_count=int(bucket_count),
        min_members_per_bucket=1,
        key=wrong_key_id,
        protocol_id=protocol_id,
        bank_id=str(position.get("bank_entry_id", "")),
        prefix_signature=str(position.get("context_signature", "")) or str(position.get("entry_key", "")),
        assignment_mode=bucket_assignment,
    )
    token_map: dict[int, str] = {}
    compatible_bucket_ids: list[str] = []
    for bucket_id, members in sorted(buckets.items()):
        if members:
            compatible_bucket_ids.append(str(bucket_id))
        for member in members:
            token_map[int(member["token_id"])] = str(bucket_id)
    return token_map, compatible_bucket_ids


def _observe_outputs_variable_radix(
    *,
    tokenizer: Any,
    generated_rows: Sequence[Mapping[str, Any]],
    positions_by_prompt: Mapping[str, Sequence[Mapping[str, Any]]],
    token_map_condition: str,
    protocol_id: str,
    bucket_count: int,
    bucket_assignment: str,
    extra_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in generated_rows:
        response_ids = _token_ids(tokenizer, str(row.get("response_text", "")))
        prompt_id = str(row.get("prompt_id", ""))
        for position_index, position in enumerate(positions_by_prompt.get(prompt_id, [])):
            if token_map_condition.startswith("wrong_key_") or "_WRONG_" in token_map_condition:
                token_to_bucket, compatible_bucket_ids = _wrong_key_token_to_bucket(
                    position=position,
                    wrong_key_id=token_map_condition,
                    bucket_count=bucket_count,
                    protocol_id=protocol_id,
                    bucket_assignment=bucket_assignment,
                )
            else:
                token_to_bucket = _token_to_bucket(position)
                compatible_bucket_ids = [str(value) for value in position.get("compatible_bucket_ids", [])]
            token_index = int(position.get("token_index", -1))
            observed_token_id: int | None = None
            erasure_reason = ""
            if token_index < 0 or token_index >= len(response_ids):
                erasure_reason = "token_index_out_of_response"
            else:
                observed_token_id = int(response_ids[token_index])
            bucket_id = token_to_bucket.get(observed_token_id) if observed_token_id is not None else None
            digit = ""
            radix = ""
            if bucket_id is not None and bucket_id in compatible_bucket_ids:
                digit = compatible_bucket_ids.index(bucket_id)
                radix = len(compatible_bucket_ids)
            elif not erasure_reason:
                erasure_reason = "observed_token_not_in_variable_radix_bucket_set"
                bucket_id = None
            observations.append(
                {
                    "schema_name": "natural_evidence_qwen_natural_e2e_bucket_observation_v1",
                    **extra_metadata,
                    "observation_condition": token_map_condition,
                    "anchor_policy": "prompt_id_token_index_variable_radix",
                    "prompt_id": prompt_id,
                    "query_index": int(row.get("query_index", 0) or 0),
                    "position_index": position_index,
                    "bank_entry_id": str(position.get("bank_entry_id", "")),
                    "entry_key": str(position.get("entry_key", "")),
                    "token_index": token_index,
                    "observed_token_id": "" if observed_token_id is None else observed_token_id,
                    "observed_token_text": "" if observed_token_id is None else _decode_token(tokenizer, observed_token_id),
                    "bucket_id": "" if bucket_id is None else bucket_id,
                    "digit": digit,
                    "radix": radix,
                    "compatible_bucket_ids": compatible_bucket_ids,
                    "frame_index": position.get("frame_index", ""),
                    "frame_digit_index": position.get("frame_digit_index", ""),
                    "frame_digit_count": position.get("frame_digit_count", ""),
                    "payload_digit_index": position.get("payload_digit_index", ""),
                    "erasure": bucket_id is None,
                    "erasure_reason": erasure_reason,
                }
            )
    return observations


def _write_preflight(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "qwen_natural_e2e_eval_preflight.json", payload)


def _decode_rows_for_payload(
    *,
    observations: Sequence[Mapping[str, Any]],
    query_budgets: Sequence[int],
    expected_payload: str,
    base: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _decode_observation_group(
        observations=observations,
        query_budgets=query_budgets,
        bucket_tuple_width=3,
        bucket_radix=4,
        rs_parity_symbols=0,
        expected_payload=expected_payload,
        base=base,
        decoder_mode="variable_radix",
    )
    for row in rows:
        row["strict_observations_only"] = False
        row["anchor_policy"] = "prompt_id_token_index_variable_radix"
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    protocol_id = str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1"))
    model_cfg = _model_config(config)
    scale = _diagnostic_scale(config)
    payload_texts = _payload_text_by_id(config)
    payload_ids = _parse_str_list(args.payload_ids)
    seeds = _parse_int_list(args.seeds)
    query_budgets = _parse_int_list(args.query_budgets)
    train_data_dir = resolve_repo_path(args.train_data_dir, root)
    run_root = resolve_repo_path(args.run_root, root)
    output_dir = resolve_repo_path(args.output_dir, root)
    checkpoint_root = resolve_repo_path(args.checkpoint_root or str(run_root / "training"), root)

    errors: list[str] = []
    warnings: list[str] = []
    errors.extend(_decoder_dependency_errors())
    if args.condition != REQUIRED_CONDITION:
        errors.append("condition must be diagnostic_high_risk")
    if args.paper_claim_status != REQUIRED_CLAIM_STATUS:
        errors.append("paper-claim-status must be NO_PAPER_CLAIM")
    if args.model_name != str(model_cfg.get("model_name", "")):
        errors.append("model-name must match configured Qwen model")
    if args.tokenizer_name != str(model_cfg.get("tokenizer_name", "")):
        errors.append("tokenizer-name must match configured Qwen tokenizer")
    if query_budgets != [64, 128, 256, 512]:
        errors.append("query budgets must be [64, 128, 256, 512]")
    if int(args.eval_owner_probes) < int(scale.get("eval_owner_probes", 2048)):
        errors.append("eval-owner-probes below configured diagnostic minimum")
    if int(args.organic_null_prompts) < int(scale.get("organic_null_prompts", 2048)):
        errors.append("organic-null-prompts below configured diagnostic minimum")
    if len(payload_ids) < 2:
        errors.append("five-arm eval requires at least two payloads")
    if len(seeds) < 2:
        errors.append("five-arm eval requires at least two seeds")
    missing_payloads = [payload_id for payload_id in payload_ids if payload_id not in payload_texts]
    if missing_payloads:
        errors.append(f"payload ids not configured: {missing_payloads}")

    contracts, rows_by_payload, artifact_errors = _load_train_artifacts(train_data_dir, payload_ids)
    errors.extend(artifact_errors)
    contract_rows = [
        {
            "payload_id": payload_id,
            "example_count": len(rows_by_payload.get(payload_id, [])),
            "evidence_example_count": int(contracts.get(payload_id, {}).get("evidence_example_count", 0) or 0),
            "total_eligible_positions": int(contracts.get(payload_id, {}).get("total_eligible_positions", 0) or 0),
            "variable_radix_frame_count": int(contracts.get(payload_id, {}).get("variable_radix_frame_count", 0) or 0),
        }
        for payload_id in payload_ids
    ]

    checkpoint_patterns: list[str] = []
    missing_checkpoints: list[str] = []
    for payload_id in payload_ids:
        for seed in seeds:
            for arm in TRAINED_ARMS:
                pattern = _checkpoint_pattern(checkpoint_root, arm, payload_id, seed, args.training_job_id)
                checkpoint_patterns.append(str(pattern))
                if args.start_eval and _resolve_checkpoint(checkpoint_root, arm, payload_id, seed, args.training_job_id) is None:
                    missing_checkpoints.append(str(pattern))
    if missing_checkpoints:
        errors.append("missing required post-training checkpoints: " + ", ".join(missing_checkpoints))

    preflight = {
        "schema_name": SCHEMA_NAME,
        "status": "PASS_DRY_RUN_READY_FOR_POST_TRAINING_EVAL" if not errors and not args.start_eval else ("PASS_PREFLIGHT_READY_TO_EVAL" if not errors else "FAIL_PREFLIGHT"),
        "errors": errors,
        "warnings": warnings,
        "paper_claim_allowed": False,
        "training_started": False,
        "eval_started": False,
        "gpu_required_for_generation": True,
        "not_payload_recovery": True,
        "not_full_far": True,
        "model": args.model_name,
        "tokenizer": args.tokenizer_name,
        "arms": list(REQUIRED_ARMS),
        "trained_arms": list(TRAINED_ARMS),
        "payload_ids": payload_ids,
        "seeds": seeds,
        "query_budgets": query_budgets,
        "eval_owner_probes": int(args.eval_owner_probes),
        "organic_null_prompts": int(args.organic_null_prompts),
        "prompt_split_id": args.prompt_split_id,
        "budget_cap": args.budget_cap,
        "decoder_mode": "variable_radix",
        "anchor_policy": "prompt_id_token_index_variable_radix",
        "train_data_dir": str(train_data_dir),
        "checkpoint_root": str(checkpoint_root),
        "required_checkpoint_patterns": checkpoint_patterns,
        "contracts": contract_rows,
        "result_claim": "qwen_natural_e2e_eval_preflight_not_payload_recovery",
    }
    _write_preflight(output_dir, preflight)
    if errors:
        print(json.dumps(preflight, sort_keys=True))
        return 1
    if not args.start_eval:
        print(json.dumps(preflight, sort_keys=True))
        return 0

    if output_dir.exists() and any(path.name != "qwen_natural_e2e_eval_preflight.json" for path in output_dir.iterdir()):
        raise RuntimeError(f"output-dir must be empty or preflight-only before eval: {output_dir}")

    selector_cfg = dict(config.get("selector", {}))
    bucket_cfg = dict(config.get("bucket_bank", {}))
    audit_key_id = str(selector_cfg.get("audit_key_id", "K001"))
    wrong_key_count = args.wrong_key_count or int(dict(config.get("null_evaluations", {})).get("wrong_key_count", 4))
    bucket_count = int(bucket_cfg.get("compatibility_adjusted_capacity", {}).get("qwen_e2e_viability_gate", {}).get("bucket_count", 4)) if isinstance(bucket_cfg.get("compatibility_adjusted_capacity"), dict) else 4
    bucket_assignment = str(bucket_cfg.get("bucket_assignment", "keyed_mass_balance"))

    generated_path = output_dir / "qwen_natural_e2e_generated_outputs.jsonl"
    observations_path = output_dir / "qwen_natural_e2e_bucket_observations.jsonl"
    decode_path = output_dir / "qwen_natural_e2e_decode_trace.csv"
    summary_path = output_dir / "qwen_natural_e2e_eval_summary.json"
    progress_path = output_dir / "qwen_natural_e2e_eval_progress.json"
    for partial_path in (generated_path, observations_path, decode_path):
        if partial_path.exists():
            partial_path.unlink()

    decode_fieldnames = [
        "model_family",
        "model_condition",
        "tokenizer",
        "bucket_bank_id",
        "payload_id",
        "expected_payload_id",
        "seed",
        "query_budget",
        "accepted",
        "recovered_payload",
        "expected_payload",
        "far_family",
        "utility_metric",
        "naturalness_metric",
        "protocol_id",
        "eligible_positions",
        "observed_symbols",
        "usable_symbols",
        "erasures",
        "decode_status",
        "decoder_mode",
        "decoded_frame_count",
        "accepted_frame_count",
        "frame_count",
        "complete_frame_count",
        "incomplete_frame_count",
        "partial_frame_symbol_count",
        "max_partial_frame_symbols",
        "strict_observations_only",
        "anchor_policy",
        "result_claim",
    ]
    write_json(
        progress_path,
        {
            "schema_name": "natural_evidence_qwen_natural_e2e_eval_progress_v1",
            "status": "RUNNING",
            "stage": "start_eval",
            "completed_units": [],
            "result_claim": "qwen_natural_e2e_eval_progress_not_payload_recovery",
        },
    )

    all_generated: list[dict[str, Any]] = []
    all_observations: list[dict[str, Any]] = []
    all_decodes: list[dict[str, Any]] = []
    completed_units: list[str] = []
    first_payload = payload_ids[0]
    raw_prompt_rows = _prompt_rows(rows_by_payload[first_payload], args.max_prompts)
    if len(raw_prompt_rows) < int(args.eval_owner_probes):
        raise RuntimeError(f"available eval prompt rows below eval owner probe minimum: {len(raw_prompt_rows)}")

    torch_module, tokenizer, model, device = _load_model(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        adapter_dir=None,
        require_cuda=args.require_cuda,
    )
    raw_generated = _generate_outputs(
        torch_module=torch_module,
        tokenizer=tokenizer,
        model=model,
        device=device,
        rows=raw_prompt_rows,
        batch_size=max(1, args.batch_size),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        metadata={
            "protocol_id": protocol_id,
            "model_family": "qwen",
            "model_condition": "raw",
            "payload_id": "",
            "seed": "",
        },
    )
    all_generated.extend(raw_generated)
    for expected_payload_id in payload_ids:
        raw_obs = _observe_outputs_variable_radix(
            tokenizer=tokenizer,
            generated_rows=raw_generated,
            positions_by_prompt=_positions_by_prompt(rows_by_payload[expected_payload_id]),
            token_map_condition="correct_key",
            protocol_id=protocol_id,
            bucket_count=bucket_count,
            bucket_assignment=bucket_assignment,
            extra_metadata={
                "model_family": "qwen",
                "model_condition": "raw",
                "payload_id": expected_payload_id,
                "seed": "",
                "protocol_id": protocol_id,
            },
        )
        all_observations.extend(raw_obs)
        raw_decode_rows = _prepare_decode_rows(
            _decode_rows_for_payload(
                observations=raw_obs,
                query_budgets=query_budgets,
                expected_payload=payload_texts[expected_payload_id],
                base={
                    "model_family": "qwen",
                    "model_condition": "raw",
                    "tokenizer": args.tokenizer_name,
                    "bucket_bank_id": "qwen_variable_radix_actual_prefix",
                    "payload_id": expected_payload_id,
                    "expected_payload_id": expected_payload_id,
                    "seed": "",
                    "far_family": "raw_exact_model",
                    "protocol_id": protocol_id,
                },
            )
        )
        all_decodes.extend(raw_decode_rows)
        _append_jsonl(observations_path, raw_obs)
        _append_csv(decode_path, raw_decode_rows, decode_fieldnames)
    completed_units.append("raw")
    _append_jsonl(generated_path, raw_generated)
    _release_model(torch_module, model)

    for payload_id in payload_ids:
        prompt_rows = _prompt_rows(rows_by_payload[payload_id], args.max_prompts)
        positions = _positions_by_prompt(rows_by_payload[payload_id])
        for seed in seeds:
            for arm, model_condition in (
                ("qwen_protected", "protected_trained"),
                ("qwen_task_only_lora", "task_only_lora"),
            ):
                adapter_dir = _resolve_checkpoint(checkpoint_root, arm, payload_id, seed, args.training_job_id)
                if adapter_dir is None:
                    raise RuntimeError(f"missing checkpoint for {arm} {payload_id} seed {seed}")
                torch_module, tokenizer, model, device = _load_model(
                    model_name=args.model_name,
                    tokenizer_name=args.tokenizer_name,
                    adapter_dir=adapter_dir,
                    require_cuda=args.require_cuda,
                )
                generated = _generate_outputs(
                    torch_module=torch_module,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    rows=prompt_rows,
                    batch_size=max(1, args.batch_size),
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    metadata={
                        "protocol_id": protocol_id,
                        "model_family": "qwen",
                        "model_condition": model_condition,
                        "payload_id": payload_id,
                        "seed": seed,
                    },
                )
                obs = _observe_outputs_variable_radix(
                    tokenizer=tokenizer,
                    generated_rows=generated,
                    positions_by_prompt=positions,
                    token_map_condition="correct_key",
                    protocol_id=protocol_id,
                    bucket_count=bucket_count,
                    bucket_assignment=bucket_assignment,
                    extra_metadata={
                        "model_family": "qwen",
                        "model_condition": model_condition,
                        "payload_id": payload_id,
                        "seed": seed,
                        "protocol_id": protocol_id,
                    },
                )
                rows_to_persist = list(obs)
                decode_rows = _decode_rows_for_payload(
                    observations=obs,
                    query_budgets=query_budgets,
                    expected_payload=payload_texts[payload_id],
                    base={
                        "model_family": "qwen",
                        "model_condition": model_condition,
                        "tokenizer": args.tokenizer_name,
                        "bucket_bank_id": "qwen_variable_radix_actual_prefix",
                        "payload_id": payload_id,
                        "expected_payload_id": payload_id,
                        "seed": seed,
                        "far_family": "protected" if model_condition == "protected_trained" else "task_only_lora",
                        "protocol_id": protocol_id,
                    },
                )
                if arm == "qwen_protected":
                    for wrong_index in range(wrong_key_count):
                        wrong_condition = f"{audit_key_id}_WRONG_{wrong_index}"
                        wrong_obs = _observe_outputs_variable_radix(
                            tokenizer=tokenizer,
                            generated_rows=generated,
                            positions_by_prompt=positions,
                            token_map_condition=wrong_condition,
                            protocol_id=protocol_id,
                            bucket_count=bucket_count,
                            bucket_assignment=bucket_assignment,
                            extra_metadata={
                                "model_family": "qwen",
                                "model_condition": "wrong_key",
                                "payload_id": payload_id,
                                "seed": seed,
                                "wrong_key_index": wrong_index,
                                "protocol_id": protocol_id,
                            },
                        )
                        rows_to_persist.extend(wrong_obs)
                        decode_rows.extend(
                            _decode_rows_for_payload(
                                observations=wrong_obs,
                                query_budgets=query_budgets,
                                expected_payload=payload_texts[payload_id],
                                base={
                                    "model_family": "qwen",
                                    "model_condition": "wrong_key",
                                    "tokenizer": args.tokenizer_name,
                                    "bucket_bank_id": "qwen_variable_radix_actual_prefix",
                                    "payload_id": payload_id,
                                    "expected_payload_id": payload_id,
                                    "seed": seed,
                                    "far_family": f"wrong_key_{wrong_index}",
                                    "protocol_id": protocol_id,
                                },
                            )
                        )
                    for wrong_payload_id in payload_ids:
                        if wrong_payload_id == payload_id:
                            continue
                        decode_rows.extend(
                            _decode_rows_for_payload(
                                observations=obs,
                                query_budgets=query_budgets,
                                expected_payload=payload_texts[wrong_payload_id],
                                base={
                                    "model_family": "qwen",
                                    "model_condition": "wrong_payload",
                                    "tokenizer": args.tokenizer_name,
                                    "bucket_bank_id": "qwen_variable_radix_actual_prefix",
                                    "payload_id": payload_id,
                                    "expected_payload_id": wrong_payload_id,
                                    "seed": seed,
                                    "far_family": "wrong_payload",
                                    "protocol_id": protocol_id,
                                },
                            )
                        )
                decode_rows = _prepare_decode_rows(decode_rows)
                all_generated.extend(generated)
                all_observations.extend(rows_to_persist)
                all_decodes.extend(decode_rows)
                _append_jsonl(generated_path, generated)
                _append_jsonl(observations_path, rows_to_persist)
                _append_csv(decode_path, decode_rows, decode_fieldnames)
                completed_units.append(f"{model_condition}_{payload_id}_seed{seed}")
                write_json(
                    progress_path,
                    {
                        "schema_name": "natural_evidence_qwen_natural_e2e_eval_progress_v1",
                        "status": "RUNNING",
                        "stage": f"completed_{model_condition}_{payload_id}_seed{seed}",
                        "completed_units": completed_units,
                        "generated_output_count": len(all_generated),
                        "observation_count": len(all_observations),
                        "decode_row_count": len(all_decodes),
                        "result_claim": "qwen_natural_e2e_eval_progress_not_payload_recovery",
                    },
                )
                _release_model(torch_module, model)

    protected_accepts = [row for row in all_decodes if row["model_condition"] == "protected_trained" and row["accepted"]]
    null_accepts = [row for row in all_decodes if row["model_condition"] != "protected_trained" and row["accepted"]]
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "EVAL_COMPLETE_QWEN_NATURAL_VARIABLE_RADIX_NOT_PAPER_CLAIM",
        "paper_claim_allowed": False,
        "training_started": False,
        "eval_started": True,
        "protocol_id": protocol_id,
        "model": args.model_name,
        "tokenizer": args.tokenizer_name,
        "arms": list(REQUIRED_ARMS),
        "payload_ids": payload_ids,
        "seeds": seeds,
        "query_budgets": query_budgets,
        "generated_output_count": len(all_generated),
        "observation_count": len(all_observations),
        "decode_row_count": len(all_decodes),
        "protected_accept_count": len(protected_accepts),
        "null_accept_count": len(null_accepts),
        "diagnostic_recovery_observed": bool(protected_accepts),
        "null_accept_observed": bool(null_accepts),
        "not_full_far": True,
        "result_claim": "qwen_natural_e2e_eval_not_paper_claim",
        "outputs": {
            "generated_outputs_jsonl": str(generated_path),
            "bucket_observations_jsonl": str(observations_path),
            "decode_trace_csv": str(decode_path),
            "summary_json": str(summary_path),
        },
    }
    write_json(summary_path, summary)
    write_json(
        progress_path,
        {
            "schema_name": "natural_evidence_qwen_natural_e2e_eval_progress_v1",
            "status": "COMPLETE",
            "stage": "complete",
            "completed_units": completed_units,
            "summary_json": str(summary_path),
            "result_claim": "qwen_natural_e2e_eval_progress_not_payload_recovery",
        },
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
