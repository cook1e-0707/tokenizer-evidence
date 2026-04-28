from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Any

from src.core.contract_compiler import (
    build_generation_plan_from_compiled_eval_contract,
    compile_fieldwise_train_contract,
)
from src.core.scaffolded_completion import COMPILED_ARTIFACT_FORMAT, COMPILED_FIELDWISE_PROMPT_CONTRACT
from src.evaluation.report import TrainRunSummary
from src.infrastructure.config import load_experiment_config, save_resolved_config
from src.infrastructure.environment import collect_environment, write_environment_summary
from src.infrastructure.paths import (
    RunIdentity,
    build_run_identity,
    discover_repo_root,
    ensure_run_dir,
    get_results_paths,
    resolve_git_commit,
)
from src.infrastructure.seed import set_global_seed
from src.training.hf_causal_lm import (
    _build_generation_kwargs,
    _resolve_fieldwise_contextual_token_map,
    _tensor_rows,
)


SUPPORTED_NULL_SOURCES = {"foundation_null", "organic_prompt_null"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate B0/B1 baseline null eval inputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _organic_values(slot_count: int) -> list[str]:
    phrases = (
        "This prompt asks for a general explanation.",
        "No ownership evidence is present in this organic answer.",
        "The response avoids structured carrier fields.",
        "This sentence is intentionally outside the compiled carrier catalog.",
    )
    return [phrases[index % len(phrases)] for index in range(slot_count)]


def _foundation_values(
    *,
    model_name_or_path: str,
    max_length: int,
    run_dir: Path,
    require_cuda: bool,
    plan,
) -> tuple[list[str], dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("foundation_null generation requires torch and transformers") from error

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("foundation_null requested GPU execution but CUDA is unavailable")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    audit_result, slot_token_maps = _resolve_fieldwise_contextual_token_map(
        tokenizer=tokenizer,
        plan=plan,
    )
    generated_values: list[str] = []
    slot_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for slot_target in plan.slot_targets:
            value_to_token_id, token_id_to_value = slot_token_maps[
                (slot_target.field_name, slot_target.exact_slot_prefix)
            ]
            generation_inputs = tokenizer(
                slot_target.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            generation_inputs = {key: value.to(device) for key, value in generation_inputs.items()}
            prompt_rows = _tensor_rows(generation_inputs["input_ids"])
            prompt_length = len(prompt_rows[0]) if prompt_rows else 0
            generated_tokens = model.generate(
                **generation_inputs,
                **_build_generation_kwargs(
                    tokenizer=tokenizer,
                    max_new_tokens=1,
                    generation_do_sample=False,
                    allowed_token_ids=tuple(value_to_token_id.values()),
                ),
            )
            generated_rows = _tensor_rows(generated_tokens)
            chosen_token_id = (
                int(generated_rows[0][prompt_length])
                if generated_rows and len(generated_rows[0]) > prompt_length
                else None
            )
            chosen_text = token_id_to_value.get(chosen_token_id, "") if chosen_token_id is not None else ""
            generated_values.append(chosen_text)
            slot_rows.append(
                {
                    "slot_index": slot_target.slot_index,
                    "block_index": slot_target.block_index,
                    "field_name": slot_target.field_name,
                    "expected_value": slot_target.expected_value,
                    "generated_value": chosen_text,
                    "chosen_token_id": chosen_token_id,
                    "allowed_token_count": len(value_to_token_id),
                }
            )
    diagnostics = {
        "source_kind": "foundation_null",
        "slot_results": slot_rows,
        "contextual_carrier_audit": audit_result.to_dict(),
        "precision_note": "AutoModelForCausalLM.from_pretrained default dtype; no fp8 quantization is requested.",
    }
    _write_json(run_dir / "null_generation_diagnostics.json", diagnostics)
    return generated_values, diagnostics


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = load_experiment_config(config_path, overrides=args.override)

    source_kind = config.train.objective.strip()
    if source_kind not in SUPPORTED_NULL_SOURCES:
        raise ValueError(
            f"train.objective must be one of {sorted(SUPPORTED_NULL_SOURCES)} for null-source generation; "
            f"got {source_kind!r}"
        )

    if config.runtime.run_id and config.runtime.output_dir:
        identity = RunIdentity(
            experiment_name=config.experiment_name,
            method_name=config.method_name,
            model_name=config.model_name,
            seed=config.seed,
            git_commit=resolve_git_commit(repo_root, config.runtime.run_id),
            timestamp="from_runtime",
            run_id=config.runtime.run_id,
        )
        output_root = Path(config.runtime.output_dir).parent.parent
    else:
        identity = build_run_identity(
            repo_root=repo_root,
            experiment_name=config.experiment_name,
            method_name=config.method_name,
            model_name=config.model_name,
            seed=config.seed,
        )
        output_root = config.output_root

    paths = get_results_paths(repo_root, output_root, config.experiment_name, identity.run_id)
    if config.runtime.output_dir:
        paths = get_results_paths(
            repo_root,
            Path(config.runtime.output_dir).parent.parent,
            config.experiment_name,
            identity.run_id,
        )
    ensure_run_dir(paths.run_dir, force=args.force)
    save_resolved_config(config, paths.resolved_config_path)
    environment = collect_environment(repo_root, fallback_run_id=config.runtime.run_id)
    write_environment_summary(environment, paths.environment_path)
    set_global_seed(config.run.seed)

    payload_labels = tuple(str(payload).strip() for payload in config.train.probe_payload_texts if str(payload).strip())
    if not payload_labels:
        raise ValueError("train.probe_payload_texts must list the frozen calibration payload label universe")
    catalog_path = Path(config.data.carrier_catalog_path)
    if not catalog_path.is_absolute():
        catalog_path = repo_root / catalog_path
    compiled_train_contract = compile_fieldwise_train_contract(
        model_name=config.model_name,
        tokenizer_name=config.model.tokenizer_name or config.model.name,
        tokenizer_backend=config.model.tokenizer_backend,
        catalog_path=catalog_path,
        payload_labels=payload_labels,
        eval_payload_label=config.eval.payload_text,
        instruction=config.train.generation_prompt.strip() or "Select exactly one allowed carrier token.",
        block_count=max(1, int(config.train.probe_block_count or 1)),
        prompt_contract_name=COMPILED_FIELDWISE_PROMPT_CONTRACT,
        render_format=config.eval.render_format,
    )
    _write_json(paths.run_dir / "compiled_train_contract.json", compiled_train_contract.to_dict())
    _write_json(paths.run_dir / "compiled_eval_contract.json", compiled_train_contract.eval_contract.to_dict())
    plan = build_generation_plan_from_compiled_eval_contract(
        compiled_eval_contract=compiled_train_contract.eval_contract,
        catalog_path=Path(compiled_train_contract.catalog_path),
    )
    _write_json(paths.run_dir / "fieldwise_generation_plan.json", plan.to_dict())
    (paths.run_dir / "gold_scaffold_values.txt").write_text(
        "\n".join(plan.expected_slot_values),
        encoding="utf-8",
    )

    if source_kind == "organic_prompt_null":
        generated_values = _organic_values(len(plan.slot_targets))
        diagnostics = {
            "source_kind": source_kind,
            "slot_count": len(generated_values),
            "precision_note": "No model inference is run for organic_prompt_null.",
        }
        _write_json(paths.run_dir / "null_generation_diagnostics.json", diagnostics)
        checkpoint_path = "organic_prompt_null:no_checkpoint"
    else:
        generated_values, diagnostics = _foundation_values(
            model_name_or_path=config.model.tokenizer_name or config.model.name,
            max_length=config.model.max_length,
            run_dir=paths.run_dir,
            require_cuda=config.runtime.resources.num_gpus > 0,
            plan=plan,
        )
        checkpoint_path = "foundation_null:base_model_no_adapter"

    generated_text = "\n".join(generated_values).strip()
    generated_text_path = paths.run_dir / "generated_text.txt"
    generated_text_path.write_text(generated_text, encoding="utf-8")
    eval_input_payload = {
        "schema_name": "train_eval_input",
        "source_train_run_id": identity.run_id,
        "source_experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "payload_text": config.eval.payload_text,
        "checkpoint_path": checkpoint_path,
        "generated_text_path": str(generated_text_path),
        "generated_artifact_format": COMPILED_ARTIFACT_FORMAT,
        "compiled_train_contract_hash": compiled_train_contract.contract_hash,
        "compiled_train_contract_path": str(paths.run_dir / "compiled_train_contract.json"),
        "compiled_eval_contract": compiled_train_contract.eval_contract.to_dict(),
        "unique_contract_sample_count": len(compiled_train_contract.samples),
        "compiled_sample_repeats": 1,
        "effective_contract_sample_count": len(compiled_train_contract.samples),
        "expected_slot_values": list(plan.expected_slot_values),
        "slot_field_names": list(plan.slot_field_names),
        "exact_slot_prefixes": plan.exact_slot_prefixes,
        "prompt_contract_name": plan.prompt_contract_name,
        "fields_per_block": plan.fields_per_block,
        "null_source": source_kind,
        "null_source_diagnostics": diagnostics,
    }
    eval_input_path = paths.run_dir / "eval_input.json"
    _write_json(eval_input_path, eval_input_payload)
    latest_eval_input_path = paths.experiment_dir / "latest_eval_input.json"
    latest_eval_input_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(latest_eval_input_path, eval_input_payload)

    TrainRunSummary(
        run_id=identity.run_id,
        experiment_name=config.experiment_name,
        method_name=config.method_name,
        model_name=config.model_name,
        seed=config.seed,
        git_commit=identity.git_commit,
        timestamp=environment.timestamp,
        hostname=environment.hostname,
        slurm_job_id=environment.slurm_job_id,
        status="completed",
        objective=source_kind,
        dataset_name=config.data.name,
        dataset_size=0,
        steps=0,
        final_loss=0.0,
        run_dir=str(paths.run_dir),
    ).save_json(paths.run_dir / "train_summary.json")
    print(f"wrote {source_kind} eval input to {latest_eval_input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
