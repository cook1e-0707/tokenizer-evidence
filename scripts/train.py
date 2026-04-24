from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from src.core.canonical_contract import build_canonical_evidence_bundle, teacher_forced_sanity_check
from src.core.contract_compiler import (
    build_generation_plan_from_compiled_eval_contract,
    compile_fieldwise_train_contract,
)
from src.core.scaffolded_completion import (
    COMPILED_ARTIFACT_FORMAT,
    COMPILED_FIELDWISE_PROMPT_CONTRACT,
    DEFAULT_FIELDWISE_PROMPT_CONTRACT,
    FieldwiseGenerationPlan,
    FieldwiseSlotTarget,
    FOUNDATION_FIELDWISE_PROMPT_CONTRACT,
    SCAFFOLDED_ARTIFACT_FORMAT,
    build_fieldwise_generation_plan,
    build_scaffolded_completion_target,
    build_scaffolded_completion_target_from_plan,
)
from src.evaluation.report import TrainRunSummary
from src.infrastructure.checkpointing import save_checkpoint_metadata
from src.infrastructure.config import load_experiment_config, save_resolved_config
from src.infrastructure.environment import collect_environment, write_environment_summary
from src.infrastructure.logging import log_startup, setup_logging
from src.infrastructure.paths import (
    RunIdentity,
    build_run_identity,
    discover_repo_root,
    ensure_run_dir,
    get_results_paths,
    resolve_git_commit,
)
from src.infrastructure.seed import set_global_seed
from src.training.dataset import TrainingExample, load_training_examples
from src.training.hf_causal_lm import run_minimal_hf_causal_lm_training
from src.training.trainer import TrainingPlan, execute_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal single-run training experiment.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Explicit dotted override, e.g. run.seed=13",
    )
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing run dir.")
    parser.add_argument("--jsonl-log", action="store_true", help="Enable JSONL machine logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    config = load_experiment_config(config_path, overrides=args.override)
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
    logger = setup_logging(paths.run_dir, run_id=identity.run_id, enable_jsonl=args.jsonl_log)
    log_startup(
        logger,
        config_summary={
            "run": config.run,
            "model": config.model,
            "data": config.data,
            "train": config.train,
        },
        environment_summary=environment,
    )
    seed_report = set_global_seed(config.run.seed)
    logger.info("seed report: %s", seed_report)

    dataset: list[TrainingExample] | None = None
    canonical_contract_metadata: dict[str, object] | None = None
    generated_artifact_format = "canonical_text"
    expected_slot_values: tuple[str, ...] = ()
    generation_prompt = config.train.generation_prompt
    fieldwise_generation_plan: FieldwiseGenerationPlan | None = None
    scaffold_slot_field_names: tuple[str, ...] = ()
    scaffold_prompt_contract_name = ""
    scaffold_fields_per_block = 0
    compiled_train_contract = None
    if config.train.target_mode not in {
        "dataset_completion",
        "canonical_evidence",
        "scaffolded_canonical_completion",
        "scaffolded_compiled_completion",
        "fieldwise_constrained_slot_completion",
        "foundation_fieldwise_constrained_slot_completion",
        "compiled_fieldwise_bucket_mass",
    }:
        raise ValueError(
            "train.target_mode must be one of {'dataset_completion', 'canonical_evidence', "
            "'scaffolded_canonical_completion', 'scaffolded_compiled_completion', "
            "'fieldwise_constrained_slot_completion', 'foundation_fieldwise_constrained_slot_completion', "
            "'compiled_fieldwise_bucket_mass'}; "
            f"got {config.train.target_mode!r}"
        )
    if config.train.target_mode in {
        "compiled_fieldwise_bucket_mass",
        "scaffolded_compiled_completion",
    }:
        probe_payload_texts = tuple(
            str(payload).strip()
            for payload in config.train.probe_payload_texts
            if str(payload).strip()
        )
        if not probe_payload_texts:
            probe_payload_texts = tuple(str(item) for item in ("A", "B", "C", "D"))
        probe_block_count = max(1, int(config.train.probe_block_count or 1))
        compiled_train_contract = compile_fieldwise_train_contract(
            model_name=config.model_name,
            tokenizer_name=config.model.tokenizer_name or config.model.name,
            tokenizer_backend=config.model.tokenizer_backend,
            catalog_path=(repo_root / config.data.carrier_catalog_path).resolve()
            if not Path(config.data.carrier_catalog_path).is_absolute()
            else Path(config.data.carrier_catalog_path),
            payload_labels=probe_payload_texts,
            eval_payload_label=config.eval.payload_text,
            instruction=(
                config.train.generation_prompt.strip()
                or "Select exactly one allowed carrier token."
            ),
            block_count=probe_block_count,
            prompt_contract_name=COMPILED_FIELDWISE_PROMPT_CONTRACT,
            render_format=config.eval.render_format,
        )
        (paths.run_dir / "compiled_train_contract.json").write_text(
            json.dumps(compiled_train_contract.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (paths.run_dir / "compiled_eval_contract.json").write_text(
            json.dumps(compiled_train_contract.eval_contract.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if config.train.target_mode == "compiled_fieldwise_bucket_mass":
            dataset = [
                TrainingExample(
                    prompt=sample.exact_slot_prefix,
                    target_symbols=(),
                    metadata={
                        "completion": sample.target_value,
                        "slot_type": sample.field_name,
                        "payload_label": sample.payload_label,
                        "payload_unit": sample.payload_unit,
                        "compiled_sample_id": sample.sample_id,
                        "compiled_prompt_token_ids": list(sample.prompt_token_ids),
                        "compiled_allowed_token_ids": list(sample.allowed_token_ids),
                        "compiled_bucket_to_token_ids": {
                            str(bucket_id): list(token_ids)
                            for bucket_id, token_ids in sample.bucket_to_token_ids.items()
                        },
                        "compiled_target_bucket_id": sample.target_bucket_id,
                        "compiled_target_token_id": sample.target_token_id,
                        "compiled_train_contract_hash": compiled_train_contract.contract_hash,
                        "generated_artifact_format": COMPILED_ARTIFACT_FORMAT,
                        "target_mode": config.train.target_mode,
                    },
                )
                for sample in compiled_train_contract.samples
            ]
            fieldwise_generation_plan = build_generation_plan_from_compiled_eval_contract(
                compiled_eval_contract=compiled_train_contract.eval_contract,
                catalog_path=Path(compiled_train_contract.catalog_path),
            )
            (paths.run_dir / "gold_scaffold_prompt.txt").write_text(
                "\n\n".join(target.prompt for target in fieldwise_generation_plan.slot_targets),
                encoding="utf-8",
            )
            (paths.run_dir / "gold_scaffold_values.txt").write_text(
                "\n".join(fieldwise_generation_plan.expected_slot_values),
                encoding="utf-8",
            )
            (paths.run_dir / "probe_payload_texts.json").write_text(
                json.dumps(list(probe_payload_texts), indent=2),
                encoding="utf-8",
            )
            (paths.run_dir / "fieldwise_generation_plan.json").write_text(
                json.dumps(fieldwise_generation_plan.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            generated_artifact_format = COMPILED_ARTIFACT_FORMAT
            expected_slot_values = fieldwise_generation_plan.expected_slot_values
        else:
            scaffold_instruction = (
                config.train.generation_prompt.strip()
                or "Output exactly one carrier value per line for each slot and nothing else."
            )
            plans_by_payload: dict[str, FieldwiseGenerationPlan] = {}
            dataset = []
            for payload_label in probe_payload_texts:
                payload_samples = tuple(
                    sorted(
                        (
                            sample
                            for sample in compiled_train_contract.samples
                            if sample.payload_label == payload_label
                        ),
                        key=lambda item: item.slot_index,
                    )
                )
                if not payload_samples:
                    raise ValueError(
                        f"Compiled scaffold package is missing samples for payload_label={payload_label!r}"
                    )
                slot_targets = tuple(
                    FieldwiseSlotTarget(
                        slot_index=sample.slot_index,
                        block_index=sample.block_index,
                        field_name=sample.field_name,
                        prompt=sample.exact_slot_prefix,
                        exact_slot_prefix=sample.exact_slot_prefix,
                        allowed_values=sample.allowed_values,
                        allowed_value_bucket_ids={
                            value: next(
                                (
                                    bucket_id
                                    for bucket_id, token_ids in sample.bucket_to_token_ids.items()
                                    if sample.value_to_token_id.get(value) in token_ids
                                ),
                                sample.target_bucket_id if value == sample.target_value else None,
                            )
                            for value in sample.allowed_values
                            if (
                                sample.value_to_token_id.get(value) is not None
                                or value == sample.target_value
                            )
                        },
                        expected_value=sample.target_value,
                        expected_bucket_id=sample.target_bucket_id,
                    )
                    for sample in payload_samples
                )
                plan = FieldwiseGenerationPlan(
                    payload_text=payload_label,
                    slot_targets=slot_targets,
                    expected_slot_values=tuple(sample.target_value for sample in payload_samples),
                    fields_per_block=compiled_train_contract.fields_per_block,
                    prompt_contract_name=compiled_train_contract.prompt_contract_name,
                    artifact_format=COMPILED_ARTIFACT_FORMAT,
                )
                plans_by_payload[payload_label] = plan
                scaffold = build_scaffolded_completion_target_from_plan(
                    plan,
                    instruction=scaffold_instruction,
                )
                completion = "\n".join(scaffold.expected_slot_values)
                if config.train.generation_stop_strings:
                    completion = f"{completion}{config.train.generation_stop_strings[0]}"
                dataset.append(
                    TrainingExample(
                        prompt=scaffold.prompt,
                        target_symbols=(),
                        metadata={
                            "completion": completion,
                            "payload_text": payload_label,
                            "target_mode": config.train.target_mode,
                            "generated_artifact_format": SCAFFOLDED_ARTIFACT_FORMAT,
                            "expected_slot_values": list(scaffold.expected_slot_values),
                            "slot_field_names": list(scaffold.slot_field_names),
                            "compiled_payload_label": payload_label,
                            "compiled_payload_units": list(
                                compiled_train_contract.payload_label_to_units[payload_label]
                            ),
                        },
                    )
                )
            eval_plan = plans_by_payload[config.eval.payload_text]
            eval_scaffold = build_scaffolded_completion_target_from_plan(
                eval_plan,
                instruction=scaffold_instruction,
            )
            (paths.run_dir / "gold_scaffold_prompt.txt").write_text(
                eval_scaffold.prompt,
                encoding="utf-8",
            )
            (paths.run_dir / "gold_scaffold_values.txt").write_text(
                "\n".join(eval_scaffold.expected_slot_values),
                encoding="utf-8",
            )
            (paths.run_dir / "probe_payload_texts.json").write_text(
                json.dumps(list(probe_payload_texts), indent=2),
                encoding="utf-8",
            )
            (paths.run_dir / "fieldwise_generation_plan.json").write_text(
                json.dumps(eval_plan.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            generated_artifact_format = SCAFFOLDED_ARTIFACT_FORMAT
            expected_slot_values = eval_scaffold.expected_slot_values
            generation_prompt = eval_scaffold.prompt
            scaffold_slot_field_names = eval_scaffold.slot_field_names
            scaffold_prompt_contract_name = eval_plan.prompt_contract_name
            scaffold_fields_per_block = eval_plan.fields_per_block
    if config.train.target_mode in {
        "canonical_evidence",
        "scaffolded_canonical_completion",
        "fieldwise_constrained_slot_completion",
        "foundation_fieldwise_constrained_slot_completion",
    }:
        bundle, sanity_result = teacher_forced_sanity_check(config, repo_root)
        (paths.run_dir / "train_contract_summary.json").write_text(
            json.dumps(bundle.contract.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (paths.run_dir / "gold_canonical_evidence.txt").write_text(
            bundle.rendered.text,
            encoding="utf-8",
        )
        (paths.run_dir / "teacher_forced_sanity.json").write_text(
            json.dumps(
                {
                    "canonical_contract": bundle.contract.to_dict(),
                    "verifier_success": sanity_result.success,
                    "decoded_payload": sanity_result.decoded_payload,
                    "messages": list(sanity_result.messages),
                    "details": sanity_result.details,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if not sanity_result.success:
            raise ValueError(
                "Teacher-forced canonical sanity check failed before training. "
                f"messages={list(sanity_result.messages)}"
            )
        canonical_contract_metadata = bundle.contract.to_dict()
        if config.train.target_mode in {
            "fieldwise_constrained_slot_completion",
            "foundation_fieldwise_constrained_slot_completion",
        }:
            probe_payload_texts = tuple(
                str(payload).strip()
                for payload in config.train.probe_payload_texts
                if str(payload).strip()
            )
            if not probe_payload_texts:
                probe_payload_texts = (config.eval.payload_text,)
            prompt_contract_name = (
                FOUNDATION_FIELDWISE_PROMPT_CONTRACT
                if config.train.target_mode == "foundation_fieldwise_constrained_slot_completion"
                else DEFAULT_FIELDWISE_PROMPT_CONTRACT
            )
            probe_block_count = config.train.probe_block_count or (
                1 if config.train.target_mode == "foundation_fieldwise_constrained_slot_completion" else 0
            )
            dataset = []
            for payload_text in dict.fromkeys(probe_payload_texts):
                probe_bundle = build_canonical_evidence_bundle(
                    config,
                    repo_root,
                    payload_text=payload_text,
                )
                probe_plan = build_fieldwise_generation_plan(
                    probe_bundle,
                    instruction=(
                        config.train.generation_prompt.strip()
                        or "Output exactly one allowed carrier value for the requested slot."
                    ),
                    prompt_contract_name=prompt_contract_name,
                    max_blocks=probe_block_count or None,
                )
                for target in probe_plan.slot_targets:
                    dataset.append(
                        TrainingExample(
                            prompt=target.prompt,
                            target_symbols=(),
                            metadata={
                                "completion": target.expected_value,
                                "payload_text": payload_text,
                                "slot_index": target.slot_index,
                                "slot_type": target.field_name,
                                "block_index": target.block_index,
                                "allowed_values": list(target.allowed_values),
                                "expected_bucket_id": target.expected_bucket_id,
                                "canonical_contract": probe_bundle.contract.to_dict(),
                                "target_mode": config.train.target_mode,
                                "generated_artifact_format": probe_plan.artifact_format,
                            },
                        )
                    )
            fieldwise_generation_plan = build_fieldwise_generation_plan(
                bundle,
                instruction=(
                    config.train.generation_prompt.strip()
                    or "Output exactly one allowed carrier value for the requested slot."
                ),
                prompt_contract_name=prompt_contract_name,
                max_blocks=probe_block_count or None,
            )
            (paths.run_dir / "gold_scaffold_prompt.txt").write_text(
                "\n\n".join(target.prompt for target in fieldwise_generation_plan.slot_targets),
                encoding="utf-8",
            )
            (paths.run_dir / "gold_scaffold_values.txt").write_text(
                "\n".join(fieldwise_generation_plan.expected_slot_values),
                encoding="utf-8",
            )
            (paths.run_dir / "probe_payload_texts.json").write_text(
                json.dumps(list(dict.fromkeys(probe_payload_texts)), indent=2),
                encoding="utf-8",
            )
            (paths.run_dir / "fieldwise_generation_plan.json").write_text(
                json.dumps(fieldwise_generation_plan.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (paths.run_dir / "slot_prompt_contract.json").write_text(
                json.dumps(
                    {
                        "prompt_contract_name": fieldwise_generation_plan.prompt_contract_name,
                        "exact_slot_prefixes": fieldwise_generation_plan.exact_slot_prefixes,
                        "probe_block_count": probe_block_count or bundle.contract.block_count,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            generated_artifact_format = fieldwise_generation_plan.artifact_format
            expected_slot_values = fieldwise_generation_plan.expected_slot_values
        elif config.train.target_mode == "scaffolded_canonical_completion":
            scaffold = build_scaffolded_completion_target(
                bundle,
                instruction=(
                    config.train.generation_prompt.strip()
                    or "Output exactly one carrier value per line for each slot and nothing else."
                ),
            )
            completion = "\n".join(scaffold.expected_slot_values)
            if config.train.generation_stop_strings:
                completion = f"{completion}{config.train.generation_stop_strings[0]}"
            (paths.run_dir / "gold_scaffold_prompt.txt").write_text(
                scaffold.prompt,
                encoding="utf-8",
            )
            (paths.run_dir / "gold_scaffold_values.txt").write_text(
                "\n".join(scaffold.expected_slot_values),
                encoding="utf-8",
            )
            dataset = [
                TrainingExample(
                    prompt=scaffold.prompt,
                    target_symbols=(),
                    metadata={
                        "completion": completion,
                        "canonical_contract": bundle.contract.to_dict(),
                        "target_mode": config.train.target_mode,
                        "generated_artifact_format": scaffold.artifact_format,
                        "expected_slot_values": list(scaffold.expected_slot_values),
                    },
                )
            ]
            generated_artifact_format = scaffold.artifact_format
            expected_slot_values = scaffold.expected_slot_values
            generation_prompt = scaffold.prompt
        else:
            completion = bundle.rendered.text
            if config.train.generation_stop_strings:
                completion = f"{completion}{config.train.generation_stop_strings[0]}"
            dataset = [
                TrainingExample(
                    prompt=(
                        config.train.generation_prompt.strip()
                        or "Emit canonical ownership evidence only:"
                    ),
                    target_symbols=(),
                    metadata={
                        "completion": completion,
                        "canonical_contract": bundle.contract.to_dict(),
                        "target_mode": config.train.target_mode,
                    },
                )
            ]
    elif dataset is None:
        if not str(config.data.train_path).strip():
            raise ValueError(
                "data.train_path is required when train.target_mode does not synthesize its own dataset"
            )
        data_path = Path(config.data.train_path)
        if not data_path.is_absolute():
            data_path = repo_root / data_path
        dataset = load_training_examples(data_path)
    assert dataset is not None
    if (
        config.train.target_mode == "compiled_fieldwise_bucket_mass"
        and config.train.evidence_loss_normalization != "per_slot_mean"
    ):
        raise ValueError(
            "G3a-v2 compiled bucket training requires train.evidence_loss_normalization=per_slot_mean; "
            f"got {config.train.evidence_loss_normalization!r}"
        )
    checkpoint_path: str
    generated_text: str
    if config.model.family == "huggingface-causal-lm":
        training_result = run_minimal_hf_causal_lm_training(
            model_name_or_path=config.model.tokenizer_name or config.model.name,
            max_length=config.model.max_length,
            dataset=dataset,
            batch_size=config.train.batch_size,
            epochs=config.train.epochs,
            learning_rate=config.train.learning_rate,
            run_dir=paths.run_dir,
            require_cuda=config.runtime.resources.num_gpus > 0,
            generation_prompt=generation_prompt,
            generation_do_sample=config.train.generation_do_sample,
            generation_max_new_tokens=config.train.generation_max_new_tokens,
            generation_stop_strings=config.train.generation_stop_strings,
            generation_bad_words=config.train.generation_bad_words,
            generation_suppress_tokens=config.train.generation_suppress_tokens,
            generation_sequence_bias=config.train.generation_sequence_bias,
            adapter_mode=config.train.adapter_mode,
            lora_r=config.train.lora_r,
            lora_alpha=config.train.lora_alpha,
            lora_dropout=config.train.lora_dropout,
            lora_target_modules=config.train.lora_target_modules,
            fieldwise_generation_plan=fieldwise_generation_plan,
            use_compiled_bucket_objective=config.train.target_mode == "compiled_fieldwise_bucket_mass",
            compiled_objective_mode=config.train.objective,
            compiled_lambda_set=config.train.lambda_set,
            checkpoint_selection_metric=config.train.checkpoint_selection_metric,
            checkpoint_selection_mode=config.train.checkpoint_selection_mode,
            checkpoint_selection_use_best_for_eval=config.train.checkpoint_selection_use_best_for_eval,
            checkpoint_selection_save_best=config.train.checkpoint_selection_save_best,
        )
        status = training_result.status
        steps = training_result.steps
        examples_seen = training_result.examples_seen
        final_loss = training_result.final_loss
        checkpoint_path = training_result.checkpoint_dir
        generated_text = training_result.generated_text
        if training_result.generation_diagnostics:
            (paths.run_dir / "fieldwise_generation_diagnostics.json").write_text(
                json.dumps(training_result.generation_diagnostics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            contextual_carrier_audit = training_result.generation_diagnostics.get(
                "contextual_carrier_audit"
            )
            if isinstance(contextual_carrier_audit, dict):
                (paths.run_dir / "contextual_carrier_audit.json").write_text(
                    json.dumps(contextual_carrier_audit, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        if training_result.health_diagnostics:
            (paths.run_dir / "training_health.json").write_text(
                json.dumps(training_result.health_diagnostics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            checkpoint_selection = training_result.health_diagnostics.get("checkpoint_selection")
            if isinstance(checkpoint_selection, dict):
                (paths.run_dir / "checkpoint_selection.json").write_text(
                    json.dumps(checkpoint_selection, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
    else:
        plan = TrainingPlan(
            dataset_name=config.data.name,
            objective=config.train.objective,
            batch_size=config.train.batch_size,
            epochs=config.train.epochs,
            learning_rate=config.train.learning_rate,
        )
        outcome = execute_training(plan, dataset_size=len(dataset))
        status = outcome.status
        steps = outcome.steps
        examples_seen = outcome.examples_seen
        final_loss = outcome.final_loss
        checkpoint_path = str(save_checkpoint_metadata(paths.run_dir, "latest", outcome.to_dict()))
        generated_text = dataset[0].prompt if dataset else config.eval.payload_text

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
        "generated_artifact_format": generated_artifact_format,
    }
    if compiled_train_contract is not None:
        eval_input_payload["compiled_train_contract_hash"] = compiled_train_contract.contract_hash
        eval_input_payload["compiled_train_contract_path"] = str(paths.run_dir / "compiled_train_contract.json")
        eval_input_payload["compiled_eval_contract"] = compiled_train_contract.eval_contract.to_dict()
        if config.train.checkpoint_selection_metric:
            eval_input_payload["checkpoint_selection"] = {
                "metric": config.train.checkpoint_selection_metric,
                "mode": config.train.checkpoint_selection_mode,
                "use_best_for_eval": config.train.checkpoint_selection_use_best_for_eval,
                "save_best": config.train.checkpoint_selection_save_best,
            }
    if canonical_contract_metadata is not None:
        eval_input_payload["canonical_contract"] = canonical_contract_metadata
    if expected_slot_values:
        eval_input_payload["expected_slot_values"] = list(expected_slot_values)
    if fieldwise_generation_plan is not None:
        eval_input_payload["slot_field_names"] = list(fieldwise_generation_plan.slot_field_names)
        eval_input_payload["exact_slot_prefixes"] = fieldwise_generation_plan.exact_slot_prefixes
        eval_input_payload["prompt_contract_name"] = fieldwise_generation_plan.prompt_contract_name
        eval_input_payload["fields_per_block"] = fieldwise_generation_plan.fields_per_block
    elif scaffold_slot_field_names:
        eval_input_payload["slot_field_names"] = list(scaffold_slot_field_names)
        eval_input_payload["prompt_contract_name"] = scaffold_prompt_contract_name
        eval_input_payload["fields_per_block"] = scaffold_fields_per_block
    eval_input_path = paths.run_dir / "eval_input.json"
    eval_input_path.write_text(
        json.dumps(eval_input_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    latest_eval_input_path = paths.experiment_dir / "latest_eval_input.json"
    latest_eval_input_path.parent.mkdir(parents=True, exist_ok=True)
    latest_eval_input_path.write_text(
        json.dumps(eval_input_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    save_checkpoint_metadata(
        paths.run_dir,
        "latest",
        {
            "status": status,
            "steps": steps,
            "examples_seen": examples_seen,
            "final_loss": final_loss,
            "checkpoint_path": checkpoint_path,
            "generated_text_path": str(generated_text_path),
            "eval_input_path": str(eval_input_path),
            "latest_eval_input_path": str(latest_eval_input_path),
        },
    )

    summary = TrainRunSummary(
        run_id=identity.run_id,
        experiment_name=config.experiment_name,
        method_name=config.method_name,
        model_name=config.model_name,
        seed=config.seed,
        git_commit=identity.git_commit,
        timestamp=environment.timestamp,
        hostname=environment.hostname,
        slurm_job_id=environment.slurm_job_id,
        status=status,
        objective=config.train.objective,
        dataset_name=config.data.name,
        dataset_size=len(dataset),
        steps=steps,
        final_loss=final_loss,
        run_dir=str(paths.run_dir),
    )
    summary.save_json(paths.run_dir / "train_summary.json")
    logger.info("wrote training summary to %s", paths.run_dir / "train_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
