from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from src.baselines.base import build_baseline_adapter
from src.core.canonical_contract import (
    CanonicalContract,
    build_canonical_contract,
    ensure_matching_canonical_contract,
)
from src.core.catalog_freeze import load_required_frozen_catalog
from src.core.contract_compiler import CompiledEvalContract
from src.core.payload_codec import BucketPayloadCodec
from src.core.render import render_bucket_tuples, render_config_from_name
from src.core.scaffolded_completion import (
    COMPILED_ARTIFACT_FORMAT,
    FOUNDATION_ARTIFACT_FORMAT,
    SCAFFOLDED_ARTIFACT_FORMAT,
    evaluate_foundation_completion,
    parse_scaffolded_completion,
)
from src.core.verifier import (
    VerificationConfig,
    VerificationResult,
    verify_canonical_rendered_text,
    verify_fixture,
)
from src.evaluation.canonical_source import load_canonical_evidence_source
from src.evaluation.report import EvalRunSummary, load_result_json
from src.evaluation.utility_eval import evaluate_utility
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
from src.core.tokenizer_utils import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation for our method or a baseline adapter.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing run dir.")
    parser.add_argument("--jsonl-log", action="store_true", help="Enable JSONL machine logs.")
    return parser.parse_args()


def _resolve_run_paths(repo_root: Path, config: object, force: bool) -> tuple[RunIdentity, object]:
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
            repo_root,
            config.experiment_name,
            config.model_name,
            config.seed,
            method_name=config.method_name,
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
    ensure_run_dir(paths.run_dir, force=force)
    return identity, paths


def _verification_config(config: object) -> VerificationConfig:
    return VerificationConfig(
        verification_mode=config.eval.verification_mode,
        render_format=config.eval.render_format,
        min_score=config.eval.min_score,
        max_candidates=config.eval.max_candidates,
        min_match_ratio=1.0,
        scan_windows=True,
        require_all_fields=True,
        decode_as_bytes=True,
        apply_rs=False,
    )


def _load_foundation_gate_summary(summary_path: Path) -> EvalRunSummary:
    summary = load_result_json(summary_path)
    if not isinstance(summary, EvalRunSummary):
        raise ValueError(f"foundation_eval_summary_path is not an eval summary: {summary_path}")
    if summary.verification_mode != "foundation_gate":
        raise ValueError(
            "foundation_eval_summary_path must point to a foundation_gate eval summary: "
            f"{summary_path}"
        )
    return summary


def _resolve_foundation_gate_diagnostics(config: object, repo_root: Path) -> dict[str, object]:
    if not config.eval.require_foundation_gate:
        return {}
    if not str(config.data.foundation_eval_summary_path).strip():
        raise ValueError(
            "canonical eval requires a passing foundation gate summary but "
            "data.foundation_eval_summary_path is empty"
        )
    summary_path = Path(config.data.foundation_eval_summary_path)
    if not summary_path.is_absolute():
        summary_path = (repo_root / summary_path).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"foundation_eval_summary_path does not exist: {summary_path}")
    foundation_summary = _load_foundation_gate_summary(summary_path)
    gate_passed = bool(foundation_summary.diagnostics.get("foundation_gate_passed"))
    if not (foundation_summary.accepted and foundation_summary.verifier_success and gate_passed):
        raise ValueError(
            "foundation gate did not pass; canonical eval remains blocked until F1 succeeds"
        )
    return {
        "foundation_gate_required": True,
        "foundation_eval_summary_path": str(summary_path),
        "foundation_gate_run_id": foundation_summary.run_id,
        "foundation_gate_passed": gate_passed,
    }


def _load_compiled_eval_contract_from_diagnostics(diagnostics: dict[str, object]) -> CompiledEvalContract:
    payload = diagnostics.get("compiled_eval_contract")
    if not isinstance(payload, dict):
        raise ValueError(
            "compiled_gate evaluation requires eval_input.json to carry compiled_eval_contract metadata"
        )
    return CompiledEvalContract.from_dict(payload)


def _run_foundation_eval(
    config: object,
    repo_root: Path,
    run_dir: Path,
) -> tuple[VerificationResult, dict[str, object]]:
    evidence_source = load_canonical_evidence_source(
        repo_root=repo_root,
        eval_path=config.data.eval_path,
        default_payload_text=config.eval.payload_text,
    )
    diagnostics = dict(evidence_source.diagnostics)
    artifact_format = diagnostics.get("generated_artifact_format", "canonical_text")
    if artifact_format != FOUNDATION_ARTIFACT_FORMAT:
        raise ValueError(
            "foundation_gate evaluation requires generated_artifact_format=foundation_slot_values"
        )
    generated_text = evidence_source.evidence_text
    if generated_text is None:
        raise ValueError("foundation_gate evaluation requires generated_text_path-backed evidence")

    catalog_path = Path(config.data.carrier_catalog_path)
    if not catalog_path.is_absolute():
        catalog_path = repo_root / catalog_path
    layout = load_required_frozen_catalog(catalog_path)
    expected_slot_values = tuple(str(item) for item in diagnostics.get("expected_slot_values", []))
    if not expected_slot_values:
        raise ValueError("foundation_gate evaluation requires expected_slot_values metadata")
    slot_field_names = tuple(str(item) for item in diagnostics.get("slot_field_names", []))
    exact_slot_prefixes = {
        str(key): str(value)
        for key, value in dict(diagnostics.get("exact_slot_prefixes", {})).items()
    }
    if not exact_slot_prefixes:
        exact_slot_prefixes = {field_name: f"{field_name}=" for field_name in layout.field_names}
    prompt_contract_name = str(diagnostics.get("prompt_contract_name", "foundation_v1"))
    tokenizer = load_tokenizer(
        config.model.tokenizer_backend,
        config.model.tokenizer_name or config.model.name,
    )
    foundation_result = evaluate_foundation_completion(
        generated_text,
        layout=layout,
        expected_slot_values=expected_slot_values,
        exact_slot_prefixes=exact_slot_prefixes,
        tokenizer=tokenizer,
        prompt_contract_name=prompt_contract_name,
        render_format=config.eval.render_format,
        slot_field_names=slot_field_names,
    )
    (run_dir / "foundation_gate_result.json").write_text(
        json.dumps(foundation_result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if foundation_result.rendered_canonical_text:
        (run_dir / "foundation_rendered_canonical.txt").write_text(
            foundation_result.rendered_canonical_text,
            encoding="utf-8",
        )

    codec = BucketPayloadCodec(bucket_radices=layout.radices)
    render_verification: VerificationResult | None = None
    render_verifier_success = False
    if foundation_result.rendered_bucket_tuples:
        expected_units = codec.decode_units(foundation_result.rendered_bucket_tuples, apply_rs=False)
        render_verification = verify_canonical_rendered_text(
            text=foundation_result.rendered_canonical_text,
            bucket_layout=layout,
            payload_codec=codec,
            expected_payload=expected_units,
            config=VerificationConfig(
                verification_mode="canonical_render",
                render_format=config.eval.render_format,
                min_score=config.eval.min_score,
                max_candidates=config.eval.max_candidates,
                min_match_ratio=1.0,
                scan_windows=True,
                require_all_fields=True,
                decode_as_bytes=False,
                apply_rs=False,
            ),
        )
        render_verifier_success = render_verification.success
        render_verification.save_json(run_dir / "foundation_render_verifier_result.json")

    foundation_gate_passed = foundation_result.foundation_gate_passed and render_verifier_success
    messages = list(foundation_result.messages)
    if not render_verifier_success:
        messages.append("deterministic canonical render did not pass verifier")
    verification_result = VerificationResult(
        success=foundation_gate_passed,
        verification_mode="foundation_gate",
        render_format=config.eval.render_format,
        decoded_units=render_verification.decoded_units if render_verification else (),
        decoded_payload=render_verification.decoded_payload if render_verification else None,
        decoded_bucket_tuples=(
            render_verification.decoded_bucket_tuples if render_verification else foundation_result.rendered_bucket_tuples
        ),
        parsed_blocks=render_verification.parsed_blocks if render_verification else (),
        parsed_carriers=render_verification.parsed_carriers if render_verification else (),
        unresolved_fields=render_verification.unresolved_fields if render_verification else (),
        bucket_mismatches=render_verification.bucket_mismatches if render_verification else (),
        messages=tuple(messages),
        expected_payload_units=render_verification.expected_payload_units if render_verification else (),
        details={
            "field_valid_rate": foundation_result.field_valid_rate,
            "bucket_correct_rate": foundation_result.bucket_correct_rate,
            "slot_exact_rate": foundation_result.slot_exact_rate,
            "per_field_accuracy": foundation_result.per_field_accuracy,
            "contextual_audit_pass": foundation_result.contextual_audit_pass,
            "foundation_gate_passed": foundation_gate_passed,
            "render_verifier_success": render_verifier_success,
        },
        match_ratio=foundation_result.slot_exact_rate,
        observed_count=len(foundation_result.parsed_slot_values),
        malformed_count=sum(1 for item in foundation_result.slot_diagnostics if not item.is_field_valid),
    )
    return verification_result, {
        **diagnostics,
        "generated_artifact_format": artifact_format,
        "field_valid_rate": foundation_result.field_valid_rate,
        "bucket_correct_rate": foundation_result.bucket_correct_rate,
        "slot_exact_rate": foundation_result.slot_exact_rate,
        "per_field_accuracy": foundation_result.per_field_accuracy,
        "contextual_audit_pass": foundation_result.contextual_audit_pass,
        "foundation_gate_passed": foundation_gate_passed,
        "render_verifier_success": render_verifier_success,
        "valid_canonical_block_count": foundation_result.valid_canonical_block_count,
        "slot_diagnostics": [item.to_dict() for item in foundation_result.slot_diagnostics],
        "chosen_token_vs_allowed_token_set": [
            {
                "slot_index": item.slot_index,
                "slot_type": item.slot_type,
                "allowed_token_ids": list(item.allowed_token_ids),
                "chosen_token_id": item.chosen_token_id,
                "chosen_token_text": item.chosen_token_text,
                "is_field_valid": item.is_field_valid,
                "is_bucket_correct": item.is_bucket_correct,
            }
            for item in foundation_result.slot_diagnostics
        ],
    }


def _run_compiled_gate_eval(
    config: object,
    repo_root: Path,
    run_dir: Path,
) -> tuple[VerificationResult, dict[str, object]]:
    evidence_source = load_canonical_evidence_source(
        repo_root=repo_root,
        eval_path=config.data.eval_path,
        default_payload_text=config.eval.payload_text,
    )
    diagnostics = dict(evidence_source.diagnostics)
    artifact_format = diagnostics.get("generated_artifact_format", "canonical_text")
    if artifact_format != COMPILED_ARTIFACT_FORMAT:
        raise ValueError(
            "compiled_gate evaluation requires generated_artifact_format=compiled_slot_values"
        )
    generated_text = evidence_source.evidence_text
    if generated_text is None:
        raise ValueError("compiled_gate evaluation requires generated_text_path-backed evidence")

    compiled_eval_contract = _load_compiled_eval_contract_from_diagnostics(diagnostics)
    catalog_path = Path(config.data.carrier_catalog_path)
    if not catalog_path.is_absolute():
        catalog_path = repo_root / catalog_path
    layout = load_required_frozen_catalog(catalog_path)
    tokenizer = load_tokenizer(
        config.model.tokenizer_backend,
        config.model.tokenizer_name or config.model.name,
    )
    compiled_result = evaluate_foundation_completion(
        generated_text,
        layout=layout,
        expected_slot_values=compiled_eval_contract.expected_slot_values,
        exact_slot_prefixes=compiled_eval_contract.exact_slot_prefixes,
        tokenizer=tokenizer,
        prompt_contract_name=compiled_eval_contract.prompt_contract_name,
        render_format=compiled_eval_contract.render_format,
        slot_field_names=compiled_eval_contract.slot_field_names,
        artifact_format=COMPILED_ARTIFACT_FORMAT,
    )
    (run_dir / "compiled_gate_result.json").write_text(
        json.dumps(compiled_result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if compiled_result.rendered_canonical_text:
        (run_dir / "compiled_rendered_canonical.txt").write_text(
            compiled_result.rendered_canonical_text,
            encoding="utf-8",
        )

    codec = BucketPayloadCodec(bucket_radices=layout.radices)
    render_verification: VerificationResult | None = None
    render_verifier_success = False
    if compiled_result.rendered_bucket_tuples:
        render_verification = verify_canonical_rendered_text(
            text=compiled_result.rendered_canonical_text,
            bucket_layout=layout,
            payload_codec=codec,
            expected_payload=tuple(int(unit) for unit in compiled_eval_contract.payload_units),
            config=VerificationConfig(
                verification_mode="canonical_render",
                render_format=compiled_eval_contract.render_format,
                min_score=config.eval.min_score,
                max_candidates=config.eval.max_candidates,
                min_match_ratio=1.0,
                scan_windows=True,
                require_all_fields=True,
                decode_as_bytes=False,
                apply_rs=False,
            ),
        )
        render_verifier_success = render_verification.success
        render_verification.save_json(run_dir / "compiled_render_verifier_result.json")

    compiled_gate_passed = compiled_result.foundation_gate_passed and render_verifier_success
    messages = list(compiled_result.messages)
    if not render_verifier_success:
        messages.append("deterministic compiled render did not pass verifier")

    verification_result = VerificationResult(
        success=compiled_gate_passed,
        verification_mode="compiled_gate",
        render_format=compiled_eval_contract.render_format,
        decoded_units=render_verification.decoded_units if render_verification else (),
        decoded_payload=(
            compiled_eval_contract.payload_label
            if render_verification and render_verification.success
            else None
        ),
        decoded_bucket_tuples=(
            render_verification.decoded_bucket_tuples
            if render_verification
            else compiled_result.rendered_bucket_tuples
        ),
        parsed_blocks=render_verification.parsed_blocks if render_verification else (),
        parsed_carriers=render_verification.parsed_carriers if render_verification else (),
        unresolved_fields=render_verification.unresolved_fields if render_verification else (),
        bucket_mismatches=render_verification.bucket_mismatches if render_verification else (),
        messages=tuple(messages),
        expected_payload_units=tuple(int(unit) for unit in compiled_eval_contract.payload_units),
        details={
            "field_valid_rate": compiled_result.field_valid_rate,
            "bucket_correct_rate": compiled_result.bucket_correct_rate,
            "slot_exact_rate": compiled_result.slot_exact_rate,
            "per_field_accuracy": compiled_result.per_field_accuracy,
            "contextual_audit_pass": compiled_result.contextual_audit_pass,
            "compiled_gate_passed": compiled_gate_passed,
            "render_verifier_success": render_verifier_success,
            "compiled_train_contract_hash": diagnostics.get("compiled_train_contract_hash"),
        },
        match_ratio=compiled_result.slot_exact_rate,
        observed_count=len(compiled_result.parsed_slot_values),
        malformed_count=sum(1 for item in compiled_result.slot_diagnostics if not item.is_field_valid),
    )
    return verification_result, {
        **diagnostics,
        "generated_artifact_format": artifact_format,
        "compiled_eval_contract": compiled_eval_contract.to_dict(),
        "payload_label": compiled_eval_contract.payload_label,
        "payload_units": list(compiled_eval_contract.payload_units),
        "field_valid_rate": compiled_result.field_valid_rate,
        "bucket_correct_rate": compiled_result.bucket_correct_rate,
        "slot_exact_rate": compiled_result.slot_exact_rate,
        "per_field_accuracy": compiled_result.per_field_accuracy,
        "contextual_audit_pass": compiled_result.contextual_audit_pass,
        "compiled_gate_passed": compiled_gate_passed,
        "render_verifier_success": render_verifier_success,
        "valid_canonical_block_count": compiled_result.valid_canonical_block_count,
        "slot_diagnostics": [item.to_dict() for item in compiled_result.slot_diagnostics],
        "chosen_token_vs_allowed_token_set": [
            {
                "slot_index": item.slot_index,
                "slot_type": item.slot_type,
                "allowed_token_ids": list(item.allowed_token_ids),
                "chosen_token_id": item.chosen_token_id,
                "chosen_token_text": item.chosen_token_text,
                "is_field_valid": item.is_field_valid,
                "is_bucket_correct": item.is_bucket_correct,
            }
            for item in compiled_result.slot_diagnostics
        ],
    }


def _run_our_method_eval(config: object, repo_root: Path, run_dir: Path) -> tuple[VerificationResult, dict[str, object]]:
    if config.eval.verification_mode == "foundation_gate":
        return _run_foundation_eval(config, repo_root, run_dir)
    if config.eval.verification_mode == "compiled_gate":
        return _run_compiled_gate_eval(config, repo_root, run_dir)

    verify_config = _verification_config(config)
    if config.eval.verification_mode == "synthetic_fixture":
        fixture_path = Path(config.data.eval_path)
        if not fixture_path.is_absolute():
            fixture_path = repo_root / fixture_path
        result = verify_fixture(fixture_path, config=verify_config)
        return result, {
            "fixture_path": str(fixture_path),
            "verification_mode": config.eval.verification_mode,
        }

    if config.eval.verification_mode == "canonical_render":
        promotion_gate_diagnostics = _resolve_foundation_gate_diagnostics(config, repo_root)
        catalog_path = Path(config.data.carrier_catalog_path)
        if not catalog_path.is_absolute():
            catalog_path = repo_root / catalog_path
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"carrier_catalog_path does not exist for canonical_render mode: {catalog_path}"
            )

        layout = load_required_frozen_catalog(catalog_path)
        codec = BucketPayloadCodec(bucket_radices=layout.radices)
        render_config = render_config_from_name(config.eval.render_format)
        active_contract = build_canonical_contract(config, repo_root)
        evidence_source = load_canonical_evidence_source(
            repo_root=repo_root,
            eval_path=config.data.eval_path,
            default_payload_text=config.eval.payload_text,
        )

        verifier_text = evidence_source.evidence_text
        diagnostics = dict(evidence_source.diagnostics)
        artifact_format = diagnostics.get("generated_artifact_format", "canonical_text")
        if artifact_format == FOUNDATION_ARTIFACT_FORMAT:
            raise ValueError(
                "canonical_render eval was given foundation_slot_values from a foundation-stage train run. "
                "Run exp_train__qwen2_5_7b__v1 first and use that main-path latest_eval_input.json for "
                "promotion into canonical clean eval."
            )
        source_contract_payload = diagnostics.get("canonical_contract")
        if diagnostics.get("evidence_source") == "generated_text_path" and not isinstance(
            source_contract_payload, dict
        ):
            raise ValueError(
                "generated-text evaluation requires eval_input.json to carry canonical_contract "
                "metadata from the source train run"
            )
        if isinstance(source_contract_payload, dict):
            ensure_matching_canonical_contract(
                CanonicalContract.from_dict(source_contract_payload),
                active_contract,
                expected_label="train_eval_input",
                observed_label="eval_config",
            )
        (run_dir / "expected_contract_summary.json").write_text(
            json.dumps(active_contract.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if isinstance(source_contract_payload, dict):
            (run_dir / "contract_alignment.json").write_text(
                json.dumps(
                    {
                        "matched": True,
                        "expected_contract": active_contract.to_dict(),
                        "source_contract": source_contract_payload,
                        "differences": [],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        if verifier_text is not None and artifact_format == SCAFFOLDED_ARTIFACT_FORMAT:
            expected_slot_values = tuple(str(item) for item in diagnostics.get("expected_slot_values", []))
            parse_result = parse_scaffolded_completion(
                verifier_text,
                layout=layout,
                slot_field_names=active_contract.field_names * active_contract.block_count,
                expected_slot_values=expected_slot_values,
            )
            (run_dir / "scaffolded_completion_diagnostics.json").write_text(
                json.dumps(parse_result.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            verifier_text = parse_result.reconstructed_text
            diagnostics.update(
                {
                    "generated_artifact_format": artifact_format,
                    "first_field_prefix_hit_rate": parse_result.first_field_prefix_hit_rate,
                    "valid_canonical_block_count": parse_result.valid_canonical_block_count,
                    "field_order_exact_rate": parse_result.field_order_exact_rate,
                    "value_slot_exact_rate": parse_result.value_slot_exact_rate,
                    "per_slot_exact_rate": parse_result.per_slot_exact_rate,
                    "parse_success_rate": parse_result.parse_success_rate,
                    "per_field_accuracy": dict(parse_result.per_field_accuracy),
                    "first_divergence_position": parse_result.first_divergence_position,
                    "parsed_slot_values": list(parse_result.parsed_slot_values),
                    "valid_slot_values": list(parse_result.valid_slot_values),
                    "malformed_slot_values": list(parse_result.malformed_slot_values),
                    "ignored_generated_lines": list(parse_result.ignored_generated_lines),
                    "slot_diagnostics": list(parse_result.slot_diagnostics),
                }
            )
        if verifier_text is None:
            encoding = codec.encode_bytes(evidence_source.expected_payload_bytes, apply_rs=False)
            rendered = render_bucket_tuples(layout, encoding.bucket_tuples, config=render_config)
            verifier_text = rendered.text
            diagnostics["num_rendered_blocks"] = len(rendered.bucket_tuples)
            (run_dir / "rendered_evidence.txt").write_text(rendered.text, encoding="utf-8")
            (run_dir / "rendered_evidence.json").write_text(
                json.dumps(rendered.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        (run_dir / "verifier_input.txt").write_text(verifier_text, encoding="utf-8")
        (run_dir / "verifier_input.json").write_text(
            json.dumps(
                {
                    "evidence_source": diagnostics.get("evidence_source"),
                    "payload_source": diagnostics.get("payload_source"),
                    "text": verifier_text,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        result = verify_canonical_rendered_text(
            text=verifier_text,
            bucket_layout=layout,
            payload_codec=codec,
            expected_payload=evidence_source.expected_payload_bytes,
            config=verify_config,
        )
        if artifact_format == SCAFFOLDED_ARTIFACT_FORMAT:
            diagnostics["decode_success_rate"] = 1.0 if result.success else 0.0
        return result, {
            "carrier_catalog_path": str(catalog_path),
            "payload_text": evidence_source.expected_payload_bytes.decode("utf-8", errors="replace"),
            "render_format": render_config.format_name,
            "num_verifier_blocks": len(result.decoded_bucket_tuples),
            "verification_mode": config.eval.verification_mode,
            "canonical_contract": active_contract.to_dict(),
            **promotion_gate_diagnostics,
            **diagnostics,
        }

    raise ValueError(f"Unsupported evaluation verification_mode: {config.eval.verification_mode}")


def _build_eval_summary(
    config: object,
    identity: RunIdentity,
    environment: object,
    run_dir: Path,
    verification_result: VerificationResult,
    diagnostics: dict[str, object],
) -> EvalRunSummary:
    utility_summary = evaluate_utility([verification_result.success])
    sample_count = len(verification_result.decoded_bucket_tuples) or 1
    return EvalRunSummary(
        run_id=identity.run_id,
        experiment_name=config.experiment_name,
        method_name=config.method_name,
        model_name=config.model_name,
        seed=config.seed,
        git_commit=identity.git_commit,
        timestamp=environment.timestamp,
        hostname=environment.hostname,
        slurm_job_id=environment.slurm_job_id,
        status="completed" if verification_result.success else "failed",
        dataset_name=config.data.name,
        sample_count=sample_count,
        accepted=verification_result.accepted,
        match_ratio=verification_result.match_ratio,
        threshold=config.eval.min_score,
        verification_mode=verification_result.verification_mode,
        render_format=verification_result.render_format,
        verifier_success=verification_result.success,
        decoded_payload=verification_result.decoded_payload,
        decoded_unit_count=len(verification_result.decoded_units),
        decoded_block_count=len(verification_result.decoded_bucket_tuples),
        unresolved_field_count=len(verification_result.unresolved_fields),
        malformed_count=verification_result.malformed_count,
        utility_acceptance_rate=utility_summary.acceptance_rate,
        notes="evaluation completed",
        diagnostics={
            **diagnostics,
            "verifier_details": verification_result.details,
            "messages": list(verification_result.messages),
            "bucket_mismatches": list(verification_result.bucket_mismatches),
            "unresolved_fields": list(verification_result.unresolved_fields),
        },
        run_dir=str(run_dir),
    )


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    config = load_experiment_config(config_path, overrides=args.override)
    identity, paths = _resolve_run_paths(repo_root, config, force=args.force)

    save_resolved_config(config, paths.resolved_config_path)
    environment = collect_environment(repo_root, fallback_run_id=config.runtime.run_id)
    write_environment_summary(environment, paths.environment_path)
    logger = setup_logging(paths.run_dir, run_id=identity.run_id, enable_jsonl=args.jsonl_log)
    log_startup(
        logger,
        config_summary={"run": config.run, "model": config.model, "eval": config.eval},
        environment_summary=environment,
    )
    set_global_seed(config.run.seed)

    if config.run.method == "our_method":
        verification_result, diagnostics = _run_our_method_eval(config, repo_root, paths.run_dir)
        verification_result.save_json(paths.run_dir / "verifier_result.json")
        summary = _build_eval_summary(
            config=config,
            identity=identity,
            environment=environment,
            run_dir=paths.run_dir,
            verification_result=verification_result,
            diagnostics=diagnostics,
        )
    else:
        adapter = build_baseline_adapter(config.run.method)
        adapter_response = adapter.verify({"config": config.to_dict()}, paths.run_dir)
        summary = EvalRunSummary(
            run_id=identity.run_id,
            experiment_name=config.experiment_name,
            method_name=config.method_name,
            model_name=config.model_name,
            seed=config.seed,
            git_commit=identity.git_commit,
            timestamp=environment.timestamp,
            hostname=environment.hostname,
            slurm_job_id=environment.slurm_job_id,
            status=adapter_response.status,
            dataset_name=config.data.name,
            sample_count=0,
            accepted=False,
            match_ratio=0.0,
            threshold=config.eval.min_score,
            verification_mode="baseline_adapter",
            render_format=None,
            verifier_success=False,
            decoded_payload=None,
            decoded_unit_count=0,
            decoded_block_count=0,
            unresolved_field_count=0,
            malformed_count=0,
            utility_acceptance_rate=0.0,
            notes=adapter_response.message,
            diagnostics=adapter_response.payload,
            run_dir=str(paths.run_dir),
        )

    summary_path = paths.run_dir / "eval_summary.json"
    summary.save_json(summary_path)
    logger.info("wrote eval summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
