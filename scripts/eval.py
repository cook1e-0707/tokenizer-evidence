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
from src.core.payload_codec import BucketPayloadCodec
from src.core.render import render_bucket_tuples, render_config_from_name
from src.core.scaffolded_completion import (
    SCAFFOLDED_ARTIFACT_FORMAT,
    parse_scaffolded_completion,
)
from src.core.verifier import (
    VerificationConfig,
    VerificationResult,
    verify_canonical_rendered_text,
    verify_fixture,
)
from src.evaluation.canonical_source import load_canonical_evidence_source
from src.evaluation.report import EvalRunSummary
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


def _run_our_method_eval(config: object, repo_root: Path, run_dir: Path) -> tuple[VerificationResult, dict[str, object]]:
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
        artifact_format = diagnostics.get("generated_artifact_format", "canonical_text")
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
                    "parse_success_rate": parse_result.parse_success_rate,
                    "first_divergence_position": parse_result.first_divergence_position,
                    "parsed_slot_values": list(parse_result.parsed_slot_values),
                    "valid_slot_values": list(parse_result.valid_slot_values),
                    "malformed_slot_values": list(parse_result.malformed_slot_values),
                    "ignored_generated_lines": list(parse_result.ignored_generated_lines),
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
