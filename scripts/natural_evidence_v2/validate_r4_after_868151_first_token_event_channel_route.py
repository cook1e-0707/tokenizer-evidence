from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate(config: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    require(
        config.get("schema_name") == "natural_evidence_v2_r4_after_868151_first_token_event_channel_route_v1",
        "unexpected schema_name",
        errors,
    )
    require(str(config.get("source_failed_job_id")) == "868151", "source_failed_job_id must be 868151", errors)

    execution = config.get("execution_policy", {})
    if not isinstance(execution, Mapping):
        errors.append("execution_policy must be a mapping")
        execution = {}
    require(execution.get("artifact_only_plan") is True, "artifact_only_plan must be true", errors)
    for key in (
        "slurm_allowed",
        "generation_allowed",
        "model_forward_allowed",
        "scoring_allowed",
        "training_allowed",
        "llama_allowed",
        "same_family_null_allowed",
        "sanitizer_allowed",
        "far_allowed",
        "paper_claim_allowed",
    ):
        require(execution.get(key) is False, f"execution_policy.{key} must be false", errors)

    extraction = config.get("event_extraction", {})
    if not isinstance(extraction, Mapping):
        errors.append("event_extraction must be a mapping")
        extraction = {}
    require(
        extraction.get("primary_event") == "first_generated_token_id_after_prefix_native_boundary",
        "primary event must be token-id backed for future positives",
        errors,
    )
    require(extraction.get("row_local_side_mapping") is True, "row_local_side_mapping must be true", errors)
    require(extraction.get("future_positive_requires_token_id_trace") is True, "future positives must require token-id trace", errors)

    paths = {
        "source_failed_review": resolve(str(config.get("source_failed_review", ""))),
        "source_failure_analysis": resolve(str(config.get("source_failure_analysis", ""))),
        "source_first_token_oracle_v2": resolve(str(config.get("source_first_token_oracle_v2", ""))),
        "route_doc": resolve(str(config.get("route_doc", ""))),
        "pivot_decision_doc": resolve(str(config.get("pivot_decision_doc", ""))),
        "score_rows": resolve(str(config.get("score_rows", ""))),
        "codebook": resolve(str(config.get("codebook", ""))),
        "source_decoder_spec": resolve(str(config.get("source_decoder_spec", ""))),
        "event_decoder_spec": resolve(str(config.get("event_decoder_spec", ""))),
    }
    for label, path in paths.items():
        require(path.exists(), f"missing {label}: {path}", errors)

    if errors:
        return errors

    review = read_json(paths["source_failed_review"])
    failure = read_json(paths["source_failure_analysis"])
    oracle = read_json(paths["source_first_token_oracle_v2"])
    codebook = read_json(paths["codebook"])
    event_spec = read_json(paths["event_decoder_spec"])

    require(
        review.get("status") == "FAIL_R4_AFTER_868016_CONTROLLER_GENERATION_DIAGNOSTIC_GATE",
        "source review must be the failed 868151 full-phrase diagnostic",
        errors,
    )
    require(int(review.get("protected_accepts_format_scrub_all", -1)) == 0, "868151 protected accepts must be 0", errors)
    require(int(review.get("forbidden_public_surface_count_format_scrub_all", -1)) > 0, "868151 forbidden count must stay recorded", errors)
    require(
        failure.get("status") == "FAILURE_ANALYSIS_RECORDED_R4_AFTER_868016_CONTROLLER_GENERATION_868151",
        "failure analysis status mismatch",
        errors,
    )
    require(int(failure.get("matched_surface_count_format_scrub_all", -1)) == 0, "matched surface count must be 0", errors)
    require(int(failure.get("selected_coordinates_observed_format_scrub_all", -1)) == 0, "selected coordinates must be 0", errors)
    require(
        oracle.get("status") == "FIRST_TOKEN_EVENT_ORACLE_V2_RECORDED_ARTIFACT_ONLY_NOT_PRECOMMITTED_NOT_POSITIVE",
        "first-token oracle v2 status mismatch",
        errors,
    )
    require(str(oracle.get("source_job_id")) == "868151", "oracle source job must be 868151", errors)
    require(int(oracle.get("protected_accepts_ignoring_forbidden", -1)) == 4, "oracle should record protected 4/4 ignoring forbidden", errors)
    for key in (
        "raw_accepts_ignoring_forbidden",
        "task_only_accepts_ignoring_forbidden",
        "wrong_key_accepts_ignoring_forbidden",
        "wrong_payload_accepts_ignoring_forbidden",
    ):
        require(int(oracle.get(key, -1)) == 0, f"oracle {key} must be 0", errors)
    require(codebook.get("status") == "PASS_R4_AFTER_868016_COORDINATE_PIVOT_CODEBOOK_PRECOMMITTED", "codebook status mismatch", errors)
    require(
        event_spec.get("status") == "PRECOMMITTED_ARTIFACT_ONLY_FIRST_TOKEN_EVENT_DECODER_SPEC_NO_COMPUTE",
        "event decoder spec status mismatch",
        errors,
    )
    require(event_spec.get("source_failed_job_id") == "868151", "event decoder spec source mismatch", errors)
    require(event_spec.get("future_positive_requires_token_id_trace") is True, "event spec must require token-id trace", errors)
    require(event_spec.get("reclassifies_868151") is False, "event spec must not reclassify 868151", errors)
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 after-868151 first-token event-channel plan.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    output_dir = resolve(args.output_dir)
    config = read_yaml(config_path)
    errors = validate(config)
    status = (
        "PASS_R4_AFTER_868151_FIRST_TOKEN_EVENT_CHANNEL_ROUTE_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_868151_FIRST_TOKEN_EVENT_CHANNEL_ROUTE_VALIDATION_NO_SUBMIT"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_868151_first_token_event_channel_route_validation_v1",
        "status": status,
        "config": str(config_path),
        "errors": errors,
        "source_failed_job_id": "868151",
        "event_decoder_spec": str(resolve(str(config.get("event_decoder_spec", "")))),
        "slurm_started": False,
        "generation_started": False,
        "model_forward_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# R4 After-868151 First-Token Event Channel Route Validation",
        "",
        f"Status: `{status}`",
        "",
        "- source failed job id: `868151`",
        f"- config: `{config_path}`",
        f"- event decoder spec: `{summary['event_decoder_spec']}`",
        f"- errors: `{len(errors)}`",
        "",
        "This is artifact-only route validation. No Slurm, generation, scoring, training, or paper claim was started.",
    ]
    if errors:
        report.extend(["", "## Errors", ""])
        report.extend(f"- {error}" for error in errors)
    (output_dir / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
