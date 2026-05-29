from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import read_json, sha256_file, write_json_new, write_text_new  # noqa: E402
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_870987_prefar_null_expansion_route.yaml"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_after_870987_prefar_null_expansion_route_validation_20260519"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the artifact-only R4 after-870987 Qwen first-token event "
            "pre-FAR null expansion route. This does not submit Slurm, generate, "
            "train, enable an allowlist entry, or make paper-facing claims."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve(path_like: Any) -> Path:
    path = Path(str(path_like))
    return path if path.is_absolute() else ROOT / path


def as_mapping(value: Any, field: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be a mapping")
        return {}
    return value


def int_field(data: Mapping[str, Any], field: str, default: int = -1) -> int:
    try:
        return int(data.get(field, default))
    except (TypeError, ValueError):
        return default


def float_field(data: Mapping[str, Any], field: str, default: float = -1.0) -> float:
    try:
        return float(data.get(field, default))
    except (TypeError, ValueError):
        return default


def require_path(config_path: Any, field: str, errors: list[str]) -> Path:
    path = resolve(config_path)
    if not path.exists():
        errors.append(f"{field} missing: {path}")
    return path


def validate_route(config: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_after_870987_prefar_null_expansion_route_v1":
        errors.append("schema_name mismatch")
    if config.get("route_id") != "r4_after_870987_prefar_null_expansion_v1":
        errors.append("route_id mismatch")
    if config.get("phase") != "V2_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE_PASSED_PREFAR_NULL_ROUTE_PLANNING":
        errors.append("phase mismatch")

    source = as_mapping(config.get("source_artifacts"), "source_artifacts", errors)
    review_path = require_path(source.get("locked_scale_review", ""), "source_artifacts.locked_scale_review", errors)
    summary_path = require_path(source.get("locked_scale_summary", ""), "source_artifacts.locked_scale_summary", errors)
    blocks_path = require_path(
        source.get("locked_scale_first_token_blocks", ""),
        "source_artifacts.locked_scale_first_token_blocks",
        errors,
    )
    trace_path = require_path(
        source.get("locked_scale_trace_binding_by_shard", ""),
        "source_artifacts.locked_scale_trace_binding_by_shard",
        errors,
    )
    if source.get("locked_scale_jobs") != ["870210", "870987"]:
        errors.append("locked_scale_jobs must be ['870210', '870987']")

    summary = read_json(summary_path) if summary_path.exists() else {}
    if summary.get("status") != "PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE":
        errors.append("locked-scale summary status must be PASS")
    if summary.get("scale_gate_pass") is not True:
        errors.append("locked-scale scale_gate_pass must be true")
    if summary.get("all_shards_complete") is not True or int_field(summary, "complete_shard_count") != 96:
        errors.append("locked-scale aggregate must be complete 96/96")

    by_arm = as_mapping(summary.get("first_token_event_summary_by_arm"), "summary.first_token_event_summary_by_arm", errors)
    protected = as_mapping(by_arm.get("protected"), "summary.first_token_event_summary_by_arm.protected", errors)
    if int_field(protected, "blocks") != 96:
        errors.append("protected blocks must be 96")
    if int_field(protected, "accepts") < 85:
        errors.append("protected strict accepts must be >= 85/96")
    if int_field(protected, "accepts_ignoring_quality") < 90:
        errors.append("protected ignoring-quality accepts must be >= 90/96")
    if int_field(protected, "duplicate_response_hash_count") != 0:
        errors.append("protected duplicate_response_hash_count must be 0")
    if int_field(protected, "forbidden_public_surface_count") != 0:
        errors.append("protected forbidden_public_surface_count must be 0")

    standard_control_arms = ["raw", "task_only", "wrong_key", "wrong_payload"]
    control_blocks: dict[str, int] = {}
    control_accepts: dict[str, int] = {}
    for arm in standard_control_arms:
        arm_summary = as_mapping(by_arm.get(arm), f"summary.first_token_event_summary_by_arm.{arm}", errors)
        control_blocks[arm] = int_field(arm_summary, "blocks")
        control_accepts[arm] = int_field(arm_summary, "accepts")
        if control_blocks[arm] != 96:
            errors.append(f"{arm} blocks must be 96")
        if control_accepts[arm] != 0:
            errors.append(f"{arm} accepts must be 0")
        if int_field(arm_summary, "accepts_ignoring_quality") != 0:
            errors.append(f"{arm} accepts_ignoring_quality must be 0")
        if int_field(arm_summary, "duplicate_response_hash_count") != 0:
            errors.append(f"{arm} duplicate_response_hash_count must be 0")
        if int_field(arm_summary, "forbidden_public_surface_count") != 0:
            errors.append(f"{arm} forbidden_public_surface_count must be 0")

    duplicates = as_mapping(summary.get("generation_duplicate_summary"), "summary.generation_duplicate_summary", errors)
    if int_field(duplicates, "global_duplicate_response_hash_extra_rows") != 0:
        errors.append("global_duplicate_response_hash_extra_rows must be 0")
    if int_field(duplicates, "duplicate_generation_id_extra_rows") != 0:
        errors.append("duplicate_generation_id_extra_rows must be 0")

    trace = as_mapping(summary.get("trace_binding"), "summary.trace_binding", errors)
    if int_field(trace, "invalid_rows") != 0:
        errors.append("trace binding invalid_rows must be 0")
    if int_field(trace, "checked_rows") <= 0:
        errors.append("trace binding checked_rows must be positive")

    claim_control = as_mapping(summary.get("claim_control"), "summary.claim_control", errors)
    for field in (
        "far_aggregation_allowed",
        "llama_allowed",
        "paper_claim_allowed",
        "payload_diversity_tested",
        "same_family_null_allowed",
        "sanitizer_allowed",
        "text_only_phrase_decoder_success_claim",
        "training_allowed",
    ):
        if claim_control.get(field) is not False:
            errors.append(f"locked-scale claim_control.{field} must be false")

    scope = as_mapping(config.get("route_scope"), "route_scope", errors)
    if scope.get("package_name") != "qwen_first_token_event_pre_far_null_v1":
        errors.append("route_scope.package_name mismatch")
    if scope.get("result_scope") != "qwen_only_same_contract_provider_side_first_token_event_trace":
        errors.append("route_scope.result_scope mismatch")
    if scope.get("contract_id") != "a55e":
        errors.append("route_scope.contract_id must be a55e")
    for field in ("payload_diversity_tested", "text_only_phrase_decoder_claim", "paper_facing_claim", "full_far_claim"):
        if scope.get(field) is not False:
            errors.append(f"route_scope.{field} must be false")

    target = as_mapping(config.get("target_null_package"), "target_null_package", errors)
    if target.get("standard_control_arms") != standard_control_arms:
        errors.append("target_null_package.standard_control_arms mismatch")
    target_per_arm = int_field(target, "target_block_equivalent_per_standard_control_arm")
    existing_per_arm = int_field(target, "existing_block_equivalent_per_standard_control_arm")
    additional_per_arm = int_field(target, "additional_block_equivalent_required_per_standard_control_arm")
    if target_per_arm != 256:
        errors.append("target standard null blocks per arm must be 256")
    if existing_per_arm != 96:
        errors.append("existing standard null blocks per arm must be 96")
    if additional_per_arm != target_per_arm - existing_per_arm:
        errors.append("additional required standard null blocks must equal target-existing")
    if additional_per_arm != 160:
        errors.append("additional required standard null blocks must be 160")
    if int_field(target, "organic_null_target_block_equivalent") != 256:
        errors.append("organic null target must be 256 block-equivalent")
    if int_field(target, "organic_null_existing_block_equivalent") != 0:
        errors.append("organic null existing block-equivalent must be 0")
    if target.get("organic_null_requires_new_prompt_bank") is not True:
        errors.append("organic null must require a new prompt bank")

    pass_gate = as_mapping(target.get("pass_gate"), "target_null_package.pass_gate", errors)
    for field in ("raw_accepts", "task_only_accepts", "wrong_key_accepts", "wrong_payload_accepts", "organic_null_accepts"):
        if int_field(pass_gate, field) != 0:
            errors.append(f"target_null_package.pass_gate.{field} must be 0")
    if int_field(pass_gate, "technical_forbidden_public_surface_count") != 0:
        errors.append("technical forbidden public surface gate must be 0")
    if int_field(pass_gate, "global_duplicate_response_hash_extra_rows") != 0:
        errors.append("global duplicate response hash extra rows gate must be 0")
    if float_field(pass_gate, "trace_binding_validity") != 1.0:
        errors.append("trace binding validity gate must be 1.0")

    compute = as_mapping(config.get("compute_policy"), "compute_policy", errors)
    if compute.get("slurm_allowed_now") is not False:
        errors.append("compute_policy.slurm_allowed_now must be false")
    if compute.get("allowlist_enabled_now") is not False:
        errors.append("compute_policy.allowlist_enabled_now must be false")
    if compute.get("allowlist_entry") != "" or compute.get("wrapper") != "":
        errors.append("route planning package must not name a submit-ready allowlist entry/wrapper yet")
    for field, expected in (
        ("partition", "pomplun"),
        ("qos", "pomplun"),
        ("account", "cs_yinxin.wan"),
        ("gres", "gpu:h200:1"),
        ("max_time", "30-00:00:00"),
        ("array_throttle_default", "%6"),
    ):
        if compute.get(field) != expected:
            errors.append(f"compute_policy.{field} mismatch")
    if compute.get("no_a100") is not True:
        errors.append("compute_policy.no_a100 must be true")

    prereqs = as_mapping(config.get("required_before_any_slurm_submission"), "required_before_any_slurm_submission", errors)
    for field, value in prereqs.items():
        if value is not True:
            errors.append(f"required_before_any_slurm_submission.{field} must be true")
    locked = as_mapping(config.get("not_unlocked_by_this_route_package"), "not_unlocked_by_this_route_package", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_package.{field} must be true")

    status = (
        "PASS_R4_AFTER_870987_PREFAR_NULL_EXPANSION_ROUTE_PLAN_ONLY_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_870987_PREFAR_NULL_EXPANSION_ROUTE_PLAN_ONLY_NO_SUBMIT"
    )
    return {
        "schema_name": "natural_evidence_v2_r4_after_870987_prefar_null_expansion_route_validation_v1",
        "status": status,
        "errors": errors,
        "locked_scale_summary": str(summary_path.relative_to(ROOT)) if summary_path.exists() and summary_path.is_relative_to(ROOT) else str(summary_path),
        "locked_scale_summary_sha256": sha256_file(summary_path) if summary_path.exists() else "",
        "locked_scale_review": str(review_path.relative_to(ROOT)) if review_path.exists() and review_path.is_relative_to(ROOT) else str(review_path),
        "locked_scale_first_token_blocks": str(blocks_path.relative_to(ROOT)) if blocks_path.exists() and blocks_path.is_relative_to(ROOT) else str(blocks_path),
        "locked_scale_trace_binding_by_shard": str(trace_path.relative_to(ROOT)) if trace_path.exists() and trace_path.is_relative_to(ROOT) else str(trace_path),
        "existing_control_blocks_per_arm": control_blocks,
        "existing_control_accepts_per_arm": control_accepts,
        "target_control_blocks_per_arm": 256,
        "additional_control_blocks_required_per_arm": {arm: 160 for arm in standard_control_arms},
        "organic_null_target_block_equivalent": 256,
        "slurm_allowed": False,
        "generation_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Build artifact-only additional standard-control and organic-null row/prompt banks, "
            "then run tokenizer/controller preflight and wrapper review before any Slurm submission."
        ),
    }


def write_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    controls = summary.get("existing_control_blocks_per_arm", {})
    additional = summary.get("additional_control_blocks_required_per_arm", {})
    text = f"""# R4 After-870987 Pre-FAR Null Expansion Route Validation

Date: 2026-05-19

Status: `{summary['status']}`

This is an artifact-only route validation after the R4 Qwen same-contract
first-token event locked-scale generation package passed. It does not submit
Slurm, generate outputs, train, enable an allowlist entry, or create a
paper-facing claim.

```text
locked-scale control blocks per arm: {controls}
additional standard-control blocks required per arm: {additional}
organic null target block-equivalent: {summary['organic_null_target_block_equivalent']}
slurm_allowed: {summary['slurm_allowed']}
paper_claim_allowed: {summary['paper_claim_allowed']}
```

Next allowed action: build artifact-only additional standard-control and
organic-null row/prompt banks, then run tokenizer/controller preflight and
full-wrapper review before any Slurm submission.
"""
    write_text_new(output_dir / "route_validation_report.md", text)


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = validate_route(load_yaml(config_path))
    write_json_new(output_dir / "route_validation_summary.json", summary)
    write_report(output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
