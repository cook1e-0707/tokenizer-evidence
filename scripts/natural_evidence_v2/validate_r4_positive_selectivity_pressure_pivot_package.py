from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_positive_selectivity_pressure_pivot_package.yaml"
REQUIRED_JOBS = {"857795", "858019", "859277", "859491"}
LOCKED_FALSE_FIELDS = (
    "slurm_allowed",
    "allowlist_enablement_allowed",
    "generation_allowed",
    "model_scoring_allowed",
    "training_allowed",
    "qwen_e2e_allowed",
    "llama_allowed",
    "same_family_null_allowed",
    "sanitizer_allowed",
    "far_aggregation_allowed",
    "payload_diversity_allowed",
    "paper_claim_allowed",
)
REQUIRED_REUSE_TRUE = (
    "forbid_post_hoc_phrase_mining_from_859491",
    "forbid_threshold_tuning_to_relabel_859491",
    "forbid_key_payload_remap_to_rescue_859491",
    "forbid_unchanged_resubmission",
)
REQUIRED_STATIC_TRUE = (
    "compare_existing_teacher_forced_and_generation_failures",
    "define_future_metric_before_generation",
    "preserve_wrong_key_and_wrong_payload_controls",
    "require_primary_format_scrub_all_for_later_generation",
    "require_public_template_leakage_check",
    "require_h200_pomplun_policy_for_future_gpu",
    "require_one_reviewed_submission_for_future_compute",
    "require_hermes_tg_email_notification_for_future_compute",
)


def _mapping(value: Any, field: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be a mapping")
        return {}
    return value


def _path_exists(root: Path, value: Any, field: str, errors: list[str]) -> None:
    path = root / str(value)
    if not path.exists():
        errors.append(f"{field} missing: {path}")


def validate_package(config: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_positive_selectivity_pressure_pivot_package_v1":
        errors.append("schema_name mismatch")
    if config.get("package_id") != "r4_positive_selectivity_pressure_pivot_v1":
        errors.append("package_id mismatch")

    _path_exists(root, config.get("source_route", ""), "source_route", errors)
    _path_exists(root, config.get("source_route_summary", ""), "source_route_summary", errors)

    diagnostics = config.get("source_failed_diagnostics")
    if not isinstance(diagnostics, list):
        errors.append("source_failed_diagnostics must be a list")
        diagnostics = []
    seen_jobs = {str(item.get("job_id")) for item in diagnostics if isinstance(item, Mapping)}
    if seen_jobs != REQUIRED_JOBS:
        errors.append(f"source_failed_diagnostics job set mismatch: {sorted(seen_jobs)}")
    for item in diagnostics:
        if not isinstance(item, Mapping):
            errors.append("source_failed_diagnostics entries must be mappings")
            continue
        job_id = str(item.get("job_id", ""))
        if not str(item.get("role", "")):
            errors.append(f"diagnostic {job_id} missing role")
        _path_exists(root, item.get("review", ""), f"diagnostic {job_id} review", errors)
        _path_exists(root, item.get("failure_analysis", ""), f"diagnostic {job_id} failure_analysis", errors)

    permissions = _mapping(config.get("current_permissions"), "current_permissions", errors)
    for field in LOCKED_FALSE_FIELDS:
        if permissions.get(field) is not False:
            errors.append(f"current_permissions.{field} must be false")

    routes = _mapping(config.get("candidate_next_routes"), "candidate_next_routes", errors)
    for route_name in ("teacher_forced_pressure_controller", "metric_exact_objective_repair", "explicit_stop_record"):
        route = _mapping(routes.get(route_name), f"candidate_next_routes.{route_name}", errors)
        if route.get("artifact_only_design_allowed") is not True:
            errors.append(f"{route_name} must be artifact-only design allowed")
        if route.get("generation_allowed_in_route") is not False:
            errors.append(f"{route_name}.generation_allowed_in_route must be false")
    controller = _mapping(routes.get("teacher_forced_pressure_controller"), "teacher_forced_pressure_controller", errors)
    if controller.get("future_compute_type") != "teacher_forced_scoring_only":
        errors.append("teacher_forced_pressure_controller future_compute_type mismatch")
    if controller.get("requires_wrong_key_control") is not True:
        errors.append("teacher_forced_pressure_controller must require wrong-key control")
    if controller.get("requires_wrong_payload_control") is not True:
        errors.append("teacher_forced_pressure_controller must require wrong-payload control")
    if controller.get("primary_scrub_mode_for_later_generation") != "all":
        errors.append("later generation must keep format_scrub=all primary")

    reuse = _mapping(config.get("reuse_policy"), "reuse_policy", errors)
    for field in REQUIRED_REUSE_TRUE:
        if reuse.get(field) is not True:
            errors.append(f"reuse_policy.{field} must be true")

    static = _mapping(config.get("static_requirements"), "static_requirements", errors)
    for field in REQUIRED_STATIC_TRUE:
        if static.get(field) is not True:
            errors.append(f"static_requirements.{field} must be true")

    compute = _mapping(config.get("future_compute_policy"), "future_compute_policy", errors)
    if compute.get("partition") != "pomplun" or compute.get("qos") != "pomplun":
        errors.append("future compute must use pomplun partition/qos")
    if compute.get("account") != "cs_yinxin.wan":
        errors.append("future compute account must be cs_yinxin.wan")
    if str(compute.get("gpu", "")).lower() != "h200":
        errors.append("future compute gpu must be h200")
    if compute.get("max_time") != "30-00:00:00":
        errors.append("future compute must use max H200 time")
    if compute.get("allowlist_enabled_now") is not False:
        errors.append("allowlist_enabled_now must be false")
    if compute.get("future_exactly_one_allowlist_entry") is not True:
        errors.append("future_exactly_one_allowlist_entry must be true")
    if compute.get("future_immediate_allowlist_disable_after_sbatch") is not True:
        errors.append("future_immediate_allowlist_disable_after_sbatch must be true")

    locked = _mapping(config.get("not_unlocked_by_this_package"), "not_unlocked_by_this_package", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_package.{field} must be true")

    status = (
        "PASS_R4_POSITIVE_SELECTIVITY_PRESSURE_PIVOT_PACKAGE_STATIC_VALIDATION_NO_COMPUTE"
        if not errors
        else "FAIL_R4_POSITIVE_SELECTIVITY_PRESSURE_PIVOT_PACKAGE_STATIC_VALIDATION_NO_COMPUTE"
    )
    return {
        "status": status,
        "errors": errors,
        "package_id": config.get("package_id"),
        "source_failed_jobs": sorted(seen_jobs),
        "current_compute_unlocked": False,
        "slurm_job_submitted": False,
        "allowlist_enabled": False,
        "generation_started": False,
        "training_started": False,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R4 selectivity pressure-pivot artifact package.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_package(load_yaml(args.config))
    if args.output_dir is not None:
        write_json(args.output_dir / "pressure_pivot_package_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
