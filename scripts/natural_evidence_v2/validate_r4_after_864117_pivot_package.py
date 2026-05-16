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

DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_864117_pivot_package.yaml"
REQUIRED_JOBS = {"859672", "863274", "864117"}
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
REQUIRED_OBJECTIVE_CONTRACT_TRUE = (
    "disabled_by_default",
    "no_behavior_change_when_flags_disabled",
    "protected_arm_only_target_pressure",
    "task_only_arm_cannot_receive_target_pressure",
    "exact_prefix_native_target_first_token_ids",
    "target_other_overlap_hard_fail",
    "stratum_weights_reviewed_artifact_only",
    "no_generated_transcript_phrase_mining",
)
REQUIRED_BEFORE_TRAINING_TRUE = (
    "route_doc_recorded",
    "objective_code_review_recorded",
    "toy_logit_tests_pass",
    "training_wrapper_plan_only_pass",
    "local_zero_enabled_allowlist_pass",
    "remote_hash_preflight_pass",
    "hermes_tg_email_notification_pass",
    "exactly_one_allowlist_entry_enabled_for_submission",
    "immediate_allowlist_disablement_after_sbatch",
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
    if config.get("schema_name") != "natural_evidence_v2_r4_after_864117_pivot_package_v1":
        errors.append("schema_name mismatch")
    if config.get("package_id") != "r4_after_864117_metric_exact_objective_pivot_v1":
        errors.append("package_id mismatch")

    reviews = config.get("source_reviews")
    if not isinstance(reviews, list):
        errors.append("source_reviews must be a list")
        reviews = []
    seen_jobs = {str(item.get("job_id")) for item in reviews if isinstance(item, Mapping)}
    if seen_jobs != REQUIRED_JOBS:
        errors.append(f"source_reviews job set mismatch: {sorted(seen_jobs)}")
    for item in reviews:
        if not isinstance(item, Mapping):
            errors.append("source_reviews entries must be mappings")
            continue
        job_id = str(item.get("job_id", ""))
        if not str(item.get("role", "")):
            errors.append(f"source review {job_id} missing role")
        _path_exists(root, item.get("review", ""), f"source review {job_id} review", errors)
        _path_exists(root, item.get("failure_artifact", ""), f"source review {job_id} failure_artifact", errors)

    permissions = _mapping(config.get("current_permissions"), "current_permissions", errors)
    for field in LOCKED_FALSE_FIELDS:
        if permissions.get(field) is not False:
            errors.append(f"current_permissions.{field} must be false")

    decision = _mapping(config.get("controller_route_decision"), "controller_route_decision", errors)
    if decision.get("scalar_additive_controller_exhausted_for_current_candidate") is not True:
        errors.append("controller_route_decision.scalar_additive_controller_exhausted_for_current_candidate must be true")
    if decision.get("no_additional_scalar_controller_grid_without_new_design") is not True:
        errors.append("controller_route_decision.no_additional_scalar_controller_grid_without_new_design must be true")
    if decision.get("wrong_controls_clean_in_latest_run") is not True:
        errors.append("controller_route_decision.wrong_controls_clean_in_latest_run must be true")
    if decision.get("positive_pressure_insufficient_in_latest_run") is not True:
        errors.append("controller_route_decision.positive_pressure_insufficient_in_latest_run must be true")
    if decision.get("selected_next_route") != "metric_exact_objective_repair":
        errors.append("controller_route_decision.selected_next_route must be metric_exact_objective_repair")
    if float(decision.get("best_latest_lift_vs_base", 1.0)) >= float(decision.get("required_lift_vs_base", 0.0)):
        errors.append("best_latest_lift_vs_base must remain below required_lift_vs_base")

    objective = _mapping(config.get("metric_exact_objective_repair"), "metric_exact_objective_repair", errors)
    if objective.get("artifact_only_design_allowed") is not True:
        errors.append("metric_exact_objective_repair.artifact_only_design_allowed must be true")
    if objective.get("future_compute_type") != "protected_micro_overfit_train_and_teacher_forced_score_only_after_review":
        errors.append("metric_exact_objective_repair.future_compute_type mismatch")
    for field in ("generation_allowed_in_route", "qwen_e2e_allowed_in_route", "llama_allowed_in_route"):
        if objective.get(field) is not False:
            errors.append(f"metric_exact_objective_repair.{field} must be false")
    contract = _mapping(objective.get("required_code_contract"), "metric_exact_objective_repair.required_code_contract", errors)
    for field in REQUIRED_OBJECTIVE_CONTRACT_TRUE:
        if contract.get(field) is not True:
            errors.append(f"required_code_contract.{field} must be true")
    gate = _mapping(objective.get("future_teacher_forced_gate"), "metric_exact_objective_repair.future_teacher_forced_gate", errors)
    if float(gate.get("protected_lift_vs_base_min", 0.0)) < 0.15:
        errors.append("future_teacher_forced_gate.protected_lift_vs_base_min must be >= 0.15")
    if float(gate.get("protected_lift_vs_task_only_min", 0.0)) < 0.10:
        errors.append("future_teacher_forced_gate.protected_lift_vs_task_only_min must be >= 0.10")
    if float(gate.get("protected_rank1_min", 0.0)) < 0.75:
        errors.append("future_teacher_forced_gate.protected_rank1_min must be >= 0.75")
    if gate.get("task_only_lift_anomaly_allowed") is not False:
        errors.append("future_teacher_forced_gate.task_only_lift_anomaly_allowed must be false")
    if int(gate.get("scorer_boundary_failures_max", -1)) != 0:
        errors.append("future_teacher_forced_gate.scorer_boundary_failures_max must be 0")
    if float(gate.get("target_other_overlap_rate_max", 1.0)) != 0.0:
        errors.append("future_teacher_forced_gate.target_other_overlap_rate_max must be 0")

    prereqs = _mapping(config.get("required_before_any_training_slurm"), "required_before_any_training_slurm", errors)
    for field in REQUIRED_BEFORE_TRAINING_TRUE:
        if prereqs.get(field) is not True:
            errors.append(f"required_before_any_training_slurm.{field} must be true")

    compute = _mapping(config.get("future_compute_policy"), "future_compute_policy", errors)
    if compute.get("partition") != "pomplun" or compute.get("qos") != "pomplun":
        errors.append("future_compute_policy must use pomplun partition/qos")
    if compute.get("account") != "cs_yinxin.wan":
        errors.append("future_compute_policy.account must be cs_yinxin.wan")
    if str(compute.get("gpu", "")).lower() != "h200":
        errors.append("future_compute_policy.gpu must be h200")
    if compute.get("max_time") != "30-00:00:00":
        errors.append("future_compute_policy.max_time must be 30-00:00:00")
    if compute.get("allowlist_enabled_now") is not False:
        errors.append("future_compute_policy.allowlist_enabled_now must be false")
    if compute.get("future_exactly_one_allowlist_entry") != "v2_r4_candidate_v3_micro_overfit_h200":
        errors.append("future_compute_policy.future_exactly_one_allowlist_entry mismatch")

    locked = _mapping(config.get("not_unlocked_by_this_package"), "not_unlocked_by_this_package", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_package.{field} must be true")

    status = (
        "PASS_R4_AFTER_864117_PIVOT_PACKAGE_STATIC_VALIDATION_NO_COMPUTE"
        if not errors
        else "FAIL_R4_AFTER_864117_PIVOT_PACKAGE_STATIC_VALIDATION_NO_COMPUTE"
    )
    return {
        "status": status,
        "errors": errors,
        "package_id": config.get("package_id"),
        "selected_next_route": decision.get("selected_next_route"),
        "source_failed_jobs": sorted(seen_jobs),
        "current_compute_unlocked": False,
        "allowlist_enabled": False,
        "slurm_job_submitted": False,
        "model_scoring_started": False,
        "generation_started": False,
        "training_started": False,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 after-864117 artifact-only pivot package.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_package(load_yaml(args.config))
    if args.output_dir is not None:
        write_json(args.output_dir / "after_864117_pivot_package_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
