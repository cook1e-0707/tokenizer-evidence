from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_positive_selectivity_pressure_controller_route.yaml"
REQUIRED_CONDITIONS_BY_SET = {
    "pressure_controls": {
        "base",
        "task_only",
        "controlled_protected",
        "wrong_key_controlled",
        "wrong_payload_controlled",
    },
    "controller_only_controls": {
        "base",
        "task_only",
        "controlled_base",
        "wrong_key_controlled_base",
        "wrong_payload_controlled_base",
    },
}
VALID_ROUTE_IDS = {
    "r4_positive_selectivity_pressure_controller_teacher_forced_v1",
    "r4_positive_selectivity_controller_only_teacher_forced_v1",
    "r4_controller_only_safety_bound_pressure_teacher_forced_v1",
    "r4_after_867621_reliability_controller_safety_bound_teacher_forced_v1",
}
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, field: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be a mapping")
        return {}
    return value


def _sorted_float_list(value: Any, field: str, errors: list[str]) -> list[float]:
    if not isinstance(value, list) or not value:
        errors.append(f"{field} must be a non-empty list")
        return []
    numbers = [float(item) for item in value]
    if numbers != sorted(numbers):
        errors.append(f"{field} must be sorted ascending")
    if len(set(numbers)) != len(numbers):
        errors.append(f"{field} must contain unique values")
    return numbers


def validate_route(config: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_positive_selectivity_pressure_controller_route_plan_v1":
        errors.append("schema_name mismatch")
    if config.get("route_id") not in VALID_ROUTE_IDS:
        errors.append("route_id mismatch")
    if config.get("contract_id") != "a55e":
        errors.append("contract_id must remain same-contract a55e")
    if config.get("model_family") != "qwen_only":
        errors.append("model_family must remain qwen_only")
    if config.get("scorer_integration_required_before_slurm") is not True:
        errors.append("scorer_integration_required_before_slurm must be true")

    for field in ("source_route_selection", "controller_helper"):
        path = root / str(config.get(field, ""))
        if not path.exists():
            errors.append(f"{field} missing: {path}")

    scope = _mapping(config.get("future_scoring_scope"), "future_scoring_scope", errors)
    expected_score_rows = int(scope.get("max_rows", 8192))
    score_rows = root / str(config.get("score_rows", ""))
    row_count = 0
    observed_hash = None
    if not score_rows.exists():
        errors.append(f"score_rows missing: {score_rows}")
    else:
        observed_hash = sha256_file(score_rows)
        if observed_hash != str(config.get("score_rows_sha256", "")):
            errors.append("score_rows_sha256 mismatch")
        with score_rows.open("r", encoding="utf-8") as handle:
            row_count = sum(1 for line in handle if line.strip())
        if row_count != expected_score_rows:
            errors.append(f"score_rows must have {expected_score_rows} rows, observed {row_count}")

    permissions = _mapping(config.get("current_permissions"), "current_permissions", errors)
    for field in LOCKED_FALSE_FIELDS:
        if permissions.get(field) is not False:
            errors.append(f"current_permissions.{field} must be false")

    condition_set = str(config.get("controller_condition_set", "pressure_controls"))
    if condition_set not in REQUIRED_CONDITIONS_BY_SET:
        errors.append("controller_condition_set must be pressure_controls or controller_only_controls")
    if scope.get("scoring_only") is not True:
        errors.append("future_scoring_scope.scoring_only must be true")
    if scope.get("generation_allowed_in_route") is not False:
        errors.append("future_scoring_scope.generation_allowed_in_route must be false")
    if scope.get("training_allowed_in_route") is not False:
        errors.append("future_scoring_scope.training_allowed_in_route must be false")
    if set(scope.get("conditions", [])) != REQUIRED_CONDITIONS_BY_SET.get(condition_set, set()):
        errors.append("future_scoring_scope.conditions mismatch")
    if scope.get("partition") != "pomplun" or scope.get("qos") != "pomplun":
        errors.append("future scoring scope must use pomplun partition/qos")
    if scope.get("account") != "cs_yinxin.wan":
        errors.append("future scoring scope account must be cs_yinxin.wan")
    if str(scope.get("gpu", "")).lower() != "h200":
        errors.append("future scoring scope gpu must be h200")
    if scope.get("max_time") != "30-00:00:00":
        errors.append("future scoring scope must use max H200 time")
    if int(scope.get("max_rows", 0)) <= 0:
        errors.append("future scoring scope max_rows must be positive")

    grid = _mapping(config.get("controller_grid"), "controller_grid", errors)
    bonus = _sorted_float_list(grid.get("bonus_nats"), "controller_grid.bonus_nats", errors)
    penalty = _sorted_float_list(grid.get("penalty_nats"), "controller_grid.penalty_nats", errors)
    max_target_mass = _sorted_float_list(grid.get("max_target_mass"), "controller_grid.max_target_mass", errors)
    max_kl = _sorted_float_list(grid.get("max_kl_budget"), "controller_grid.max_kl_budget", errors)
    if not bonus or min(bonus) <= 0.0 or max(bonus) > 2.0:
        errors.append("controller_grid.bonus_nats must stay within (0, 2.0]")
    if penalty and min(penalty) < 0.0:
        errors.append("controller_grid.penalty_nats must be non-negative")
    if max_target_mass and max(max_target_mass) > 0.50:
        errors.append("controller_grid.max_target_mass must not exceed 0.50")
    if max_kl and max(max_kl) > 0.20:
        errors.append("controller_grid.max_kl_budget must not exceed 0.20")
    if grid.get("controller_mode") != "additive":
        errors.append("controller_grid.controller_mode must be additive")

    mapping = _mapping(config.get("controller_control_mapping"), "controller_control_mapping", errors)
    if mapping.get("scorer_target_remains_committed") is not True:
        errors.append("controller_control_mapping.scorer_target_remains_committed must be true")
    if condition_set == "pressure_controls":
        if mapping.get("controlled_protected_policy") != "committed":
            errors.append("controller_control_mapping.controlled_protected_policy must be committed")
        if mapping.get("wrong_payload_controlled_policy") != "complement":
            errors.append("controller_control_mapping.wrong_payload_controlled_policy must be complement")
        if mapping.get("wrong_key_controlled_policy") != "coordinate_hash_v1":
            errors.append("controller_control_mapping.wrong_key_controlled_policy must be coordinate_hash_v1")
    if condition_set == "controller_only_controls":
        if mapping.get("protected_adapter_loaded_for_controller_arms") is not False:
            errors.append("controller_control_mapping.protected_adapter_loaded_for_controller_arms must be false")
        if mapping.get("controlled_base_policy") != "committed":
            errors.append("controller_control_mapping.controlled_base_policy must be committed")
        if mapping.get("wrong_payload_controlled_base_policy") != "complement":
            errors.append("controller_control_mapping.wrong_payload_controlled_base_policy must be complement")
        if mapping.get("wrong_key_controlled_base_policy") != "coordinate_hash_v1":
            errors.append("controller_control_mapping.wrong_key_controlled_base_policy must be coordinate_hash_v1")
    if mapping.get("wrong_key_hash_salt") != "r4_wrong_key_controller_v1":
        errors.append("controller_control_mapping.wrong_key_hash_salt must be r4_wrong_key_controller_v1")
    if mapping.get("no_posthoc_key_payload_remap") is not True:
        errors.append("controller_control_mapping.no_posthoc_key_payload_remap must be true")
    if mapping.get("no_transcript_conditioned_mapping") is not True:
        errors.append("controller_control_mapping.no_transcript_conditioned_mapping must be true")

    gate = _mapping(config.get("future_teacher_forced_gate"), "future_teacher_forced_gate", errors)
    lift_base_field = "protected_lift_vs_base_min" if condition_set == "pressure_controls" else "controlled_lift_vs_base_min"
    lift_task_field = (
        "protected_lift_vs_task_only_min" if condition_set == "pressure_controls" else "controlled_lift_vs_task_only_min"
    )
    rank_field = "protected_rank1_min" if condition_set == "pressure_controls" else "controlled_rank1_min"
    if float(gate.get(lift_base_field, 0.0)) < 0.15:
        errors.append(f"future_teacher_forced_gate.{lift_base_field} must be >= 0.15")
    if float(gate.get(lift_task_field, 0.0)) < 0.10:
        errors.append(f"future_teacher_forced_gate.{lift_task_field} must be >= 0.10")
    if float(gate.get(rank_field, 0.0)) < 0.75:
        errors.append(f"future_teacher_forced_gate.{rank_field} must be >= 0.75")
    if int(gate.get("wrong_key_accepts_max", -1)) != 0:
        errors.append("future_teacher_forced_gate.wrong_key_accepts_max must be 0")
    if int(gate.get("wrong_payload_accepts_max", -1)) != 0:
        errors.append("future_teacher_forced_gate.wrong_payload_accepts_max must be 0")
    if float(gate.get("max_single_surface_mass_max", 1.0)) > 0.50:
        errors.append("future_teacher_forced_gate.max_single_surface_mass_max must be <= 0.50")
    if float(gate.get("target_other_overlap_rate_max", 1.0)) != 0.0:
        errors.append("future_teacher_forced_gate.target_other_overlap_rate_max must be 0")
    if int(gate.get("scorer_boundary_failures_max", -1)) != 0:
        errors.append("future_teacher_forced_gate.scorer_boundary_failures_max must be 0")

    prereqs = _mapping(config.get("required_before_slurm"), "required_before_slurm", errors)
    for field, value in prereqs.items():
        if value is not True:
            errors.append(f"required_before_slurm.{field} must be true")

    locked = _mapping(config.get("not_unlocked_by_this_route_plan"), "not_unlocked_by_this_route_plan", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_plan.{field} must be true")

    status = (
        "PASS_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_PLAN_NO_COMPUTE"
        if not errors
        else "FAIL_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_PLAN_NO_COMPUTE"
    )
    return {
        "status": status,
        "errors": errors,
        "route_id": config.get("route_id"),
        "score_rows_sha256_observed": observed_hash,
        "score_row_count": row_count,
        "current_compute_unlocked": False,
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
    parser = argparse.ArgumentParser(description="Validate the R4 pressure-controller route plan without compute.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_route(load_yaml(args.config))
    if args.output_dir is not None:
        write_json(args.output_dir / "pressure_controller_route_plan_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
