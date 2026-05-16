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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_metric_exact_floor_dominant_micro_overfit_route.yaml"
EXPECTED_ALLOWLIST_ENTRY = "v2_r4_candidate_v3_floor_dominant_micro_overfit_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch"


def _mapping(value: Any, field: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be a mapping")
        return {}
    return value


def _path_exists(root: Path, value: Any, field: str, errors: list[str]) -> None:
    path = root / str(value)
    if not path.exists():
        errors.append(f"{field} missing: {path}")


def _expect_false(mapping: Mapping[str, Any], field: str, errors: list[str], prefix: str) -> None:
    if mapping.get(field) is not False:
        errors.append(f"{prefix}.{field} must be false")


def validate_route(config: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_metric_exact_floor_dominant_micro_overfit_route_v1":
        errors.append("schema_name mismatch")
    if config.get("package_id") != "r4_metric_exact_floor_dominant_micro_overfit_after_864332_v1":
        errors.append("package_id mismatch")

    source = _mapping(config.get("source_failure"), "source_failure", errors)
    if source.get("job_id") != "864332":
        errors.append("source_failure.job_id must be 864332")
    if source.get("failure_status") != "FAIL_R4_METRIC_EXACT_MICRO_OVERFIT_864332_TEACHER_FORCED_GATE":
        errors.append("source_failure.failure_status mismatch")
    _path_exists(root, source.get("review_doc", ""), "source_failure.review_doc", errors)
    _path_exists(root, source.get("review_summary", ""), "source_failure.review_summary", errors)
    if float(source.get("protected_lift_vs_base", 1.0)) >= 0.15:
        errors.append("source_failure.protected_lift_vs_base must be below gate")
    if float(source.get("protected_lift_vs_task_only", 1.0)) >= 0.10:
        errors.append("source_failure.protected_lift_vs_task_only must be below gate")
    if float(source.get("protected_rank1_rate", 0.0)) < 0.75:
        errors.append("source_failure.protected_rank1_rate should show rank-ordering solved")
    if float(source.get("final_floor_loss", 0.0)) < 0.15:
        errors.append("source_failure.final_floor_loss should show floor remained unsatisfied")

    interpretation = _mapping(config.get("failure_interpretation"), "failure_interpretation", errors)
    required_true = (
        "clean_slurm_completion",
        "rank_ordering_solved",
        "absolute_target_mass_insufficient",
        "target_mass_floor_unsatisfied",
        "ce_loss_dominated_previous_route",
    )
    for field in required_true:
        if interpretation.get(field) is not True:
            errors.append(f"failure_interpretation.{field} must be true")
    if interpretation.get("tokenizer_or_boundary_failure") is not False:
        errors.append("failure_interpretation.tokenizer_or_boundary_failure must be false")
    if interpretation.get("selected_repair") != "floor_dominant_metric_exact_micro_overfit":
        errors.append("failure_interpretation.selected_repair mismatch")

    route = _mapping(config.get("route_parameters"), "route_parameters", errors)
    if route.get("allowlist_entry") != EXPECTED_ALLOWLIST_ENTRY:
        errors.append("route_parameters.allowlist_entry mismatch")
    if route.get("wrapper") != EXPECTED_WRAPPER:
        errors.append("route_parameters.wrapper mismatch")
    _path_exists(root, route.get("wrapper", ""), "route_parameters.wrapper", errors)
    command = str(route.get("command_pattern", ""))
    required_command_fragments = (
        "SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus",
        "TASK_CE_WEIGHT=0.0",
        "TARGET_MASS_FLOOR_LAMBDA=50.0",
        "MARGIN_LAMBDA=1.0",
        "MAX_STEPS=128",
        "LEARNING_RATE=1e-4",
        EXPECTED_WRAPPER,
    )
    for fragment in required_command_fragments:
        if fragment not in command:
            errors.append(f"route_parameters.command_pattern missing {fragment}")
    if route.get("surface_margin_loss_mode") != "logsumexp_softplus":
        errors.append("route_parameters.surface_margin_loss_mode mismatch")
    if float(route.get("task_ce_weight", -1)) != 0.0:
        errors.append("route_parameters.task_ce_weight must be 0.0")
    if float(route.get("target_mass_floor_lambda", 0.0)) < 50.0:
        errors.append("route_parameters.target_mass_floor_lambda must be >= 50")
    if float(route.get("margin_lambda", 999.0)) > 1.0:
        errors.append("route_parameters.margin_lambda must be <= 1.0 for floor-dominant route")
    if int(route.get("max_steps", 0)) < 128:
        errors.append("route_parameters.max_steps must be >= 128")

    wrapper_path = root / EXPECTED_WRAPPER
    wrapper_text = wrapper_path.read_text(encoding="utf-8") if wrapper_path.exists() else ""
    wrapper = _mapping(config.get("required_wrapper_contract"), "required_wrapper_contract", errors)
    if wrapper.get("exposes_task_ce_weight") is not True:
        errors.append("required_wrapper_contract.exposes_task_ce_weight must be true")
    if 'TASK_CE_WEIGHT="${TASK_CE_WEIGHT:-1.0}"' not in wrapper_text:
        errors.append("wrapper does not expose TASK_CE_WEIGHT env default")
    if '--task-ce-weight "$TASK_CE_WEIGHT"' not in wrapper_text:
        errors.append("wrapper does not pass --task-ce-weight")
    if 'VALIDATE_PLAN_ONLY=1: exiting before model/tokenizer loading' not in wrapper_text:
        errors.append("wrapper plan-only guard missing")
    for field in ("exposes_surface_margin_loss_mode", "h200_pomplun_policy", "plan_only_exits_before_model_loading", "refuses_existing_output_dir"):
        if wrapper.get(field) is not True:
            errors.append(f"required_wrapper_contract.{field} must be true")

    compute = _mapping(config.get("compute_policy"), "compute_policy", errors)
    if compute.get("partition") != "pomplun" or compute.get("qos") != "pomplun":
        errors.append("compute_policy must use pomplun partition/qos")
    if compute.get("account") != "cs_yinxin.wan":
        errors.append("compute_policy.account must be cs_yinxin.wan")
    if compute.get("gres") != "gpu:h200:1":
        errors.append("compute_policy.gres must be gpu:h200:1")
    if compute.get("max_time") != "30-00:00:00":
        errors.append("compute_policy.max_time must be 30-00:00:00")
    _expect_false(compute, "allowlist_enabled_now", errors, "compute_policy")
    if compute.get("exactly_one_submission_when_unlocked") is not True:
        errors.append("compute_policy.exactly_one_submission_when_unlocked must be true")

    gate = _mapping(config.get("future_teacher_forced_gate"), "future_teacher_forced_gate", errors)
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

    locked = _mapping(config.get("not_unlocked_by_this_route_package"), "not_unlocked_by_this_route_package", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_package.{field} must be true")

    status = (
        "PASS_R4_METRIC_EXACT_FLOOR_DOMINANT_ROUTE_STATIC_VALIDATION_NO_COMPUTE"
        if not errors
        else "FAIL_R4_METRIC_EXACT_FLOOR_DOMINANT_ROUTE_STATIC_VALIDATION_NO_COMPUTE"
    )
    return {
        "status": status,
        "errors": errors,
        "package_id": config.get("package_id"),
        "source_job_id": source.get("job_id"),
        "selected_repair": interpretation.get("selected_repair"),
        "allowlist_entry": route.get("allowlist_entry"),
        "command_pattern": route.get("command_pattern"),
        "current_compute_unlocked": False,
        "allowlist_enabled": False,
        "slurm_job_submitted": False,
        "training_started": False,
        "generation_started": False,
        "model_scoring_started": False,
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 floor-dominant metric-exact route package.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_route(load_yaml(args.config))
    if args.output_dir is not None:
        write_json_new(args.output_dir / "floor_dominant_route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
