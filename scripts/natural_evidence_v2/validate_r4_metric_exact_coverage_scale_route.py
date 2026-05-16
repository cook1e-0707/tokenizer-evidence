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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_metric_exact_coverage_scale_micro_overfit_route.yaml"
EXPECTED_ALLOWLIST_ENTRY = "v2_r4_candidate_v3_coverage_scale_micro_overfit_h200"
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


def validate_route(config: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_metric_exact_coverage_scale_micro_overfit_route_v1":
        errors.append("schema_name mismatch")
    if config.get("package_id") != "r4_metric_exact_coverage_scale_micro_overfit_after_864705_v1":
        errors.append("package_id mismatch")

    source = _mapping(config.get("source_failure"), "source_failure", errors)
    if source.get("job_id") != "864705":
        errors.append("source_failure.job_id must be 864705")
    if source.get("failure_status") != "FAIL_R4_METRIC_EXACT_FLOOR_DOMINANT_864705_TEACHER_FORCED_GATE":
        errors.append("source_failure.failure_status mismatch")
    _path_exists(root, source.get("review_doc", ""), "source_failure.review_doc", errors)
    _path_exists(root, source.get("review_summary", ""), "source_failure.review_summary", errors)
    if float(source.get("protected_lift_vs_base", 1.0)) >= 0.15:
        errors.append("source_failure.protected_lift_vs_base must be below gate")
    if float(source.get("protected_lift_vs_task_only", 1.0)) >= 0.10:
        errors.append("source_failure.protected_lift_vs_task_only must be below gate")
    if float(source.get("protected_rank1_rate", 0.0)) < 0.75:
        errors.append("source_failure.protected_rank1_rate should show rank-ordering solved")

    interpretation = _mapping(config.get("failure_interpretation"), "failure_interpretation", errors)
    for field in (
        "clean_slurm_completion",
        "rank_ordering_solved",
        "absolute_target_mass_insufficient",
        "floor_dominant_pressure_directionally_effective",
        "training_coverage_insufficient",
    ):
        if interpretation.get(field) is not True:
            errors.append(f"failure_interpretation.{field} must be true")
    if interpretation.get("tokenizer_or_boundary_failure") is not False:
        errors.append("failure_interpretation.tokenizer_or_boundary_failure must be false")
    if interpretation.get("selected_repair") != "coverage_scale_floor_dominant_metric_exact_micro_overfit":
        errors.append("failure_interpretation.selected_repair mismatch")

    route = _mapping(config.get("route_parameters"), "route_parameters", errors)
    if route.get("allowlist_entry") != EXPECTED_ALLOWLIST_ENTRY:
        errors.append("route_parameters.allowlist_entry mismatch")
    if route.get("wrapper") != EXPECTED_WRAPPER:
        errors.append("route_parameters.wrapper mismatch")
    _path_exists(root, route.get("wrapper", ""), "route_parameters.wrapper", errors)
    command = str(route.get("command_pattern", ""))
    for fragment in (
        "SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus",
        "TASK_CE_WEIGHT=0.0",
        "TARGET_MASS_FLOOR=0.25",
        "TARGET_MASS_FLOOR_LAMBDA=75.0",
        "MAX_TRAIN_ROWS=8192",
        "MAX_SCORE_ROWS=8192",
        "MAX_STEPS=4096",
        "BATCH_SIZE=2",
        "GRADIENT_ACCUMULATION_STEPS=8",
        EXPECTED_WRAPPER,
    ):
        if fragment not in command:
            errors.append(f"route_parameters.command_pattern missing {fragment}")
    if route.get("surface_margin_loss_mode") != "logsumexp_softplus":
        errors.append("route_parameters.surface_margin_loss_mode mismatch")
    if float(route.get("task_ce_weight", -1)) != 0.0:
        errors.append("route_parameters.task_ce_weight must be 0.0")
    if float(route.get("target_mass_floor", 0.0)) < 0.25:
        errors.append("route_parameters.target_mass_floor must be >= 0.25")
    if float(route.get("target_mass_floor_lambda", 0.0)) < 75.0:
        errors.append("route_parameters.target_mass_floor_lambda must be >= 75")
    max_train_rows = int(route.get("max_train_rows", 0))
    max_steps = int(route.get("max_steps", 0))
    batch_size = int(route.get("batch_size", 0))
    if max_train_rows < 8192:
        errors.append("route_parameters.max_train_rows must be >= 8192")
    if max_steps * batch_size < max_train_rows:
        errors.append("route_parameters.max_steps * batch_size must cover max_train_rows")

    wrapper_path = root / EXPECTED_WRAPPER
    wrapper_text = wrapper_path.read_text(encoding="utf-8") if wrapper_path.exists() else ""
    for text, label in (
        ('TASK_CE_WEIGHT="${TASK_CE_WEIGHT:-1.0}"', "TASK_CE_WEIGHT env default"),
        ('MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-512}"', "MAX_TRAIN_ROWS env default"),
        ('BATCH_SIZE="${BATCH_SIZE:-1}"', "BATCH_SIZE env default"),
        ('GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"', "GRADIENT_ACCUMULATION_STEPS env default"),
        ('--max-rows "$MAX_TRAIN_ROWS"', "max rows forwarding"),
        ('--batch-size "$BATCH_SIZE"', "batch size forwarding"),
        ('--gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"', "grad accumulation forwarding"),
        ('VALIDATE_PLAN_ONLY=1: exiting before model/tokenizer loading', "plan-only guard"),
    ):
        if text not in wrapper_text:
            errors.append(f"wrapper missing {label}")

    compute = _mapping(config.get("compute_policy"), "compute_policy", errors)
    if compute.get("partition") != "pomplun" or compute.get("qos") != "pomplun":
        errors.append("compute_policy must use pomplun partition/qos")
    if compute.get("account") != "cs_yinxin.wan":
        errors.append("compute_policy.account must be cs_yinxin.wan")
    if compute.get("gres") != "gpu:h200:1":
        errors.append("compute_policy.gres must be gpu:h200:1")
    if compute.get("max_time") != "30-00:00:00":
        errors.append("compute_policy.max_time must be 30-00:00:00")
    if compute.get("allowlist_enabled_now") is not False:
        errors.append("compute_policy.allowlist_enabled_now must be false")
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

    locked = _mapping(config.get("not_unlocked_by_this_route_package"), "not_unlocked_by_this_route_package", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_package.{field} must be true")

    status = (
        "PASS_R4_METRIC_EXACT_COVERAGE_SCALE_ROUTE_STATIC_VALIDATION_NO_COMPUTE"
        if not errors
        else "FAIL_R4_METRIC_EXACT_COVERAGE_SCALE_ROUTE_STATIC_VALIDATION_NO_COMPUTE"
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
    parser = argparse.ArgumentParser(description="Validate the R4 coverage-scale metric-exact route package.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_route(load_yaml(args.config))
    if args.output_dir is not None:
        write_json_new(args.output_dir / "coverage_scale_route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
