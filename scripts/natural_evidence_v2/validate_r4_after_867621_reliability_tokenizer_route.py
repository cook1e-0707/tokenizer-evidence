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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_867621_reliability_tokenizer_preflight_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_h200.sbatch"
EXPECTED_ROWS = 4096


def _mapping(value: Any, field: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be a mapping")
        return {}
    return value


def _path(value: Any, field: str, errors: list[str]) -> Path:
    path = ROOT / str(value)
    if not path.exists():
        errors.append(f"{field} missing: {path}")
    return path


def _read_json(path: Path, field: str, errors: list[str]) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        errors.append(f"{field} unreadable JSON: {exc}")
        return {}
    if not isinstance(payload, Mapping):
        errors.append(f"{field} must be a JSON object")
        return {}
    return payload


def _expect_false(mapping: Mapping[str, Any], field: str, errors: list[str], prefix: str) -> None:
    if mapping.get(field) is not False:
        errors.append(f"{prefix}.{field} must be false")


def _expect_true(mapping: Mapping[str, Any], field: str, errors: list[str], prefix: str) -> None:
    if mapping.get(field) is not True:
        errors.append(f"{prefix}.{field} must be true")


def _find_allowlist_entry(allowlist: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        entries = allowlist.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("name") == name:
                return entry
    return None


def validate_route(config: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_after_867621_reliability_tokenizer_preflight_route_v1":
        errors.append("schema_name mismatch")
    if config.get("package_id") != "r4_after_867621_reliability_tokenizer_preflight_route_v1":
        errors.append("package_id mismatch")

    source_failure = _mapping(config.get("source_failure"), "source_failure", errors)
    review_path = _path(source_failure.get("review_summary", ""), "source_failure.review_summary", errors)
    failure_path = _path(source_failure.get("failure_analysis", ""), "source_failure.failure_analysis", errors)
    review = _read_json(review_path, "source_failure.review_summary", errors) if review_path.exists() else {}
    failure = _read_json(failure_path, "source_failure.failure_analysis", errors) if failure_path.exists() else {}
    if source_failure.get("job_id") != "867621":
        errors.append("source_failure.job_id must be 867621")
    if review.get("status") != source_failure.get("expected_review_status"):
        errors.append("source_failure review status mismatch")
    if failure.get("status") != source_failure.get("expected_failure_status"):
        errors.append("source_failure failure-analysis status mismatch")
    if failure.get("root_cause") != source_failure.get("expected_root_cause"):
        errors.append("source_failure root_cause mismatch")

    source_rows = _mapping(config.get("source_rows"), "source_rows", errors)
    rows_path = _path(source_rows.get("rows", ""), "source_rows.rows", errors)
    rows_summary_path = _path(source_rows.get("summary", ""), "source_rows.summary", errors)
    rows_summary = _read_json(rows_summary_path, "source_rows.summary", errors) if rows_summary_path.exists() else {}
    if rows_summary.get("status") != source_rows.get("expected_status"):
        errors.append("source_rows summary status mismatch")
    if int(rows_summary.get("row_count", 0)) != EXPECTED_ROWS:
        errors.append("source_rows row_count must be 4096")
    if int(source_rows.get("expected_rows", 0)) != EXPECTED_ROWS:
        errors.append("source_rows.expected_rows must be 4096")
    if int(rows_summary.get("selected_coordinate_count", 0)) != int(source_rows.get("expected_selected_coordinates", -1)):
        errors.append("source_rows selected_coordinate_count mismatch")
    if int(rows_summary.get("selected_coordinate_count", 0)) != 16:
        errors.append("source_rows selected_coordinate_count must be 16")
    if rows_summary.get("current_two_way_scorer_compatible") is not True:
        errors.append("source_rows must be current_two_way_scorer_compatible=true")
    for field in ("tokenizer_validation_started", "model_scoring_started", "training_started", "generation_started", "slurm_submitted"):
        _expect_false(rows_summary, field, errors, "source_rows.summary")
    if rows_path.exists():
        row_count = sum(1 for _ in rows_path.open("r", encoding="utf-8"))
        if row_count != EXPECTED_ROWS:
            errors.append("source_rows JSONL line count must be 4096")

    static = _mapping(config.get("source_static_preflight"), "source_static_preflight", errors)
    static_summary_path = _path(static.get("summary", ""), "source_static_preflight.summary", errors)
    static_summary = _read_json(static_summary_path, "source_static_preflight.summary", errors) if static_summary_path.exists() else {}
    if static_summary.get("status") != static.get("expected_status"):
        errors.append("source_static_preflight status mismatch")
    if int(static_summary.get("checked_row_count", 0)) != EXPECTED_ROWS:
        errors.append("source_static_preflight checked_row_count must be 4096")
    if int(static_summary.get("failed_row_count", -1)) != 0:
        errors.append("source_static_preflight failed_row_count must be 0")
    for field in ("qwen_tokenizer_preflight_started", "model_forward_pass_started", "scoring_job_submitted", "training_started", "generation_started"):
        _expect_false(static_summary, field, errors, "source_static_preflight.summary")

    route = _mapping(config.get("route"), "route", errors)
    if route.get("allowlist_entry") != EXPECTED_ENTRY:
        errors.append("route.allowlist_entry mismatch")
    if route.get("wrapper") != EXPECTED_WRAPPER:
        errors.append("route.wrapper mismatch")
    wrapper_path = _path(route.get("wrapper", ""), "route.wrapper", errors)
    if route.get("command_pattern") != f"sbatch {EXPECTED_WRAPPER}":
        errors.append("route.command_pattern mismatch")
    if route.get("run_qwen_tokenizer") is not True:
        errors.append("route.run_qwen_tokenizer must be true")
    if int(route.get("max_rows", 0)) != EXPECTED_ROWS:
        errors.append("route.max_rows must be 4096")
    for field in ("model_forward_allowed", "scoring_allowed", "generation_allowed", "training_allowed"):
        _expect_false(route, field, errors, "route")

    wrapper_text = wrapper_path.read_text(encoding="utf-8") if wrapper_path.exists() else ""
    required_wrapper_fragments = (
        "#SBATCH --partition=pomplun",
        "#SBATCH --account=cs_yinxin.wan",
        "#SBATCH --qos=pomplun",
        "#SBATCH --gres=gpu:h200:1",
        "#SBATCH --time=30-00:00:00",
        "r4_after_867621_reliability_surface_mass_rows_20260516/reliability_surface_mass_rows.jsonl",
        "model_forward_started=false",
        "scoring_started=false",
        "generation_started=false",
        "training_started=false",
        "--run-qwen-tokenizer",
    )
    for fragment in required_wrapper_fragments:
        if fragment not in wrapper_text:
            errors.append(f"wrapper missing fragment: {fragment}")

    compute = _mapping(config.get("compute_policy"), "compute_policy", errors)
    if compute.get("partition") != "pomplun" or compute.get("qos") != "pomplun":
        errors.append("compute_policy must use pomplun")
    if compute.get("account") != "cs_yinxin.wan":
        errors.append("compute_policy.account mismatch")
    if compute.get("gres") != "gpu:h200:1":
        errors.append("compute_policy.gres mismatch")
    if compute.get("max_time") != "30-00:00:00":
        errors.append("compute_policy.max_time mismatch")
    _expect_false(compute, "allowlist_enabled_now", errors, "compute_policy")
    for field in (
        "exactly_one_submission_when_unlocked",
        "remote_hash_preflight_required",
        "hermes_notification_required",
        "post_submit_allowlist_shutdown_required",
    ):
        _expect_true(compute, field, errors, "compute_policy")

    gate = _mapping(config.get("future_tokenizer_gate"), "future_tokenizer_gate", errors)
    if int(gate.get("checked_rows", 0)) != EXPECTED_ROWS:
        errors.append("future_tokenizer_gate.checked_rows must be 4096")
    for field in ("failed_rows_max", "empty_target_id_row_count_max", "empty_other_id_row_count_max", "target_other_overlap_row_count_max"):
        if int(gate.get(field, -1)) != 0:
            errors.append(f"future_tokenizer_gate.{field} must be 0")

    future = _mapping(config.get("future_after_tokenizer_pass"), "future_after_tokenizer_pass", errors)
    if future.get("next_route") != "reviewed_h200_teacher_forced_surface_mass_scoring_only":
        errors.append("future_after_tokenizer_pass.next_route mismatch")
    _expect_false(future, "generation_allowed", errors, "future_after_tokenizer_pass")

    locked = _mapping(config.get("not_unlocked_by_this_route_package"), "not_unlocked_by_this_route_package", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_package.{field} must be true")

    allowlist = load_yaml(ALLOWLIST)
    entry = _find_allowlist_entry(allowlist, EXPECTED_ENTRY)
    if entry is None:
        errors.append("allowlist entry missing")
    else:
        if entry.get("enabled") is not False:
            errors.append("allowlist entry must be disabled")
        if entry.get("command_pattern") != f"sbatch {EXPECTED_WRAPPER}":
            errors.append("allowlist command_pattern mismatch")

    status = (
        "PASS_R4_AFTER_867621_RELIABILITY_TOKENIZER_PREFLIGHT_ROUTE_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_867621_RELIABILITY_TOKENIZER_PREFLIGHT_ROUTE_VALIDATION_NO_SUBMIT"
    )
    return {
        "status": status,
        "errors": errors,
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
        "score_rows": source_rows.get("rows"),
        "expected_rows": EXPECTED_ROWS,
        "current_compute_unlocked": False,
        "allowlist_enabled": False,
        "slurm_job_submitted": False,
        "tokenizer_validation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate after-867621 reliability tokenizer preflight route.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_route(load_yaml(args.config))
    if args.output_dir is not None:
        write_json_new(args.output_dir / "reliability_tokenizer_route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
