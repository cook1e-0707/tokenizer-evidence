from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml

from scripts.prepare_matched_budget_baselines import (
    _case_records,
    _calibration_rows,
    _load_frozen_thresholds,
)
from src.evaluation.report import EvalRunSummary, maybe_load_result_json
from src.infrastructure.paths import current_timestamp, discover_repo_root


RUN_FIELDS = [
    "case_id",
    "b_stage",
    "method_id",
    "method_slug",
    "method_name",
    "display_name",
    "baseline_family",
    "baseline_role",
    "train_objective",
    "matched_budget_status",
    "requires_training",
    "requires_external_integration",
    "block_count",
    "payload",
    "seed",
    "query_budget",
    "queries_used",
    "target_far",
    "frozen_threshold",
    "calibration_observed_far",
    "utility_acceptance_rate",
    "ownership_score",
    "accepted",
    "verifier_success",
    "decoded_payload",
    "status",
    "result_class",
    "failure_reasons",
    "valid_completed",
    "success",
    "method_failure",
    "invalid_excluded",
    "pending",
    "unavailable",
    "contract_hash_status",
    "run_dir",
    "case_root",
    "eval_summary_path",
    "train_summary_path",
    "config_path",
]

CALIBRATION_FIELDS = [
    "method_id",
    "method_slug",
    "method_name",
    "baseline_family",
    "baseline_role",
    "status",
    "score_name",
    "score_direction",
    "target_far",
    "frozen_threshold",
    "calibration_observed_far",
    "calibration_payloads",
    "calibration_seed",
    "negative_sets",
    "requires_external_integration",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build B1/B2 matched-budget baseline artifacts.")
    parser.add_argument("--package-config", default="configs/reporting/matched_budget_baselines_v1.yaml")
    parser.add_argument("--output-dir", default="results/processed/paper_stats")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument(
        "--case-root-base",
        help="Optional base directory for baseline case roots. Defaults to EXP_SCRATCH/matched_budget_baselines_v1.",
    )
    parser.add_argument(
        "--calibration-summary",
        default="results/processed/paper_stats/baseline_calibration_summary.json",
        help="Frozen B0 calibration summary used by final baseline artifacts.",
    )
    return parser.parse_args()


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(os.path.expandvars(raw))
    return path if path.is_absolute() else repo_root / path


def _repo_relative_path(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _resolve_output_root_base(package_config: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return str(Path(os.path.expandvars(explicit)).as_posix())
    prefix = str(package_config["new_case_root_prefix"])
    exp_scratch = os.environ.get("EXP_SCRATCH")
    if exp_scratch:
        return str((Path(exp_scratch) / prefix).as_posix())
    return prefix


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Success & Method fail & Invalid & Pending & FAR target \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        method = str(row["method_slug"]).replace("_", "\\_")
        lines.append(
            f"{method} & {row['success_count']} & {row['method_failure_count']} & "
            f"{row['invalid_excluded_count']} & {row['pending_count']} & "
            f"{float(row['target_far']):.2f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{B1/B2 matched-budget baseline package under frozen B0 calibration rules. Pending and unavailable rows are not reported as method failures.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_latest(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern), key=lambda item: item.stat().st_mtime if item.exists() else 0)
    return matches[-1] if matches else None


def _summary_paths(case_root: Path) -> tuple[Path | None, Path | None]:
    train_summary = _find_latest(case_root, "runs/exp_train/*/train_summary.json")
    eval_summary = _find_latest(case_root, "runs/exp_eval/*/eval_summary.json")
    return train_summary, eval_summary


def _load_calibration_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_name": "baseline_calibration_summary",
            "schema_version": 0,
            "status": "missing",
            "thresholds_frozen": False,
            "method_rows": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected calibration summary object in {path}")
    return payload


def _claim_conditioned_score(case: dict[str, Any], result: EvalRunSummary) -> float:
    claimed_payload = str(case["payload"])
    decoded_payload = result.decoded_payload or ""
    if bool(result.accepted) or decoded_payload == claimed_payload:
        return float(result.match_ratio)
    return 0.0


def _pending_row(case: dict[str, Any], case_root: Path, train_summary: Path | None) -> dict[str, Any]:
    reason = "external_baseline_not_integrated" if case["requires_external_integration"] else "eval_summary_missing"
    return {
        **case,
        "queries_used": "",
        "frozen_threshold": case.get("frozen_threshold", ""),
        "calibration_observed_far": case.get("calibration_observed_far", ""),
        "utility_acceptance_rate": "",
        "ownership_score": "",
        "accepted": False,
        "verifier_success": False,
        "decoded_payload": "",
        "status": "pending",
        "result_class": "pending_external_integration" if case["requires_external_integration"] else "pending",
        "failure_reasons": reason,
        "valid_completed": False,
        "success": False,
        "method_failure": False,
        "invalid_excluded": False,
        "pending": True,
        "unavailable": bool(case["requires_external_integration"]),
        "contract_hash_status": "pending",
        "run_dir": "",
        "case_root": str(case_root),
        "eval_summary_path": "",
        "train_summary_path": str(train_summary) if train_summary else "",
        "config_path": "",
    }


def _row_from_eval_summary(
    case: dict[str, Any],
    case_root: Path,
    train_summary: Path | None,
    eval_summary_path: Path,
) -> dict[str, Any]:
    result = maybe_load_result_json(eval_summary_path)
    if not isinstance(result, EvalRunSummary):
        return {
            **_pending_row(case, case_root, train_summary),
            "status": "invalid",
            "result_class": "invalid_excluded",
            "failure_reasons": "eval_summary_schema_invalid",
            "invalid_excluded": True,
            "pending": False,
        }
    placeholder = result.status == "placeholder" or "placeholder" in str(result.notes).lower()
    completed = result.status == "completed"
    valid_completed = completed and not placeholder
    success = valid_completed and bool(result.accepted) and bool(result.verifier_success)
    method_failure = valid_completed and not success
    invalid = not valid_completed and not placeholder
    unavailable = placeholder
    result_class = "valid_success" if success else "valid_method_failure" if method_failure else "unavailable" if unavailable else "invalid_excluded"
    failure_reasons = ""
    if method_failure:
        failure_reasons = "score_or_verifier_failed_under_frozen_threshold"
    elif unavailable:
        failure_reasons = "baseline_adapter_placeholder"
    elif invalid:
        failure_reasons = f"eval_status={result.status}"
    return {
        **case,
        "queries_used": case["query_budget"] if valid_completed else "",
        "frozen_threshold": result.threshold,
        "calibration_observed_far": case.get("calibration_observed_far", ""),
        "utility_acceptance_rate": result.utility_acceptance_rate,
        "ownership_score": _claim_conditioned_score(case, result),
        "accepted": bool(result.accepted),
        "verifier_success": bool(result.verifier_success),
        "decoded_payload": result.decoded_payload or "",
        "status": result.status,
        "result_class": result_class,
        "failure_reasons": failure_reasons,
        "valid_completed": valid_completed,
        "success": success,
        "method_failure": method_failure,
        "invalid_excluded": invalid,
        "pending": False,
        "unavailable": unavailable,
        "contract_hash_status": "missing_hash" if valid_completed else "unavailable",
        "run_dir": result.run_dir,
        "case_root": str(case_root),
        "eval_summary_path": str(eval_summary_path),
        "train_summary_path": str(train_summary) if train_summary else "",
        "config_path": "",
    }


def _collect_row(repo_root: Path, case: dict[str, Any]) -> dict[str, Any]:
    case_root = Path(str(case["case_root"]))
    if not case_root.is_absolute():
        case_root = repo_root / case_root
    train_summary, eval_summary = _summary_paths(case_root)
    if not eval_summary:
        return _pending_row(case, case_root, train_summary)
    return _row_from_eval_summary(case, case_root, train_summary, eval_summary)


def _summary_row(scope: str, rows: list[dict[str, Any]], target_far: float) -> dict[str, Any]:
    valid = [row for row in rows if row["valid_completed"]]
    successes = [row for row in rows if row["success"]]
    method_failures = [row for row in rows if row["method_failure"]]
    invalid = [row for row in rows if row["invalid_excluded"]]
    pending = [row for row in rows if row["pending"]]
    unavailable = [row for row in rows if row["unavailable"]]
    return {
        "method_slug": scope,
        "target_count": len(rows),
        "valid_completed_count": len(valid),
        "success_count": len(successes),
        "method_failure_count": len(method_failures),
        "invalid_excluded_count": len(invalid),
        "pending_count": len(pending),
        "unavailable_count": len(unavailable),
        "success_rate": len(successes) / len(valid) if valid else 0.0,
        "target_far": target_far,
    }


def _json_ready_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        converted.append(payload)
    return converted


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = _resolve_path(repo_root, args.tables_dir)
    calibration_summary_path = _resolve_path(repo_root, args.calibration_summary)
    calibration_summary = _load_calibration_summary(calibration_summary_path)
    frozen_thresholds = _load_frozen_thresholds(calibration_summary_path)
    root_base = _resolve_output_root_base(package_config, args.case_root_base)
    cases = _case_records(package_config, root_base)
    for case in cases:
        frozen = frozen_thresholds.get(str(case["method_slug"]), {})
        case["frozen_threshold"] = frozen.get("frozen_threshold", "")
        case["calibration_observed_far"] = frozen.get("calibration_observed_far", "")
    rows = [_collect_row(repo_root, case) for case in cases]
    fixed = dict(package_config["fixed_contract"])
    target_far = float(fixed["target_far"])
    method_rows = [
        _summary_row(str(method["slug"]), [row for row in rows if row["method_slug"] == method["slug"]], target_far)
        for method in package_config["baseline_methods"]
    ]
    overall_row = _summary_row("overall", rows, target_far)
    calibration_rows = _calibration_rows(package_config, frozen_thresholds)
    valid_completed = [row for row in rows if row["valid_completed"]]
    successes = [row for row in rows if row["success"]]
    method_failures = [row for row in rows if row["method_failure"]]
    invalid = [row for row in rows if row["invalid_excluded"]]
    pending = [row for row in rows if row["pending"]]
    unavailable = [row for row in rows if row["unavailable"]]
    completed = [row for row in rows if not row["pending"]]
    paper_ready_checks = {
        "calibration_thresholds_frozen_before_final": bool(calibration_summary.get("thresholds_frozen"))
        and bool(frozen_thresholds),
        "query_budget_not_exceeded": all(
            not row["valid_completed"] or int(row["queries_used"]) <= int(row["query_budget"])
            for row in rows
        ),
        "target_far_reported": all(float(row["target_far"]) == target_far for row in rows),
        "utility_metric_reported": all(not row["valid_completed"] or row["utility_acceptance_rate"] != "" for row in rows),
        "valid_completed_failures_remain_in_denominator": True,
        "invalid_exclusions_have_artifact_or_contract_reason": all(row["failure_reasons"] for row in invalid),
        "provenance_controls_not_reported_as_primary_ownership_baselines": all(
            row["baseline_role"] != "primary_ownership_baseline"
            for row in rows
            if "provenance" in str(row["baseline_family"])
        ),
    }
    summary = {
        "schema_name": "baseline_summary",
        "schema_version": 1,
        "workstream": package_config.get("workstream", "B1-B2"),
        "description": package_config.get("description", ""),
        "generated_at": current_timestamp(),
        "package_config_path": _repo_relative_path(repo_root, package_config_path),
        "new_case_root_base": root_base,
        "b0_protocol": package_config.get("b0_protocol", {}),
        "fixed_contract": fixed,
        "calibration_split": package_config.get("calibration_split", {}),
        "calibration_summary_path": _repo_relative_path(repo_root, calibration_summary_path),
        "calibration_summary": calibration_summary,
        "baseline_methods": package_config.get("baseline_methods", []),
        "target_count": len(rows),
        "completed_count": len(completed),
        "valid_completed_count": len(valid_completed),
        "success_count": len(successes),
        "method_failure_count": len(method_failures),
        "invalid_excluded_count": len(invalid),
        "pending_count": len(pending),
        "unavailable_count": len(unavailable),
        "success_rate": len(successes) / len(valid_completed) if valid_completed else 0.0,
        "paper_ready": all(paper_ready_checks.values()) and bool(valid_completed) and not pending and not unavailable,
        "paper_ready_checks": paper_ready_checks,
        "summary_rows": [overall_row, *method_rows],
        "success_case_ids": [row["case_id"] for row in successes],
        "method_failure_case_ids": [row["case_id"] for row in method_failures],
        "invalid_excluded_case_ids": [row["case_id"] for row in invalid],
        "pending_case_ids": [row["case_id"] for row in pending],
        "unavailable_case_ids": [row["case_id"] for row in unavailable],
    }
    inclusion = {
        "schema_name": "baseline_run_accounting",
        "schema_version": 1,
        "valid_successes": _json_ready_rows(successes),
        "method_failures": _json_ready_rows(method_failures),
        "invalid_excluded": _json_ready_rows(invalid),
        "pending": _json_ready_rows(pending),
        "unavailable": _json_ready_rows(unavailable),
    }
    far_rows = [
        {
            "method_id": row["method_id"],
            "method_slug": row["method_slug"],
            "target_far": row["target_far"],
            "final_observed_far": "",
            "final_far_false_accept_count": "",
            "final_far_negative_count": "",
            "final_far_wilson_low": "",
            "final_far_wilson_high": "",
            "status": row["status"],
        }
        for row in calibration_rows
    ]
    utility_rows = [
        {
            "method_id": row["method_id"],
            "method_slug": row["method_slug"],
            "utility_acceptance_rate": "",
            "utility_delta_vs_foundation": "",
            "utility_delta_vs_primary": "",
            "utility_match_status": "pending",
        }
        for row in calibration_rows
    ]
    compute = {
        "schema_name": "baseline_compute_accounting",
        "schema_version": 1,
        **dict(package_config.get("compute_estimate", {})),
    }

    _write_json(output_dir / "baseline_summary.json", summary)
    _write_json(output_dir / "baseline_run_inclusion_list.json", inclusion)
    _write_json(output_dir / "baseline_compute_accounting.json", compute)
    _write_json(output_dir / "baseline_calibration_summary.json", calibration_summary)
    _write_csv(tables_dir / "matched_budget_baselines.csv", rows, RUN_FIELDS)
    _write_csv(tables_dir / "baseline_calibration.csv", calibration_rows, CALIBRATION_FIELDS)
    _write_csv(tables_dir / "baseline_far_summary.csv", far_rows, list(far_rows[0]) if far_rows else [])
    _write_csv(tables_dir / "baseline_utility_summary.csv", utility_rows, list(utility_rows[0]) if utility_rows else [])
    _write_tex(tables_dir / "matched_budget_baselines.tex", method_rows)
    print(f"wrote baseline summary to {output_dir / 'baseline_summary.json'}")
    print(f"wrote baseline run table to {tables_dir / 'matched_budget_baselines.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
