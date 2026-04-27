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

from scripts.prepare_matched_budget_baseline_calibration import _eval_cases
from src.evaluation.report import EvalRunSummary, maybe_load_result_json
from src.infrastructure.paths import current_timestamp, discover_repo_root


CASE_FIELDS = [
    "case_id",
    "method_id",
    "method_slug",
    "method_name",
    "baseline_family",
    "baseline_role",
    "owner_payload",
    "claim_payload",
    "label",
    "eval_kind",
    "negative_set",
    "seed",
    "query_budget",
    "target_far",
    "status",
    "result_class",
    "score_name",
    "ownership_score",
    "accepted",
    "verifier_success",
    "decoded_payload",
    "utility_acceptance_rate",
    "case_root",
    "eval_summary_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build B1/B2 baseline calibration artifacts.")
    parser.add_argument("--package-config", default="configs/reporting/matched_budget_baselines_v1.yaml")
    parser.add_argument("--output-dir", default="results/processed/paper_stats")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument(
        "--case-root-base",
        help="Optional base directory for baseline calibration roots. Defaults to EXP_SCRATCH/matched_budget_baselines_v1.",
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


def _find_latest_eval(case_root: Path) -> Path | None:
    matches = sorted(
        case_root.glob("runs/exp_eval/*/eval_summary.json"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0,
    )
    return matches[-1] if matches else None


def _pending_row(case: dict[str, Any], case_root: Path) -> dict[str, Any]:
    return {
        **case,
        "status": "pending",
        "result_class": "pending",
        "score_name": "ownership_score",
        "ownership_score": "",
        "accepted": False,
        "verifier_success": False,
        "decoded_payload": "",
        "utility_acceptance_rate": "",
        "case_root": str(case_root),
        "eval_summary_path": "",
    }


def _row_from_summary(case: dict[str, Any], case_root: Path, eval_summary_path: Path) -> dict[str, Any]:
    result = maybe_load_result_json(eval_summary_path)
    if not isinstance(result, EvalRunSummary):
        return {
            **_pending_row(case, case_root),
            "status": "invalid",
            "result_class": "invalid_eval_summary",
            "eval_summary_path": str(eval_summary_path),
        }
    return {
        **case,
        "status": result.status,
        "result_class": "completed" if result.status == "completed" else result.status,
        "score_name": "ownership_score",
        "ownership_score": result.match_ratio,
        "accepted": bool(result.accepted),
        "verifier_success": bool(result.verifier_success),
        "decoded_payload": result.decoded_payload or "",
        "utility_acceptance_rate": result.utility_acceptance_rate,
        "case_root": str(case_root),
        "eval_summary_path": str(eval_summary_path),
    }


def _collect_row(repo_root: Path, case: dict[str, Any]) -> dict[str, Any]:
    case_root = Path(str(case["case_root"]))
    if not case_root.is_absolute():
        case_root = repo_root / case_root
    eval_summary_path = _find_latest_eval(case_root)
    if not eval_summary_path:
        return _pending_row(case, case_root)
    return _row_from_summary(case, case_root, eval_summary_path)


def _method_status(method_slug: str, rows: list[dict[str, Any]], target_far: float) -> dict[str, Any]:
    method_rows = [row for row in rows if row["method_slug"] == method_slug]
    positives = [row for row in method_rows if row["label"] is True]
    negatives = [row for row in method_rows if row["label"] is False]
    completed = [row for row in method_rows if row["result_class"] == "completed"]
    pending = [row for row in method_rows if row["result_class"] == "pending"]
    return {
        "method_slug": method_slug,
        "target_far": target_far,
        "case_count": len(method_rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "completed_count": len(completed),
        "pending_count": len(pending),
        "threshold_status": "blocked_missing_negative_sets_or_scores",
        "frozen_threshold": "",
        "calibration_observed_far": "",
    }


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = _resolve_path(repo_root, args.tables_dir)
    root_base = _resolve_output_root_base(package_config, args.case_root_base)
    rows = [_collect_row(repo_root, case) for case in _eval_cases(package_config, root_base)]
    fixed = dict(package_config["fixed_contract"])
    target_far = float(fixed["target_far"])
    available_negative_sets = sorted(
        {str(row["negative_set"]) for row in rows if row["negative_set"]}
    )
    required_negative_sets = [str(item) for item in package_config["calibration_split"]["negative_sets"]]
    missing_negative_sets = [
        item for item in required_negative_sets if item not in set(available_negative_sets)
    ]
    completed = [row for row in rows if row["result_class"] == "completed"]
    pending = [row for row in rows if row["result_class"] == "pending"]
    method_rows = [
        _method_status(str(method["slug"]), rows, target_far)
        for method in package_config["baseline_methods"]
        if bool(method["requires_training"]) and not bool(method["requires_external_integration"])
    ]
    summary = {
        "schema_name": "baseline_calibration_summary",
        "schema_version": 2,
        "generated_at": current_timestamp(),
        "package_config_path": _repo_relative_path(repo_root, package_config_path),
        "new_case_root_base": root_base,
        "target_far": target_far,
        "status": "pending_real_calibration_scores",
        "thresholds_frozen": False,
        "threshold_freeze_allowed": False,
        "threshold_freeze_blockers": [
            *[f"missing_negative_set:{item}" for item in missing_negative_sets],
            "pending_calibration_eval_summaries" if pending else "",
        ],
        "available_negative_sets": available_negative_sets,
        "missing_negative_sets": missing_negative_sets,
        "case_count": len(rows),
        "completed_count": len(completed),
        "pending_count": len(pending),
        "method_rows": method_rows,
    }
    far_rows = [
        {
            "method_slug": row["method_slug"],
            "target_far": target_far,
            "negative_set": negative_set,
            "observed_far": "",
            "false_accept_count": "",
            "negative_count": "",
            "status": "pending",
        }
        for row in method_rows
        for negative_set in required_negative_sets
    ]
    utility_rows = [
        {
            "method_slug": row["method_slug"],
            "utility_acceptance_rate": "",
            "utility_delta_vs_foundation": "",
            "utility_delta_vs_primary": "",
            "utility_match_status": "pending",
        }
        for row in method_rows
    ]
    _write_json(output_dir / "baseline_calibration_summary.json", summary)
    _write_csv(tables_dir / "baseline_calibration_cases.csv", rows, CASE_FIELDS)
    _write_csv(tables_dir / "baseline_far_summary.csv", far_rows, list(far_rows[0]) if far_rows else [])
    _write_csv(tables_dir / "baseline_utility_summary.csv", utility_rows, list(utility_rows[0]) if utility_rows else [])
    print(f"wrote baseline calibration summary to {output_dir / 'baseline_calibration_summary.json'}")
    print(f"wrote baseline calibration cases to {tables_dir / 'baseline_calibration_cases.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
