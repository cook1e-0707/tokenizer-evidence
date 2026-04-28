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
from src.infrastructure.registry import RegistryRecord, latest_registry_by_manifest_id, load_registry


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
    parser.add_argument(
        "--eval-registry",
        action="append",
        default=None,
        help="Optional calibration eval job registry used for exact manifest_id to output_dir mapping.",
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


def _eval_manifest_id(case: dict[str, Any]) -> str:
    return (
        f"baseline-calibration-eval-{case['method_slug']}-{str(case['owner_payload']).lower()}"
        f"-claim-{str(case['claim_payload']).lower()}-s{case['seed']}"
    )


def _resolved_eval_payload(config_path: Path) -> str:
    if not config_path.exists():
        return ""
    try:
        payload = _load_yaml(config_path)
    except (OSError, ValueError, yaml.YAMLError):
        return ""
    eval_section = payload.get("eval", {})
    if not isinstance(eval_section, dict):
        return ""
    return str(eval_section.get("payload_text", ""))


def _find_eval_summary(
    case_root: Path,
    case: dict[str, Any],
    registry_by_manifest_id: dict[str, RegistryRecord],
) -> Path | None:
    manifest_id = _eval_manifest_id(case)
    record = registry_by_manifest_id.get(manifest_id)
    if record is not None and record.output_dir:
        candidate = Path(record.output_dir) / "eval_summary.json"
        if candidate.exists():
            return candidate

    matches = [
        item
        for item in case_root.glob("runs/exp_eval/*/eval_summary.json")
        if _resolved_eval_payload(item.parent / "config.resolved.yaml") == str(case["claim_payload"])
    ]
    if not matches:
        all_matches = list(case_root.glob("runs/exp_eval/*/eval_summary.json"))
        if len(all_matches) == 1:
            return all_matches[0]
        return None
    matches = sorted(matches, key=lambda item: item.stat().st_mtime if item.exists() else 0)
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
        "result_class": "valid_completed",
        "score_name": "ownership_score",
        "ownership_score": result.match_ratio,
        "accepted": bool(result.accepted),
        "verifier_success": bool(result.verifier_success),
        "decoded_payload": result.decoded_payload or "",
        "utility_acceptance_rate": result.utility_acceptance_rate,
        "case_root": str(case_root),
        "eval_summary_path": str(eval_summary_path),
    }


def _score(row: dict[str, Any]) -> float | None:
    try:
        return float(row["ownership_score"])
    except (TypeError, ValueError):
        return None


def _select_threshold(rows: list[dict[str, Any]], target_far: float) -> dict[str, Any]:
    positives = [row for row in rows if row["label"] is True and row["result_class"] == "valid_completed"]
    negatives = [row for row in rows if row["label"] is False and row["result_class"] == "valid_completed"]
    scores = sorted(
        {
            score
            for row in [*positives, *negatives]
            if (score := _score(row)) is not None
        }
    )
    if not positives or not negatives or not scores:
        return {
            "threshold_status": "blocked_missing_scores",
            "frozen_threshold": "",
            "calibration_observed_far": "",
            "false_accept_count": "",
            "negative_count": len(negatives),
            "true_accept_count": "",
            "positive_count": len(positives),
            "calibration_sensitivity": "",
        }
    candidates = sorted({*scores, max(scores) + 1e-12})
    selected: float | None = None
    selected_false_accepts = 0
    selected_true_accepts = 0
    for candidate in candidates:
        false_accepts = sum(
            1
            for row in negatives
            if (score := _score(row)) is not None and score >= candidate
        )
        observed_far = false_accepts / len(negatives) if negatives else 1.0
        if observed_far <= target_far:
            selected = candidate
            selected_false_accepts = false_accepts
            selected_true_accepts = sum(
                1
                for row in positives
                if (score := _score(row)) is not None and score >= candidate
            )
            break
    if selected is None:
        raise RuntimeError("internal error: threshold candidates must include a strict max-score cutoff")
    threshold_status = "frozen"
    observed_far = selected_false_accepts / len(negatives) if negatives else 1.0
    sensitivity = selected_true_accepts / len(positives) if positives else 0.0
    return {
        "threshold_status": threshold_status,
        "frozen_threshold": selected,
        "calibration_observed_far": observed_far,
        "false_accept_count": selected_false_accepts,
        "negative_count": len(negatives),
        "true_accept_count": selected_true_accepts,
        "positive_count": len(positives),
        "calibration_sensitivity": sensitivity,
    }


def _collect_row(
    repo_root: Path,
    case: dict[str, Any],
    registry_by_manifest_id: dict[str, RegistryRecord],
) -> dict[str, Any]:
    case_root = Path(str(case["case_root"]))
    if not case_root.is_absolute():
        case_root = repo_root / case_root
    eval_summary_path = _find_eval_summary(case_root, case, registry_by_manifest_id)
    if not eval_summary_path:
        return _pending_row(case, case_root)
    return _row_from_summary(case, case_root, eval_summary_path)


def _method_status(method_slug: str, rows: list[dict[str, Any]], target_far: float) -> dict[str, Any]:
    method_rows = [row for row in rows if row["method_slug"] == method_slug]
    positives = [row for row in method_rows if row["label"] is True]
    negatives = [row for row in method_rows if row["label"] is False]
    completed = [row for row in method_rows if row["result_class"] == "valid_completed"]
    pending = [row for row in method_rows if row["result_class"] == "pending"]
    invalid = [
        row
        for row in method_rows
        if row["result_class"] not in {"valid_completed", "pending"}
    ]
    threshold = _select_threshold(method_rows, target_far) if not pending and not invalid else {
        "threshold_status": "blocked_pending_or_invalid_scores",
        "frozen_threshold": "",
        "calibration_observed_far": "",
        "false_accept_count": "",
        "negative_count": len(negatives),
        "true_accept_count": "",
        "positive_count": len(positives),
        "calibration_sensitivity": "",
    }
    return {
        "method_slug": method_slug,
        "target_far": target_far,
        "case_count": len(method_rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "completed_count": len(completed),
        "pending_count": len(pending),
        "invalid_count": len(invalid),
        **threshold,
    }


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = _resolve_path(repo_root, args.tables_dir)
    root_base = _resolve_output_root_base(package_config, args.case_root_base)
    eval_registry_paths = args.eval_registry or [
        "manifests/matched_budget_baselines/calibration_eval_job_registry.jsonl"
    ]
    registry_records: list[RegistryRecord] = []
    for eval_registry_raw in eval_registry_paths:
        eval_registry_path = _resolve_path(repo_root, eval_registry_raw)
        registry_records.extend(load_registry(eval_registry_path))
    registry_by_manifest_id = latest_registry_by_manifest_id(registry_records)
    rows = [
        _collect_row(repo_root, case, registry_by_manifest_id)
        for case in _eval_cases(package_config, root_base)
    ]
    fixed = dict(package_config["fixed_contract"])
    target_far = float(fixed["target_far"])
    available_negative_sets = sorted(
        {str(row["negative_set"]) for row in rows if row["negative_set"]}
    )
    required_negative_sets = [str(item) for item in package_config["calibration_split"]["negative_sets"]]
    missing_negative_sets = [
        item for item in required_negative_sets if item not in set(available_negative_sets)
    ]
    completed = [row for row in rows if row["result_class"] == "valid_completed"]
    pending = [row for row in rows if row["result_class"] == "pending"]
    invalid = [
        row
        for row in rows
        if row["result_class"] not in {"valid_completed", "pending"}
    ]
    method_rows = [
        _method_status(str(method["slug"]), rows, target_far)
        for method in package_config["baseline_methods"]
        if bool(method["requires_training"]) and not bool(method["requires_external_integration"])
    ]
    frozen_methods = [
        row for row in method_rows if row["threshold_status"] == "frozen"
    ]
    threshold_freeze_allowed = (
        not pending
        and not invalid
        and not missing_negative_sets
        and len(frozen_methods) == len(method_rows)
    )
    thresholds_frozen = threshold_freeze_allowed
    threshold_freeze_blockers = [
        *[f"missing_negative_set:{item}" for item in missing_negative_sets],
        "pending_calibration_eval_summaries" if pending else "",
        "invalid_calibration_eval_summaries" if invalid else "",
        "method_thresholds_not_selected" if len(frozen_methods) != len(method_rows) else "",
    ]
    threshold_freeze_blockers = [item for item in threshold_freeze_blockers if item]
    summary = {
        "schema_name": "baseline_calibration_summary",
        "schema_version": 2,
        "generated_at": current_timestamp(),
        "package_config_path": _repo_relative_path(repo_root, package_config_path),
        "new_case_root_base": root_base,
        "target_far": target_far,
        "status": "thresholds_frozen" if thresholds_frozen else "pending_real_calibration_scores",
        "thresholds_frozen": thresholds_frozen,
        "threshold_freeze_allowed": threshold_freeze_allowed,
        "threshold_freeze_blockers": threshold_freeze_blockers,
        "available_negative_sets": available_negative_sets,
        "missing_negative_sets": missing_negative_sets,
        "case_count": len(rows),
        "completed_count": len(completed),
        "pending_count": len(pending),
        "invalid_count": len(invalid),
        "method_rows": method_rows,
    }
    far_rows: list[dict[str, Any]] = []
    for method_row in method_rows:
        method_threshold = method_row["frozen_threshold"]
        for negative_set in required_negative_sets:
            negative_rows = [
                row
                for row in rows
                if row["method_slug"] == method_row["method_slug"]
                and row["negative_set"] == negative_set
                and row["result_class"] == "valid_completed"
            ]
            false_accept_count: int | str = ""
            observed_far: float | str = ""
            status = "pending"
            if method_threshold != "" and negative_rows:
                false_accept_count = sum(
                    1
                    for row in negative_rows
                    if (score := _score(row)) is not None and score >= float(method_threshold)
                )
                observed_far = false_accept_count / len(negative_rows)
                status = "completed"
            far_rows.append(
                {
                    "method_slug": method_row["method_slug"],
                    "target_far": target_far,
                    "negative_set": negative_set,
                    "observed_far": observed_far,
                    "false_accept_count": false_accept_count,
                    "negative_count": len(negative_rows) if negative_rows else "",
                    "status": status,
                }
            )
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
