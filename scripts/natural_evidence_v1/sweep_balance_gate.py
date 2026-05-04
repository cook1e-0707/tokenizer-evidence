from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.common import read_jsonl, resolve_repo_path, write_csv, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep build-time balance thresholds for a natural_evidence_v1 "
            "opportunity bank. This is a coverage diagnostic, not a bank rebuild "
            "and not a fingerprint claim."
        )
    )
    parser.add_argument("--entries", required=True)
    parser.add_argument("--rejections", required=True)
    parser.add_argument("--target-entries", type=int, default=24576)
    parser.add_argument("--min-bucket-mass-values", default="0.0025,0.005,0.01")
    parser.add_argument("--max-bucket-mass-ratio-values", default="5,8,10,15,20,50")
    parser.add_argument("--min-bucket-entropy-fraction-values", default="0.75,0.80,0.85,0.90")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args(argv)


def _parse_float_list(payload: str) -> list[float]:
    values = [float(item.strip()) for item in payload.split(",") if item.strip()]
    if not values:
        raise ValueError("threshold value list must not be empty")
    return values


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _accepted_rows(entries_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(entries_path):
        metrics = dict(row.get("bucket_mass_summary", {}))
        min_mass = _float_or_none(metrics.get("min_bucket_mass"))
        ratio = _float_or_none(metrics.get("bucket_mass_ratio"))
        entropy = _float_or_none(metrics.get("bucket_entropy_fraction"))
        if min_mass is None or ratio is None or entropy is None:
            continue
        rows.append(
            {
                "source": "accepted",
                "prompt_id": str(row.get("prompt_id", "")),
                "min_bucket_mass": min_mass,
                "bucket_mass_ratio": ratio,
                "bucket_entropy_fraction": entropy,
            }
        )
    return rows


def _rejected_rows(rejections_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with rejections_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            min_mass = _float_or_none(row.get("min_bucket_mass"))
            ratio = _float_or_none(row.get("bucket_mass_ratio"))
            entropy = _float_or_none(row.get("bucket_entropy_fraction"))
            if min_mass is None or ratio is None or entropy is None:
                continue
            rows.append(
                {
                    "source": "rejected",
                    "prompt_id": str(row.get("prompt_id", "")),
                    "rejection_reason": str(row.get("rejection_reason", "")),
                    "min_bucket_mass": min_mass,
                    "bucket_mass_ratio": ratio,
                    "bucket_entropy_fraction": entropy,
                }
            )
    return rows


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _sweep_rows(
    *,
    metric_rows: list[dict[str, Any]],
    min_mass_values: list[float],
    ratio_values: list[float],
    entropy_values: list[float],
    target_entries: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for min_mass in min_mass_values:
        for max_ratio in ratio_values:
            for min_entropy in entropy_values:
                passed = [
                    row
                    for row in metric_rows
                    if float(row["min_bucket_mass"]) >= min_mass
                    and float(row["bucket_mass_ratio"]) <= max_ratio
                    and float(row["bucket_entropy_fraction"]) >= min_entropy
                ]
                min_masses = [float(row["min_bucket_mass"]) for row in passed]
                ratios = [float(row["bucket_mass_ratio"]) for row in passed]
                entropies = [float(row["bucket_entropy_fraction"]) for row in passed]
                rows.append(
                    {
                        "schema_name": "natural_balance_gate_sweep_row_v1",
                        "result_claim": "threshold_coverage_diagnostic_not_fingerprint_count",
                        "min_bucket_mass_threshold": min_mass,
                        "max_bucket_mass_ratio_threshold": max_ratio,
                        "min_bucket_entropy_fraction_threshold": min_entropy,
                        "accepted_entries": len(passed),
                        "target_entries": target_entries,
                        "coverage_complete": len(passed) >= target_entries,
                        "prompt_count": len({str(row.get("prompt_id", "")) for row in passed if row.get("prompt_id")}),
                        "accepted_fraction_of_metric_records": len(passed) / len(metric_rows) if metric_rows else 0.0,
                        "accepted_min_bucket_mass_min": min(min_masses) if min_masses else 0.0,
                        "accepted_bucket_mass_ratio_max": max(ratios) if ratios else 0.0,
                        "accepted_bucket_entropy_fraction_min": min(entropies) if entropies else 0.0,
                        "accepted_min_bucket_mass_median": _median(min_masses),
                        "accepted_bucket_mass_ratio_median": _median(ratios),
                        "accepted_bucket_entropy_fraction_median": _median(entropies),
                    }
                )
    return rows


def _recommended_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if bool(row["coverage_complete"])]
    if complete:
        return sorted(
            complete,
            key=lambda row: (
                -float(row["min_bucket_mass_threshold"]),
                -float(row["min_bucket_entropy_fraction_threshold"]),
                float(row["max_bucket_mass_ratio_threshold"]),
                int(row["accepted_entries"]),
            ),
        )[0]
    return sorted(
        rows,
        key=lambda row: (
            int(row["accepted_entries"]),
            float(row["min_bucket_mass_threshold"]),
            float(row["min_bucket_entropy_fraction_threshold"]),
            -float(row["max_bucket_mass_ratio_threshold"]),
        ),
        reverse=True,
    )[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    entries_path = resolve_repo_path(args.entries, root)
    rejections_path = resolve_repo_path(args.rejections, root)
    output_csv = resolve_repo_path(args.output_csv, root)
    summary_json = resolve_repo_path(args.summary_json, root)

    metric_rows = _accepted_rows(entries_path) + _rejected_rows(rejections_path)
    rows = _sweep_rows(
        metric_rows=metric_rows,
        min_mass_values=_parse_float_list(args.min_bucket_mass_values),
        ratio_values=_parse_float_list(args.max_bucket_mass_ratio_values),
        entropy_values=_parse_float_list(args.min_bucket_entropy_fraction_values),
        target_entries=args.target_entries,
    )
    fieldnames = [
        "schema_name",
        "result_claim",
        "min_bucket_mass_threshold",
        "max_bucket_mass_ratio_threshold",
        "min_bucket_entropy_fraction_threshold",
        "accepted_entries",
        "target_entries",
        "coverage_complete",
        "prompt_count",
        "accepted_fraction_of_metric_records",
        "accepted_min_bucket_mass_min",
        "accepted_bucket_mass_ratio_max",
        "accepted_bucket_entropy_fraction_min",
        "accepted_min_bucket_mass_median",
        "accepted_bucket_mass_ratio_median",
        "accepted_bucket_entropy_fraction_median",
    ]
    write_csv(output_csv, rows, fieldnames)
    recommended = _recommended_row(rows) if rows else {}
    summary = {
        "schema_name": "natural_balance_gate_sweep_summary_v1",
        "entries_path": str(entries_path),
        "rejections_path": str(rejections_path),
        "output_csv": str(output_csv),
        "metric_records": len(metric_rows),
        "target_entries": args.target_entries,
        "threshold_grid_rows": len(rows),
        "any_threshold_reaches_target": any(bool(row["coverage_complete"]) for row in rows),
        "recommended_row": recommended,
        "result_claim": "threshold_coverage_diagnostic_not_fingerprint_count",
        "next_decision": (
            "relaxed_threshold_candidate_available"
            if recommended and bool(recommended.get("coverage_complete"))
            else "expand_candidate_supply_or_expand_sweep_grid"
        ),
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
