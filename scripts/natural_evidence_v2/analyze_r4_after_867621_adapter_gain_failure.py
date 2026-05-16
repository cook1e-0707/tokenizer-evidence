from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping


def read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def iter_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    yield payload


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_group(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    masses = [float(row["target_mass"]) for row in rows]
    margins = [float(row["target_margin"]) for row in rows]
    rank1 = [bool(row.get("target_surface_rank1", False)) for row in rows]
    return {
        "row_count": len(rows),
        "mean_target_mass": mean(masses),
        "median_target_margin": median(margins) if margins else 0.0,
        "rank1_rate": mean([1.0 if value else 0.0 for value in rank1]),
    }


def condition_sort_key(condition: str) -> tuple[int, float, str]:
    if condition == "base":
        return (0, 0.0, condition)
    if condition == "task_only":
        return (2, 0.0, condition)
    if condition.startswith("protected_gain_"):
        raw = condition.removeprefix("protected_gain_").replace("_", ".")
        return (1, float(raw), condition)
    return (3, 0.0, condition)


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def analyze(summary_path: Path, rows_path: Path, output_dir: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_condition_coordinate: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in iter_rows(rows_path):
        condition = str(row["condition"])
        coordinate_id = int(row["coordinate_id"])
        by_condition[condition].append(row)
        by_condition_coordinate[(condition, coordinate_id)].append(row)

    condition_rows: list[dict[str, Any]] = []
    base_mass = float(summary["condition_summary"]["base"]["mean_target_mass"])
    task_mass = float(summary["condition_summary"]["task_only"]["mean_target_mass"])
    for condition in sorted(by_condition, key=condition_sort_key):
        stats = summarize_group(by_condition[condition])
        condition_rows.append(
            {
                "condition": condition,
                "mean_target_mass": stats["mean_target_mass"],
                "lift_vs_base": stats["mean_target_mass"] - base_mass,
                "lift_vs_task_only": stats["mean_target_mass"] - task_mass,
                "rank1_rate": stats["rank1_rate"],
                "median_target_margin": stats["median_target_margin"],
                "row_count": stats["row_count"],
            }
        )

    protected_rows = [row for row in condition_rows if str(row["condition"]).startswith("protected_gain_")]
    best_by_mass = max(protected_rows, key=lambda row: float(row["mean_target_mass"]))
    best_by_lift = max(protected_rows, key=lambda row: float(row["lift_vs_base"]))
    passing_gains = [
        row
        for row in protected_rows
        if float(row["lift_vs_base"]) >= 0.15
        and float(row["lift_vs_task_only"]) >= 0.10
        and float(row["rank1_rate"]) >= 0.75
        and float(row["median_target_margin"]) > 0.0
    ]

    coord_rows: list[dict[str, Any]] = []
    for (condition, coordinate_id), grouped in sorted(
        by_condition_coordinate.items(), key=lambda item: (condition_sort_key(item[0][0]), item[0][1])
    ):
        stats = summarize_group(grouped)
        coord_rows.append(
            {
                "condition": condition,
                "coordinate_id": coordinate_id,
                "mean_target_mass": stats["mean_target_mass"],
                "rank1_rate": stats["rank1_rate"],
                "median_target_margin": stats["median_target_margin"],
                "row_count": stats["row_count"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "gain_by_condition.csv",
        condition_rows,
        [
            "condition",
            "mean_target_mass",
            "lift_vs_base",
            "lift_vs_task_only",
            "rank1_rate",
            "median_target_margin",
            "row_count",
        ],
    )
    write_csv(
        output_dir / "gain_by_coordinate.csv",
        coord_rows,
        ["condition", "coordinate_id", "mean_target_mass", "rank1_rate", "median_target_margin", "row_count"],
    )

    result = {
        "schema_name": "natural_evidence_v2_r4_after_867621_adapter_gain_failure_analysis_v1",
        "status": "FAIL_R4_AFTER_867621_RELIABILITY_ADAPTER_GAIN_SWEEP_NO_GENERATION",
        "source_job_id": "867897",
        "source_summary": str(summary_path),
        "source_rows": str(rows_path),
        "score_row_count": int(summary.get("score_row_count", 0)),
        "scored_row_count": int(summary.get("scored_row_count", 0)),
        "teacher_forced_surface_gate_status": summary.get("teacher_forced_surface_gate_status"),
        "best_gain_by_mass": best_by_mass,
        "best_gain_by_lift_vs_base": best_by_lift,
        "passing_gain_count": len(passing_gains),
        "passing_gains": passing_gains,
        "adapter_gain_rescued_gate": False,
        "generation_unlocked": False,
        "training_unlocked": False,
        "next_allowed_action": "artifact-only repair or pivot route decision; do not run generation from this failed gain sweep",
    }
    write_json(output_dir / "failure_analysis_summary.json", result)

    report = f"""# R4 After 867621 Adapter-Gain Failure Analysis

status: `{result["status"]}`

Job `867897` completed cleanly, but no protected-adapter gain satisfied the
teacher-forced surface-mass gate.

Best protected gain by mean target mass:

```text
condition: {best_by_mass["condition"]}
mean target mass: {float(best_by_mass["mean_target_mass"]):.6f}
lift vs base: {float(best_by_mass["lift_vs_base"]):.6f}
lift vs task_only: {float(best_by_mass["lift_vs_task_only"]):.6f}
rank1 rate: {float(best_by_mass["rank1_rate"]):.6f}
median target margin: {float(best_by_mass["median_target_margin"]):.8f}
```

Gate targets were `+0.15` lift vs base, `+0.10` lift vs task-only, rank1
`>=0.75`, and median target margin `>0`. The best observed lift vs base was
only `{float(best_by_lift["lift_vs_base"]):.6f}`.

Interpretation:

```text
adapter_gain_rescued_gate: false
generation_unlocked: false
training_unlocked: false
```

Increasing the protected adapter gain did not produce monotonic improvement.
The best point was `protected_gain_0_5`; larger gains reduced target mass and
rank1. This points away from a simple insufficient-scale explanation and toward
an adapter-direction or objective/surface mismatch. This result must not unlock
generation.
"""
    write_text(output_dir / "failure_analysis.md", report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze R4 after-867621 adapter-gain sweep failure.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(args.summary, args.rows, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
