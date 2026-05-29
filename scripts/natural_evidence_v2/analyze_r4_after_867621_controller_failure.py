from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"expected JSON object row: {path}")
            yield payload


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize_rows(rows_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_condition_coordinate: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in read_jsonl(rows_path):
        condition = str(row.get("condition", ""))
        coordinate_id = int(row.get("coordinate_id", -1))
        by_condition[condition].append(row)
        by_condition_coordinate[(condition, coordinate_id)].append(row)

    condition_rows: list[dict[str, Any]] = []
    for condition, condition_items in sorted(by_condition.items()):
        applied_items = [row for row in condition_items if bool(row.get("controller_applied"))]
        cap_counter: Counter[str] = Counter()
        for row in applied_items:
            reasons = row.get("controller_cap_reasons", [])
            if isinstance(reasons, list):
                cap_counter.update(str(reason) for reason in reasons)
        condition_rows.append(
            {
                "condition": condition,
                "row_count": len(condition_items),
                "mean_target_mass": mean([float(row.get("target_mass", 0.0)) for row in condition_items]),
                "rank1_rate": mean([1.0 if bool(row.get("target_surface_rank1")) else 0.0 for row in condition_items]),
                "median_target_margin": median([float(row.get("target_margin", 0.0)) for row in condition_items]),
                "controller_applied_rows": len(applied_items),
                "mean_controller_scale": mean([float(row.get("controller_scale", 0.0)) for row in applied_items]),
                "min_controller_scale": min([float(row.get("controller_scale", 0.0)) for row in applied_items], default=0.0),
                "capped_rows": sum(
                    1
                    for row in applied_items
                    if isinstance(row.get("controller_cap_reasons", []), list)
                    and len(row.get("controller_cap_reasons", [])) > 0
                ),
                "cap_reason_counts": ";".join(f"{key}:{value}" for key, value in sorted(cap_counter.items())),
                "mean_controller_kl_to_base": mean(
                    [float(row.get("controller_kl_to_base", 0.0)) for row in applied_items]
                ),
            }
        )

    coordinate_rows: list[dict[str, Any]] = []
    for (condition, coordinate_id), items in sorted(by_condition_coordinate.items()):
        coordinate_rows.append(
            {
                "condition": condition,
                "coordinate_id": coordinate_id,
                "row_count": len(items),
                "mean_target_mass": mean([float(row.get("target_mass", 0.0)) for row in items]),
                "rank1_rate": mean([1.0 if bool(row.get("target_surface_rank1")) else 0.0 for row in items]),
                "median_target_margin": median([float(row.get("target_margin", 0.0)) for row in items]),
                "mean_controller_scale": mean(
                    [
                        float(row.get("controller_scale", 0.0))
                        for row in items
                        if bool(row.get("controller_applied"))
                    ]
                ),
            }
        )
    return condition_rows, coordinate_rows


def analyze(aggregate_path: Path, output_dir: Path) -> dict[str, Any]:
    aggregate = read_json(aggregate_path)
    best_rows = aggregate.get("best_by_controlled_lift_vs_base", [])
    if not isinstance(best_rows, list) or not best_rows:
        raise ValueError("aggregate best_by_controlled_lift_vs_base missing")
    best = best_rows[0]
    best_grid_path = Path(str(best["path"]))
    rows_path = best_grid_path.parent / "r4_teacher_forced_surface_mass_rows.jsonl"

    condition_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    if rows_path.exists():
        condition_rows, coordinate_rows = summarize_rows(rows_path)

    controlled_lift = float(best["controlled_lift_vs_base"])
    task_lift = float(best["controlled_lift_vs_task_only"])
    rank1 = float(best["controlled_rank1_rate"])
    median_margin = float(best["controlled_median_target_margin"])
    lift_deficit_vs_base = max(0.0, 0.15 - controlled_lift)
    lift_deficit_vs_task_only = max(0.0, 0.10 - task_lift)
    rank1_gate_pass = rank1 >= 0.75
    wrong_controls_clean = int(aggregate.get("wrong_control_failure_count", -1)) == 0

    controlled_condition = next((row for row in condition_rows if row["condition"] == "controlled_base"), {})
    capped_rows = int(controlled_condition.get("capped_rows", 0) or 0)
    applied_rows = int(controlled_condition.get("controller_applied_rows", 0) or 0)
    capped_rate = float(capped_rows / applied_rows) if applied_rows else 0.0

    status = "FAIL_R4_AFTER_867621_CONTROLLER_867939_ANALYZED_NO_GENERATION"
    suggested_pivot = (
        "artifact-only reviewed controller feasibility-envelope route"
        if wrong_controls_clean and rank1_gate_pass and controlled_lift > 0.0
        else "artifact-only target construction/objective repair"
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_867621_controller_failure_analysis_v1",
        "status": status,
        "source_job_id": str(aggregate.get("source_job_id", "867939")),
        "aggregate_path": str(aggregate_path),
        "best_grid_index": int(best["grid_index"]),
        "best_grid_summary_path": str(best_grid_path),
        "best_grid_rows_path": str(rows_path),
        "best_bonus_nats": float(best["bonus_nats"]),
        "best_penalty_nats": float(best["penalty_nats"]),
        "best_max_target_mass": float(best["max_target_mass"]),
        "best_max_kl_budget": float(best["max_kl_budget"]),
        "best_controlled_lift_vs_base": controlled_lift,
        "best_controlled_lift_vs_task_only": task_lift,
        "best_controlled_rank1_rate": rank1,
        "best_controlled_median_margin": median_margin,
        "lift_deficit_vs_base_gate": lift_deficit_vs_base,
        "lift_deficit_vs_task_only_gate": lift_deficit_vs_task_only,
        "rank1_gate_pass": rank1_gate_pass,
        "wrong_controls_clean": wrong_controls_clean,
        "passing_grid_count": int(aggregate.get("passing_grid_count", 0)),
        "wrong_control_failure_count": int(aggregate.get("wrong_control_failure_count", -1)),
        "best_grid_controlled_capped_rate": capped_rate,
        "best_grid_condition_rows_written": bool(condition_rows),
        "best_grid_coordinate_rows_written": bool(coordinate_rows),
        "generation_unlocked": False,
        "training_unlocked": False,
        "paper_claim_allowed": False,
        "next_allowed_action": f"{suggested_pivot}; no generation",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if condition_rows:
        write_csv(
            output_dir / "best_grid_condition_summary.csv",
            condition_rows,
            [
                "condition",
                "row_count",
                "mean_target_mass",
                "rank1_rate",
                "median_target_margin",
                "controller_applied_rows",
                "mean_controller_scale",
                "min_controller_scale",
                "capped_rows",
                "cap_reason_counts",
                "mean_controller_kl_to_base",
            ],
        )
    if coordinate_rows:
        write_csv(
            output_dir / "best_grid_coordinate_summary.csv",
            coordinate_rows,
            [
                "condition",
                "coordinate_id",
                "row_count",
                "mean_target_mass",
                "rank1_rate",
                "median_target_margin",
                "mean_controller_scale",
            ],
        )
    write_json(output_dir / "failure_analysis_summary.json", summary)
    report = f"""# R4 After 867621 Controller 867939 Failure Analysis

Status: `{status}`

Best grid:

```text
grid_index: {summary["best_grid_index"]}
bonus_nats: {summary["best_bonus_nats"]}
penalty_nats: {summary["best_penalty_nats"]}
max_target_mass: {summary["best_max_target_mass"]}
max_kl_budget: {summary["best_max_kl_budget"]}
controlled lift vs base: {controlled_lift:.6f}
controlled lift vs task_only: {task_lift:.6f}
controlled rank1: {rank1:.6f}
controlled median margin: {median_margin:.6f}
```

Gate deficits:

```text
lift deficit vs +0.15 base gate: {lift_deficit_vs_base:.6f}
lift deficit vs +0.10 task_only gate: {lift_deficit_vs_task_only:.6f}
rank1 gate pass at 0.75: {rank1_gate_pass}
wrong controls clean: {wrong_controls_clean}
passing grids: {summary["passing_grid_count"]}
```

Interpretation: the bounded controller improves rank1 and keeps wrong controls
clean, but the mass lift remains far below gate. This is not a generation
unlock. The next step remains artifact-only: record a reviewed repair/pivot
route before any additional Slurm submission.

Next allowed action: `{summary["next_allowed_action"]}`.
"""
    report_path = output_dir / "failure_analysis.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {report_path}")
    report_path.write_text(report, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze R4 after-867621 controller 867939 failure artifacts.")
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = analyze(args.aggregate, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
