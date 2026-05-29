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
                raise ValueError(f"expected JSON object row in {path}")
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


def load_rows(path: Path) -> list[Mapping[str, Any]]:
    return list(read_jsonl(path)) if path.exists() else []


def summarize_condition(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition", ""))].append(row)

    output: list[dict[str, Any]] = []
    for condition, items in sorted(by_condition.items()):
        applied = [row for row in items if bool(row.get("controller_applied"))]
        cap_counter: Counter[str] = Counter()
        for row in applied:
            reasons = row.get("controller_cap_reasons", [])
            if isinstance(reasons, list):
                cap_counter.update(str(reason) for reason in reasons)
        scales = [float(row.get("controller_scale", 0.0) or 0.0) for row in applied]
        kls = [float(row.get("controller_kl_to_base", 0.0) or 0.0) for row in applied]
        output.append(
            {
                "condition": condition,
                "row_count": len(items),
                "mean_target_mass": mean([float(row.get("target_mass", 0.0)) for row in items]),
                "rank1_rate": mean([1.0 if bool(row.get("target_surface_rank1")) else 0.0 for row in items]),
                "median_target_margin": median([float(row.get("target_margin", 0.0)) for row in items]),
                "controller_applied_rows": len(applied),
                "mean_controller_scale": mean(scales),
                "min_controller_scale": min(scales, default=0.0),
                "median_controller_scale": median(scales),
                "max_controller_scale": max(scales, default=0.0),
                "mean_controller_kl_to_base": mean(kls),
                "max_controller_kl_to_base": max(kls, default=0.0),
                "capped_rows": sum(1 for row in applied if row.get("controller_cap_reasons")),
                "cap_reason_counts": ";".join(f"{key}:{value}" for key, value in sorted(cap_counter.items())),
            }
        )
    return output


def summarize_group(
    rows: list[Mapping[str, Any]],
    *,
    key: str,
    condition: str = "controlled_base",
) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("condition")) == condition]
    by_key: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_key[str(row.get(key, ""))].append(row)

    output: list[dict[str, Any]] = []
    for value, items in sorted(by_key.items()):
        scales = [float(row.get("controller_scale", 0.0) or 0.0) for row in items]
        output.append(
            {
                key: value,
                "condition": condition,
                "row_count": len(items),
                "mean_target_mass": mean([float(row.get("target_mass", 0.0)) for row in items]),
                "rank1_rate": mean([1.0 if bool(row.get("target_surface_rank1")) else 0.0 for row in items]),
                "median_target_margin": median([float(row.get("target_margin", 0.0)) for row in items]),
                "mean_controller_scale": mean(scales),
                "capped_rate": mean([1.0 if row.get("controller_cap_reasons") else 0.0 for row in items]),
            }
        )
    return output


def metrics_for_rows(rows: list[Mapping[str, Any]], *, excluded_coordinates: set[int]) -> dict[str, dict[str, float]]:
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        coordinate_id = int(row.get("coordinate_id", -1))
        if coordinate_id in excluded_coordinates:
            continue
        by_condition[str(row.get("condition", ""))].append(row)

    metrics: dict[str, dict[str, float]] = {}
    for condition, items in by_condition.items():
        metrics[condition] = {
            "row_count": float(len(items)),
            "mean_target_mass": mean([float(row.get("target_mass", 0.0)) for row in items]),
            "rank1_rate": mean([1.0 if bool(row.get("target_surface_rank1")) else 0.0 for row in items]),
            "median_target_margin": median([float(row.get("target_margin", 0.0)) for row in items]),
        }
    return metrics


def simulate_coordinate_exclusions(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    controlled_rows = [row for row in rows if str(row.get("condition")) == "controlled_base"]
    coordinate_rows = summarize_group(rows, key="coordinate_id")
    ranked = sorted(coordinate_rows, key=lambda row: float(row["mean_target_mass"]))
    excluded: set[int] = set()
    output: list[dict[str, Any]] = []

    for step in range(min(8, len(ranked)) + 1):
        metrics = metrics_for_rows(rows, excluded_coordinates=excluded)
        base = metrics.get("base", {})
        task = metrics.get("task_only", {})
        controlled = metrics.get("controlled_base", {})
        wrong_key = metrics.get("wrong_key_controlled_base", {})
        wrong_payload = metrics.get("wrong_payload_controlled_base", {})
        output.append(
            {
                "excluded_coordinate_count": len(excluded),
                "excluded_coordinates": ",".join(str(item) for item in sorted(excluded)),
                "remaining_controlled_rows": int(controlled.get("row_count", 0.0)),
                "controlled_mean_target_mass": controlled.get("mean_target_mass", 0.0),
                "controlled_lift_vs_base": controlled.get("mean_target_mass", 0.0) - base.get("mean_target_mass", 0.0),
                "controlled_lift_vs_task_only": controlled.get("mean_target_mass", 0.0)
                - task.get("mean_target_mass", 0.0),
                "controlled_rank1_rate": controlled.get("rank1_rate", 0.0),
                "controlled_median_target_margin": controlled.get("median_target_margin", 0.0),
                "wrong_key_lift_vs_base": wrong_key.get("mean_target_mass", 0.0) - base.get("mean_target_mass", 0.0),
                "wrong_key_rank1_rate": wrong_key.get("rank1_rate", 0.0),
                "wrong_payload_lift_vs_base": wrong_payload.get("mean_target_mass", 0.0)
                - base.get("mean_target_mass", 0.0),
                "wrong_payload_rank1_rate": wrong_payload.get("rank1_rate", 0.0),
            }
        )
        if step < len(ranked):
            excluded.add(int(ranked[step]["coordinate_id"]))
    return output


def analyze(aggregate_path: Path, output_dir: Path) -> dict[str, Any]:
    aggregate = read_json(aggregate_path)
    best_rows = aggregate.get("best_by_controlled_lift_vs_base", [])
    if not isinstance(best_rows, list) or not best_rows:
        raise ValueError("missing best_by_controlled_lift_vs_base")
    best = best_rows[0]
    best_grid_path = Path(str(best["path"]))
    rows_path = best_grid_path.parent / "r4_teacher_forced_surface_mass_rows.jsonl"
    rows = load_rows(rows_path)

    condition_rows = summarize_condition(rows) if rows else []
    coordinate_rows = summarize_group(rows, key="coordinate_id") if rows else []
    prefix_rows = summarize_group(rows, key="assistant_prefix_model_text") if rows else []
    exclusion_rows = simulate_coordinate_exclusions(rows) if rows else []

    controlled_lift_vs_base = float(best["controlled_lift_vs_base"])
    controlled_lift_vs_task_only = float(best["controlled_lift_vs_task_only"])
    controlled_rank1 = float(best["controlled_rank1_rate"])
    controlled_margin = float(best["controlled_median_target_margin"])
    base_lift_deficit = max(0.0, 0.15 - controlled_lift_vs_base)
    task_lift_deficit = max(0.0, 0.10 - controlled_lift_vs_task_only)
    rank_gate_pass = controlled_rank1 >= 0.75
    wrong_controls_clean = int(aggregate.get("wrong_control_failure_count", -1)) == 0

    weak_coordinates = [
        int(row["coordinate_id"])
        for row in coordinate_rows
        if float(row["mean_target_mass"]) < 0.10 or float(row["rank1_rate"]) < 0.90
    ]
    first_exclusion_gate_pass = next(
        (
            row
            for row in exclusion_rows
            if float(row["controlled_lift_vs_base"]) >= 0.15
            and float(row["controlled_lift_vs_task_only"]) >= 0.10
            and float(row["controlled_rank1_rate"]) >= 0.75
            and float(row["wrong_key_lift_vs_base"]) < 0.15
            and float(row["wrong_payload_lift_vs_base"]) < 0.15
        ),
        None,
    )

    status = "FAIL_R4_AFTER_867621_CONTROLLER_FEASIBILITY_868016_ANALYZED_NO_GENERATION"
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_867621_controller_feasibility_failure_analysis_v1",
        "status": status,
        "source_job_id": str(aggregate.get("source_job_id", "868016")),
        "aggregate_path": str(aggregate_path),
        "best_grid_index": int(best["grid_index"]),
        "best_grid_summary_path": str(best_grid_path),
        "best_grid_rows_path": str(rows_path),
        "best_bonus_nats": float(best["bonus_nats"]),
        "best_penalty_nats": float(best["penalty_nats"]),
        "best_max_target_mass": float(best["max_target_mass"]),
        "best_max_kl_budget": float(best["max_kl_budget"]),
        "best_controlled_mean_target_mass": float(best["controlled_mean_target_mass"]),
        "best_controlled_lift_vs_base": controlled_lift_vs_base,
        "best_controlled_lift_vs_task_only": controlled_lift_vs_task_only,
        "best_controlled_rank1_rate": controlled_rank1,
        "best_controlled_median_margin": controlled_margin,
        "lift_deficit_vs_base_gate": base_lift_deficit,
        "lift_deficit_vs_task_only_gate": task_lift_deficit,
        "rank1_gate_pass": rank_gate_pass,
        "wrong_controls_clean": wrong_controls_clean,
        "wrong_control_failure_count": int(aggregate.get("wrong_control_failure_count", -1)),
        "passing_grid_count": int(aggregate.get("passing_grid_count", 0)),
        "row_level_analysis_available": bool(rows),
        "weak_coordinates_under_mass_0_10_or_rank_0_90": weak_coordinates,
        "first_posthoc_coordinate_exclusion_gate_pass": first_exclusion_gate_pass,
        "posthoc_exclusion_is_diagnostic_only": True,
        "generation_unlocked": False,
        "training_unlocked": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "artifact-only reviewed reliability-coordinate pivot route; no generation"
            if first_exclusion_gate_pass
            else "artifact-only target construction/controller repair route; no generation"
        ),
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
                "median_controller_scale",
                "max_controller_scale",
                "mean_controller_kl_to_base",
                "max_controller_kl_to_base",
                "capped_rows",
                "cap_reason_counts",
            ],
        )
    if coordinate_rows:
        write_csv(
            output_dir / "best_grid_coordinate_summary.csv",
            coordinate_rows,
            [
                "coordinate_id",
                "condition",
                "row_count",
                "mean_target_mass",
                "rank1_rate",
                "median_target_margin",
                "mean_controller_scale",
                "capped_rate",
            ],
        )
    if prefix_rows:
        write_csv(
            output_dir / "best_grid_prefix_summary.csv",
            prefix_rows,
            [
                "assistant_prefix_model_text",
                "condition",
                "row_count",
                "mean_target_mass",
                "rank1_rate",
                "median_target_margin",
                "mean_controller_scale",
                "capped_rate",
            ],
        )
    if exclusion_rows:
        write_csv(
            output_dir / "posthoc_coordinate_exclusion_diagnostic.csv",
            exclusion_rows,
            [
                "excluded_coordinate_count",
                "excluded_coordinates",
                "remaining_controlled_rows",
                "controlled_mean_target_mass",
                "controlled_lift_vs_base",
                "controlled_lift_vs_task_only",
                "controlled_rank1_rate",
                "controlled_median_target_margin",
                "wrong_key_lift_vs_base",
                "wrong_key_rank1_rate",
                "wrong_payload_lift_vs_base",
                "wrong_payload_rank1_rate",
            ],
        )
    write_json(output_dir / "failure_analysis_summary.json", summary)

    report = f"""# R4 After 867621 Controller Feasibility 868016 Failure Analysis

Status: `{status}`

Job `868016` was a teacher-forced controller feasibility-envelope run only. It
does not unlock generation, training, Llama, null expansion, sanitizer, FAR, or
paper-facing claims.

Best grid:

```text
grid_index: {summary["best_grid_index"]}
bonus_nats: {summary["best_bonus_nats"]}
penalty_nats: {summary["best_penalty_nats"]}
max_target_mass: {summary["best_max_target_mass"]}
max_kl_budget: {summary["best_max_kl_budget"]}
controlled mean target mass: {summary["best_controlled_mean_target_mass"]:.6f}
controlled lift vs base: {controlled_lift_vs_base:.6f}
controlled lift vs task_only: {controlled_lift_vs_task_only:.6f}
controlled rank1: {controlled_rank1:.6f}
controlled median margin: {controlled_margin:.6f}
```

Gate result:

```text
passing grids: {summary["passing_grid_count"]}
wrong-control failures: {summary["wrong_control_failure_count"]}
lift deficit vs +0.15 base gate: {base_lift_deficit:.6f}
lift deficit vs +0.10 task-only gate: {task_lift_deficit:.6f}
rank1 gate pass at 0.75: {rank_gate_pass}
wrong controls clean: {wrong_controls_clean}
```

Row-level interpretation:

```text
row-level analysis available: {bool(rows)}
weak coordinates under mass 0.10 or rank 0.90: {weak_coordinates}
first post-hoc coordinate exclusion gate pass: {first_exclusion_gate_pass}
```

The best controller grid is close on aggregate mass and strong on rank1, but it
still fails the precommitted lift gate. The row-level diagnostic shows that
failure is concentrated in weak coordinates rather than uniformly distributed
across all reliability surfaces. Any coordinate exclusion shown here is
diagnostic only: it cannot reclassify job `868016` as passing because the
exclusion was derived after reviewing the job output.

Next allowed action: `{summary["next_allowed_action"]}`.
"""
    report_path = output_dir / "failure_analysis.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {report_path}")
    report_path.write_text(report, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze R4 after-867621 controller feasibility 868016 failure.")
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
