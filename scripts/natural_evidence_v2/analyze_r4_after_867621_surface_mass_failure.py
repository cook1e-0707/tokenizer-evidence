from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORE_DIR = ROOT / "results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_score_867849"
DEFAULT_OUTPUT_DIR = ROOT / "results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_failure_analysis_867849_20260516"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(payload)
    return rows


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def median(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.median(items) if items else 0.0


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "mean_target_mass": mean(float(row.get("target_mass", 0.0)) for row in rows),
        "median_target_mass": median(float(row.get("target_mass", 0.0)) for row in rows),
        "mean_target_margin": mean(float(row.get("target_margin", 0.0)) for row in rows),
        "median_target_margin": median(float(row.get("target_margin", 0.0)) for row in rows),
        "rank1_rate": mean(1.0 if row.get("target_surface_rank1") else 0.0 for row in rows),
    }


def group_rows(rows: list[Mapping[str, Any]], key_fields: tuple[str, ...]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in key_fields)].append(row)
    return grouped


def stratum_rows(rows: list[Mapping[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped = group_rows(rows, ("condition",) + key_fields)
    condition_by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for key, items in grouped.items():
        condition = str(key[0])
        stratum_key = key[1:]
        condition_by_key[stratum_key][condition] = summarize(items)

    output: list[dict[str, Any]] = []
    for stratum_key, by_condition in sorted(condition_by_key.items(), key=lambda item: str(item[0])):
        base = by_condition.get("base", {})
        protected = by_condition.get("protected", {})
        task_only = by_condition.get("task_only", {})
        row: dict[str, Any] = {
            field: value for field, value in zip(key_fields, stratum_key, strict=True)
        }
        row.update(
            {
                "base_mean_target_mass": base.get("mean_target_mass"),
                "protected_mean_target_mass": protected.get("mean_target_mass"),
                "task_only_mean_target_mass": task_only.get("mean_target_mass"),
                "protected_lift_vs_base": float(protected.get("mean_target_mass", 0.0))
                - float(base.get("mean_target_mass", 0.0)),
                "protected_lift_vs_task_only": float(protected.get("mean_target_mass", 0.0))
                - float(task_only.get("mean_target_mass", 0.0)),
                "base_rank1_rate": base.get("rank1_rate"),
                "protected_rank1_rate": protected.get("rank1_rate"),
                "task_only_rank1_rate": task_only.get("rank1_rate"),
                "protected_median_target_margin": protected.get("median_target_margin"),
                "row_count_per_condition": protected.get("row_count"),
            }
        )
        output.append(row)
    return output


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    text = f"""# R4 After 867621 Surface-Mass Failure Analysis

status: `{summary['status']}`

Job `867849` completed cleanly, but the teacher-forced surface-mass gate failed.

```text
protected lift vs base: {summary['protected_lift_vs_base']:.6f}
protected lift vs task_only: {summary['protected_lift_vs_task_only']:.6f}
protected rank1 rate: {summary['protected_rank1_rate']:.6f}
protected median target margin: {summary['protected_median_target_margin']:.8f}
task_only lift vs base: {summary['task_only_lift_vs_base']:.6f}
```

Gate targets were `+0.15` lift vs base, `+0.10` lift vs task-only, rank1
`>=0.75`, and median target margin `>0`. The protected adapter produced only a
small positive target-mass lift and had lower rank1 than both base and task-only.

Interpretation:

```text
clean_slurm_completion: true
tokenizer_boundary_valid: true
task_only_leakage: false
protected_pressure_sufficient: false
generation_unlocked: false
```

This failure should not trigger generation. The next allowed step is an
artifact-only repair/pivot decision using the per-coordinate, per-prefix, and
per-surface failure tables.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze R4 after-867621 surface-mass scoring failure.")
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    score_dir = args.score_dir if args.score_dir.is_absolute() else ROOT / args.score_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")

    summary = read_json(score_dir / "r4_teacher_forced_surface_mass_summary.json")
    rows = read_jsonl(score_dir / "r4_teacher_forced_surface_mass_rows.jsonl")
    by_condition = {condition: summarize([row for row in rows if row.get("condition") == condition]) for condition in ("base", "protected", "task_only")}

    base = by_condition["base"]
    protected = by_condition["protected"]
    task_only = by_condition["task_only"]
    protected_lift_vs_base = protected["mean_target_mass"] - base["mean_target_mass"]
    protected_lift_vs_task_only = protected["mean_target_mass"] - task_only["mean_target_mass"]
    task_only_lift_vs_base = task_only["mean_target_mass"] - base["mean_target_mass"]

    output_summary = {
        "schema_name": "natural_evidence_v2_r4_after_867621_surface_mass_failure_analysis_v1",
        "status": "FAILURE_ANALYSIS_RECORDED_NO_GENERATION",
        "source_score_summary": str((score_dir / "r4_teacher_forced_surface_mass_summary.json").relative_to(ROOT)),
        "source_score_rows": str((score_dir / "r4_teacher_forced_surface_mass_rows.jsonl").relative_to(ROOT)),
        "score_row_count": int(summary.get("score_row_count", 0)),
        "scored_row_count": int(summary.get("scored_row_count", 0)),
        "teacher_forced_surface_gate_status": summary.get("teacher_forced_surface_gate_status"),
        "protected_lift_vs_base": protected_lift_vs_base,
        "protected_lift_vs_task_only": protected_lift_vs_task_only,
        "task_only_lift_vs_base": task_only_lift_vs_base,
        "protected_rank1_rate": protected["rank1_rate"],
        "protected_median_target_margin": protected["median_target_margin"],
        "base_rank1_rate": base["rank1_rate"],
        "task_only_rank1_rate": task_only["rank1_rate"],
        "clean_slurm_completion": True,
        "tokenizer_boundary_valid": True,
        "protected_pressure_sufficient": False,
        "generation_unlocked": False,
        "next_allowed_action": "artifact-only repair/pivot decision; do not run generation from this failed gate",
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "failure_analysis_summary.json", output_summary)
    write_csv(output_dir / "by_coordinate.csv", stratum_rows(rows, ("coordinate_id",)))
    write_csv(output_dir / "by_prefix.csv", stratum_rows(rows, ("assistant_prefix_model_text",)))
    write_csv(output_dir / "by_surface.csv", stratum_rows(rows, ("target_surface_label",)))
    write_csv(output_dir / "by_coordinate_prefix.csv", stratum_rows(rows, ("coordinate_id", "assistant_prefix_model_text")))
    write_report(output_dir / "failure_analysis.md", output_summary)
    print(json.dumps(output_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
