from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_ROWS = Path(
    "results/natural_evidence_v2/status/r4_prefix_native_candidate_v3_surface_mass_score_856453_review/"
    "remote_artifacts/r4_teacher_forced_surface_mass_rows.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("results/natural_evidence_v2/status/r4_candidate_v3_mass_elasticity_20260513")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def logistic_delta(current: float, goal: float) -> float:
    eps = 1e-12
    current = min(max(float(current), eps), 1.0 - eps)
    goal = min(max(float(goal), eps), 1.0 - eps)
    if current >= goal:
        return 0.0
    return math.log(goal * (1.0 - current) / (current * (1.0 - goal)))


def group_rows(rows: Iterable[Mapping[str, Any]], key_fields: tuple[str, ...]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in key_fields)].append(row)
    return grouped


def summarize_group(rows: list[Mapping[str, Any]], key_fields: tuple[str, ...], key: tuple[Any, ...]) -> dict[str, Any]:
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition"))].append(row)

    def mean_mass(condition: str) -> float:
        values = [float(row.get("target_mass", 0.0)) for row in by_condition.get(condition, [])]
        return statistics.fmean(values) if values else 0.0

    def rank1(condition: str) -> float:
        values = [1.0 if row.get("target_surface_rank1") else 0.0 for row in by_condition.get(condition, [])]
        return statistics.fmean(values) if values else 0.0

    base_mass = mean_mass("base")
    task_mass = mean_mass("task_only")
    protected_mass = mean_mass("protected")
    goal_vs_base = min(0.95, base_mass + 0.15)
    goal_vs_task = min(0.95, task_mass + 0.10)
    goal = max(goal_vs_base, goal_vs_task)
    result = {
        "row_count": len(rows),
        "base_mean_target_mass": base_mass,
        "task_only_mean_target_mass": task_mass,
        "protected_mean_target_mass": protected_mass,
        "protected_lift_vs_base": protected_mass - base_mass,
        "protected_lift_vs_task_only": protected_mass - task_mass,
        "protected_rank1_rate": rank1("protected"),
        "target_mass_goal": goal,
        "required_delta_nats_to_gate": logistic_delta(protected_mass, goal),
        "requires_gt_1_5_nats": logistic_delta(protected_mass, goal) > 1.5,
        "requires_gt_2_0_nats": logistic_delta(protected_mass, goal) > 2.0,
        "requires_gt_4_0_nats": logistic_delta(protected_mass, goal) > 4.0,
    }
    for field, value in zip(key_fields, key):
        result[field] = value
    return result


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze R4 candidate v3 target-mass elasticity from existing scored rows.")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {args.output_dir}")
    rows = read_jsonl(args.rows)
    if not rows:
        raise ValueError("no rows loaded")
    outputs: dict[str, list[dict[str, Any]]] = {}
    grouping_specs = {
        "overall": tuple(),
        "by_prefix": ("assistant_prefix_model_text",),
        "by_surface": ("target_surface",),
        "by_coordinate": ("coordinate_id",),
        "by_target_bit": ("target_bit",),
        "by_prefix_surface": ("assistant_prefix_model_text", "target_surface"),
    }
    for name, fields in grouping_specs.items():
        grouped = group_rows(rows, fields)
        outputs[name] = [summarize_group(group, fields, key) for key, group in sorted(grouped.items(), key=lambda item: str(item[0]))]
        write_csv(args.output_dir / f"{name}.csv", outputs[name])

    overall = outputs["overall"][0]
    by_coordinate = outputs["by_coordinate"]
    summary = {
        "artifact_role": "r4_candidate_v3_mass_elasticity_artifact_only",
        "generation_started": False,
        "input_rows": len(rows),
        "model_scoring_started": False,
        "overall": overall,
        "paper_claim_allowed": False,
        "status": "ARTIFACT_ONLY_R4_CANDIDATE_V3_MASS_ELASTICITY_RECORDED_NO_RUN",
        "training_started": False,
        "coordinate_requires_gt_1_5_nats": sum(1 for row in by_coordinate if row["requires_gt_1_5_nats"]),
        "coordinate_requires_gt_2_0_nats": sum(1 for row in by_coordinate if row["requires_gt_2_0_nats"]),
        "coordinate_requires_gt_4_0_nats": sum(1 for row in by_coordinate if row["requires_gt_4_0_nats"]),
    }
    report = [
        "# R4 Candidate v3 Mass-Elasticity Audit",
        "",
        "Artifact-only analysis over existing scored rows from job `856453`.",
        "",
        f"- input rows: `{len(rows)}`",
        f"- overall protected mass: `{overall['protected_mean_target_mass']}`",
        f"- overall base mass: `{overall['base_mean_target_mass']}`",
        f"- overall target mass goal: `{overall['target_mass_goal']}`",
        f"- required delta nats to gate: `{overall['required_delta_nats_to_gate']}`",
        f"- coordinates requiring >1.5 nats: `{summary['coordinate_requires_gt_1_5_nats']}`",
        f"- coordinates requiring >2.0 nats: `{summary['coordinate_requires_gt_2_0_nats']}`",
        f"- coordinates requiring >4.0 nats: `{summary['coordinate_requires_gt_4_0_nats']}`",
        "",
        "No training, model scoring, generation, Slurm, FAR, sanitizer, Llama, or paper-facing claim was started.",
    ]
    write_json(args.output_dir / "mass_elasticity_summary.json", summary)
    (args.output_dir / "mass_elasticity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
