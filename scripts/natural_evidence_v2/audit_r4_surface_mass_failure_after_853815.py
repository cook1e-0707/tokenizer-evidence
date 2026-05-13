from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SCORE_ROWS = (
    ROOT
    / "results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_score_853815/"
    / "r4_teacher_forced_surface_mass_rows.jsonl"
)
DEFAULT_PROBE_ROWS = (
    ROOT
    / "results/natural_evidence_v2/status/r4_surface_teacher_forced_probe_preflight_binary_repair_20260513/"
    / "r4_surface_teacher_forced_probe_rows.jsonl"
)
DEFAULT_BANK = (
    ROOT
    / "results/natural_evidence_v2/status/r4_binary_surface_bank_repair_plan_20260513/"
    / "candidate_binary_surface_bank.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_surface_mass_failure_diagnosis_after_853815"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-only diagnosis for failed R4 teacher-forced surface-mass "
            "gate after job 853815. This script reads existing rows only; it "
            "does not score models, submit Slurm, train, generate, or make claims."
        )
    )
    parser.add_argument("--score-rows", type=Path, default=DEFAULT_SCORE_ROWS)
    parser.add_argument("--probe-rows", type=Path, default=DEFAULT_PROBE_ROWS)
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def p95(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return float(ordered[index])


def key_for(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["prompt_id"]), int(row["coordinate_id"])


def group_by(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, ""))].append(row)
    return dict(grouped)


def summarize_condition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target_mass = [float(row["target_mass"]) for row in rows]
    other_mass = [float(row["other_mass"]) for row in rows]
    margin = [float(row["target_margin"]) for row in rows]
    rank1 = [1.0 if bool(row.get("target_surface_rank1")) else 0.0 for row in rows]
    return {
        "row_count": len(rows),
        "mean_target_mass": mean(target_mass),
        "median_target_mass": median(target_mass),
        "p95_target_mass": p95(target_mass),
        "mean_other_mass": mean(other_mass),
        "mean_target_margin": mean(margin),
        "median_target_margin": median(margin),
        "rank1_rate": mean(rank1),
    }


def condition_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    return group_by(rows, "condition")


def row_lift_records(score_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in score_rows:
        by_key[key_for(row)][str(row["condition"])] = row
    records: list[dict[str, Any]] = []
    for (prompt_id, coordinate_id), by_condition in sorted(by_key.items()):
        if not {"base", "protected", "task_only"}.issubset(by_condition):
            continue
        base = by_condition["base"]
        protected = by_condition["protected"]
        task = by_condition["task_only"]
        records.append(
            {
                "prompt_id": prompt_id,
                "coordinate_id": coordinate_id,
                "target_bit": int(protected["target_bit"]),
                "target_surface": str(protected.get("target_surface", "")),
                "base_target_mass": float(base["target_mass"]),
                "protected_target_mass": float(protected["target_mass"]),
                "task_only_target_mass": float(task["target_mass"]),
                "protected_lift_vs_base": float(protected["target_mass"]) - float(base["target_mass"]),
                "protected_lift_vs_task_only": float(protected["target_mass"]) - float(task["target_mass"]),
                "task_only_lift_vs_base": float(task["target_mass"]) - float(base["target_mass"]),
                "protected_margin": float(protected["target_margin"]),
                "target_surface_rank1_protected": bool(protected.get("target_surface_rank1")),
                "first_token_overlap_count": len(
                    set(int(x) for x in protected.get("target_first_token_ids", []))
                    & set(int(x) for x in protected.get("other_first_token_ids", []))
                ),
                "target_first_token_count": len(protected.get("target_first_token_ids", [])),
                "other_first_token_count": len(protected.get("other_first_token_ids", [])),
            }
        )
    return records


def summarize_lifts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lift_base = [float(row["protected_lift_vs_base"]) for row in records]
    lift_task = [float(row["protected_lift_vs_task_only"]) for row in records]
    protected_mass = [float(row["protected_target_mass"]) for row in records]
    protected_margin = [float(row["protected_margin"]) for row in records]
    return {
        "record_count": len(records),
        "mean_protected_lift_vs_base": mean(lift_base),
        "median_protected_lift_vs_base": median(lift_base),
        "positive_lift_vs_base_rate": mean([1.0 if value > 0 else 0.0 for value in lift_base]),
        "mean_protected_lift_vs_task_only": mean(lift_task),
        "median_protected_lift_vs_task_only": median(lift_task),
        "positive_lift_vs_task_only_rate": mean([1.0 if value > 0 else 0.0 for value in lift_task]),
        "mean_protected_target_mass": mean(protected_mass),
        "median_protected_target_mass": median(protected_mass),
        "p95_protected_target_mass": p95(protected_mass),
        "positive_protected_margin_rate": mean([1.0 if value > 0 else 0.0 for value in protected_margin]),
    }


def aggregate_lifts(
    records: Sequence[Mapping[str, Any]], probe_by_key: Mapping[tuple[str, int], Mapping[str, Any]], field: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        probe = probe_by_key.get((str(record["prompt_id"]), int(record["coordinate_id"])), {})
        grouped[str(probe.get(field, ""))].append(record)
    rows: list[dict[str, Any]] = []
    for value, items in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        summary = summarize_lifts(items)
        rows.append({"group": value, **summary})
    return rows


def coordinate_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["coordinate_id"])].append(record)
    rows: list[dict[str, Any]] = []
    for coordinate_id, items in sorted(grouped.items()):
        target_bits = sorted({int(row["target_bit"]) for row in items})
        summary = summarize_lifts(items)
        rows.append(
            {
                "coordinate_id": coordinate_id,
                "target_bits": "|".join(str(bit) for bit in target_bits),
                **summary,
            }
        )
    return rows


def surface_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["target_surface"])].append(record)
    rows: list[dict[str, Any]] = []
    for surface, items in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        summary = summarize_lifts(items)
        rows.append({"target_surface": surface, **summary})
    return rows


def bucket_overlap_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overlaps = [int(row["first_token_overlap_count"]) for row in records]
    target_counts = [int(row["target_first_token_count"]) for row in records]
    other_counts = [int(row["other_first_token_count"]) for row in records]
    return {
        "any_target_other_first_token_overlap_rate": mean([1.0 if value > 0 else 0.0 for value in overlaps]),
        "max_target_other_first_token_overlap_count": max(overlaps) if overlaps else 0,
        "mean_target_first_token_count": mean([float(value) for value in target_counts]),
        "mean_other_first_token_count": mean([float(value) for value in other_counts]),
    }


def bank_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    by_coordinate_bit: dict[tuple[int, int], int] = defaultdict(int)
    source_rules: dict[str, int] = defaultdict(int)
    for entry in entries:
        by_coordinate_bit[(int(entry["coordinate_id"]), int(entry["bucket_id"]))] += 1
        source_rules[str(entry.get("source_rule_id", ""))] += 1
    missing_sides = [
        {"coordinate_id": coordinate, "missing_bit": bit}
        for coordinate in range(int(payload.get("num_coordinates", 0)))
        for bit in (0, 1)
        if by_coordinate_bit.get((coordinate, bit), 0) == 0
    ]
    return {
        "path": str(path),
        "schema_name": payload.get("schema_name"),
        "contract_id": payload.get("contract_id"),
        "entry_count": len(entries),
        "num_coordinates": payload.get("num_coordinates"),
        "phrase_level": payload.get("phrase_level"),
        "first_word_only": payload.get("first_word_only"),
        "missing_coordinate_sides": missing_sides,
        "source_rule_counts": dict(sorted(source_rules.items())),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    score_rows = read_jsonl(args.score_rows)
    probe_rows = read_jsonl(args.probe_rows)
    probe_by_key = {key_for(row): row for row in probe_rows}
    records = row_lift_records(score_rows)
    rows_by_condition = condition_map(score_rows)
    condition_summary = {
        condition: summarize_condition(rows)
        for condition, rows in sorted(rows_by_condition.items())
    }
    coordinate_table = coordinate_rows(records)
    surface_table = surface_rows(records)
    prefix_table = aggregate_lifts(records, probe_by_key, "assistant_prefix_before_surface")
    prompt_family_table = aggregate_lifts(records, probe_by_key, "split")
    overlap_summary = bucket_overlap_summary(records)
    bank = bank_summary(args.surface_bank)

    protected_lift_vs_base = float(summarize_lifts(records)["mean_protected_lift_vs_base"])
    protected_lift_vs_task = float(summarize_lifts(records)["mean_protected_lift_vs_task_only"])
    diagnosis = {
        "schema_name": "natural_evidence_v2_r4_surface_mass_failure_diagnosis_v1",
        "status": "FAIL_SURFACE_MASS_GATE_ARTIFACT_DIAGNOSED",
        "score_rows": str(args.score_rows),
        "probe_rows": str(args.probe_rows),
        "surface_bank": str(args.surface_bank),
        "score_row_count": len(score_rows),
        "probe_row_count": len(probe_rows),
        "joined_records": len(records),
        "condition_summary": condition_summary,
        "lift_summary": summarize_lifts(records),
        "bucket_overlap_summary": overlap_summary,
        "bank_summary": bank,
        "coordinate_count": len(coordinate_table),
        "surface_count": len(surface_table),
        "prefix_shape_count": len(prefix_table),
        "method_failure_classification": {
            "slurm_or_provider_failure": False,
            "formal_binary_bank_missing_side_failure": bool(bank["missing_coordinate_sides"]),
            "first_token_overlap_primary_failure": overlap_summary[
                "any_target_other_first_token_overlap_rate"
            ]
            > 0.0,
            "protected_adapter_learned_surface_channel": protected_lift_vs_base >= 0.15
            and protected_lift_vs_task >= 0.10,
            "primary_failure": (
                "target phrase surfaces have near-zero probability and the existing "
                "protected adapter does not increase their mass under teacher-forced "
                "R4 prefixes"
            ),
        },
        "next_allowed_action": (
            "Artifact-only repair design: revise R4 target construction and surface-bank/prefix "
            "shape before any more Slurm scoring, generation, training, Llama, FAR, sanitizer, "
            "or paper-claim action."
        ),
        "generation_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "llama_started": False,
        "far_aggregation_started": False,
        "paper_claim_allowed": False,
    }

    write_json_new(output_dir / "surface_mass_failure_diagnosis_summary.json", diagnosis)
    write_csv_new(
        output_dir / "per_coordinate_surface_mass_lift.csv",
        coordinate_table,
        [
            "coordinate_id",
            "target_bits",
            "record_count",
            "mean_protected_lift_vs_base",
            "median_protected_lift_vs_base",
            "positive_lift_vs_base_rate",
            "mean_protected_lift_vs_task_only",
            "median_protected_lift_vs_task_only",
            "positive_lift_vs_task_only_rate",
            "mean_protected_target_mass",
            "median_protected_target_mass",
            "p95_protected_target_mass",
            "positive_protected_margin_rate",
        ],
    )
    write_csv_new(
        output_dir / "per_target_surface_mass_lift.csv",
        surface_table,
        [
            "target_surface",
            "record_count",
            "mean_protected_lift_vs_base",
            "median_protected_lift_vs_base",
            "positive_lift_vs_base_rate",
            "mean_protected_lift_vs_task_only",
            "median_protected_lift_vs_task_only",
            "positive_lift_vs_task_only_rate",
            "mean_protected_target_mass",
            "median_protected_target_mass",
            "p95_protected_target_mass",
            "positive_protected_margin_rate",
        ],
    )
    write_csv_new(
        output_dir / "per_prefix_shape_mass_lift.csv",
        prefix_table,
        [
            "group",
            "record_count",
            "mean_protected_lift_vs_base",
            "median_protected_lift_vs_base",
            "positive_lift_vs_base_rate",
            "mean_protected_lift_vs_task_only",
            "median_protected_lift_vs_task_only",
            "positive_lift_vs_task_only_rate",
            "mean_protected_target_mass",
            "median_protected_target_mass",
            "p95_protected_target_mass",
            "positive_protected_margin_rate",
        ],
    )
    write_csv_new(
        output_dir / "per_split_mass_lift.csv",
        prompt_family_table,
        [
            "group",
            "record_count",
            "mean_protected_lift_vs_base",
            "median_protected_lift_vs_base",
            "positive_lift_vs_base_rate",
            "mean_protected_lift_vs_task_only",
            "median_protected_lift_vs_task_only",
            "positive_lift_vs_task_only_rate",
            "mean_protected_target_mass",
            "median_protected_target_mass",
            "p95_protected_target_mass",
            "positive_protected_margin_rate",
        ],
    )

    worst_coordinates = sorted(
        coordinate_table,
        key=lambda row: float(row["mean_protected_lift_vs_base"]),
    )[:5]
    best_coordinates = sorted(
        coordinate_table,
        key=lambda row: float(row["mean_protected_lift_vs_base"]),
        reverse=True,
    )[:5]
    top_surfaces = sorted(
        surface_table,
        key=lambda row: float(row["mean_protected_target_mass"]),
        reverse=True,
    )[:8]
    lines = [
        "# R4 surface-mass failure diagnosis after 853815",
        "",
        "This is an artifact-only diagnosis. It reads existing 853815 score rows,",
        "the binary surface-bank repair candidate, and the frozen teacher-forced",
        "probe rows. It does not train, generate, score models, submit Slurm, run",
        "Llama, aggregate FAR, or make paper claims.",
        "",
        "## Gate Result",
        "",
        "- status: `FAIL_SURFACE_MASS_GATE_ARTIFACT_DIAGNOSED`",
        f"- scored rows: `{len(score_rows)}`",
        f"- joined base/protected/task-only records: `{len(records)}`",
        f"- protected mean target mass: `{condition_summary['protected']['mean_target_mass']:.10f}`",
        f"- protected-vs-base mean lift: `{protected_lift_vs_base:.10f}`",
        f"- protected-vs-task-only mean lift: `{protected_lift_vs_task:.10f}`",
        f"- protected rank-1 rate: `{condition_summary['protected']['rank1_rate']:.4f}`",
        "",
        "## Diagnostic Interpretation",
        "",
        "The binary candidate bank has both bit sides for every coordinate, so the",
        "previous one-sided-bank formal problem is not the active blocker. The",
        "first-token target/other overlap rate is not the primary failure either.",
        "The active blocker is that the selected phrase-level target cylinders are",
        "extremely low probability under the relevant prefixes, and the existing",
        "protected adapter does not increase their mass.",
        "",
        "## Worst Coordinates By Protected-vs-Base Lift",
        "",
    ]
    for row in worst_coordinates:
        lines.append(
            f"- coordinate `{row['coordinate_id']}`: lift `{float(row['mean_protected_lift_vs_base']):.10f}`, "
            f"protected mass `{float(row['mean_protected_target_mass']):.10f}`"
        )
    lines.extend(["", "## Best Coordinates By Protected-vs-Base Lift", ""])
    for row in best_coordinates:
        lines.append(
            f"- coordinate `{row['coordinate_id']}`: lift `{float(row['mean_protected_lift_vs_base']):.10f}`, "
            f"protected mass `{float(row['mean_protected_target_mass']):.10f}`"
        )
    lines.extend(["", "## Highest Protected-Mass Surfaces", ""])
    for row in top_surfaces:
        lines.append(
            f"- `{row['target_surface']}`: protected mass `{float(row['mean_protected_target_mass']):.10f}`, "
            f"lift vs base `{float(row['mean_protected_lift_vs_base']):.10f}`"
        )
    lines.extend(
        [
            "",
            "## Next Allowed Action",
            "",
            "Artifact-only repair design for R4 target construction, surface-bank",
            "selection, and prefix shapes. Do not submit another scoring job or run",
            "generation/training/Llama/FAR/sanitizer/paper-claim actions from this state.",
            "",
        ]
    )
    write_text_new(output_dir / "surface_mass_failure_diagnosis_report.md", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
