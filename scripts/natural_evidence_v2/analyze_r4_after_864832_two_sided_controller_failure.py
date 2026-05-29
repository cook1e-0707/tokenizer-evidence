#!/usr/bin/env python3
"""Artifact-only failure attribution for the R4 two-sided controller run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_864832_two_sided_controller_only_score_865434_review"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_864832_two_sided_controller_only_failure_attribution_865434_20260516"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out):
        return 0.0
    return out


@dataclass(frozen=True)
class RowKey:
    prompt_index: int
    coordinate_id: int
    prefix: str
    surface: str
    target_bit: int


def row_key(row: dict[str, Any]) -> RowKey:
    return RowKey(
        prompt_index=int(row["prompt_index"]),
        coordinate_id=int(row["coordinate_id"]),
        prefix=str(row["assistant_prefix_model_text"]),
        surface=str(row["target_surface_label"]),
        target_bit=int(row["target_bit"]),
    )


def aggregate_stratum(
    paired: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        buckets[tuple(row[field] for field in key_fields)].append(row)

    output: list[dict[str, Any]] = []
    for key, rows in sorted(buckets.items()):
        base_mass = [r["base_target_mass"] for r in rows]
        controlled_mass = [r["controlled_target_mass"] for r in rows]
        task_mass = [r["task_only_target_mass"] for r in rows]
        controlled_margin = [r["controlled_target_margin"] for r in rows]
        output.append(
            {
                **dict(zip(key_fields, key)),
                "row_count": len(rows),
                "base_mean_target_mass": fmean(base_mass),
                "task_only_mean_target_mass": fmean(task_mass),
                "controlled_mean_target_mass": fmean(controlled_mass),
                "controlled_lift_vs_base": fmean([r["controlled_lift_vs_base"] for r in rows]),
                "controlled_lift_vs_task_only": fmean(
                    [r["controlled_lift_vs_task_only"] for r in rows]
                ),
                "controlled_rank1_rate": fmean(
                    [1.0 if r["controlled_target_surface_rank1"] else 0.0 for r in rows]
                ),
                "controlled_median_margin": median(controlled_margin) if controlled_margin else 0.0,
                "controlled_mean_kl_to_base": fmean([r["controlled_kl_to_base"] for r in rows]),
            }
        )
    output.sort(
        key=lambda r: (
            safe_float(r["controlled_lift_vs_base"]),
            safe_float(r["controlled_rank1_rate"]),
        )
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    aggregate = read_json(args.review_dir / "aggregate_summary.json")
    best = aggregate["best_by_controlled_lift_vs_base"][0]
    rows_path = (ROOT / best["path"]).with_name("r4_teacher_forced_surface_mass_rows.jsonl")
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)

    by_key: dict[RowKey, dict[str, dict[str, Any]]] = defaultdict(dict)
    cap_reasons: Counter[str] = Counter()
    with rows_path.open() as f:
        for line in f:
            row = json.loads(line)
            condition = row["condition"]
            by_key[row_key(row)][condition] = row
            if condition == "controlled_base":
                cap_reasons.update(row.get("controller_cap_reasons") or [])

    paired: list[dict[str, Any]] = []
    missing_conditions = 0
    for key, conds in by_key.items():
        required = {"base", "task_only", "controlled_base"}
        if not required.issubset(conds):
            missing_conditions += 1
            continue
        base = conds["base"]
        task = conds["task_only"]
        controlled = conds["controlled_base"]
        paired.append(
            {
                "prompt_index": key.prompt_index,
                "coordinate_id": key.coordinate_id,
                "assistant_prefix_model_text": key.prefix,
                "target_surface_label": key.surface,
                "target_bit": key.target_bit,
                "base_target_mass": safe_float(base["target_mass"]),
                "task_only_target_mass": safe_float(task["target_mass"]),
                "controlled_target_mass": safe_float(controlled["target_mass"]),
                "controlled_lift_vs_base": safe_float(controlled["target_mass"])
                - safe_float(base["target_mass"]),
                "controlled_lift_vs_task_only": safe_float(controlled["target_mass"])
                - safe_float(task["target_mass"]),
                "controlled_target_surface_rank1": bool(controlled["target_surface_rank1"]),
                "controlled_target_margin": safe_float(controlled["target_margin"]),
                "controlled_kl_to_base": safe_float(controlled.get("controller_kl_to_base")),
            }
        )

    by_coordinate = aggregate_stratum(paired, ("coordinate_id",))
    by_prefix = aggregate_stratum(paired, ("assistant_prefix_model_text",))
    by_surface = aggregate_stratum(paired, ("target_surface_label",))
    by_prefix_surface = aggregate_stratum(
        paired, ("assistant_prefix_model_text", "target_surface_label")
    )

    summary = {
        "schema_name": "r4_after_864832_two_sided_controller_failure_attribution_v1",
        "status": "FAILURE_ATTRIBUTION_RECORDED_NO_COMPUTE",
        "source_review": str(args.review_dir),
        "source_rows": str(rows_path),
        "selected_grid_index": best["grid_index"],
        "selected_grid_bonus_nats": best["bonus_nats"],
        "selected_grid_penalty_nats": best["penalty_nats"],
        "selected_grid_max_target_mass": best["max_target_mass"],
        "selected_grid_max_kl_budget": best["max_kl_budget"],
        "paired_rows": len(paired),
        "missing_condition_groups": missing_conditions,
        "controller_cap_reasons": dict(cap_reasons),
        "overall": {
            "base_mean_target_mass": best["base_mean_target_mass"],
            "task_only_mean_target_mass": best["task_only_mean_target_mass"],
            "controlled_mean_target_mass": best["controlled_mean_target_mass"],
            "controlled_lift_vs_base": best["controlled_lift_vs_base"],
            "controlled_lift_vs_task_only": best["controlled_lift_vs_task_only"],
            "controlled_rank1_rate": best["controlled_rank1_rate"],
            "controlled_median_target_margin": best["controlled_median_target_margin"],
            "wrong_key_mean_target_mass": best["wrong_key_mean_target_mass"],
            "wrong_payload_mean_target_mass": best["wrong_payload_mean_target_mass"],
        },
        "weakest_coordinates_by_lift": by_coordinate[:8],
        "strongest_coordinates_by_lift": list(reversed(by_coordinate[-8:])),
        "weakest_prefixes_by_lift": by_prefix[:8],
        "strongest_prefixes_by_lift": list(reversed(by_prefix[-8:])),
        "weakest_surfaces_by_lift": by_surface[:12],
        "strongest_surfaces_by_lift": list(reversed(by_surface[-12:])),
        "interpretation": (
            "Best controller grid remains below the teacher-forced selective gate. "
            "Failure attribution is artifact-only and does not unlock generation."
        ),
        "next_allowed_action": (
            "Design reviewed repair or pivot route using this attribution; no Slurm submission "
            "until a new route decision and allowlist preflight are recorded."
        ),
    }

    write_json(args.output_dir / "failure_attribution_summary.json", summary)
    write_csv(
        args.output_dir / "by_coordinate.csv",
        by_coordinate,
        [
            "coordinate_id",
            "row_count",
            "base_mean_target_mass",
            "task_only_mean_target_mass",
            "controlled_mean_target_mass",
            "controlled_lift_vs_base",
            "controlled_lift_vs_task_only",
            "controlled_rank1_rate",
            "controlled_median_margin",
            "controlled_mean_kl_to_base",
        ],
    )
    write_csv(
        args.output_dir / "by_prefix.csv",
        by_prefix,
        [
            "assistant_prefix_model_text",
            "row_count",
            "base_mean_target_mass",
            "task_only_mean_target_mass",
            "controlled_mean_target_mass",
            "controlled_lift_vs_base",
            "controlled_lift_vs_task_only",
            "controlled_rank1_rate",
            "controlled_median_margin",
            "controlled_mean_kl_to_base",
        ],
    )
    write_csv(
        args.output_dir / "by_surface.csv",
        by_surface,
        [
            "target_surface_label",
            "row_count",
            "base_mean_target_mass",
            "task_only_mean_target_mass",
            "controlled_mean_target_mass",
            "controlled_lift_vs_base",
            "controlled_lift_vs_task_only",
            "controlled_rank1_rate",
            "controlled_median_margin",
            "controlled_mean_kl_to_base",
        ],
    )
    write_csv(
        args.output_dir / "by_prefix_surface.csv",
        by_prefix_surface,
        [
            "assistant_prefix_model_text",
            "target_surface_label",
            "row_count",
            "base_mean_target_mass",
            "task_only_mean_target_mass",
            "controlled_mean_target_mass",
            "controlled_lift_vs_base",
            "controlled_lift_vs_task_only",
            "controlled_rank1_rate",
            "controlled_median_margin",
            "controlled_mean_kl_to_base",
        ],
    )

    weakest_coord = summary["weakest_coordinates_by_lift"][0] if by_coordinate else {}
    strongest_coord = summary["strongest_coordinates_by_lift"][0] if by_coordinate else {}
    lines = [
        "# R4 After 864832 Two-Sided Controller Failure Attribution",
        "",
        "Status: `FAILURE_ATTRIBUTION_RECORDED_NO_COMPUTE`",
        "",
        f"Source review: `{args.review_dir}`",
        f"Selected grid: `{best['grid_index']}` "
        f"(bonus={best['bonus_nats']}, penalty={best['penalty_nats']}, "
        f"max_target_mass={best['max_target_mass']}, max_kl_budget={best['max_kl_budget']})",
        "",
        "## Overall",
        "",
        f"- controlled mean target mass: `{best['controlled_mean_target_mass']:.6f}`",
        f"- lift vs base: `{best['controlled_lift_vs_base']:.6f}`",
        f"- lift vs task-only: `{best['controlled_lift_vs_task_only']:.6f}`",
        f"- rank1: `{best['controlled_rank1_rate']:.6f}`",
        f"- median margin: `{best['controlled_median_target_margin']:.6f}`",
        f"- wrong-key mean target mass: `{best['wrong_key_mean_target_mass']:.6f}`",
        f"- wrong-payload mean target mass: `{best['wrong_payload_mean_target_mass']:.6f}`",
        "",
        "The controller improved target mass and median margin but remains well below the",
        "`+0.15` lift and `0.75` rank1 gate. Null controls remain below their basic gates.",
        "",
        "## Strata",
        "",
        f"Weakest coordinate by lift: `{weakest_coord.get('coordinate_id')}` "
        f"lift `{safe_float(weakest_coord.get('controlled_lift_vs_base')):.6f}`, "
        f"rank1 `{safe_float(weakest_coord.get('controlled_rank1_rate')):.6f}`.",
        f"Strongest coordinate by lift: `{strongest_coord.get('coordinate_id')}` "
        f"lift `{safe_float(strongest_coord.get('controlled_lift_vs_base')):.6f}`, "
        f"rank1 `{safe_float(strongest_coord.get('controlled_rank1_rate')):.6f}`.",
        "",
        "Detailed CSVs are written for coordinate, prefix, surface, and prefix-surface strata.",
        "",
        "## Next",
        "",
        "This attribution does not unlock generation or training. The next step is a reviewed",
        "repair/pivot route based on these strata, with fresh precommit, allowlist safety,",
        "Hermes notification, and H200 Slurm-only execution if compute is later needed.",
    ]
    (args.output_dir / "failure_attribution.md").write_text("\n".join(lines) + "\n")

    print(summary["status"])
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
