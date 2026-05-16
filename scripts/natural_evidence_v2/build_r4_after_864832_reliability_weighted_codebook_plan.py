#!/usr/bin/env python3
"""Build an artifact-only reliability-weighted codebook plan after job 866147."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATTRIBUTION = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_864832_two_sided_controller_safety_bound_failure_attribution_866147_20260516"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_864832_reliability_weighted_codebook_plan_20260516"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution-dir", type=Path, default=DEFAULT_ATTRIBUTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-lift", type=float, default=0.03)
    parser.add_argument("--min-rank1", type=float, default=0.80)
    parser.add_argument("--min-margin", type=float, default=0.0)
    args = parser.parse_args()

    coord_rows = read_csv(args.attribution_dir / "by_coordinate.csv")
    by_coord = {int(row["coordinate_id"]): row for row in coord_rows}

    reliable = {
        coord
        for coord, row in by_coord.items()
        if f(row, "controlled_lift_vs_base") >= args.min_lift
        and f(row, "controlled_rank1_rate") >= args.min_rank1
        and f(row, "controlled_median_margin") > args.min_margin
    }

    pair_rows: list[dict[str, Any]] = []
    for base_coord in range(16):
        mate = base_coord + 16
        left = by_coord.get(base_coord)
        right = by_coord.get(mate)
        if left is None or right is None:
            continue
        pair_reliable = base_coord in reliable and mate in reliable
        pair_rows.append(
            {
                "pair_id": len(pair_rows),
                "coordinate_a": base_coord,
                "coordinate_b": mate,
                "pair_reliable": pair_reliable,
                "min_lift_vs_base": min(
                    f(left, "controlled_lift_vs_base"),
                    f(right, "controlled_lift_vs_base"),
                ),
                "min_rank1": min(
                    f(left, "controlled_rank1_rate"),
                    f(right, "controlled_rank1_rate"),
                ),
                "min_margin": min(
                    f(left, "controlled_median_margin"),
                    f(right, "controlled_median_margin"),
                ),
                "mean_controlled_mass": (
                    f(left, "controlled_mean_target_mass")
                    + f(right, "controlled_mean_target_mass")
                )
                / 2.0,
            }
        )

    selected_pairs = [row for row in pair_rows if row["pair_reliable"]]
    selected_pairs.sort(
        key=lambda row: (row["min_rank1"], row["min_lift_vs_base"], row["min_margin"]),
        reverse=True,
    )
    selected_pairs = selected_pairs[:8]
    selected_coordinates = [
        coord
        for row in selected_pairs
        for coord in (int(row["coordinate_a"]), int(row["coordinate_b"]))
    ]

    viable_8_pair_codebook = len(selected_pairs) >= 8
    plan_status = (
        "PASS_RELIABILITY_WEIGHTED_CODEBOOK_PLAN_8_PAIRS_AVAILABLE_NO_COMPUTE"
        if viable_8_pair_codebook
        else "FAIL_RELIABILITY_WEIGHTED_CODEBOOK_PLAN_INSUFFICIENT_PAIRS_NO_COMPUTE"
    )

    codebook_plan = {
        "schema_name": "r4_after_864832_reliability_weighted_codebook_plan_v1",
        "status": plan_status,
        "source_attribution_dir": str(args.attribution_dir),
        "selection_thresholds": {
            "min_lift_vs_base": args.min_lift,
            "min_rank1": args.min_rank1,
            "min_median_margin": args.min_margin,
        },
        "selected_pair_count": len(selected_pairs),
        "selected_coordinate_count": len(selected_coordinates),
        "selected_pairs": selected_pairs,
        "selected_coordinates": selected_coordinates,
        "candidate_contract": {
            "contract_id": "a55e",
            "payload_bits": 4,
            "checksum_bits": 4,
            "coordinate_repetition": 2,
            "total_coordinates": 16,
            "decoder": "pair_majority_then_checksum",
            "format_scrub_primary": "all",
        },
        "claim_scope": {
            "generation_unlocked": False,
            "training_unlocked": False,
            "llama_unlocked": False,
            "far_unlocked": False,
            "paper_claim_unlocked": False,
        },
        "next_allowed_action": (
            "If reviewed, freeze a new reliability-weighted precommit and run a "
            "teacher-forced decoder oracle or small generation route only after "
            "the usual route decision and control-plane preflight."
        ),
    }

    write_json(args.output_dir / "codebook_plan_summary.json", codebook_plan)
    write_csv(
        args.output_dir / "coordinate_pair_reliability.csv",
        pair_rows,
        [
            "pair_id",
            "coordinate_a",
            "coordinate_b",
            "pair_reliable",
            "min_lift_vs_base",
            "min_rank1",
            "min_margin",
            "mean_controlled_mass",
        ],
    )
    write_json(
        args.output_dir / "candidate_codebook.json",
        {
            "schema_name": "r4_after_864832_candidate_reliability_weighted_codebook_v1",
            "status": "CANDIDATE_NOT_PRECOMMITTED",
            "selected_coordinates": selected_coordinates,
            "selected_pairs": selected_pairs,
            "payload_bits": 4,
            "checksum_bits": 4,
            "coordinate_repetition": 2,
            "requires_review_before_use": True,
        },
    )

    lines = [
        "# R4 After 864832 Reliability-Weighted Codebook Plan",
        "",
        f"Status: `{plan_status}`",
        "",
        "This is artifact-only planning from reviewed `866147` dev scoring artifacts.",
        "It does not run Slurm, generation, training, Llama, FAR, sanitizer, or claims.",
        "",
        "## Selection",
        "",
        f"- min lift vs base: `{args.min_lift}`",
        f"- min rank1: `{args.min_rank1}`",
        f"- min median margin: `>{args.min_margin}`",
        f"- selected pairs: `{len(selected_pairs)}`",
        f"- selected coordinates: `{selected_coordinates}`",
        "",
        "## Candidate Contract",
        "",
        "- 4 payload bits",
        "- 4 checksum bits",
        "- 2 coordinates per bit",
        "- pair-majority then checksum decoder",
        "- primary reporting must remain `format_scrub=all`",
        "",
        "## Next",
        "",
        "The candidate codebook is not precommitted and cannot be used for compute",
        "until a separate reviewed route decision freezes it and passes control-plane",
        "preflight.",
    ]
    (args.output_dir / "codebook_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(plan_status)
    print(args.output_dir)
    return 0 if viable_8_pair_codebook else 1


if __name__ == "__main__":
    raise SystemExit(main())
