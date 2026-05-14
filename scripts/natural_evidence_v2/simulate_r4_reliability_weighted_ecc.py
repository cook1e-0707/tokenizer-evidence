from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ELASTICITY = Path("results/natural_evidence_v2/status/r4_candidate_v3_mass_elasticity_20260513/by_coordinate.csv")
DEFAULT_OUTPUT_DIR = Path("results/natural_evidence_v2/status/r4_candidate_v3_channel_capacity_analysis_20260513")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def reliability_weight(required_delta_nats: float) -> float:
    if required_delta_nats <= 1.0:
        return 1.0
    if required_delta_nats <= 1.5:
        return 0.75
    if required_delta_nats <= 2.0:
        return 0.5
    if required_delta_nats <= 4.0:
        return 0.25
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate reliability-weighted ECC viability from R4 candidate v3 coordinate elasticity.")
    parser.add_argument("--coordinate-elasticity", type=Path, default=DEFAULT_ELASTICITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {args.output_dir}")
    rows = read_csv(args.coordinate_elasticity)
    if not rows:
        raise ValueError("no coordinate rows loaded")
    out_rows = []
    for row in rows:
        delta = float(row["required_delta_nats_to_gate"])
        weight = reliability_weight(delta)
        out_rows.append(
            {
                **row,
                "coordinate_reliability_weight": weight,
                "coordinate_viability": (
                    "high" if weight >= 0.75 else "medium" if weight >= 0.5 else "low" if weight > 0 else "no_go"
                ),
            }
        )
    total_weight = sum(float(row["coordinate_reliability_weight"]) for row in out_rows)
    summary = {
        "artifact_role": "r4_candidate_v3_reliability_weighted_ecc_simulation_artifact_only",
        "coordinate_count": len(out_rows),
        "generation_started": False,
        "model_scoring_started": False,
        "paper_claim_allowed": False,
        "status": "ARTIFACT_ONLY_R4_RELIABILITY_WEIGHTED_ECC_SIMULATION_RECORDED_NO_RUN",
        "total_reliability_weight": total_weight,
        "training_started": False,
        "viability_counts": {
            label: sum(1 for row in out_rows if row["coordinate_viability"] == label)
            for label in ("high", "medium", "low", "no_go")
        },
    }
    write_csv(args.output_dir / "coordinate_capacity_summary.csv", out_rows)
    write_json(args.output_dir / "ecc_simulation_summary.json", summary)
    report = [
        "# R4 Reliability-Weighted ECC Simulation",
        "",
        "Artifact-only simulation from coordinate-level mass-elasticity rows.",
        "",
        f"- coordinates: `{summary['coordinate_count']}`",
        f"- total reliability weight: `{summary['total_reliability_weight']}`",
        f"- viability counts: `{summary['viability_counts']}`",
        "",
        "No model scoring, training, generation, Slurm, FAR, sanitizer, Llama, or paper-facing claim was started.",
    ]
    (args.output_dir / "coordinate_capacity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
