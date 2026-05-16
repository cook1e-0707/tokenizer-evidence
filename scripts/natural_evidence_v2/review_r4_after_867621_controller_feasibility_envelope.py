from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping


GRID_RE = re.compile(r"grid_(\d+)$")


def read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def grid_index(path: Path) -> int:
    match = GRID_RE.match(path.parent.name)
    if not match:
        raise ValueError(f"cannot infer grid index from {path}")
    return int(match.group(1))


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


def review(input_dir: Path, output_dir: Path, *, expected_grid_count: int, source_job_id: str) -> dict[str, Any]:
    summary_paths = sorted(input_dir.glob("grid_*/r4_teacher_forced_surface_mass_summary.json"))
    if len(summary_paths) != expected_grid_count:
        raise ValueError(f"expected {expected_grid_count} grid summaries, observed {len(summary_paths)}")

    rows: list[dict[str, Any]] = []
    for path in summary_paths:
        payload = read_json(path)
        ctrl = payload.get("controller_only_summary")
        if not isinstance(ctrl, Mapping):
            raise ValueError(f"missing controller_only_summary: {path}")
        config = payload.get("controller_config")
        if not isinstance(config, Mapping):
            raise ValueError(f"missing controller_config: {path}")
        rows.append(
            {
                "grid_index": grid_index(path),
                "path": str(path),
                "bonus_nats": float(config["bonus_nats"]),
                "penalty_nats": float(config["penalty_nats"]),
                "max_target_mass": float(config["max_target_mass"]),
                "max_kl_budget": float(config["max_kl_budget"]),
                "controlled_mean_target_mass": float(ctrl["controlled_base_mean_target_mass"]),
                "controlled_lift_vs_base": float(ctrl["controlled_base_lift_vs_base"]),
                "controlled_lift_vs_task_only": float(ctrl["controlled_base_lift_vs_task_only"]),
                "controlled_rank1_rate": float(ctrl["controlled_base_rank1_rate"]),
                "controlled_median_target_margin": float(ctrl["controlled_base_median_target_margin"]),
                "controlled_basic_gate_pass": bool(ctrl["controlled_basic_gate_pass"]),
                "wrong_key_mean_target_mass": float(ctrl["wrong_key_mean_target_mass"]),
                "wrong_key_lift_vs_base": float(ctrl["wrong_key_lift_vs_base"]),
                "wrong_key_rank1_rate": float(ctrl["wrong_key_rank1_rate"]),
                "wrong_key_basic_gate_pass": bool(ctrl["wrong_key_basic_gate_pass"]),
                "wrong_payload_mean_target_mass": float(ctrl["wrong_payload_mean_target_mass"]),
                "wrong_payload_lift_vs_base": float(ctrl["wrong_payload_lift_vs_base"]),
                "wrong_payload_rank1_rate": float(ctrl["wrong_payload_rank1_rate"]),
                "wrong_payload_basic_gate_pass": bool(ctrl["wrong_payload_basic_gate_pass"]),
                "overall_selective_gate_pass": bool(ctrl["overall_selective_gate_pass"]),
            }
        )

    rows.sort(key=lambda row: (row["controlled_lift_vs_base"], row["controlled_rank1_rate"]), reverse=True)
    best = rows[0]
    passing = [row for row in rows if row["overall_selective_gate_pass"]]
    wrong_control_failures = [
        row for row in rows if row["wrong_key_basic_gate_pass"] or row["wrong_payload_basic_gate_pass"]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "grid_index",
        "bonus_nats",
        "penalty_nats",
        "max_target_mass",
        "max_kl_budget",
        "controlled_mean_target_mass",
        "controlled_lift_vs_base",
        "controlled_lift_vs_task_only",
        "controlled_rank1_rate",
        "controlled_median_target_margin",
        "controlled_basic_gate_pass",
        "wrong_key_mean_target_mass",
        "wrong_key_lift_vs_base",
        "wrong_key_rank1_rate",
        "wrong_key_basic_gate_pass",
        "wrong_payload_mean_target_mass",
        "wrong_payload_lift_vs_base",
        "wrong_payload_rank1_rate",
        "wrong_payload_basic_gate_pass",
        "overall_selective_gate_pass",
        "path",
    ]
    write_csv(output_dir / "grid_summary.csv", rows, fields)
    aggregate = {
        "schema_name": "natural_evidence_v2_r4_after_867621_controller_feasibility_envelope_review_v1",
        "status": (
            "PASS_R4_AFTER_867621_CONTROLLER_FEASIBILITY_ENVELOPE_TEACHER_FORCED_GATE"
            if passing
            else "FAIL_R4_AFTER_867621_CONTROLLER_FEASIBILITY_ENVELOPE_TEACHER_FORCED_GATE"
        ),
        "source_job_id": source_job_id,
        "input_dir": str(input_dir),
        "grid_count": len(rows),
        "passing_grid_count": len(passing),
        "wrong_control_failure_count": len(wrong_control_failures),
        "best_by_controlled_lift_vs_base": rows[:5],
        "passing_grids": passing,
        "generation_unlocked": bool(passing),
        "training_unlocked": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "reviewed small generation route planning"
            if passing
            else "artifact-only failure analysis or repair/pivot route; no generation"
        ),
    }
    write_json(output_dir / "aggregate_summary.json", aggregate)

    report = f"""# R4 After 867621 Controller Feasibility-Envelope Review

Status: `{aggregate["status"]}`

Job `{source_job_id}` completed as a {expected_grid_count}-grid H200/pomplun
controller-only teacher-forced scoring array.

Best grid by controlled lift vs base:

```text
grid_index: {best["grid_index"]}
bonus_nats: {best["bonus_nats"]}
penalty_nats: {best["penalty_nats"]}
max_target_mass: {best["max_target_mass"]}
max_kl_budget: {best["max_kl_budget"]}
controlled lift vs base: {best["controlled_lift_vs_base"]:.6f}
controlled lift vs task_only: {best["controlled_lift_vs_task_only"]:.6f}
controlled rank1: {best["controlled_rank1_rate"]:.6f}
controlled median margin: {best["controlled_median_target_margin"]:.6f}
wrong-key basic gate pass: {best["wrong_key_basic_gate_pass"]}
wrong-payload basic gate pass: {best["wrong_payload_basic_gate_pass"]}
overall selective gate pass: {best["overall_selective_gate_pass"]}
```

Passing grids: `{len(passing)}` / `{len(rows)}`.
Wrong-control gate failures: `{len(wrong_control_failures)}`.

This is teacher-forced scoring only. It is not a natural-output positive result
and does not by itself support paper-facing claims.
"""
    report_path = output_dir / "review.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {report_path}")
    report_path.write_text(report, encoding="utf-8")
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review after-867621 controller feasibility-envelope grid summaries.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-grid-count", type=int, default=30)
    parser.add_argument("--source-job-id", default="868016")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aggregate = review(
        args.input_dir,
        args.output_dir,
        expected_grid_count=args.expected_grid_count,
        source_job_id=str(args.source_job_id),
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
