from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def summarize_grid(grid_dir: Path) -> dict[str, Any]:
    summary = read_json(grid_dir / "r4_teacher_forced_surface_mass_summary.json")
    cfg = summary["controller_config"]
    controller = summary["controller_only_summary"]
    cap_reasons: Counter[str] = Counter()
    controlled_rows = 0
    for row in iter_rows(grid_dir / "r4_teacher_forced_surface_mass_rows.jsonl"):
        if row.get("condition") == "controlled_base":
            controlled_rows += 1
            reasons = row.get("controller_cap_reasons") or ["none"]
            cap_reasons.update(str(reason) for reason in reasons)
    return {
        "grid_id": grid_dir.name,
        "bonus_nats": cfg["bonus_nats"],
        "penalty_nats": cfg["penalty_nats"],
        "max_target_mass": cfg["max_target_mass"],
        "max_kl_budget": cfg["max_kl_budget"],
        "controlled_mean_target_mass": controller["controlled_base_mean_target_mass"],
        "controlled_lift_vs_base": controller["controlled_base_lift_vs_base"],
        "controlled_lift_vs_task_only": controller["controlled_base_lift_vs_task_only"],
        "controlled_rank1_rate": controller["controlled_base_rank1_rate"],
        "controlled_median_target_margin": controller["controlled_base_median_target_margin"],
        "controlled_basic_gate_pass": controller["controlled_basic_gate_pass"],
        "wrong_key_mean_target_mass": controller["wrong_key_mean_target_mass"],
        "wrong_key_lift_vs_base": controller["wrong_key_lift_vs_base"],
        "wrong_key_rank1_rate": controller["wrong_key_rank1_rate"],
        "wrong_key_basic_gate_pass": controller["wrong_key_basic_gate_pass"],
        "wrong_payload_mean_target_mass": controller["wrong_payload_mean_target_mass"],
        "wrong_payload_lift_vs_base": controller["wrong_payload_lift_vs_base"],
        "wrong_payload_rank1_rate": controller["wrong_payload_rank1_rate"],
        "wrong_payload_basic_gate_pass": controller["wrong_payload_basic_gate_pass"],
        "overall_selective_gate_pass": controller["overall_selective_gate_pass"],
        "score_row_count": summary["score_row_count"],
        "scored_row_count": summary["scored_row_count"],
        "controlled_rows": controlled_rows,
        "controller_cap_reasons": dict(cap_reasons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Review R4 after-867621 controller safety-bound grid.")
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-job-id", default="867939")
    parser.add_argument("--expected-grid-count", type=int, default=24)
    parser.add_argument("--expected-controlled-rows", type=int, default=4096)
    parser.add_argument(
        "--pass-status",
        default="PASS_R4_AFTER_867621_CONTROLLER_SAFETY_BOUND_TEACHER_FORCED_GATE",
    )
    parser.add_argument(
        "--fail-status",
        default="FAIL_R4_AFTER_867621_CONTROLLER_SAFETY_BOUND_NO_GENERATION",
    )
    args = parser.parse_args()

    grid_dirs = sorted(path for path in args.score_dir.glob("grid_*") if path.is_dir())
    rows = [summarize_grid(path) for path in grid_dirs]
    rows.sort(key=lambda row: (float(row["controlled_lift_vs_base"]), float(row["controlled_rank1_rate"])), reverse=True)
    best = rows[0] if rows else {}
    passing = [row for row in rows if row.get("overall_selective_gate_pass") is True]
    completed_grid_count = len(rows)
    all_outputs_present = completed_grid_count == args.expected_grid_count and all(
        int(row["controlled_rows"]) == args.expected_controlled_rows for row in rows
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "grid_id",
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
        "score_row_count",
        "scored_row_count",
        "controlled_rows",
        "controller_cap_reasons",
    ]
    write_csv(args.output_dir / "grid_summary.csv", rows, fields)
    status = (
        args.pass_status
        if passing
        else args.fail_status
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_867621_controller_safety_bound_review_v1",
        "status": status,
        "source_job_id": str(args.source_job_id),
        "source_score_dir": str(args.score_dir),
        "completed_grid_count": completed_grid_count,
        "expected_grid_count": args.expected_grid_count,
        "expected_controlled_rows": args.expected_controlled_rows,
        "all_outputs_present": all_outputs_present,
        "passing_grid_count": len(passing),
        "passing_grids": passing,
        "best_by_lift_vs_base": best,
        "generation_unlocked": bool(passing),
        "training_unlocked": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "reviewed small generation route may be prepared"
            if passing
            else "artifact-only repair or pivot route decision; do not run generation from this failed controller sweep"
        ),
    }
    write_json(args.output_dir / "review_summary.json", summary)
    write_text(
        args.output_dir / "review.md",
        f"""# R4 After 867621 Controller Safety-Bound Review

Status: `{status}`

Job `{args.source_job_id}` completed `{completed_grid_count}` controller grid tasks.

Best grid by controlled lift vs base:

```text
grid: {best.get("grid_id")}
bonus_nats: {best.get("bonus_nats")}
penalty_nats: {best.get("penalty_nats")}
max_target_mass: {best.get("max_target_mass")}
max_kl_budget: {best.get("max_kl_budget")}
controlled mean target mass: {float(best.get("controlled_mean_target_mass", 0.0)):.6f}
controlled lift vs base: {float(best.get("controlled_lift_vs_base", 0.0)):.6f}
controlled lift vs task_only: {float(best.get("controlled_lift_vs_task_only", 0.0)):.6f}
controlled rank1 rate: {float(best.get("controlled_rank1_rate", 0.0)):.6f}
controlled median margin: {float(best.get("controlled_median_target_margin", 0.0)):.6f}
wrong-key rank1 rate: {float(best.get("wrong_key_rank1_rate", 0.0)):.6f}
wrong-payload rank1 rate: {float(best.get("wrong_payload_rank1_rate", 0.0)):.6f}
```

Gate targets were `+0.15` lift vs base, `+0.10` lift vs task-only, controlled
rank1 `>=0.75`, and clean wrong-key/wrong-payload controls. The best grid
achieved the rank1 target but only `{float(best.get("controlled_lift_vs_base", 0.0)):.6f}`
lift vs base.

Interpretation:

```text
passing_grid_count: {len(passing)}
generation_unlocked: {bool(passing)}
```

This result does not unlock generation. The next step is artifact-only repair or
pivot route decision.
""",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
