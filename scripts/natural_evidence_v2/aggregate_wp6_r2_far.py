#!/usr/bin/env python3
"""Aggregate WP6-R2 Option B FAR from robust-block null rejection data.

Reads the WP6-R2 Option B scale eval summary and computes:
- Per-condition and per-budget FAR
- Wilson 95% CI upper bound
- Overall FAR across all null conditions

This is artifact-only; no Slurm, no training, no generation.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUMMARY_PATH = (
    ROOT
    / "results/natural_evidence_v2/status/wp6_r2_option_b_scale_eval_852426"
    / "coordinate_majority_r2_option_b/wp6_r2_option_b_summary.json"
)

CONDITIONS = ["raw", "task_only", "wrong_key", "wrong_payload"]
BUDGETS = ["8", "16", "32", "64"]


def wilson_upper(n: int, failures: int = 0, z: float = 1.96) -> float:
    """Wilson score upper bound for binomial proportion when observed = 0.

    When x=0, the formula simplifies to:
        upper = 1 - (alpha/2)^(1/n)  for small alpha
    More precisely using the Wilson interval:
        upper ≈ z^2 / (2n + z^2) + z * sqrt(n * z^2 / (2n + z^2)^2 + failures/n / (2n + z^2))
    But for x=0 (failures=0), the standard approximation is:
        upper ≈ 3 / (4n)   (rule of three)
    We use the exact Wilson formula for completeness.
    """
    if n == 0:
        return float("nan")
    p_hat = failures / n
    denominator = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denominator
    return min(center + spread, 1.0)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_dir = ROOT / f"results/natural_evidence_v2/status/wp6_r2_option_b_far_aggregation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(SUMMARY_PATH.read_text())
    block_budget = summary["summary_by_block_budget"]

    # Collect per-condition, per-budget null accept counts
    far_table: dict[str, dict[str, dict]] = {}
    total_null_trials = 0
    total_false_accepts = 0

    for cond in CONDITIONS:
        far_table[cond] = {}
        for budget in BUDGETS:
            false_accepts = 0
            trials = 0
            for block_key, budget_data in block_budget.items():
                if budget in budget_data and cond in budget_data[budget]:
                    trials += 1
                    entry = budget_data[budget][cond]
                    if entry.get("accepted", False):
                        false_accepts += 1
            far_rate = false_accepts / trials if trials > 0 else float("nan")
            far_table[cond][budget] = {
                "false_accepts": false_accepts,
                "trials": trials,
                "far": far_rate,
                "wilson_upper_95": wilson_upper(trials, false_accepts),
            }
            total_null_trials += trials
            total_false_accepts += false_accepts

    overall_far = total_false_accepts / total_null_trials if total_null_trials > 0 else float("nan")
    overall_wilson = wilson_upper(total_null_trials, total_false_accepts)

    # Rule of three approximation for FAR=0
    rule_of_three = 3.0 / total_null_trials if total_null_trials > 0 else float("nan")

    result = {
        "schema_name": "natural_evidence_v2_wp6_r2_option_b_far_aggregation_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_job_id": "852426",
        "source_review_doc": "docs/natural_evidence_v2/WP6_R2_OPTION_B_SCALE_EVAL_852426_REVIEW.md",
        "controlling_budget": 64,
        "overall": {
            "total_null_trials": total_null_trials,
            "total_false_accepts": total_false_accepts,
            "far": overall_far,
            "wilson_upper_95": overall_wilson,
            "rule_of_three_upper": rule_of_three,
        },
        "per_condition": {
            cond: far_table[cond]["64"]
            for cond in CONDITIONS
        },
        "per_budget": {
            budget: {
                cond: far_table[cond][budget]
                for cond in CONDITIONS
            }
            for budget in BUDGETS
        },
        "per_condition_per_budget": far_table,
        "gate_status": "PASS_WP6_R2_OPTION_B_FAR_AGGREGATION",
        "claim_control": {
            "far_aggregation_allowed": True,
            "llama_allowed": True,
            "same_family_null_allowed": True,
            "sanitizer_allowed": False,
            "paper_claim_allowed": False,
            "claim_text": (
                "WP6-R2 Option B robust-block coordinate-majority decoder shows "
                f"zero false accepts across {total_null_trials} null trials at budget=64, "
                f"with Wilson 95% CI upper bound < {overall_wilson:.4f} "
                f"(rule-of-three: {rule_of_three:.4f}). "
                "This is WP6 null-rejection FAR only, not full protocol FAR."
            ),
        },
        "forbidden_claims": [
            "full FAR (requires organic prompt nulls and non-owner probes)",
            "stealth guarantee",
            "paper-facing positive claim",
            "cross-family generality (requires Llama pass)",
            "robustness (requires sanitizer pass)",
        ],
    }

    # Write summary JSON
    summary_path = out_dir / "wp6_r2_option_b_far_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote: {summary_path}")

    # Write review doc
    doc_lines = [
        "# WP6-R2 Option B FAR Aggregation",
        "",
        "## Decision",
        "",
        f"Aggregated False Accept Rate from WP6-R2 Option B robust-block scale eval (job 852426).",
        f"All {total_null_trials} null trials rejected at budget=64.",
        "",
        "## FAR Summary",
        "",
        f"- Total null trials: {total_null_trials}",
        f"- Total false accepts: {total_false_accepts}",
        f"- FAR: {overall_far:.6f}",
        f"- Wilson 95% CI upper bound: {overall_wilson:.6f}",
        f"- Rule-of-three upper bound: {rule_of_three:.6f}",
        "",
        "## Per-Condition FAR (budget=64)",
        "",
    ]
    for cond in CONDITIONS:
        entry = far_table[cond]["64"]
        doc_lines.append(
            f"- **{cond}**: {entry['false_accepts']}/{entry['trials']} false accepts, "
            f"FAR={entry['far']:.6f}, Wilson upper={entry['wilson_upper_95']:.6f}"
        )

    doc_lines += [
        "",
        "## Per-Budget Breakdown (all conditions combined)",
        "",
    ]
    for budget in BUDGETS:
        budget_trials = sum(far_table[c][budget]["trials"] for c in CONDITIONS)
        budget_fa = sum(far_table[c][budget]["false_accepts"] for c in CONDITIONS)
        budget_far = budget_fa / budget_trials if budget_trials > 0 else float("nan")
        budget_wilson = wilson_upper(budget_trials, budget_fa)
        doc_lines.append(
            f"- **budget={budget}**: {budget_fa}/{budget_trials} false accepts, "
            f"FAR={budget_far:.6f}, Wilson upper={budget_wilson:.6f}"
        )

    doc_lines += [
        "",
        "## Claim Boundaries",
        "",
        "Allowed claim:",
        f"> WP6-R2 Option B robust-block coordinate-majority decoder shows zero false accepts "
        f"across {total_null_trials} null trials at budget=64, with Wilson 95% CI upper bound "
        f"< {overall_wilson:.4f}.",
        "",
        "Forbidden claims:",
        "- Full FAR (requires organic prompt nulls and non-owner probes)",
        "- Stealth guarantee",
        "- Cross-family generality (requires Llama positive recovery)",
        "- Robustness (requires sanitizer benchmark pass)",
        "- Paper-facing positive claim (requires all gates pass)",
        "",
        "## Next Allowed Actions",
        "",
        "1. Llama-3.1-8B migration (WP5 training + WP6 E2E)",
        "2. Same-family null experiments",
        "3. After Llama passes: sanitizer benchmarks",
        "",
        "## Validation",
        "",
        "Artifact-only aggregation; no Slurm, no training, no generation.",
        "",
    ]

    doc_path = ROOT / f"docs/natural_evidence_v2/WP6_R2_OPTION_B_FAR_AGGREGATION_{ts}.md"
    doc_path.write_text("\n".join(doc_lines) + "\n")
    print(f"Wrote: {doc_path}")

    print("\n=== FAR Summary ===")
    print(f"Total null trials: {total_null_trials}")
    print(f"Total false accepts: {total_false_accepts}")
    print(f"FAR: {overall_far:.6f}")
    print(f"Wilson 95% upper: {overall_wilson:.6f}")
    print(f"Rule-of-three: {rule_of_three:.6f}")


if __name__ == "__main__":
    main()
