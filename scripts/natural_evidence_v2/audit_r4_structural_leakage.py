from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (
    line_features,
    mann_whitney_auc,
    read_jsonl,
    resolve,
    write_csv_new,
    write_json_new,
    write_text_new,
)


DEFAULT_INPUT = ROOT / "results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/r3_2_generated_outputs.jsonl"
DEFAULT_OUTPUT = ROOT / "results/natural_evidence_v2/status/r4_structural_leakage_audit_20260512"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit public structural leakage in generated outputs.")
    parser.add_argument("--generated-outputs", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(resolve(args.generated_outputs))
    feature_rows = []
    by_condition: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        features = line_features(str(row.get("response_text", "")))
        condition = str(row.get("generation_condition", "unknown"))
        feature_rows.append(
            {
                "generation_id": row.get("generation_id", ""),
                "generation_condition": condition,
                "replicate_group_id": row.get("replicate_group_id", ""),
                **features,
            }
        )
        by_condition[condition].append({key: float(value) for key, value in features.items()})
    protected = by_condition.get("protected", [])
    raw = by_condition.get("raw", [])
    auc_rows = []
    for feature_name in [
        "line_count",
        "bullet_count",
        "numbered_label_count",
        "step_token_count",
        "step_label_count",
        "repeated_label_count",
        "mean_line_length",
        "line_length_variance",
        "heading_count",
        "colon_after_label_count",
        "first_token_entropy",
        "technical_literal_count",
    ]:
        auc_rows.append(
            {
                "feature": feature_name,
                "protected_vs_raw_auc": mann_whitney_auc(
                    [row[feature_name] for row in protected],
                    [row[feature_name] for row in raw],
                ),
            }
        )
    max_auc = max((float(row["protected_vs_raw_auc"]) for row in auc_rows), default=0.5)
    summary = {
        "schema_name": "natural_evidence_v2_r4_structural_leakage_audit_v1",
        "input_generated_outputs": str(args.generated_outputs),
        "row_count": len(rows),
        "condition_counts": {condition: len(items) for condition, items in sorted(by_condition.items())},
        "max_protected_vs_raw_shallow_feature_auc": max_auc,
        "hard_gate_auc_threshold": 0.60,
        "hard_gate_pass_if_new_protocol": max_auc <= 0.60,
        "note": "This is an audit of R3.2 853524 artifacts; R4 prompts must remove Step-label dependence.",
        "slurm_submitted": False,
        "paper_claim_allowed": False,
    }
    output_dir = resolve(args.output_dir)
    write_csv_new(
        output_dir / "shallow_feature_classifier.csv",
        auc_rows,
        ["feature", "protected_vs_raw_auc"],
    )
    write_json_new(output_dir / "structural_leakage_summary.json", summary)
    report = "\n".join(
        [
            "# R4 structural leakage audit",
            "",
            f"Input: `{args.generated_outputs}`",
            "",
            f"Rows: `{len(rows)}`",
            "",
            f"Max protected-vs-raw shallow feature AUC: `{max_auc:.3f}`",
            "",
            "This is diagnostic evidence from the failed R3.2 Step-label protocol. "
            "It does not authorize R4 generation or reruns.",
            "",
        ]
    )
    write_text_new(output_dir / "structural_leakage_report.md", report)
    write_csv_new(
        output_dir / "structural_features_by_output.csv",
        feature_rows,
        [
            "generation_id",
            "generation_condition",
            "replicate_group_id",
            "line_count",
            "bullet_count",
            "numbered_label_count",
            "step_token_count",
            "step_label_count",
            "repeated_label_count",
            "mean_line_length",
            "line_length_variance",
            "heading_count",
            "colon_after_label_count",
            "first_token_entropy",
            "technical_literal_count",
        ],
    )
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
