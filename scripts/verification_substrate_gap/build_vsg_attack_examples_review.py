#!/usr/bin/env python3
"""Build a small VSG attack-example review table from existing artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


STATUS = "PASS_VSG_ATTACK_EXAMPLES_REVIEW_BUILT_ARTIFACT_ONLY_NO_NEW_CLAIMS"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_excerpt(value: str, limit: int = 150) -> str:
    text = " ".join(value.replace("\n", " ").split())
    return text[: limit - 3] + "..." if len(text) > limit else text


def pick_guided(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    wanted = [("qwen", "raw"), ("qwen", "task_only"), ("llama", "raw")]
    out = []
    for model, arm in wanted:
        matches = [
            row
            for row in rows
            if row["model_family"] == model
            and row["candidate_arm"] == arm
            and row["rank_index"] == "1"
        ]
        if matches:
            out.append(matches[0])
    return out


def build(guided_path: Path, output_dir: Path) -> dict:
    guided = pick_guided(read_csv(guided_path))
    rows = []
    for row in guided:
        rows.append(
            {
                "source_id": row["source_id"],
                "model_family": row["model_family"],
                "candidate_arm": row["candidate_arm"],
                "attack_mode": row["attack_mode"],
                "rank_index": row["rank_index"],
                "query_cost_to_here": row["public_predicate_query_cost_to_here"],
                "original_score": f"{float(row['original_score']):.3f}",
                "rewrite_score": f"{float(row['best_rewrite_score']):.3f}",
                "threshold": f"{float(row['threshold']):.3f}",
                "transform_id": row["best_transform_id"],
                "original_excerpt": clean_excerpt(row["original_excerpt"]),
                "rewrite_excerpt": clean_excerpt(row["best_rewrite_excerpt"]),
                "naturalness_caveat": "report_only_no_semantic_naturalness_gate",
                "claim_scope": row["claim_scope"],
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "attack_examples_review.csv"
    write_csv(csv_path, rows)
    md_lines = [
        "# VSG Attack Examples Review",
        "",
        "This table is artifact-only. Rows are source-mismatch public-predicate accepts, not protected success and not codeword recovery.",
        "",
        "| model | arm | attack | score -> rewrite | threshold | naturalness caveat |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| {model_family} | {candidate_arm} | {attack_mode} | {original_score} -> {rewrite_score} | {threshold} | {naturalness_caveat} |".format(
                **row
            )
        )
    md_lines.append("")
    md_path = output_dir / "attack_examples_review.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    summary = {
        "schema_name": "vsg_attack_examples_review_v1",
        "status": STATUS,
        "guided_examples_source": str(guided_path),
        "output_dir": str(output_dir),
        "review_rows": len(rows),
        "naturalness_claimed": False,
        "semantic_naturalness_gate_applied": False,
        "source_mismatch_rows_only": True,
        "protected_success_claimed": False,
        "codeword_recovery_claimed": False,
        "new_compute_started": False,
        "slurm_submitted": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
    }
    summary_path = output_dir / "attack_examples_review_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_name": "vsg_attack_examples_review_manifest_v1",
        "status": STATUS,
        "files": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in [csv_path, md_path, summary_path]
        ],
    }
    manifest_path = output_dir / "attack_examples_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--guided-examples",
        type=Path,
        default=Path("results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/surrogate_guided_rewrite_examples.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/verification_substrate_gap/paper_attack_examples_20260531"),
    )
    args = parser.parse_args()
    print(json.dumps(build(args.guided_examples, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
