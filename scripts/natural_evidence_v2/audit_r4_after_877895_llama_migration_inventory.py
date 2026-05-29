#!/usr/bin/env python3
"""Inventory existing Llama artifacts before R4 second-family planning.

This is artifact-only. It does not submit Slurm, score a model, generate text,
train, run sanitizer/FAR, or make claims.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_877895_llama_migration_inventory_20260526"
)
SEARCH_ROOTS = (
    "configs",
    "docs",
    "scripts",
    "results/natural_evidence_v1/status",
    "results/natural_evidence_v2/status",
)


@dataclass(frozen=True)
class InventoryRow:
    path: str
    kind: str
    canonical_for_r4: bool
    use_policy: str
    reason: str


def classify(path: Path) -> InventoryRow:
    rel = path.relative_to(ROOT).as_posix()
    lower = rel.lower()
    suffix = path.suffix.lower()

    if "__pycache__" in lower:
        return InventoryRow(rel, "cache", False, "ignore", "Python bytecode cache.")
    if "baseline" in lower or "perinucleus" in lower:
        return InventoryRow(
            rel,
            "baseline_or_comparison",
            False,
            "baseline_reference_only",
            "Baseline/comparison artifact, not an R4 first-token event route.",
        )
    if "llama_v2_wp5" in lower or "llama_v2_wp6" in lower:
        return InventoryRow(
            rel,
            "old_wp5_wp6_slurm_or_plan",
            False,
            "do_not_submit_as_canonical_r4",
            "Old Step-label/WP5/WP6 route; current R4 route needs tokenizer-native first-token event planning.",
        )
    if "llama_v2_migration" in lower or rel.endswith("LLAMA_V2_MIGRATION_PLAN_20260510.md"):
        return InventoryRow(
            rel,
            "old_llama_migration_plan",
            False,
            "historical_context_only",
            "Plan predates the R4 provider-side first-token event route.",
        )
    if "build_llama_v2_bucket_bank" in lower or "patch_train_for_llama" in lower:
        return InventoryRow(
            rel,
            "old_llama_helper_script",
            False,
            "debug_hint_only",
            "Helper targets old bucket/Step-label pipeline and must not define the R4 route.",
        )
    if lower.endswith("configs/model/llama3_1_8b_instruct.yaml"):
        return InventoryRow(
            rel,
            "model_config",
            True,
            "candidate_model_reference",
            "Model identity/config reference may be reused after R4 route validation.",
        )
    if "catalog" in lower and "llama" in lower:
        return InventoryRow(
            rel,
            "catalog_or_freeze",
            False,
            "historical_reference_only",
            "Catalog/freeze artifact is not a tokenizer-native R4 route decision.",
        )
    if suffix in {".md", ".json", ".yaml", ".yml", ".py", ".sbatch"}:
        return InventoryRow(
            rel,
            "llama_related_artifact",
            False,
            "review_before_reuse",
            "Llama-related artifact requires explicit R4 compatibility review before reuse.",
        )
    return InventoryRow(rel, "other", False, "ignore", "Not directly useful for R4 route planning.")


def discover() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    seen: set[str] = set()
    for root_name in SEARCH_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel_lower = path.relative_to(ROOT).as_posix().lower()
            if "llama" not in rel_lower:
                continue
            if rel_lower in seen:
                continue
            seen.add(rel_lower)
            rows.append(classify(path))
    return sorted(rows, key=lambda row: row.path)


def write_outputs(rows: list[InventoryRow], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    canonical = [row for row in row_dicts if row["canonical_for_r4"]]
    noncanonical = [row for row in row_dicts if not row["canonical_for_r4"]]
    summary = {
        "schema_name": "r4_after_877895_llama_migration_inventory_v1",
        "status": "PASS_R4_AFTER_877895_LLAMA_MIGRATION_INVENTORY_ARTIFACT_ONLY_NO_SUBMIT",
        "artifact_only": True,
        "slurm_submitted": False,
        "allowlist_enabled": False,
        "model_scoring_started": False,
        "generation_started": False,
        "training_started": False,
        "total_llama_related_files": len(rows),
        "canonical_candidate_files": len(canonical),
        "noncanonical_or_review_before_reuse_files": len(noncanonical),
        "canonical_candidates": canonical,
        "next_allowed_action": (
            "Prepare an R4-specific second-family tokenizer-native route plan; "
            "do not submit old WP5/WP6 Llama wrappers as canonical R4 jobs."
        ),
    }
    (output_dir / "llama_inventory_rows.json").write_text(
        json.dumps(row_dicts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "llama_inventory_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    table_rows = "\n".join(
        "| {path} | {kind} | {canonical_for_r4} | {use_policy} | {reason} |".format(
            **row
        )
        for row in row_dicts
    )
    report = f"""# R4 After-877895 Llama Migration Inventory

Status: `{summary["status"]}`

This is artifact-only. It does not submit Slurm, enable allowlist, score a
model, generate text, train, run sanitizer/FAR, or make claims.

## Summary

- Llama-related files found: `{len(rows)}`
- R4 canonical candidate files: `{len(canonical)}`
- Noncanonical or review-before-reuse files: `{len(noncanonical)}`

The old Llama WP5/WP6 route and `LLAMA_V2_MIGRATION_PLAN_20260510.md` predate
the R4 provider-side first-token event route. They are historical context/debug
hints only. The next route must build a tokenizer-native second-family plan for
the current first-token event channel rather than submitting old Step-label
wrappers.

## Inventory

| Path | Kind | Canonical for R4 | Use policy | Reason |
| --- | --- | --- | --- | --- |
{table_rows}
"""
    (output_dir / "llama_inventory_report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = write_outputs(discover(), args.output_dir)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
