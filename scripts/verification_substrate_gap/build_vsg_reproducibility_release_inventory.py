#!/usr/bin/env python3
"""Build a reproducibility release inventory for the VSG manuscript snapshot.

The inventory is artifact-only. It does not copy or publish files, start new
compute, run generation, or change claim scope. Its purpose is to make the
submission/release blocker concrete: which files are release candidates, which
need anonymization or path scrubbing, and which are internal/raw artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "verification_substrate_gap" / "reproducibility_release_inventory_20260601"

INVENTORY_FIELDS = [
    "artifact_group",
    "release_role",
    "path",
    "exists",
    "tracked_by_git",
    "git_tracking_scope",
    "bytes",
    "sha256",
    "release_status",
    "requires_anonymization_review",
    "private_path_hit",
    "secret_term_hit",
    "notes",
]

PRIVATE_PATH_MARKERS = [
    "/Users/",
    "/hpcstor",
    "scratch01",
    "/home/",
]

SECRET_TERM_MARKERS = [
    "secret_key",
    "binding_hmac",
    "key_id_not_secret_key",
]


def planned_artifacts() -> list[dict[str, str]]:
    """Return release-inventory rows before filesystem inspection."""
    rows: list[dict[str, str]] = []

    def add(group: str, role: str, path: str, status: str, notes: str = "") -> None:
        rows.append(
            {
                "artifact_group": group,
                "release_role": role,
                "path": path,
                "release_status": status,
                "notes": notes,
            }
        )

    manuscript = "manuscripts/69db2644566dcc36c9da320e"
    for rel in [
        "main.tex",
        "section_01_introduction.tex",
        "section_02_related_work.tex",
        "section_03_problem_setup.tex",
        "section_04_tokenizer_alignment.tex",
        "section_05_bucket_level_injection.tex",
        "section_06_deterministic_verification.tex",
        "section_07_experiments.tex",
        "section_08_discussion_limitations.tex",
        "section_09_conclusion.tex",
        "appendix/proofs.tex",
        "appendix/formal_substrate_gap.tex",
        "appendix/attack_examples.tex",
        "appendix/extended_related_work.tex",
        "appendix/reproducibility.tex",
        "appendix/asset_licenses.tex",
        "appendix/reproducibility_commands.tex",
        "references.bib",
        "neurips_2026.sty",
    ]:
        add("manuscript_source", "submission_source", f"{manuscript}/{rel}", "release_candidate")

    for idx in range(1, 6):
        figure_name = [
            "figure_1_verification_substrate_map.png",
            "figure_2_first_divergence_diagnostic.png",
            "figure_3_controllability_vs_observability.png",
            "figure_4_public_predicate_attack_ladder.png",
            "figure_5_ownership_scenario_heatmap.png",
        ][idx - 1]
        add("manuscript_figures", "rendered_figure", f"{manuscript}/figures/{figure_name}", "release_candidate")

    for rel in [
        "trace_bound_accepts.csv",
        "public_text_verifier_baselines.csv",
        "template_leakage_summary.csv",
        "attack_ladder_summary.csv",
        "ownership_scenario_heatmap.csv",
        "claim_ledger.csv",
        "figure_data_summary.json",
        "figure_data_manifest.json",
    ]:
        add(
            "figure_data",
            "plot_input_or_claim_table",
            f"results/verification_substrate_gap/paper_figure_data_20260530/{rel}",
            "release_candidate",
        )

    for rel in [
        "combined_blocks.csv",
        "qwen_blocks.csv",
        "llama_blocks.csv",
        "corpus_manifest.json",
    ]:
        add(
            "trace_bound_corpus_summary",
            "summary_only_not_raw_trace",
            f"results/verification_substrate_gap/corpora/trace_bound_controllability/{rel}",
            "needs_anonymization_review",
            "Summary rows can contain source path fields; scrub private paths before public release.",
        )

    for rel in [
        "public_text_verifier_results.csv",
        "public_text_verifier_block_scores.csv",
        "public_text_verifier_summary.json",
        "public_text_verifier_report.md",
    ]:
        add(
            "public_text_verifier_baselines",
            "predicate_summary",
            f"results/verification_substrate_gap/public_text_verifier_remote_20260529/{rel}",
            "release_candidate",
        )

    for rel in [
        "surrogate_guided_rewrite_curve.csv",
        "surrogate_guided_transform_summary.csv",
        "surrogate_guided_rewrite_examples.csv",
        "surrogate_guided_rewrite_summary.json",
        "surrogate_guided_rewrite_report.md",
    ]:
        add(
            "public_predicate_attack_ladder",
            "source_mismatch_attack_summary",
            f"results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/{rel}",
            "release_candidate_with_caveat",
            "Examples are source-mismatch spoofing rows; not protected success.",
        )

    for rel in [
        "attack_naturalness_proxy_rows.csv",
        "attack_naturalness_proxy_by_group.csv",
        "attack_naturalness_proxy_summary.json",
        "attack_naturalness_proxy_report.md",
    ]:
        add(
            "attack_naturalness_proxy_audit",
            "readability_proxy_summary",
            f"results/verification_substrate_gap/public_predicate_attack_naturalness_audit_20260601/{rel}",
            "release_candidate",
        )

    for rel in [
        "public_text_verifier_results.csv",
        "public_text_verifier_block_scores.csv",
        "public_text_verifier_summary.json",
        "public_text_verifier_report.md",
    ]:
        add(
            "stronger_public_predicate_local_pilot",
            "local_non_claim_pilot_summary",
            f"results/verification_substrate_gap/public_text_verifier_stronger_local_pilot_20260601/{rel}",
            "release_candidate_with_scope_note",
            "Local pilot only; not adopted locked evidence.",
        )

    for rel in [
        "substrate_gap_matrix.csv",
        "substrate_gap_matrix_summary.json",
        "substrate_gap_matrix.md",
    ]:
        add("substrate_matrix", "taxonomy_summary", f"results/verification_substrate_gap/substrate_gap_matrix/{rel}", "release_candidate")

    for rel in [
        "ownership_scenario_heatmap.csv",
        "figure_data_summary.json",
    ]:
        add(
            "ownership_stress_test",
            "scenario_summary",
            f"results/verification_substrate_gap/paper_figure_data_20260530/{rel}",
            "release_candidate",
        )

    for rel in [
        "lint_claim_scope.py",
        "evaluate_public_text_verifier.py",
        "audit_public_predicate_attack_naturalness.py",
        "verify_vsg_expert_review_packet.py",
        "audit_vsg_expert_handoff.py",
        "build_vsg_reproducibility_release_inventory.py",
    ]:
        add("reproducibility_code", "script", f"scripts/verification_substrate_gap/{rel}", "release_candidate")

    for rel in [
        "public_text_verifier_baselines.yaml",
        "public_text_verifier_stronger_local_pilot.yaml",
        "text_only_observability.yaml",
        "template_leakage_audit.yaml",
        "surrogate_guided_rewrite_spoofing.yaml",
    ]:
        add("reproducibility_config", "config", f"configs/verification_substrate_gap/{rel}", "release_candidate")

    for rel in [
        "test_claim_scope_linter.py",
        "test_vsg_expert_packet_verifiers.py",
        "test_vsg_manuscript_prose_hardening.py",
        "test_public_text_verifier_stronger_baselines.py",
        "test_public_predicate_attack_naturalness_audit.py",
    ]:
        add("reproducibility_tests", "test", f"tests/verification_substrate_gap/{rel}", "release_candidate")

    for rel in [
        "VSG_CURRENT_HANDOFF_STATE_20260601.md",
        "VSG_ATTACK_NATURALNESS_PROXY_AUDIT_20260601.md",
        "VSG_PUBLIC_TEXT_STRONGER_BASELINE_LOCAL_PILOT_20260601.md",
        "VSG_EXPERT_REPLY_HARDENING_DECOMPOSITION_20260601.md",
    ]:
        add("state_and_scope_records", "scope_record", f"results/verification_substrate_gap/{rel}", "release_candidate_with_caveat")

    return rows


def git_tracked_paths(root: Path) -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return set(completed.stdout.splitlines())


def manuscript_tracked_paths(root: Path) -> set[str]:
    manuscript = root / "manuscripts" / "69db2644566dcc36c9da320e"
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=manuscript,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    prefix = manuscript.relative_to(root).as_posix()
    return {f"{prefix}/{path}" for path in completed.stdout.splitlines()}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def display_path(path: Path, root: Path = ROOT) -> str:
    """Return a repo-relative path when possible, else an absolute path."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def text_flags(path: Path) -> tuple[bool, bool]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False, False
    if path.name == "build_vsg_reproducibility_release_inventory.py":
        text = strip_inventory_marker_definitions(text)
    private_hit = any(marker in text for marker in PRIVATE_PATH_MARKERS)
    secret_hit = any(marker in text for marker in SECRET_TERM_MARKERS)
    return private_hit, secret_hit


def strip_inventory_marker_definitions(text: str) -> str:
    """Remove this inventory script's marker lists before self-scanning."""
    stripped: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line.startswith("PRIVATE_PATH_MARKERS = [") or line.startswith("SECRET_TERM_MARKERS = ["):
            skipping = True
            continue
        if skipping:
            if line.strip() == "]":
                skipping = False
            continue
        stripped.append(line)
    return "\n".join(stripped)


def inspect_rows(rows: list[dict[str, str]], root: Path) -> list[dict[str, Any]]:
    root_tracked = git_tracked_paths(root)
    manuscript_tracked = manuscript_tracked_paths(root)
    inspected: list[dict[str, Any]] = []
    for row in rows:
        rel = row["path"]
        path = root / rel
        exists = path.is_file()
        if rel in root_tracked:
            tracked_by_git = True
            tracking_scope = "root_git"
        elif rel in manuscript_tracked:
            tracked_by_git = True
            tracking_scope = "manuscript_git"
        else:
            tracked_by_git = False
            tracking_scope = "untracked"
        private_hit = False
        secret_hit = False
        if exists:
            private_hit, secret_hit = text_flags(path)
        requires_review = row["release_status"] in {
            "needs_anonymization_review",
            "release_candidate_with_caveat",
            "release_candidate_with_scope_note",
        } or private_hit or secret_hit
        inspected.append(
            {
                **row,
                "exists": exists,
                "tracked_by_git": tracked_by_git,
                "git_tracking_scope": tracking_scope,
                "bytes": path.stat().st_size if exists else "",
                "sha256": sha256_file(path) if exists else "",
                "requires_anonymization_review": requires_review,
                "private_path_hit": private_hit,
                "secret_term_hit": secret_hit,
            }
        )
    return inspected


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(output_dir: Path) -> dict[str, Any]:
    rows = inspect_rows(planned_artifacts(), ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "release_inventory.csv"
    summary_path = output_dir / "release_inventory_summary.json"
    report_path = output_dir / "release_inventory_report.md"
    manifest_path = output_dir / "release_inventory_manifest.json"
    write_csv(csv_path, rows, INVENTORY_FIELDS)
    missing = [row for row in rows if not row["exists"]]
    private_hits = [row for row in rows if row["private_path_hit"]]
    secret_hits = [row for row in rows if row["secret_term_hit"]]
    untracked = [row for row in rows if row["exists"] and not row["tracked_by_git"]]
    review_rows = [row for row in rows if row["requires_anonymization_review"]]
    summary = {
        "status": "PASS_VSG_REPRODUCIBILITY_RELEASE_INVENTORY_RECORDED_REVIEW_REQUIRED",
        "schema_name": "verification_substrate_gap_reproducibility_release_inventory_v1",
        "output_dir": display_path(output_dir),
        "row_count": len(rows),
        "existing_file_count": len(rows) - len(missing),
        "missing_file_count": len(missing),
        "untracked_existing_file_count": len(untracked),
        "requires_anonymization_review_count": len(review_rows),
        "private_path_hit_count": len(private_hits),
        "secret_term_hit_count": len(secret_hits),
        "release_ready_without_review": len(missing) == 0 and len(private_hits) == 0 and len(secret_hits) == 0,
        "new_slurm_started": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "public_text_only_verification_claimed": False,
        "ownership_proof_claimed": False,
        "missing_paths": [row["path"] for row in missing],
        "untracked_existing_paths": [row["path"] for row in untracked],
        "private_path_hit_paths": [row["path"] for row in private_hits],
        "secret_term_hit_paths": [row["path"] for row in secret_hits],
        "inventory_csv": display_path(csv_path),
    }
    write_json(summary_path, summary)
    write_report(report_path, summary, rows)
    manifest = {
        "status": summary["status"],
        "schema_name": "verification_substrate_gap_reproducibility_release_inventory_manifest_v1",
        "files": [
            {
                "path": display_path(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in [csv_path, summary_path, report_path]
        ],
    }
    write_json(manifest_path, manifest)
    return summary


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    group_counts: dict[str, int] = {}
    for row in rows:
        group_counts[row["artifact_group"]] = group_counts.get(row["artifact_group"], 0) + 1
    lines = [
        "# VSG Reproducibility Release Inventory",
        "",
        "This artifact-only inventory identifies candidate files for a future",
        "supplemental release. It does not publish the files and does not create",
        "a public text-only verification or ownership-proof claim.",
        "",
        f"Status: `{summary['status']}`",
        f"Rows: `{summary['row_count']}`",
        f"Existing files: `{summary['existing_file_count']}`",
        f"Missing files: `{summary['missing_file_count']}`",
        f"Existing files not tracked by selected git scopes: `{summary['untracked_existing_file_count']}`",
        f"Rows requiring anonymization/scope review: `{summary['requires_anonymization_review_count']}`",
        f"Private path hits: `{summary['private_path_hit_count']}`",
        f"Secret-term hits: `{summary['secret_term_hit_count']}`",
        f"Release-ready without review: `{summary['release_ready_without_review']}`",
        "",
        "## Groups",
        "",
        "| Group | Rows |",
        "| --- | ---: |",
    ]
    for group, count in sorted(group_counts.items()):
        lines.append(f"| {group} | {count} |")
    lines.extend(
        [
            "",
            "## Required Follow-Up Before Public Release",
            "",
            "- Resolve missing files or mark them intentionally out of scope.",
            "- Scrub private cluster/local paths from release candidates.",
            "- Review any files containing key/HMAC-related field names before release.",
            "- Decide whether untracked-but-existing files should be committed, copied into a release bundle, or excluded.",
            "",
            "This inventory is compatible with the current VSG claim boundary: trace-bound",
            "results remain provider-side diagnostics; public final-text predicates remain",
            "observability/spoofing diagnostics; source-mismatch accepts are not protected",
            "success and not codeword recovery.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
