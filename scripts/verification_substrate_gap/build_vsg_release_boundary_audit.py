#!/usr/bin/env python3
"""Build an artifact-only public-release boundary audit for VSG supplements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "reproducibility_release_inventory_20260601"
    / "release_inventory.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "reproducibility_release_boundary_audit_20260601"
)

OUTPUT_FIELDS = [
    "artifact_group",
    "release_role",
    "path",
    "exists",
    "tracked_by_git",
    "release_status",
    "requires_anonymization_review",
    "private_path_hit",
    "secret_term_hit",
    "boundary_decision",
    "public_supplement_action",
    "pre_release_review_required",
    "release_blocker",
    "reason",
]


def truthy(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def classify(row: dict[str, str]) -> dict[str, Any]:
    exists = truthy(row.get("exists", "False"))
    tracked = truthy(row.get("tracked_by_git", "False"))
    needs_review = truthy(row.get("requires_anonymization_review", "False"))
    private_hit = truthy(row.get("private_path_hit", "False"))
    secret_hit = truthy(row.get("secret_term_hit", "False"))
    group = row.get("artifact_group", "")
    release_status = row.get("release_status", "")

    if not exists:
        decision = "missing_or_out_of_scope"
        action = "resolve_missing_file_or_exclude_from_supplement"
        review_required = True
        blocker = True
        reason = "inventory row points to a file that is not present"
    elif group == "state_and_scope_records":
        decision = "exclude_from_public_supplement"
        action = "keep_as_internal_handoff_record_not_public_reproducibility_artifact"
        review_required = False
        blocker = False
        reason = "state/handoff records are useful for project audit but not required in the public supplement"
    elif private_hit:
        decision = "redact_or_summarize_before_release"
        action = "scrub_private_paths_or_release_a_derived_summary"
        review_required = True
        blocker = True
        reason = "file contains private local or cluster path markers"
    elif secret_hit:
        decision = "security_review_before_release"
        action = "review_key_hmac_fields_and_redact_if_needed"
        review_required = True
        blocker = True
        reason = "file contains key/HMAC-related field names"
    elif release_status == "needs_anonymization_review" or needs_review:
        decision = "scope_review_before_release"
        action = "confirm anonymization_scope_or_replace_with_derived_summary"
        review_required = True
        blocker = True
        reason = "inventory marks this row as requiring anonymization or scope review"
    elif not tracked:
        decision = "stage_or_copy_to_supplement_before_release"
        action = "commit_file_or_copy_into_a_reviewed_supplement_bundle"
        review_required = True
        blocker = True
        reason = "file exists but is not tracked by the selected git scopes"
    else:
        decision = "ready_for_reviewed_public_supplement"
        action = "include_after_final_license_and_scope_review"
        review_required = False
        blocker = False
        reason = "file is present, tracked, and has no inventory privacy/security flags"

    return {
        **row,
        "boundary_decision": decision,
        "public_supplement_action": action,
        "pre_release_review_required": review_required,
        "release_blocker": blocker,
        "reason": reason,
    }


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def summarize(rows: list[dict[str, Any]], output_dir: Path, inventory: Path) -> dict[str, Any]:
    decision_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for row in rows:
        decision_counts[row["boundary_decision"]] = decision_counts.get(row["boundary_decision"], 0) + 1
        action_counts[row["public_supplement_action"]] = action_counts.get(row["public_supplement_action"], 0) + 1
        group_counts[row["artifact_group"]] = group_counts.get(row["artifact_group"], 0) + 1
    release_blockers = [row for row in rows if truthy(row["release_blocker"])]
    review_rows = [row for row in rows if truthy(row["pre_release_review_required"])]
    ready_rows = [row for row in rows if row["boundary_decision"] == "ready_for_reviewed_public_supplement"]
    excluded_rows = [row for row in rows if row["boundary_decision"] == "exclude_from_public_supplement"]
    return {
        "status": "PASS_VSG_RELEASE_BOUNDARY_AUDIT_RECORDED_REVIEW_REQUIRED",
        "schema_name": "verification_substrate_gap_release_boundary_audit_v1",
        "source_inventory": display_path(inventory),
        "output_dir": display_path(output_dir),
        "row_count": len(rows),
        "ready_for_reviewed_public_supplement_count": len(ready_rows),
        "excluded_from_public_supplement_count": len(excluded_rows),
        "pre_release_review_required_count": len(review_rows),
        "release_blocker_count": len(release_blockers),
        "release_ready_now": len(release_blockers) == 0,
        "decision_counts": dict(sorted(decision_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "release_blocker_paths": [row["path"] for row in release_blockers],
        "excluded_paths": [row["path"] for row in excluded_rows],
        "public_supplement_publication_performed": False,
        "new_slurm_started": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "allowlist_enabled": False,
        "public_text_only_verification_claimed": False,
        "ownership_proof_claimed": False,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# VSG Release Boundary Audit",
        "",
        "This artifact-only audit converts the reproducibility release inventory",
        "into a public-supplement boundary decision table. It does not publish",
        "files, copy raw artifacts, start compute, or expand the manuscript claim",
        "boundary.",
        "",
        f"Status: `{summary['status']}`",
        f"Rows: `{summary['row_count']}`",
        f"Ready for reviewed public supplement: `{summary['ready_for_reviewed_public_supplement_count']}`",
        f"Excluded from public supplement: `{summary['excluded_from_public_supplement_count']}`",
        f"Pre-release review required: `{summary['pre_release_review_required_count']}`",
        f"Release blockers: `{summary['release_blocker_count']}`",
        f"Release-ready now: `{summary['release_ready_now']}`",
        "",
        "## Boundary Decisions",
        "",
        "| Decision | Rows |",
        "| --- | ---: |",
    ]
    for decision, count in summary["decision_counts"].items():
        lines.append(f"| {decision} | {count} |")
    lines.extend(
        [
            "",
            "## Required Before Public Supplement Release",
            "",
            "- Redact or summarize files with private path markers.",
            "- Review files with key/HMAC-related field names.",
            "- Decide whether untracked candidate files are committed, copied into a reviewed supplement bundle, or excluded.",
            "- Keep internal handoff/state records outside the public supplement unless explicitly approved.",
            "",
            "## Claim Scope",
            "",
            "This audit preserves the current VSG claim boundary: trace-bound",
            "first-divergence results remain provider-side diagnostics; public",
            "final-text predicates remain observability/spoofing diagnostics;",
            "source-mismatch accepts are not protected success and not codeword",
            "recovery.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(inventory: Path, output_dir: Path) -> dict[str, Any]:
    rows = [classify(row) for row in read_inventory(inventory)]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "release_boundary_decisions.csv"
    summary_path = output_dir / "release_boundary_summary.json"
    report_path = output_dir / "release_boundary_report.md"
    manifest_path = output_dir / "release_boundary_manifest.json"
    write_csv(csv_path, rows)
    summary = summarize(rows, output_dir, inventory)
    write_json(summary_path, summary)
    write_report(report_path, summary)
    manifest = {
        "status": summary["status"],
        "schema_name": "verification_substrate_gap_release_boundary_manifest_v1",
        "files": [
            {
                "path": display_path(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in [csv_path, summary_path, report_path]
        ],
        "manifest_self_hash_excluded": True,
    }
    write_json(manifest_path, manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = build(Path(args.inventory), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
