#!/usr/bin/env python3
"""Build a dry-run manifest for a future VSG public supplement bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_READINESS = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_readiness_audit_20260601"
    / "readiness_decisions.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_dry_run_manifest_20260601"
)

OUTPUT_FIELDS = [
    "artifact_group",
    "readiness_decision",
    "bundle_action",
    "include_in_dry_run_bundle",
    "bundle_source_path",
    "bundle_source_role",
    "planned_supplement_path",
    "review_artifact_path",
    "source_exists",
    "source_sha256",
    "review_artifact_exists",
    "manual_review_required",
    "publication_blocker",
    "remaining_action",
    "claim_scope_guard",
]


def truthy(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def path_exists(path_text: str) -> bool:
    path = resolve_path(path_text)
    return bool(path and path.is_file())


def sha256_file(path_text: str) -> str:
    path = resolve_path(path_text)
    if not path or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def action_for(row: dict[str, str]) -> tuple[str, bool, str, str, str]:
    decision = row["readiness_decision"]
    if decision == "ready_for_final_license_scope_review":
        return (
            "direct_include_after_final_license_scope_review",
            True,
            row["source_path"],
            "original_source",
            "",
        )
    if decision == "copy_or_commit_required_before_supplement_bundle":
        return (
            "copy_source_to_bundle_after_review",
            True,
            row["source_path"],
            "original_source",
            "",
        )
    if decision == "redacted_derivative_available_manual_review_required":
        return (
            "use_redacted_derivative_after_human_review",
            True,
            row["derivative_path"],
            "redacted_derivative",
            row["source_path"],
        )
    if decision == "scope_note_available_manual_review_required":
        return (
            "include_source_with_scope_note_after_human_review",
            True,
            row["source_path"],
            "original_source",
            row["derivative_path"],
        )
    if decision == "security_review_available_manual_review_required":
        return (
            "include_source_after_security_review",
            True,
            row["source_path"],
            "original_source",
            row["derivative_path"],
        )
    if decision == "excluded_from_public_supplement":
        return ("exclude_internal_record", False, "", "excluded", "")
    return ("manual_classification_required", False, "", "unknown", "")


def manifest_row(row: dict[str, str]) -> dict[str, Any]:
    action, include, source_path, source_role, review_artifact_path = action_for(row)
    source_exists = path_exists(source_path) if include else False
    review_exists = path_exists(review_artifact_path) if review_artifact_path else False
    publication_blocker = truthy(row["publication_blocker"]) or (include and not source_exists)
    remaining_action = row["remaining_action"]
    if include and not source_exists:
        remaining_action = f"{remaining_action}; resolve missing dry-run source before bundle construction"
    return {
        "artifact_group": row["artifact_group"],
        "readiness_decision": row["readiness_decision"],
        "bundle_action": action,
        "include_in_dry_run_bundle": include,
        "bundle_source_path": source_path,
        "bundle_source_role": source_role,
        "planned_supplement_path": row["planned_supplement_path"] if include else "",
        "review_artifact_path": review_artifact_path,
        "source_exists": source_exists,
        "source_sha256": sha256_file(source_path) if include and source_exists else "",
        "review_artifact_exists": review_exists,
        "manual_review_required": truthy(row["manual_review_required"]),
        "publication_blocker": publication_blocker,
        "remaining_action": remaining_action,
        "claim_scope_guard": row["claim_scope_guard"],
    }


def summarize(rows: list[dict[str, Any]], output_dir: Path, readiness_csv: Path) -> dict[str, Any]:
    action_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    target_paths = [row["planned_supplement_path"] for row in rows if row["planned_supplement_path"]]
    duplicate_targets = sorted({path for path in target_paths if target_paths.count(path) > 1})
    for row in rows:
        action_counts[row["bundle_action"]] = action_counts.get(row["bundle_action"], 0) + 1
        role_counts[row["bundle_source_role"]] = role_counts.get(row["bundle_source_role"], 0) + 1

    included_rows = [row for row in rows if truthy(row["include_in_dry_run_bundle"])]
    excluded_rows = [row for row in rows if not truthy(row["include_in_dry_run_bundle"])]
    missing_source_rows = [row for row in included_rows if not truthy(row["source_exists"])]
    review_artifact_rows = [row for row in rows if row["review_artifact_path"]]
    missing_review_artifact_rows = [row for row in review_artifact_rows if not truthy(row["review_artifact_exists"])]
    publication_blockers = [row for row in rows if truthy(row["publication_blocker"])]
    manual_review_rows = [row for row in rows if truthy(row["manual_review_required"])]
    redacted_derivatives = [row for row in rows if row["bundle_source_role"] == "redacted_derivative"]

    release_ready = (
        not publication_blockers
        and not manual_review_rows
        and not missing_source_rows
        and not missing_review_artifact_rows
        and not duplicate_targets
    )
    return {
        "status": "PASS_VSG_PUBLIC_SUPPLEMENT_DRY_RUN_MANIFEST_RECORDED_NOT_RELEASE_READY",
        "schema_name": "verification_substrate_gap_public_supplement_dry_run_manifest_v1",
        "source_readiness_decisions": display_path(readiness_csv),
        "output_dir": display_path(output_dir),
        "row_count": len(rows),
        "dry_run_bundle_entry_count": len(included_rows),
        "excluded_internal_record_count": len(excluded_rows),
        "direct_include_entry_count": action_counts.get("direct_include_after_final_license_scope_review", 0),
        "copy_required_entry_count": action_counts.get("copy_source_to_bundle_after_review", 0),
        "redacted_derivative_entry_count": len(redacted_derivatives),
        "scope_note_review_entry_count": action_counts.get("include_source_with_scope_note_after_human_review", 0),
        "security_review_entry_count": action_counts.get("include_source_after_security_review", 0),
        "manual_review_required_count": len(manual_review_rows),
        "publication_blocker_count": len(publication_blockers),
        "missing_source_count": len(missing_source_rows),
        "missing_review_artifact_count": len(missing_review_artifact_rows),
        "duplicate_planned_target_count": len(duplicate_targets),
        "duplicate_planned_targets": duplicate_targets,
        "bundle_action_counts": dict(sorted(action_counts.items())),
        "bundle_source_role_counts": dict(sorted(role_counts.items())),
        "publication_blocker_paths": [row["bundle_source_path"] or row["planned_supplement_path"] for row in publication_blockers],
        "release_ready_after_dry_run": release_ready,
        "dry_run_only": True,
        "files_copied": False,
        "public_supplement_created": False,
        "publication_performed": False,
        "new_slurm_started": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "allowlist_enabled": False,
        "overleaf_push_performed": False,
        "public_text_only_verification_claimed": False,
        "ownership_proof_claimed": False,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# VSG Public Supplement Dry-Run Bundle Manifest",
        "",
        "This artifact-only manifest resolves the readiness audit into a future",
        "bundle construction plan. It records source files, review artifacts,",
        "target supplement paths, and remaining blockers. It does not copy files,",
        "create a public supplement, publish artifacts, start compute, or expand",
        "claim scope.",
        "",
        f"Status: `{summary['status']}`",
        f"Rows: `{summary['row_count']}`",
        f"Dry-run bundle entries: `{summary['dry_run_bundle_entry_count']}`",
        f"Excluded internal records: `{summary['excluded_internal_record_count']}`",
        f"Direct include entries: `{summary['direct_include_entry_count']}`",
        f"Copy-required entries: `{summary['copy_required_entry_count']}`",
        f"Redacted derivative entries: `{summary['redacted_derivative_entry_count']}`",
        f"Scope-note review entries: `{summary['scope_note_review_entry_count']}`",
        f"Security-review entries: `{summary['security_review_entry_count']}`",
        f"Manual-review-required rows: `{summary['manual_review_required_count']}`",
        f"Publication blockers: `{summary['publication_blocker_count']}`",
        f"Missing dry-run sources: `{summary['missing_source_count']}`",
        f"Missing review artifacts: `{summary['missing_review_artifact_count']}`",
        f"Duplicate planned targets: `{summary['duplicate_planned_target_count']}`",
        f"Release-ready after dry-run: `{summary['release_ready_after_dry_run']}`",
        "",
        "## Bundle Actions",
        "",
        "| Bundle action | Rows |",
        "| --- | ---: |",
    ]
    for action, count in summary["bundle_action_counts"].items():
        lines.append(f"| {action} | {count} |")
    lines.extend(
        [
            "",
            "## Remaining Work Before Actual Bundle Construction",
            "",
            "- Copy the 21 copy-required entries into a reviewed supplement layout.",
            "- Human-review the 14 rows that depend on redaction, scope notes, or security review.",
            "- Keep the 4 internal handoff records excluded.",
            "- Preserve the non-claim guards: no public text-only verification success and no ownership proof.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, output_files: list[Path], status: str) -> None:
    manifest = {
        "status": status,
        "schema_name": "verification_substrate_gap_public_supplement_dry_run_manifest_files_v1",
        "files": [
            {
                "path": display_path(output_file),
                "sha256": sha256_file(display_path(output_file)),
                "bytes": output_file.stat().st_size,
            }
            for output_file in output_files
        ],
        "manifest_self_hash_excluded": True,
    }
    write_json(path, manifest)


def build(readiness_csv: Path, output_dir: Path) -> dict[str, Any]:
    rows = [manifest_row(row) for row in read_csv(readiness_csv)]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dry_run_bundle_manifest.csv"
    summary_path = output_dir / "dry_run_bundle_summary.json"
    report_path = output_dir / "dry_run_bundle_report.md"
    manifest_path = output_dir / "dry_run_bundle_file_manifest.json"
    write_csv(csv_path, rows)
    summary = summarize(rows, output_dir, readiness_csv)
    write_json(summary_path, summary)
    write_report(report_path, summary)
    write_manifest(manifest_path, [csv_path, summary_path, report_path], summary["status"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", default=str(DEFAULT_READINESS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = build(Path(args.readiness), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
