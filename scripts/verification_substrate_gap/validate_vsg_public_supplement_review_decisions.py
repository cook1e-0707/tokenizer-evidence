#!/usr/bin/env python3
"""Validate VSG public-supplement human review decision records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECISION_RECORDS = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_review_decision_template_20260601"
    / "review_decision_template.csv"
)
DEFAULT_SCHEMA = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_review_decision_template_20260601"
    / "review_decision_schema.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_review_decision_validation_20260601"
)

VALIDATION_FIELDS = [
    "review_id",
    "review_type",
    "decision_status",
    "source_sha256_expected",
    "source_sha256_actual",
    "review_artifact_sha256_expected",
    "review_artifact_sha256_actual",
    "approval_gate",
    "validation_status",
    "validation_errors",
]

ALLOWED_DECISIONS = {"pending_not_performed", "approved", "rejected", "hold"}
TRUTHY_VALUES = {"true", "1", "yes", "y"}
APPROVED_GATE = "approved_hash_scope_guard_validated"
PENDING_GATE = "not_approved_template_only"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


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


def sha256_file(path_text: str) -> str:
    path = resolve_path(path_text)
    if not path or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def truthy(value: str) -> bool:
    return value.strip().lower() in TRUTHY_VALUES


def utc_timestamp_valid(value: str) -> bool:
    if not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def required_fields(schema: dict[str, Any]) -> list[str]:
    return list(schema.get("required_fields_for_any_non_pending_decision", []))


def required_truthy_fields(schema: dict[str, Any]) -> list[str]:
    return list(schema.get("required_truthy_fields_for_approved", []))


def validate_pending_row(row: dict[str, str]) -> list[str]:
    errors: list[str] = []
    forbidden_non_empty = [
        "reviewer_id",
        "reviewed_at_utc",
        "source_sha256_verified",
        "review_artifact_sha256_verified",
        "reviewer_assertion_confirmed",
        "claim_scope_guard_preserved",
    ]
    for field in forbidden_non_empty:
        if row.get(field, "").strip():
            errors.append(f"pending_row_has_{field}")
    if row.get("approval_gate") != PENDING_GATE:
        errors.append("pending_row_has_non_pending_approval_gate")
    return errors


def validate_non_pending_row(row: dict[str, str], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in required_fields(schema):
        if not row.get(field, "").strip():
            errors.append(f"missing_required_field:{field}")
    if not utc_timestamp_valid(row.get("reviewed_at_utc", "")):
        errors.append("reviewed_at_utc_not_iso_utc")
    return errors


def validate_hashes(row: dict[str, str], *, require_verified: bool) -> tuple[str, str, list[str]]:
    errors: list[str] = []
    source_actual = sha256_file(row.get("source_path", ""))
    review_actual = sha256_file(row.get("review_artifact_path", ""))
    source_expected = row.get("source_sha256_expected", "")
    review_expected = row.get("review_artifact_sha256_expected", "")

    if not source_actual:
        errors.append("source_file_missing")
    elif source_actual != source_expected:
        errors.append("source_sha256_mismatch")

    if not review_actual:
        errors.append("review_artifact_file_missing")
    elif review_actual != review_expected:
        errors.append("review_artifact_sha256_mismatch")

    if require_verified:
        if not truthy(row.get("source_sha256_verified", "")):
            errors.append("source_sha256_not_marked_verified")
        if not truthy(row.get("review_artifact_sha256_verified", "")):
            errors.append("review_artifact_sha256_not_marked_verified")
    return source_actual, review_actual, errors


def validate_approved_row(row: dict[str, str], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in required_truthy_fields(schema):
        if not truthy(row.get(field, "")):
            errors.append(f"approved_missing_truthy_field:{field}")
    if row.get("approval_gate") != APPROVED_GATE:
        errors.append("approved_row_missing_validated_approval_gate")
    return errors


def validate_row(row: dict[str, str], schema: dict[str, Any]) -> dict[str, Any]:
    decision_status = row.get("decision_status", "")
    errors: list[str] = []
    if decision_status not in ALLOWED_DECISIONS:
        errors.append("invalid_decision_status")

    if decision_status == "pending_not_performed":
        errors.extend(validate_pending_row(row))
        source_actual = ""
        review_actual = ""
    else:
        errors.extend(validate_non_pending_row(row, schema))
        source_actual, review_actual, hash_errors = validate_hashes(row, require_verified=True)
        errors.extend(hash_errors)
        if decision_status == "approved":
            errors.extend(validate_approved_row(row, schema))

    return {
        "review_id": row.get("review_id", ""),
        "review_type": row.get("review_type", ""),
        "decision_status": decision_status,
        "source_sha256_expected": row.get("source_sha256_expected", ""),
        "source_sha256_actual": source_actual,
        "review_artifact_sha256_expected": row.get("review_artifact_sha256_expected", ""),
        "review_artifact_sha256_actual": review_actual,
        "approval_gate": row.get("approval_gate", ""),
        "validation_status": "valid" if not errors else "invalid",
        "validation_errors": ";".join(errors),
    }


def summarize(
    validation_rows: list[dict[str, Any]],
    *,
    decisions_csv: Path,
    schema_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in validation_rows:
        counts[row["decision_status"]] = counts.get(row["decision_status"], 0) + 1
    invalid_rows = [row for row in validation_rows if row["validation_status"] == "invalid"]
    approved_rows = [row for row in validation_rows if row["decision_status"] == "approved"]
    status = (
        "FAIL_VSG_PUBLIC_SUPPLEMENT_REVIEW_DECISIONS_INVALID"
        if invalid_rows
        else (
            "PASS_VSG_PUBLIC_SUPPLEMENT_REVIEW_DECISIONS_VALIDATED_PENDING_ONLY"
            if len(approved_rows) == 0 and counts.get("rejected", 0) == 0 and counts.get("hold", 0) == 0
            else "PASS_VSG_PUBLIC_SUPPLEMENT_REVIEW_DECISIONS_VALIDATED_NON_PENDING_RECORDS"
        )
    )
    return {
        "status": status,
        "schema_name": "verification_substrate_gap_public_supplement_review_decision_validation_v1",
        "decision_records": display_path(decisions_csv),
        "schema": display_path(schema_json),
        "output_dir": display_path(output_dir),
        "decision_row_count": len(validation_rows),
        "pending_decision_count": counts.get("pending_not_performed", 0),
        "approved_decision_count": counts.get("approved", 0),
        "rejected_decision_count": counts.get("rejected", 0),
        "hold_decision_count": counts.get("hold", 0),
        "valid_decision_count": len(validation_rows) - len(invalid_rows),
        "invalid_decision_count": len(invalid_rows),
        "invalid_review_ids": [row["review_id"] for row in invalid_rows],
        "all_decisions_valid": len(invalid_rows) == 0,
        "all_decisions_pending": counts.get("pending_not_performed", 0) == len(validation_rows),
        "review_approvals_recorded_in_input": counts.get("approved", 0) > 0,
        "review_approvals_created_by_validator": False,
        "human_reviews_performed_by_validator": False,
        "publication_blockers_resolved": False,
        "release_ready_after_validation": False,
        "artifact_only": True,
        "files_copied": False,
        "candidate_bundle_created": False,
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
        "# VSG Public Supplement Review Decision Validation",
        "",
        "This artifact-only validator checks future human review decision records.",
        "It does not create approvals, perform human review, copy files, build a",
        "candidate supplement, publish artifacts, start compute, or expand claim",
        "scope.",
        "",
        f"Status: `{summary['status']}`",
        f"Decision rows: `{summary['decision_row_count']}`",
        f"Pending decisions: `{summary['pending_decision_count']}`",
        f"Approved decisions: `{summary['approved_decision_count']}`",
        f"Rejected decisions: `{summary['rejected_decision_count']}`",
        f"Hold decisions: `{summary['hold_decision_count']}`",
        f"Invalid decisions: `{summary['invalid_decision_count']}`",
        f"Release-ready after validation: `{summary['release_ready_after_validation']}`",
        "",
        "## Boundary",
        "",
        "- Pending template rows remain valid only when reviewer and approval fields stay empty.",
        "- Approved rows require reviewer identity, UTC review timestamp, hash verification, assertion confirmation, claim-scope preservation, and validated approval gate.",
        "- Rejected and hold rows require reviewer identity, UTC review timestamp, and hash-verification fields, but do not release artifacts.",
        "- Public text-only verification success and ownership-proof claims remain disallowed.",
    ]
    if summary["invalid_review_ids"]:
        lines.extend(["", "## Invalid Review IDs", ""])
        lines.extend(f"- `{review_id}`" for review_id in summary["invalid_review_ids"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, output_files: list[Path], status: str) -> None:
    manifest = {
        "status": status,
        "schema_name": "verification_substrate_gap_public_supplement_review_decision_validation_manifest_v1",
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


def build(decisions_csv: Path, schema_json: Path, output_dir: Path) -> dict[str, Any]:
    schema = read_json(schema_json)
    decision_rows = read_csv(decisions_csv)
    validation_rows = [validate_row(row, schema) for row in decision_rows]
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_csv = output_dir / "review_decision_validation.csv"
    summary_json = output_dir / "review_decision_validation_summary.json"
    report_md = output_dir / "review_decision_validation_report.md"
    manifest_json = output_dir / "review_decision_validation_manifest.json"

    write_csv(validation_csv, validation_rows, VALIDATION_FIELDS)
    summary = summarize(validation_rows, decisions_csv=decisions_csv, schema_json=schema_json, output_dir=output_dir)
    write_json(summary_json, summary)
    write_report(report_md, summary)
    write_manifest(manifest_json, [validation_csv, summary_json, report_md], summary["status"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-records", default=str(DEFAULT_DECISION_RECORDS))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = build(Path(args.decision_records), Path(args.schema), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if summary["status"].startswith("FAIL_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
