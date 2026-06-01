#!/usr/bin/env python3
"""Build pending review-decision templates for VSG public supplement rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET_INDEX = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_human_review_packet_20260601"
    / "human_review_packet_index.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_review_decision_template_20260601"
)

DECISION_FIELDS = [
    "review_id",
    "review_type",
    "entry_id",
    "blocker_id",
    "artifact_group",
    "source_path",
    "source_sha256_expected",
    "review_artifact_path",
    "review_artifact_sha256_expected",
    "planned_supplement_path",
    "candidate_bundle_path",
    "required_evidence",
    "reviewer_assertion_required",
    "claim_scope_guard",
    "allowed_decisions",
    "decision_status",
    "reviewer_id",
    "reviewed_at_utc",
    "source_sha256_verified",
    "review_artifact_sha256_verified",
    "reviewer_assertion_confirmed",
    "claim_scope_guard_preserved",
    "failure_condition",
    "approval_gate",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def failure_condition_for(review_type: str) -> str:
    if review_type == "redaction_review":
        return (
            "reject_or_hold_if_private_fields_remain; reject_if_trace_bound_only_claim_scope_not_preserved; "
            "reject_if_source_or_review_artifact_hash_mismatch"
        )
    if review_type == "scope_note_review":
        return (
            "reject_or_hold_if_allowed_interpretation_missing; reject_if_forbidden_claims_not_preserved; "
            "reject_if_artifact_is_interpreted_as_protected_success_or_ownership_proof"
        )
    if review_type == "security_review":
        return (
            "reject_or_hold_if_secret_values_detected; reject_if_key_schema_exposure_is_not_acceptable; "
            "reject_if_source_or_review_artifact_hash_mismatch"
        )
    return "reject_or_hold_if_required_evidence_is_not_confirmed"


def decision_template_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "review_id": row["review_id"],
        "review_type": row["review_type"],
        "entry_id": row["entry_id"],
        "blocker_id": row["blocker_id"],
        "artifact_group": row["artifact_group"],
        "source_path": row["source_path"],
        "source_sha256_expected": row["source_sha256"],
        "review_artifact_path": row["review_artifact_path"],
        "review_artifact_sha256_expected": row["review_artifact_sha256"],
        "planned_supplement_path": row["planned_supplement_path"],
        "candidate_bundle_path": row["candidate_bundle_path"],
        "required_evidence": row["required_evidence"],
        "reviewer_assertion_required": row["reviewer_assertion_required"],
        "claim_scope_guard": row["claim_scope_guard"],
        "allowed_decisions": "approved;rejected;hold",
        "decision_status": "pending_not_performed",
        "reviewer_id": "",
        "reviewed_at_utc": "",
        "source_sha256_verified": "",
        "review_artifact_sha256_verified": "",
        "reviewer_assertion_confirmed": "",
        "claim_scope_guard_preserved": "",
        "failure_condition": failure_condition_for(row["review_type"]),
        "approval_gate": "not_approved_template_only",
    }


def schema() -> dict[str, Any]:
    return {
        "schema_name": "verification_substrate_gap_public_supplement_review_decision_record_v1",
        "record_type": "human_review_decision_record",
        "allowed_decisions": ["approved", "rejected", "hold"],
        "initial_decision_status": "pending_not_performed",
        "required_fields_for_any_non_pending_decision": [
            "review_id",
            "review_type",
            "reviewer_id",
            "reviewed_at_utc",
            "decision_status",
            "source_sha256_verified",
            "review_artifact_sha256_verified",
            "reviewer_assertion_confirmed",
            "claim_scope_guard_preserved",
            "failure_condition",
        ],
        "required_truthy_fields_for_approved": [
            "source_sha256_verified",
            "review_artifact_sha256_verified",
            "reviewer_assertion_confirmed",
            "claim_scope_guard_preserved",
        ],
        "approval_must_not_claim": [
            "public text-only verification success",
            "ownership proof",
            "protected success for source-mismatch rows",
            "codeword recovery for public final text",
            "naturalness-preserving rewrite unless separately reviewed",
        ],
        "review_type_specific_required_assertions": {
            "redaction_review": [
                "private fields removed",
                "trace-bound-only scope preserved",
                "redacted derivative hash matches packet index",
            ],
            "scope_note_review": [
                "allowed interpretation present",
                "forbidden claims preserved",
                "artifact not reinterpreted as protected success or ownership proof",
            ],
            "security_review": [
                "no literal secret values",
                "schema/field-name exposure acceptable",
                "configuration review remains non-claim evidence",
            ],
        },
        "template_only": True,
        "review_approvals_recorded": False,
        "human_reviews_performed": False,
    }


def summarize(rows: list[dict[str, Any]], output_dir: Path, packet_index: Path) -> dict[str, Any]:
    review_type_counts: dict[str, int] = {}
    for row in rows:
        review_type_counts[row["review_type"]] = review_type_counts.get(row["review_type"], 0) + 1
    pending_rows = [row for row in rows if row["decision_status"] == "pending_not_performed"]
    empty_reviewer_rows = [row for row in rows if not row["reviewer_id"]]
    empty_review_time_rows = [row for row in rows if not row["reviewed_at_utc"]]
    approved_rows = [row for row in rows if row["decision_status"] == "approved"]
    return {
        "status": "PASS_VSG_PUBLIC_SUPPLEMENT_REVIEW_DECISION_TEMPLATE_RECORDED_PENDING_ONLY",
        "schema_name": "verification_substrate_gap_public_supplement_review_decision_template_v1",
        "source_human_review_packet_index": display_path(packet_index),
        "output_dir": display_path(output_dir),
        "decision_template_row_count": len(rows),
        "pending_decision_count": len(pending_rows),
        "approved_decision_count": len(approved_rows),
        "redaction_review_count": review_type_counts.get("redaction_review", 0),
        "scope_note_review_count": review_type_counts.get("scope_note_review", 0),
        "security_review_count": review_type_counts.get("security_review", 0),
        "empty_reviewer_id_count": len(empty_reviewer_rows),
        "empty_reviewed_at_utc_count": len(empty_review_time_rows),
        "review_type_counts": dict(sorted(review_type_counts.items())),
        "all_decisions_pending": len(pending_rows) == len(rows),
        "schema_written": True,
        "decision_records_template_written": True,
        "review_approvals_recorded": False,
        "human_reviews_performed": False,
        "publication_blockers_resolved": False,
        "release_ready_after_template": False,
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
        "# VSG Public Supplement Review Decision Template",
        "",
        "This artifact-only template defines the fields required for future human",
        "review decisions. It does not approve any review, perform human review,",
        "copy files, create a candidate supplement, publish artifacts, start",
        "compute, or expand claim scope.",
        "",
        f"Status: `{summary['status']}`",
        f"Decision template rows: `{summary['decision_template_row_count']}`",
        f"Pending decisions: `{summary['pending_decision_count']}`",
        f"Approved decisions: `{summary['approved_decision_count']}`",
        f"Redaction reviews: `{summary['redaction_review_count']}`",
        f"Scope-note reviews: `{summary['scope_note_review_count']}`",
        f"Security reviews: `{summary['security_review_count']}`",
        f"Empty reviewer IDs: `{summary['empty_reviewer_id_count']}`",
        f"Empty reviewed_at_utc fields: `{summary['empty_reviewed_at_utc_count']}`",
        f"Review approvals recorded: `{summary['review_approvals_recorded']}`",
        f"Release-ready after template: `{summary['release_ready_after_template']}`",
        "",
        "## Review Types",
        "",
        "| Review type | Rows |",
        "| --- | ---: |",
    ]
    for review_type, count in summary["review_type_counts"].items():
        lines.append(f"| {review_type} | {count} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Every decision row remains `pending_not_performed`.",
            "- Future approvals require reviewer identity, review timestamp, hash verification, assertion confirmation, and claim-scope preservation.",
            "- The template itself does not resolve publication blockers.",
            "- Public text-only verification success and ownership-proof claims remain disallowed.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, output_files: list[Path], status: str) -> None:
    manifest = {
        "status": status,
        "schema_name": "verification_substrate_gap_public_supplement_review_decision_template_manifest_v1",
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


def build(packet_index: Path, output_dir: Path) -> dict[str, Any]:
    rows = [decision_template_row(row) for row in read_csv(packet_index)]
    output_dir.mkdir(parents=True, exist_ok=True)

    template_csv = output_dir / "review_decision_template.csv"
    schema_json = output_dir / "review_decision_schema.json"
    summary_json = output_dir / "review_decision_template_summary.json"
    report_md = output_dir / "review_decision_template_report.md"
    manifest_json = output_dir / "review_decision_template_manifest.json"

    write_csv(template_csv, rows, DECISION_FIELDS)
    write_json(schema_json, schema())
    summary = summarize(rows, output_dir, packet_index)
    write_json(summary_json, summary)
    write_report(report_md, summary)
    write_manifest(manifest_json, [template_csv, schema_json, summary_json, report_md], summary["status"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-index", default=str(DEFAULT_PACKET_INDEX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = build(Path(args.packet_index), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
