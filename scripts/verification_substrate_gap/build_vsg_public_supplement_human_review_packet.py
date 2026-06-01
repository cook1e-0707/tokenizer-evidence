#!/usr/bin/env python3
"""Build an artifact-only human-review packet for pending VSG supplement rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW_CHECKLIST = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_copy_review_plan_20260601"
    / "reviewer_facing_checklist.csv"
)
DEFAULT_DERIVATIVES_SUMMARY = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_review_derivatives_20260601"
    / "review_derivatives_summary.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_human_review_packet_20260601"
)

OUTPUT_FIELDS = [
    "review_id",
    "review_type",
    "entry_id",
    "blocker_id",
    "artifact_group",
    "source_path",
    "source_sha256",
    "source_bytes",
    "source_row_count",
    "planned_supplement_path",
    "candidate_bundle_path",
    "review_artifact_path",
    "review_artifact_sha256",
    "review_artifact_bytes",
    "review_artifact_row_count",
    "scope_note_type",
    "public_label",
    "allowed_interpretation",
    "forbidden_claims",
    "redaction_dropped_fields",
    "redaction_private_marker_hits_after_redaction",
    "security_field_name_hit_count",
    "security_secret_value_hit_count",
    "release_recommendation",
    "required_evidence",
    "reviewer_assertion_required",
    "approval_status",
    "claim_scope_guard",
]


def truthy(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


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


def file_bytes(path_text: str) -> int:
    path = resolve_path(path_text)
    if not path or not path.is_file():
        return 0
    return path.stat().st_size


def csv_row_count(path_text: str) -> int | str:
    path = resolve_path(path_text)
    if not path or not path.is_file() or path.suffix.lower() != ".csv":
        return ""
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def load_derivative_context(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scope_csv = summary.get("scope_notes", {}).get("scope_note_csv", "")
    scope_rows = read_csv(ROOT / scope_csv) if scope_csv else []
    return {
        "redaction_by_derivative": {
            row["derivative_path"]: row for row in summary.get("redacted_csv_results", [])
        },
        "redaction_by_source": {
            row["source_path"]: row for row in summary.get("redacted_csv_results", [])
        },
        "scope_by_source": {row["source_path"]: row for row in scope_rows},
        "scope_by_target": {row["planned_supplement_path"]: row for row in scope_rows},
        "security_by_source": {
            row["source_path"]: row for row in summary.get("security_reviews", [])
        },
        "summary": summary,
    }


def context_for(row: dict[str, str], derivative_context: dict[str, Any]) -> dict[str, Any]:
    review_type = row["review_type"]
    if review_type == "redaction_review":
        return derivative_context["redaction_by_derivative"].get(row["source_path"], {})
    if review_type == "scope_note_review":
        return (
            derivative_context["scope_by_source"].get(row["source_path"], {})
            or derivative_context["scope_by_target"].get(row["planned_supplement_path"], {})
        )
    if review_type == "security_review":
        return derivative_context["security_by_source"].get(row["source_path"], {})
    return {}


def packet_row(row: dict[str, str], derivative_context: dict[str, Any]) -> dict[str, Any]:
    ctx = context_for(row, derivative_context)
    review_artifact = row["review_artifact_path"]
    return {
        "review_id": row["review_id"],
        "review_type": row["review_type"],
        "entry_id": row["entry_id"],
        "blocker_id": row["blocker_id"],
        "artifact_group": row["artifact_group"],
        "source_path": row["source_path"],
        "source_sha256": sha256_file(row["source_path"]),
        "source_bytes": file_bytes(row["source_path"]),
        "source_row_count": csv_row_count(row["source_path"]),
        "planned_supplement_path": row["planned_supplement_path"],
        "candidate_bundle_path": row["candidate_bundle_path"],
        "review_artifact_path": review_artifact,
        "review_artifact_sha256": sha256_file(review_artifact),
        "review_artifact_bytes": file_bytes(review_artifact),
        "review_artifact_row_count": csv_row_count(review_artifact),
        "scope_note_type": ctx.get("scope_note_type", ""),
        "public_label": ctx.get("public_label", ""),
        "allowed_interpretation": ctx.get("allowed_interpretation", ""),
        "forbidden_claims": ctx.get("forbidden_claims", ""),
        "redaction_dropped_fields": ";".join(ctx.get("dropped_fields", [])) if ctx.get("dropped_fields") else "",
        "redaction_private_marker_hits_after_redaction": ctx.get("private_marker_hits_after_redaction", ""),
        "security_field_name_hit_count": ctx.get("field_name_hit_count", ""),
        "security_secret_value_hit_count": ctx.get("secret_value_hit_count", ""),
        "release_recommendation": ctx.get("release_recommendation", ""),
        "required_evidence": row["required_evidence"],
        "reviewer_assertion_required": row["reviewer_assertion_required"],
        "approval_status": row["approval_status"],
        "claim_scope_guard": row["claim_scope_guard"],
    }


def summarize(rows: list[dict[str, Any]], output_dir: Path, checklist: Path, derivatives_summary: Path) -> dict[str, Any]:
    review_type_counts: dict[str, int] = {}
    for row in rows:
        review_type_counts[row["review_type"]] = review_type_counts.get(row["review_type"], 0) + 1
    pending_rows = [row for row in rows if row["approval_status"] == "pending_not_performed"]
    missing_sources = [row for row in rows if not row["source_sha256"]]
    missing_review_artifacts = [row for row in rows if not row["review_artifact_sha256"]]
    redaction_rows = [row for row in rows if row["review_type"] == "redaction_review"]
    redaction_private_hits = sum(int(row["redaction_private_marker_hits_after_redaction"] or 0) for row in redaction_rows)
    security_rows = [row for row in rows if row["review_type"] == "security_review"]
    security_secret_hits = sum(int(row["security_secret_value_hit_count"] or 0) for row in security_rows)
    scope_rows = [row for row in rows if row["review_type"] == "scope_note_review"]
    missing_scope_notes = [
        row for row in scope_rows if not row["allowed_interpretation"] or not row["forbidden_claims"]
    ]
    return {
        "status": "PASS_VSG_PUBLIC_SUPPLEMENT_HUMAN_REVIEW_PACKET_RECORDED_PENDING_REVIEW",
        "schema_name": "verification_substrate_gap_public_supplement_human_review_packet_v1",
        "source_reviewer_checklist": display_path(checklist),
        "source_derivatives_summary": display_path(derivatives_summary),
        "output_dir": display_path(output_dir),
        "review_row_count": len(rows),
        "pending_review_count": len(pending_rows),
        "redaction_review_count": review_type_counts.get("redaction_review", 0),
        "scope_note_review_count": review_type_counts.get("scope_note_review", 0),
        "security_review_count": review_type_counts.get("security_review", 0),
        "missing_source_count": len(missing_sources),
        "missing_review_artifact_count": len(missing_review_artifacts),
        "missing_scope_note_context_count": len(missing_scope_notes),
        "redaction_private_marker_hits_after_redaction": redaction_private_hits,
        "security_secret_value_hit_count": security_secret_hits,
        "review_type_counts": dict(sorted(review_type_counts.items())),
        "all_reviews_pending": len(pending_rows) == len(rows),
        "all_sources_present": len(missing_sources) == 0,
        "all_review_artifacts_present": len(missing_review_artifacts) == 0,
        "all_scope_notes_resolved": len(missing_scope_notes) == 0,
        "review_packet_created": True,
        "review_approvals_recorded": False,
        "human_reviews_performed": False,
        "files_copied": False,
        "candidate_bundle_created": False,
        "publication_blockers_resolved": False,
        "release_ready_after_packet": False,
        "artifact_only": True,
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


def write_cards(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# VSG Public Supplement Human Review Packet",
        "",
        "This packet is artifact-only. It lists pending review rows, hashes,",
        "review artifacts, claim guards, and required reviewer assertions. It does",
        "not approve any row, copy files, create a public supplement, publish",
        "artifacts, start compute, or expand claim scope.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['review_id']} - {row['review_type']}",
                "",
                f"- Entry: `{row['entry_id']}`",
                f"- Blocker: `{row['blocker_id']}`",
                f"- Artifact group: `{row['artifact_group']}`",
                f"- Source: `{row['source_path']}`",
                f"- Source SHA256: `{row['source_sha256']}`",
                f"- Planned supplement path: `{row['planned_supplement_path']}`",
                f"- Review artifact: `{row['review_artifact_path']}`",
                f"- Review artifact SHA256: `{row['review_artifact_sha256']}`",
                f"- Required evidence: {row['required_evidence']}",
                f"- Reviewer assertion required: {row['reviewer_assertion_required']}",
                f"- Approval status: `{row['approval_status']}`",
                f"- Claim scope guard: {row['claim_scope_guard']}",
            ]
        )
        if row["allowed_interpretation"] or row["forbidden_claims"]:
            lines.extend(
                [
                    f"- Allowed interpretation: {row['allowed_interpretation']}",
                    f"- Forbidden claims: {row['forbidden_claims']}",
                ]
            )
        if row["redaction_dropped_fields"]:
            lines.extend(
                [
                    f"- Redaction dropped fields: `{row['redaction_dropped_fields']}`",
                    f"- Private marker hits after redaction: `{row['redaction_private_marker_hits_after_redaction']}`",
                ]
            )
        if row["release_recommendation"]:
            lines.extend(
                [
                    f"- Release recommendation: `{row['release_recommendation']}`",
                    f"- Security field-name hits: `{row['security_field_name_hit_count']}`",
                    f"- Security secret-value hits: `{row['security_secret_value_hit_count']}`",
                ]
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# VSG Public Supplement Human Review Packet Summary",
        "",
        "This artifact-only packet prepares the 14 pending human-review rows for",
        "manual inspection. It does not approve any row, copy files, create a",
        "candidate supplement, publish artifacts, start compute, or expand claim",
        "scope.",
        "",
        f"Status: `{summary['status']}`",
        f"Review rows: `{summary['review_row_count']}`",
        f"Pending reviews: `{summary['pending_review_count']}`",
        f"Redaction reviews: `{summary['redaction_review_count']}`",
        f"Scope-note reviews: `{summary['scope_note_review_count']}`",
        f"Security reviews: `{summary['security_review_count']}`",
        f"Missing sources: `{summary['missing_source_count']}`",
        f"Missing review artifacts: `{summary['missing_review_artifact_count']}`",
        f"Missing scope-note context: `{summary['missing_scope_note_context_count']}`",
        f"Redaction private marker hits after redaction: `{summary['redaction_private_marker_hits_after_redaction']}`",
        f"Security secret-value hits: `{summary['security_secret_value_hit_count']}`",
        f"Review approvals recorded: `{summary['review_approvals_recorded']}`",
        f"Release-ready after packet: `{summary['release_ready_after_packet']}`",
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
            "- All review rows remain `pending_not_performed`.",
            "- Human review approvals are not recorded by this packet.",
            "- No files are copied and no candidate supplement is created.",
            "- Public text-only verification success and ownership-proof claims remain disallowed.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, output_files: list[Path], status: str) -> None:
    manifest = {
        "status": status,
        "schema_name": "verification_substrate_gap_public_supplement_human_review_packet_manifest_v1",
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


def build(review_checklist: Path, derivatives_summary: Path, output_dir: Path) -> dict[str, Any]:
    derivative_context = load_derivative_context(derivatives_summary)
    rows = [packet_row(row, derivative_context) for row in read_csv(review_checklist)]
    output_dir.mkdir(parents=True, exist_ok=True)

    index_csv = output_dir / "human_review_packet_index.csv"
    cards_md = output_dir / "human_review_cards.md"
    report_md = output_dir / "human_review_packet_report.md"
    summary_json = output_dir / "human_review_packet_summary.json"
    manifest_json = output_dir / "human_review_packet_manifest.json"

    write_csv(index_csv, rows, OUTPUT_FIELDS)
    write_cards(cards_md, rows)
    summary = summarize(rows, output_dir, review_checklist, derivatives_summary)
    write_json(summary_json, summary)
    write_report(report_md, summary)
    write_manifest(manifest_json, [index_csv, cards_md, summary_json, report_md], summary["status"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-checklist", default=str(DEFAULT_REVIEW_CHECKLIST))
    parser.add_argument("--derivatives-summary", default=str(DEFAULT_DERIVATIVES_SUMMARY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = build(Path(args.review_checklist), Path(args.derivatives_summary), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
