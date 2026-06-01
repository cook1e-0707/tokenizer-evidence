#!/usr/bin/env python3
"""Build artifact-only review derivatives for a future VSG public supplement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGING_PLAN = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "reproducibility_release_staging_plan_20260601"
    / "release_staging_plan.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_review_derivatives_20260601"
)

PRIVATE_MARKERS = (
    "/hpcstor",
    "/Users/",
    "guanjie.lin001",
    "tokenizer-evidence/natural_evidence",
)
SECURITY_REVIEW_PATTERN = re.compile(r"(secret|key|hmac)", re.IGNORECASE)
SECRET_VALUE_PATTERNS = {
    "aws_access_key_id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "openai_style_api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
}


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


def has_private_marker(value: str) -> bool:
    return any(marker in value for marker in PRIVATE_MARKERS)


def redacted_target_path(output_dir: Path, planned_supplement_path: str) -> Path:
    return output_dir / planned_supplement_path


def redact_csv_source(row: dict[str, str], output_dir: Path) -> dict[str, Any]:
    source = ROOT / row["source_path"]
    rows = read_csv(source)
    if not rows:
        raise ValueError(f"cannot redact empty CSV: {source}")
    fields = list(rows[0].keys())
    dropped_fields = [
        field
        for field in fields
        if field == "source_shard_dir" or any(has_private_marker(record.get(field, "")) for record in rows)
    ]
    kept_fields = [field for field in fields if field not in dropped_fields]
    redacted_rows = [{field: record.get(field, "") for field in kept_fields} for record in rows]
    target = redacted_target_path(output_dir, row["planned_supplement_path"])
    write_csv(target, redacted_rows, kept_fields)
    private_hits_after = count_private_marker_hits(target)
    return {
        "source_path": row["source_path"],
        "derivative_path": display_path(target),
        "row_count": len(redacted_rows),
        "input_field_count": len(fields),
        "output_field_count": len(kept_fields),
        "dropped_fields": dropped_fields,
        "private_marker_hits_after_redaction": private_hits_after,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def scope_note_type(group: str) -> str:
    if group == "trace_bound_corpus_summary":
        return "trace_bound_scope_note"
    if group == "public_predicate_attack_ladder":
        return "source_mismatch_spoofing_scope_note"
    if group == "stronger_public_predicate_local_pilot":
        return "local_non_claim_pilot_scope_note"
    return "generic_scope_note"


def forbidden_claims_for(group: str) -> str:
    common = [
        "public text-only verification success",
        "ownership proof",
    ]
    if group == "trace_bound_corpus_summary":
        return "; ".join(common + ["public final-text codeword recovery"])
    if group == "public_predicate_attack_ladder":
        return "; ".join(common + ["protected success", "codeword recovery", "naturalness-preserving rewrite"])
    if group == "stronger_public_predicate_local_pilot":
        return "; ".join(common + ["adopted locked evidence", "paper-facing final-text claim"])
    return "; ".join(common)


def scope_label_for(group: str) -> str:
    if group == "trace_bound_corpus_summary":
        return "provider-side trace-bound diagnostic summary only"
    if group == "public_predicate_attack_ladder":
        return "source-mismatch public-predicate spoofing artifact only"
    if group == "stronger_public_predicate_local_pilot":
        return "local non-adopted/historical pilot only"
    return "scope-gated artifact"


def build_scope_notes(rows: list[dict[str, str]], output_dir: Path) -> dict[str, Any]:
    note_rows = []
    for row in rows:
        note_rows.append(
            {
                "artifact_group": row["artifact_group"],
                "source_path": row["source_path"],
                "planned_supplement_path": row["planned_supplement_path"],
                "scope_note_type": scope_note_type(row["artifact_group"]),
                "public_label": scope_label_for(row["artifact_group"]),
                "allowed_interpretation": row["release_claim_scope_note"],
                "forbidden_claims": forbidden_claims_for(row["artifact_group"]),
                "manual_review_required": row["manual_review_required"],
            }
        )
    fields = [
        "artifact_group",
        "source_path",
        "planned_supplement_path",
        "scope_note_type",
        "public_label",
        "allowed_interpretation",
        "forbidden_claims",
        "manual_review_required",
    ]
    csv_path = output_dir / "scope_notes.csv"
    write_csv(csv_path, note_rows, fields)
    md_path = output_dir / "scope_notes.md"
    write_scope_notes_md(md_path, note_rows)
    return {
        "scope_note_count": len(note_rows),
        "scope_note_csv": display_path(csv_path),
        "scope_note_md": display_path(md_path),
        "scope_note_csv_sha256": sha256_file(csv_path),
        "scope_note_md_sha256": sha256_file(md_path),
    }


def write_scope_notes_md(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# VSG Public Supplement Scope Notes",
        "",
        "These notes are artifact-only labels for future supplement review. They",
        "do not publish the referenced source files and do not expand the VSG",
        "claim boundary.",
        "",
        "| Artifact group | Source path | Public label | Forbidden claims |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["artifact_group"],
                    row["source_path"],
                    row["public_label"],
                    row["forbidden_claims"],
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def review_security_config(row: dict[str, str], output_dir: Path) -> dict[str, Any]:
    source = ROOT / row["source_path"]
    text = source.read_text(encoding="utf-8")
    field_name_hits = []
    secret_value_hits = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if SECURITY_REVIEW_PATTERN.search(line):
            field_name_hits.append({"line": line_no, "text": line.strip()})
        for name, pattern in SECRET_VALUE_PATTERNS.items():
            if pattern.search(line):
                secret_value_hits.append({"line": line_no, "pattern": name})
    review = {
        "source_path": row["source_path"],
        "planned_supplement_path": row["planned_supplement_path"],
        "field_name_hit_count": len(field_name_hits),
        "field_name_hits": field_name_hits,
        "secret_value_hit_count": len(secret_value_hits),
        "secret_value_hits": secret_value_hits,
        "manual_review_required": True,
        "release_recommendation": (
            "schema_field_review_required_no_literal_secret_values_detected"
            if not secret_value_hits
            else "do_not_release_until_secret_values_are_removed"
        ),
        "claim_scope": "configuration review only; no public text-only verification or ownership-proof claim",
    }
    json_path = output_dir / "security_review_text_only_observability.json"
    md_path = output_dir / "security_review_text_only_observability.md"
    write_json(json_path, review)
    write_security_review_md(md_path, review)
    review["security_review_json"] = display_path(json_path)
    review["security_review_md"] = display_path(md_path)
    review["security_review_json_sha256"] = sha256_file(json_path)
    review["security_review_md_sha256"] = sha256_file(md_path)
    return review


def write_security_review_md(path: Path, review: dict[str, Any]) -> None:
    lines = [
        "# VSG Config Security Review",
        "",
        f"Source: `{review['source_path']}`",
        f"Planned supplement path: `{review['planned_supplement_path']}`",
        f"Field-name hits: `{review['field_name_hit_count']}`",
        f"Secret-value hits: `{review['secret_value_hit_count']}`",
        f"Release recommendation: `{review['release_recommendation']}`",
        "",
        "## Field-Name Hits",
        "",
    ]
    for hit in review["field_name_hits"]:
        lines.append(f"- line {hit['line']}: `{hit['text']}`")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This review distinguishes key/HMAC-related schema field names from",
            "literal secret values. It does not publish the config and does not",
            "expand the VSG claim boundary.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_private_marker_hits(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return sum(text.count(marker) for marker in PRIVATE_MARKERS)


def write_manifest(path: Path, output_files: list[Path], status: str) -> None:
    manifest = {
        "status": status,
        "schema_name": "verification_substrate_gap_public_supplement_review_derivatives_manifest_v1",
        "files": [
            {
                "path": display_path(output_file),
                "sha256": sha256_file(output_file),
                "bytes": output_file.stat().st_size,
            }
            for output_file in output_files
        ],
        "manifest_self_hash_excluded": True,
    }
    write_json(path, manifest)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# VSG Public Supplement Review Derivatives",
        "",
        "This artifact-only pass creates review derivatives requested by the",
        "release staging plan: redacted trace CSVs, scope notes, and a config",
        "security review. It does not create or publish a public supplement.",
        "",
        f"Status: `{summary['status']}`",
        f"Redacted CSVs written: `{summary['redacted_csv_written_count']}`",
        f"Scope notes: `{summary['scope_note_count']}`",
        f"Security reviews: `{summary['security_review_count']}`",
        f"Private marker hits after redaction: `{summary['private_marker_hits_after_redaction']}`",
        f"Secret value hits: `{summary['security_secret_value_hit_count']}`",
        "",
        "## Scope",
        "",
        "- Trace-bound derivatives remain provider-side diagnostic summaries.",
        "- Source-mismatch attack artifacts remain spoofing evidence only.",
        "- Local-pilot artifacts remain non-adopted/historical pilot evidence only.",
        "- No public text-only verification success or ownership-proof claim is made.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(staging_plan: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_csv(staging_plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    redaction_rows = [row for row in rows if row["staging_decision"] == "redacted_derivative_candidate"]
    scope_rows = [row for row in rows if row["staging_decision"] == "scope_note_gated_candidate"]
    security_rows = [row for row in rows if row["staging_decision"] == "security_review_gated_candidate"]
    redaction_results = [redact_csv_source(row, output_dir) for row in redaction_rows]
    scope_result = build_scope_notes(scope_rows, output_dir)
    security_results = [review_security_config(row, output_dir) for row in security_rows]
    private_hits_after = sum(result["private_marker_hits_after_redaction"] for result in redaction_results)
    secret_value_hits = sum(result["secret_value_hit_count"] for result in security_results)
    status = "PASS_VSG_PUBLIC_SUPPLEMENT_REVIEW_DERIVATIVES_RECORDED_ARTIFACT_ONLY"
    summary = {
        "status": status,
        "schema_name": "verification_substrate_gap_public_supplement_review_derivatives_v1",
        "source_staging_plan": display_path(staging_plan),
        "output_dir": display_path(output_dir),
        "redacted_csv_written_count": len(redaction_results),
        "redacted_csv_results": redaction_results,
        "redacted_rows_total": sum(result["row_count"] for result in redaction_results),
        "private_marker_hits_after_redaction": private_hits_after,
        "scope_note_count": scope_result["scope_note_count"],
        "scope_notes": scope_result,
        "security_review_count": len(security_results),
        "security_reviews": security_results,
        "security_field_name_hit_count": sum(result["field_name_hit_count"] for result in security_results),
        "security_secret_value_hit_count": secret_value_hits,
        "review_derivatives_created": True,
        "source_files_copied_without_transform": False,
        "public_supplement_created": False,
        "publication_performed": False,
        "new_slurm_started": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "allowlist_enabled": False,
        "public_text_only_verification_claimed": False,
        "ownership_proof_claimed": False,
        "release_ready_after_derivatives": False,
    }
    summary_path = output_dir / "review_derivatives_summary.json"
    report_path = output_dir / "review_derivatives_report.md"
    manifest_path = output_dir / "review_derivatives_manifest.json"
    write_json(summary_path, summary)
    write_report(report_path, summary)
    output_files = [
        Path(result["derivative_path"]) if Path(result["derivative_path"]).is_absolute() else ROOT / result["derivative_path"]
        for result in redaction_results
    ]
    output_files.extend([output_dir / "scope_notes.csv", output_dir / "scope_notes.md"])
    output_files.extend([output_dir / "security_review_text_only_observability.json", output_dir / "security_review_text_only_observability.md"])
    output_files.extend([summary_path, report_path])
    write_manifest(manifest_path, output_files, status)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-plan", default=str(DEFAULT_STAGING_PLAN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = build(Path(args.staging_plan), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
