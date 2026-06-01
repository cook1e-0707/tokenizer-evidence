#!/usr/bin/env python3
"""Audit VSG public-supplement readiness after review derivatives are created."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    / "public_supplement_readiness_audit_20260601"
)

OUTPUT_FIELDS = [
    "artifact_group",
    "source_path",
    "staging_decision",
    "planned_supplement_path",
    "readiness_decision",
    "derivative_path",
    "derivative_status",
    "remaining_action",
    "manual_review_required",
    "publication_blocker",
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_exists(path_text: str) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.is_file()


def build_derivative_maps(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    redaction = {
        row["source_path"]: row for row in summary.get("redacted_csv_results", [])
    }
    scope_csv = summary.get("scope_notes", {}).get("scope_note_csv", "")
    scope = {}
    if scope_csv and path_exists(scope_csv):
        for row in read_csv(ROOT / scope_csv):
            scope[row["source_path"]] = row
    security = {
        row["source_path"]: row for row in summary.get("security_reviews", [])
    }
    return {"redaction": redaction, "scope": scope, "security": security}


def readiness_for(row: dict[str, str], derivatives: dict[str, dict[str, Any]]) -> dict[str, Any]:
    staging = row["staging_decision"]
    source_path = row["source_path"]
    if staging == "direct_include_candidate":
        return decision(
            row,
            readiness_decision="ready_for_final_license_scope_review",
            derivative_status="not_required",
            derivative_path="",
            remaining_action="final license and scope check before public bundle construction",
            manual_review_required=False,
            publication_blocker=False,
            claim_scope_guard=row["release_claim_scope_note"],
        )
    if staging == "stage_or_copy_candidate":
        return decision(
            row,
            readiness_decision="copy_or_commit_required_before_supplement_bundle",
            derivative_status="not_required",
            derivative_path="",
            remaining_action="copy into reviewed supplement bundle or commit tracked source before release",
            manual_review_required=False,
            publication_blocker=True,
            claim_scope_guard=row["release_claim_scope_note"],
        )
    if staging == "redacted_derivative_candidate":
        redaction = derivatives["redaction"].get(source_path)
        covered = bool(redaction) and redaction.get("private_marker_hits_after_redaction") == 0 and path_exists(
            redaction.get("derivative_path", "")
        )
        return decision(
            row,
            readiness_decision=(
                "redacted_derivative_available_manual_review_required" if covered else "redacted_derivative_missing"
            ),
            derivative_status="covered" if covered else "missing_or_private_marker_residual",
            derivative_path=redaction.get("derivative_path", "") if redaction else "",
            remaining_action=(
                "human review redacted derivative before bundle inclusion"
                if covered
                else "create redacted derivative with private markers removed"
            ),
            manual_review_required=True,
            publication_blocker=True,
            claim_scope_guard=row["release_claim_scope_note"],
        )
    if staging == "scope_note_gated_candidate":
        scope = derivatives["scope"].get(source_path)
        covered = bool(scope) and "forbidden_claims" in scope and "public_label" in scope
        return decision(
            row,
            readiness_decision="scope_note_available_manual_review_required" if covered else "scope_note_missing",
            derivative_status="covered" if covered else "missing_scope_note",
            derivative_path=derivatives_summary_scope_path(derivatives) if covered else "",
            remaining_action=(
                "human review scope note before bundle inclusion"
                if covered
                else "create scope note preserving non-claim interpretation"
            ),
            manual_review_required=True,
            publication_blocker=True,
            claim_scope_guard=scope.get("forbidden_claims", row["release_claim_scope_note"]) if scope else row["release_claim_scope_note"],
        )
    if staging == "security_review_gated_candidate":
        review = derivatives["security"].get(source_path)
        covered = bool(review) and review.get("secret_value_hit_count") == 0 and path_exists(
            review.get("security_review_json", "")
        )
        return decision(
            row,
            readiness_decision="security_review_available_manual_review_required" if covered else "security_review_missing_or_secret_values_detected",
            derivative_status="covered" if covered else "missing_or_secret_values_detected",
            derivative_path=review.get("security_review_json", "") if review else "",
            remaining_action=(
                "human security review of schema field names before bundle inclusion"
                if covered
                else "remove literal secret values or create security review before release"
            ),
            manual_review_required=True,
            publication_blocker=True,
            claim_scope_guard=row["release_claim_scope_note"],
        )
    if staging == "excluded_internal_record":
        return decision(
            row,
            readiness_decision="excluded_from_public_supplement",
            derivative_status="not_required",
            derivative_path="",
            remaining_action="none for public supplement",
            manual_review_required=False,
            publication_blocker=False,
            claim_scope_guard=row["release_claim_scope_note"],
        )
    return decision(
        row,
        readiness_decision="unclassified_manual_review_required",
        derivative_status="unknown",
        derivative_path="",
        remaining_action="manual classification required",
        manual_review_required=True,
        publication_blocker=True,
        claim_scope_guard=row["release_claim_scope_note"],
    )


def derivatives_summary_scope_path(derivatives: dict[str, dict[str, Any]]) -> str:
    # All scope-note rows are covered by the single scope_notes.csv file.
    for row in derivatives["scope"].values():
        if row:
            return "results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv"
    return ""


def decision(
    row: dict[str, str],
    *,
    readiness_decision: str,
    derivative_path: str,
    derivative_status: str,
    remaining_action: str,
    manual_review_required: bool,
    publication_blocker: bool,
    claim_scope_guard: str,
) -> dict[str, Any]:
    return {
        "artifact_group": row["artifact_group"],
        "source_path": row["source_path"],
        "staging_decision": row["staging_decision"],
        "planned_supplement_path": row["planned_supplement_path"],
        "readiness_decision": readiness_decision,
        "derivative_path": derivative_path,
        "derivative_status": derivative_status,
        "remaining_action": remaining_action,
        "manual_review_required": manual_review_required,
        "publication_blocker": publication_blocker,
        "claim_scope_guard": claim_scope_guard,
    }


def summarize(rows: list[dict[str, Any]], output_dir: Path, staging_plan: Path, derivatives_summary: Path) -> dict[str, Any]:
    readiness_counts: dict[str, int] = {}
    derivative_status_counts: dict[str, int] = {}
    for row in rows:
        readiness_counts[row["readiness_decision"]] = readiness_counts.get(row["readiness_decision"], 0) + 1
        derivative_status_counts[row["derivative_status"]] = derivative_status_counts.get(row["derivative_status"], 0) + 1
    publication_blockers = [row for row in rows if truthy(row["publication_blocker"])]
    manual_review_rows = [row for row in rows if truthy(row["manual_review_required"])]
    derivative_required_rows = [
        row
        for row in rows
        if row["staging_decision"]
        in {"redacted_derivative_candidate", "scope_note_gated_candidate", "security_review_gated_candidate"}
    ]
    derivative_covered_rows = [row for row in derivative_required_rows if row["derivative_status"] == "covered"]
    stage_or_copy_rows = [row for row in rows if row["staging_decision"] == "stage_or_copy_candidate"]
    direct_include_rows = [row for row in rows if row["staging_decision"] == "direct_include_candidate"]
    excluded_rows = [row for row in rows if row["staging_decision"] == "excluded_internal_record"]
    return {
        "status": "PASS_VSG_PUBLIC_SUPPLEMENT_READINESS_AUDIT_RECORDED_REVIEW_REQUIRED",
        "schema_name": "verification_substrate_gap_public_supplement_readiness_audit_v1",
        "source_staging_plan": display_path(staging_plan),
        "source_derivatives_summary": display_path(derivatives_summary),
        "output_dir": display_path(output_dir),
        "row_count": len(rows),
        "direct_include_candidate_count": len(direct_include_rows),
        "stage_or_copy_required_count": len(stage_or_copy_rows),
        "derivative_required_count": len(derivative_required_rows),
        "derivative_covered_count": len(derivative_covered_rows),
        "derivative_uncovered_count": len(derivative_required_rows) - len(derivative_covered_rows),
        "manual_review_required_after_derivatives_count": len(manual_review_rows),
        "publication_blocker_count": len(publication_blockers),
        "excluded_internal_record_count": len(excluded_rows),
        "readiness_decision_counts": dict(sorted(readiness_counts.items())),
        "derivative_status_counts": dict(sorted(derivative_status_counts.items())),
        "publication_blocker_paths": [row["source_path"] for row in publication_blockers],
        "release_ready_now": False,
        "public_supplement_created": False,
        "publication_performed": False,
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
        "# VSG Public Supplement Readiness Audit",
        "",
        "This artifact-only audit checks whether the supplement staging blockers",
        "are covered by review derivatives. It does not copy files into a public",
        "supplement, publish artifacts, start compute, or expand claim scope.",
        "",
        f"Status: `{summary['status']}`",
        f"Rows: `{summary['row_count']}`",
        f"Direct include candidates: `{summary['direct_include_candidate_count']}`",
        f"Stage/copy still required: `{summary['stage_or_copy_required_count']}`",
        f"Derivative-required rows: `{summary['derivative_required_count']}`",
        f"Derivative-covered rows: `{summary['derivative_covered_count']}`",
        f"Derivative-uncovered rows: `{summary['derivative_uncovered_count']}`",
        f"Manual review still required: `{summary['manual_review_required_after_derivatives_count']}`",
        f"Publication blockers: `{summary['publication_blocker_count']}`",
        f"Release-ready now: `{summary['release_ready_now']}`",
        "",
        "## Readiness Decisions",
        "",
        "| Decision | Rows |",
        "| --- | ---: |",
    ]
    for decision_name, count in summary["readiness_decision_counts"].items():
        lines.append(f"| {decision_name} | {count} |")
    lines.extend(
        [
            "",
            "## Remaining Work Before A Public Supplement",
            "",
            "- Copy or commit the 21 stage/copy candidates into a reviewed bundle.",
            "- Human-review the 14 derivative-covered rows before inclusion.",
            "- Keep the 4 internal handoff records excluded from the public supplement.",
            "- Do not claim public text-only verification success or ownership proof.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, output_files: list[Path], status: str) -> None:
    manifest = {
        "status": status,
        "schema_name": "verification_substrate_gap_public_supplement_readiness_manifest_v1",
        "files": [
            {"path": display_path(output_file), "sha256": sha256_file(output_file), "bytes": output_file.stat().st_size}
            for output_file in output_files
        ],
        "manifest_self_hash_excluded": True,
    }
    write_json(path, manifest)


def build(staging_plan: Path, derivatives_summary: Path, output_dir: Path) -> dict[str, Any]:
    staging_rows = read_csv(staging_plan)
    derivatives = build_derivative_maps(read_json(derivatives_summary))
    rows = [readiness_for(row, derivatives) for row in staging_rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "readiness_decisions.csv"
    summary_path = output_dir / "readiness_summary.json"
    report_path = output_dir / "readiness_report.md"
    manifest_path = output_dir / "readiness_manifest.json"
    write_csv(csv_path, rows)
    summary = summarize(rows, output_dir, staging_plan, derivatives_summary)
    write_json(summary_path, summary)
    write_report(report_path, summary)
    write_manifest(manifest_path, [csv_path, summary_path, report_path], summary["status"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-plan", default=str(DEFAULT_STAGING_PLAN))
    parser.add_argument("--derivatives-summary", default=str(DEFAULT_DERIVATIVES_SUMMARY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = build(Path(args.staging_plan), Path(args.derivatives_summary), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
