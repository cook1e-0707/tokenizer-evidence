#!/usr/bin/env python3
"""Build copy-command and review checklists for the VSG supplement preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFLIGHT_DIR = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_bundle_preflight_20260601"
)
DEFAULT_FUTURE_COPY_PLAN = DEFAULT_PREFLIGHT_DIR / "future_copy_plan.csv"
DEFAULT_HUMAN_REVIEW_HOLDS = DEFAULT_PREFLIGHT_DIR / "human_review_holds.csv"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_copy_review_plan_20260601"
)

COPY_FIELDS = [
    "entry_id",
    "artifact_group",
    "source_path",
    "source_sha256",
    "candidate_bundle_path",
    "planned_supplement_path",
    "target_parent_dir",
    "mkdir_command",
    "copy_command",
    "verify_sha256_command",
    "source_exists",
    "target_exists_now",
    "ready_for_future_copy_after_review",
    "execution_status",
    "claim_scope_guard",
]

REVIEW_FIELDS = [
    "review_id",
    "entry_id",
    "blocker_id",
    "artifact_group",
    "review_type",
    "source_path",
    "source_sha256",
    "candidate_bundle_path",
    "planned_supplement_path",
    "review_artifact_path",
    "review_artifact_exists",
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


def quote(path_text: str) -> str:
    return shlex.quote(path_text)


def copy_plan_row(row: dict[str, str]) -> dict[str, Any]:
    target = row["candidate_bundle_path"]
    target_parent = str(Path(target).parent)
    source_hash = sha256_file(row["bundle_source_path"])
    source_exists = path_exists(row["bundle_source_path"])
    target_exists = path_exists(target)
    return {
        "entry_id": row["entry_id"],
        "artifact_group": row["artifact_group"],
        "source_path": row["bundle_source_path"],
        "source_sha256": source_hash,
        "candidate_bundle_path": target,
        "planned_supplement_path": row["planned_supplement_path"],
        "target_parent_dir": target_parent,
        "mkdir_command": f"mkdir -p {quote(target_parent)}",
        "copy_command": f"cp -p {quote(row['bundle_source_path'])} {quote(target)}",
        "verify_sha256_command": (
            "python3 - <<'PY'\n"
            "import hashlib, pathlib, sys\n"
            f"path = pathlib.Path({target!r})\n"
            f"expected = {source_hash!r}\n"
            "actual = hashlib.sha256(path.read_bytes()).hexdigest()\n"
            "sys.exit(0 if actual == expected else 1)\n"
            "PY"
        ),
        "source_exists": source_exists,
        "target_exists_now": target_exists,
        "ready_for_future_copy_after_review": source_exists and not target_exists,
        "execution_status": "not_executed_plan_only",
        "claim_scope_guard": row["claim_scope_guard"],
    }


def review_type_for(row: dict[str, str]) -> str:
    action = row["bundle_action"]
    if action == "use_redacted_derivative_after_human_review":
        return "redaction_review"
    if action == "include_source_with_scope_note_after_human_review":
        return "scope_note_review"
    if action == "include_source_after_security_review":
        return "security_review"
    return "manual_review"


def reviewer_assertion_for(review_type: str) -> str:
    if review_type == "redaction_review":
        return "approve redacted derivative; confirm private fields removed and trace-bound-only scope preserved"
    if review_type == "scope_note_review":
        return "approve scope note; confirm artifact remains source-mismatch or non-claim evidence only"
    if review_type == "security_review":
        return "approve security review; confirm no secret values and acceptable schema/field-name exposure"
    return "approve artifact-specific release evidence before any copy"


def review_row(row: dict[str, str], review_index: int) -> dict[str, Any]:
    review_type = review_type_for(row)
    return {
        "review_id": f"PSR-{review_index:03d}",
        "entry_id": row["entry_id"],
        "blocker_id": row["blocker_id"],
        "artifact_group": row["artifact_group"],
        "review_type": review_type,
        "source_path": row["bundle_source_path"],
        "source_sha256": sha256_file(row["bundle_source_path"]),
        "candidate_bundle_path": row["candidate_bundle_path"],
        "planned_supplement_path": row["planned_supplement_path"],
        "review_artifact_path": row["review_artifact_path"],
        "review_artifact_exists": path_exists(row["review_artifact_path"]),
        "required_evidence": row["required_pre_copy_evidence"],
        "reviewer_assertion_required": reviewer_assertion_for(review_type),
        "approval_status": "pending_not_performed",
        "claim_scope_guard": row["claim_scope_guard"],
    }


def write_command_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# VSG public supplement copy command plan",
        "# PLAN ONLY: do not execute without a separate reviewed route decision.",
        "# The generator did not copy files or create a candidate bundle.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"# {row['entry_id']} -> {row['planned_supplement_path']}",
                f"# expected_sha256={row['source_sha256']}",
                f"# mkdir: {row['mkdir_command']}",
                f"# copy: {row['copy_command']}",
                "# verify:",
                *[f"#   {line}" for line in row["verify_sha256_command"].splitlines()],
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(
    copy_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    output_dir: Path,
    future_copy_plan: Path,
    human_review_holds: Path,
) -> dict[str, Any]:
    review_type_counts: dict[str, int] = {}
    for row in review_rows:
        review_type_counts[row["review_type"]] = review_type_counts.get(row["review_type"], 0) + 1
    missing_copy_sources = [row for row in copy_rows if not truthy(row["source_exists"])]
    existing_targets = [row for row in copy_rows if truthy(row["target_exists_now"])]
    missing_review_artifacts = [row for row in review_rows if not truthy(row["review_artifact_exists"])]
    pending_reviews = [row for row in review_rows if row["approval_status"] != "approved"]
    return {
        "status": "PASS_VSG_PUBLIC_SUPPLEMENT_COPY_REVIEW_PLAN_RECORDED_ARTIFACT_ONLY",
        "schema_name": "verification_substrate_gap_public_supplement_copy_review_plan_v1",
        "source_future_copy_plan": display_path(future_copy_plan),
        "source_human_review_holds": display_path(human_review_holds),
        "output_dir": display_path(output_dir),
        "copy_command_count": len(copy_rows),
        "review_checklist_count": len(review_rows),
        "redaction_review_count": review_type_counts.get("redaction_review", 0),
        "scope_note_review_count": review_type_counts.get("scope_note_review", 0),
        "security_review_count": review_type_counts.get("security_review", 0),
        "missing_copy_source_count": len(missing_copy_sources),
        "existing_target_count": len(existing_targets),
        "missing_review_artifact_count": len(missing_review_artifacts),
        "pending_review_count": len(pending_reviews),
        "review_type_counts": dict(sorted(review_type_counts.items())),
        "copy_commands_written_as_comments": True,
        "all_copy_sources_present": len(missing_copy_sources) == 0,
        "all_candidate_targets_absent": len(existing_targets) == 0,
        "all_review_artifacts_present": len(missing_review_artifacts) == 0,
        "all_reviews_pending": len(pending_reviews) == len(review_rows),
        "copy_plan_only": True,
        "files_copied": False,
        "candidate_bundle_created": False,
        "human_reviews_performed": False,
        "publication_blockers_resolved": False,
        "release_ready_after_plan": False,
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


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# VSG Public Supplement Copy / Review Plan",
        "",
        "This artifact-only plan turns the bundle preflight outputs into a",
        "commented copy-command plan and a reviewer-facing checklist. It does not",
        "copy files, create a candidate bundle, perform human review, publish",
        "artifacts, start compute, or expand claim scope.",
        "",
        f"Status: `{summary['status']}`",
        f"Copy commands: `{summary['copy_command_count']}`",
        f"Review checklist rows: `{summary['review_checklist_count']}`",
        f"Redaction reviews: `{summary['redaction_review_count']}`",
        f"Scope-note reviews: `{summary['scope_note_review_count']}`",
        f"Security reviews: `{summary['security_review_count']}`",
        f"Missing copy sources: `{summary['missing_copy_source_count']}`",
        f"Existing candidate targets: `{summary['existing_target_count']}`",
        f"Missing review artifacts: `{summary['missing_review_artifact_count']}`",
        f"Pending reviews: `{summary['pending_review_count']}`",
        f"Files copied: `{summary['files_copied']}`",
        f"Candidate bundle created: `{summary['candidate_bundle_created']}`",
        f"Release-ready after plan: `{summary['release_ready_after_plan']}`",
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
            "- The copy-command plan is comments-only and requires a separate route decision before execution.",
            "- All 14 review checklist rows remain `pending_not_performed`.",
            "- No candidate supplement directory is created by this step.",
            "- Public text-only verification success and ownership-proof claims remain disallowed.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, output_files: list[Path], status: str) -> None:
    manifest = {
        "status": status,
        "schema_name": "verification_substrate_gap_public_supplement_copy_review_plan_manifest_v1",
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


def build(future_copy_plan: Path, human_review_holds: Path, output_dir: Path) -> dict[str, Any]:
    copy_rows = [copy_plan_row(row) for row in read_csv(future_copy_plan)]
    review_rows = [review_row(row, idx) for idx, row in enumerate(read_csv(human_review_holds), start=1)]
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_csv = output_dir / "copy_command_dry_run.csv"
    review_csv = output_dir / "reviewer_facing_checklist.csv"
    command_plan = output_dir / "copy_commands_plan.txt"
    summary_path = output_dir / "copy_review_plan_summary.json"
    report_path = output_dir / "copy_review_plan_report.md"
    manifest_path = output_dir / "copy_review_plan_manifest.json"

    write_csv(copy_csv, copy_rows, COPY_FIELDS)
    write_csv(review_csv, review_rows, REVIEW_FIELDS)
    write_command_plan(command_plan, copy_rows)
    summary = summarize(copy_rows, review_rows, output_dir, future_copy_plan, human_review_holds)
    write_json(summary_path, summary)
    write_report(report_path, summary)
    write_manifest(manifest_path, [copy_csv, review_csv, command_plan, summary_path, report_path], summary["status"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--future-copy-plan", default=str(DEFAULT_FUTURE_COPY_PLAN))
    parser.add_argument("--human-review-holds", default=str(DEFAULT_HUMAN_REVIEW_HOLDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = build(Path(args.future_copy_plan), Path(args.human_review_holds), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
