#!/usr/bin/env python3
"""Build artifact-only checklists for unresolved VSG supplement blockers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRY_RUN_MANIFEST = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_dry_run_manifest_20260601"
    / "dry_run_bundle_manifest.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_blocker_checklist_20260601"
)

OUTPUT_FIELDS = [
    "blocker_id",
    "resolution_track",
    "artifact_group",
    "bundle_action",
    "bundle_source_path",
    "planned_supplement_path",
    "review_artifact_path",
    "source_exists",
    "source_sha256",
    "review_artifact_exists",
    "required_evidence_before_resolution",
    "resolution_gate",
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


def resolution_track_for(row: dict[str, str]) -> str:
    if row["bundle_action"] == "copy_source_to_bundle_after_review":
        return "copy_required"
    if truthy(row["manual_review_required"]):
        return "human_review_required"
    return "unclassified_blocker"


def evidence_for(row: dict[str, str], track: str) -> str:
    action = row["bundle_action"]
    if track == "copy_required":
        return "source file copied to planned supplement path and hash verified against dry-run source"
    if action == "use_redacted_derivative_after_human_review":
        return "human reviewer confirms redacted derivative removes private fields and preserves trace-bound-only claim scope"
    if action == "include_source_with_scope_note_after_human_review":
        return "human reviewer approves source with scope note and confirms source-mismatch/non-claim wording"
    if action == "include_source_after_security_review":
        return "human reviewer confirms security review has no secret values and field names are acceptable for release"
    return "manual resolution evidence required"


def checklist_row(row: dict[str, str], blocker_index: int) -> dict[str, Any]:
    track = resolution_track_for(row)
    source_exists = truthy(row["source_exists"]) and path_exists(row["bundle_source_path"])
    review_exists = bool(row["review_artifact_path"]) and path_exists(row["review_artifact_path"])
    return {
        "blocker_id": f"PSB-{blocker_index:03d}",
        "resolution_track": track,
        "artifact_group": row["artifact_group"],
        "bundle_action": row["bundle_action"],
        "bundle_source_path": row["bundle_source_path"],
        "planned_supplement_path": row["planned_supplement_path"],
        "review_artifact_path": row["review_artifact_path"],
        "source_exists": source_exists,
        "source_sha256": sha256_file(row["bundle_source_path"]) if source_exists else "",
        "review_artifact_exists": review_exists,
        "required_evidence_before_resolution": evidence_for(row, track),
        "resolution_gate": "not_resolved_artifact_only_checklist",
        "claim_scope_guard": row["claim_scope_guard"],
    }


def summarize(
    rows: list[dict[str, Any]],
    copy_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    output_dir: Path,
    dry_run_manifest: Path,
) -> dict[str, Any]:
    track_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for row in rows:
        track_counts[row["resolution_track"]] = track_counts.get(row["resolution_track"], 0) + 1
        action_counts[row["bundle_action"]] = action_counts.get(row["bundle_action"], 0) + 1
    missing_sources = [row for row in rows if not truthy(row["source_exists"])]
    review_artifact_rows = [row for row in rows if row["review_artifact_path"]]
    missing_review_artifacts = [row for row in review_artifact_rows if not truthy(row["review_artifact_exists"])]
    return {
        "status": "PASS_VSG_PUBLIC_SUPPLEMENT_BLOCKER_CHECKLIST_RECORDED_ARTIFACT_ONLY",
        "schema_name": "verification_substrate_gap_public_supplement_blocker_checklist_v1",
        "source_dry_run_manifest": display_path(dry_run_manifest),
        "output_dir": display_path(output_dir),
        "publication_blocker_count": len(rows),
        "copy_required_count": len(copy_rows),
        "human_review_required_count": len(review_rows),
        "missing_source_count": len(missing_sources),
        "missing_review_artifact_count": len(missing_review_artifacts),
        "unclassified_blocker_count": track_counts.get("unclassified_blocker", 0),
        "resolution_track_counts": dict(sorted(track_counts.items())),
        "bundle_action_counts": dict(sorted(action_counts.items())),
        "all_blockers_have_resolution_track": track_counts.get("unclassified_blocker", 0) == 0,
        "all_sources_present": len(missing_sources) == 0,
        "all_review_artifacts_present": len(missing_review_artifacts) == 0,
        "blockers_resolved": False,
        "release_ready_after_checklist": False,
        "artifact_only": True,
        "files_copied": False,
        "human_reviews_performed": False,
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
        "# VSG Public Supplement Blocker Checklist",
        "",
        "This artifact-only checklist splits the dry-run bundle blockers into",
        "copy-required rows and human-review-required rows. It records the evidence",
        "needed to resolve each blocker later, but does not copy files, perform",
        "reviews, create a public supplement, publish artifacts, start compute, or",
        "expand claim scope.",
        "",
        f"Status: `{summary['status']}`",
        f"Publication blockers: `{summary['publication_blocker_count']}`",
        f"Copy-required rows: `{summary['copy_required_count']}`",
        f"Human-review-required rows: `{summary['human_review_required_count']}`",
        f"Missing sources: `{summary['missing_source_count']}`",
        f"Missing review artifacts: `{summary['missing_review_artifact_count']}`",
        f"Unclassified blockers: `{summary['unclassified_blocker_count']}`",
        f"All blockers have resolution track: `{summary['all_blockers_have_resolution_track']}`",
        f"Release-ready after checklist: `{summary['release_ready_after_checklist']}`",
        "",
        "## Resolution Tracks",
        "",
        "| Track | Rows |",
        "| --- | ---: |",
    ]
    for track, count in summary["resolution_track_counts"].items():
        lines.append(f"| {track} | {count} |")
    lines.extend(
        [
            "",
            "## Required Before These Blockers Can Close",
            "",
            "- Copy-required rows need reviewed bundle placement and hash verification.",
            "- Human-review rows need explicit approval of redaction, scope note, or security review evidence.",
            "- The checklist itself does not resolve any blocker.",
            "- Public text-only verification success and ownership-proof claims remain disallowed.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, output_files: list[Path], status: str) -> None:
    manifest = {
        "status": status,
        "schema_name": "verification_substrate_gap_public_supplement_blocker_checklist_manifest_v1",
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


def build(dry_run_manifest: Path, output_dir: Path) -> dict[str, Any]:
    blocker_source_rows = [row for row in read_csv(dry_run_manifest) if truthy(row["publication_blocker"])]
    rows = [checklist_row(row, idx) for idx, row in enumerate(blocker_source_rows, start=1)]
    copy_rows = [row for row in rows if row["resolution_track"] == "copy_required"]
    review_rows = [row for row in rows if row["resolution_track"] == "human_review_required"]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_csv = output_dir / "blocker_checklist.csv"
    copy_csv = output_dir / "copy_required_checklist.csv"
    review_csv = output_dir / "human_review_checklist.csv"
    summary_path = output_dir / "blocker_checklist_summary.json"
    report_path = output_dir / "blocker_checklist_report.md"
    manifest_path = output_dir / "blocker_checklist_manifest.json"

    write_csv(all_csv, rows)
    write_csv(copy_csv, copy_rows)
    write_csv(review_csv, review_rows)
    summary = summarize(rows, copy_rows, review_rows, output_dir, dry_run_manifest)
    write_json(summary_path, summary)
    write_report(report_path, summary)
    write_manifest(manifest_path, [all_csv, copy_csv, review_csv, summary_path, report_path], summary["status"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run-manifest", default=str(DEFAULT_DRY_RUN_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = build(Path(args.dry_run_manifest), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
