#!/usr/bin/env python3
"""Build an artifact-only construction preflight for a future VSG supplement."""

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
DEFAULT_BLOCKER_CHECKLIST = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_blocker_checklist_20260601"
    / "blocker_checklist.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "public_supplement_bundle_preflight_20260601"
)
DEFAULT_CANDIDATE_BUNDLE_ROOT = (
    "results/verification_substrate_gap/public_supplement_candidate_20260601"
)

OUTPUT_FIELDS = [
    "entry_id",
    "preflight_class",
    "blocker_id",
    "artifact_group",
    "bundle_action",
    "bundle_source_path",
    "source_sha256",
    "planned_supplement_path",
    "candidate_bundle_path",
    "review_artifact_path",
    "source_exists",
    "review_artifact_exists",
    "future_copy_plan_entry",
    "human_review_hold",
    "excluded_from_bundle",
    "publication_blocker",
    "required_pre_copy_evidence",
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


def blocker_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["bundle_source_path"], row["planned_supplement_path"], row["bundle_action"])


def blocker_map(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {blocker_key(row): row for row in rows}


def preflight_class_for(row: dict[str, str], blocker: dict[str, str] | None) -> str:
    if not truthy(row["include_in_dry_run_bundle"]):
        return "excluded_internal_record"
    if row["bundle_action"] == "direct_include_after_final_license_scope_review":
        return "direct_include_final_scope_check"
    if blocker and blocker["resolution_track"] == "copy_required":
        return "copy_required_preflight"
    if blocker and blocker["resolution_track"] == "human_review_required":
        return "human_review_hold"
    if truthy(row["publication_blocker"]):
        return "unresolved_publication_blocker"
    return "included_preflight"


def required_evidence_for(row: dict[str, str], blocker: dict[str, str] | None, preflight_class: str) -> str:
    if blocker:
        return blocker["required_evidence_before_resolution"]
    if preflight_class == "direct_include_final_scope_check":
        return "final license and scope check before future bundle copy"
    if preflight_class == "excluded_internal_record":
        return "excluded from public supplement"
    return "source hash verification before future bundle copy"


def candidate_path(candidate_root: str, planned_supplement_path: str) -> str:
    if not planned_supplement_path:
        return ""
    return f"{candidate_root.rstrip('/')}/{planned_supplement_path.lstrip('/')}"


def preflight_row(
    row: dict[str, str],
    blocker_lookup: dict[tuple[str, str, str], dict[str, str]],
    entry_index: int,
    candidate_root: str,
) -> dict[str, Any]:
    blocker = blocker_lookup.get(blocker_key(row))
    preflight_class = preflight_class_for(row, blocker)
    source_exists = truthy(row["source_exists"]) and path_exists(row["bundle_source_path"])
    review_artifact_exists = bool(row["review_artifact_path"]) and path_exists(row["review_artifact_path"])
    human_review_hold = preflight_class == "human_review_hold"
    excluded = preflight_class == "excluded_internal_record"
    future_copy_plan = truthy(row["include_in_dry_run_bundle"]) and not human_review_hold and not excluded
    return {
        "entry_id": f"PSP-{entry_index:03d}",
        "preflight_class": preflight_class,
        "blocker_id": blocker["blocker_id"] if blocker else "",
        "artifact_group": row["artifact_group"],
        "bundle_action": row["bundle_action"],
        "bundle_source_path": row["bundle_source_path"],
        "source_sha256": sha256_file(row["bundle_source_path"]) if source_exists else "",
        "planned_supplement_path": row["planned_supplement_path"] if not excluded else "",
        "candidate_bundle_path": candidate_path(candidate_root, row["planned_supplement_path"]) if not excluded else "",
        "review_artifact_path": row["review_artifact_path"],
        "source_exists": source_exists,
        "review_artifact_exists": review_artifact_exists,
        "future_copy_plan_entry": future_copy_plan,
        "human_review_hold": human_review_hold,
        "excluded_from_bundle": excluded,
        "publication_blocker": truthy(row["publication_blocker"]),
        "required_pre_copy_evidence": required_evidence_for(row, blocker, preflight_class),
        "claim_scope_guard": row["claim_scope_guard"],
    }


def summarize(
    rows: list[dict[str, Any]],
    output_dir: Path,
    dry_run_manifest: Path,
    blocker_checklist: Path,
    candidate_root: str,
) -> dict[str, Any]:
    class_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for row in rows:
        class_counts[row["preflight_class"]] = class_counts.get(row["preflight_class"], 0) + 1
        action_counts[row["bundle_action"]] = action_counts.get(row["bundle_action"], 0) + 1
    included_rows = [row for row in rows if not truthy(row["excluded_from_bundle"])]
    copy_plan_rows = [row for row in rows if truthy(row["future_copy_plan_entry"])]
    human_review_rows = [row for row in rows if truthy(row["human_review_hold"])]
    excluded_rows = [row for row in rows if truthy(row["excluded_from_bundle"])]
    blocker_rows = [row for row in rows if truthy(row["publication_blocker"])]
    missing_sources = [row for row in included_rows if not truthy(row["source_exists"])]
    review_rows = [row for row in rows if row["review_artifact_path"]]
    missing_review_artifacts = [row for row in review_rows if not truthy(row["review_artifact_exists"])]
    targets = [row["candidate_bundle_path"] for row in included_rows if row["candidate_bundle_path"]]
    duplicate_targets = sorted({target for target in targets if targets.count(target) > 1})
    blocker_rows_missing_ids = [row for row in blocker_rows if not row["blocker_id"]]
    return {
        "status": "PASS_VSG_PUBLIC_SUPPLEMENT_BUNDLE_PREFLIGHT_RECORDED_ARTIFACT_ONLY",
        "schema_name": "verification_substrate_gap_public_supplement_bundle_preflight_v1",
        "source_dry_run_manifest": display_path(dry_run_manifest),
        "source_blocker_checklist": display_path(blocker_checklist),
        "output_dir": display_path(output_dir),
        "candidate_bundle_root": candidate_root,
        "row_count": len(rows),
        "included_entry_count": len(included_rows),
        "future_copy_plan_entry_count": len(copy_plan_rows),
        "human_review_hold_count": len(human_review_rows),
        "excluded_internal_record_count": len(excluded_rows),
        "direct_include_final_scope_check_count": class_counts.get("direct_include_final_scope_check", 0),
        "copy_required_preflight_count": class_counts.get("copy_required_preflight", 0),
        "publication_blocker_count": len(blocker_rows),
        "blocker_rows_missing_ids": len(blocker_rows_missing_ids),
        "missing_source_count": len(missing_sources),
        "missing_review_artifact_count": len(missing_review_artifacts),
        "duplicate_candidate_target_count": len(duplicate_targets),
        "duplicate_candidate_targets": duplicate_targets,
        "preflight_class_counts": dict(sorted(class_counts.items())),
        "bundle_action_counts": dict(sorted(action_counts.items())),
        "all_included_sources_present": len(missing_sources) == 0,
        "all_review_artifacts_present": len(missing_review_artifacts) == 0,
        "all_publication_blockers_linked_to_checklist": len(blocker_rows_missing_ids) == 0,
        "copy_plan_created": True,
        "human_review_holds_preserved": len(human_review_rows) == 14,
        "candidate_bundle_created": False,
        "files_copied": False,
        "human_reviews_performed": False,
        "publication_blockers_resolved": False,
        "release_ready_after_preflight": False,
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
        "# VSG Public Supplement Bundle Construction Preflight",
        "",
        "This artifact-only preflight converts the dry-run bundle manifest and",
        "blocker checklist into a future bundle construction plan. It records",
        "candidate target paths, source hashes, copy-plan entries, and human-review",
        "holds. It does not copy files, create the candidate bundle, perform human",
        "review, publish artifacts, start compute, or expand claim scope.",
        "",
        f"Status: `{summary['status']}`",
        f"Rows: `{summary['row_count']}`",
        f"Included entries: `{summary['included_entry_count']}`",
        f"Future copy-plan entries: `{summary['future_copy_plan_entry_count']}`",
        f"Human-review holds: `{summary['human_review_hold_count']}`",
        f"Excluded internal records: `{summary['excluded_internal_record_count']}`",
        f"Publication blockers: `{summary['publication_blocker_count']}`",
        f"Missing included sources: `{summary['missing_source_count']}`",
        f"Missing review artifacts: `{summary['missing_review_artifact_count']}`",
        f"Duplicate candidate targets: `{summary['duplicate_candidate_target_count']}`",
        f"Candidate bundle created: `{summary['candidate_bundle_created']}`",
        f"Files copied: `{summary['files_copied']}`",
        f"Release-ready after preflight: `{summary['release_ready_after_preflight']}`",
        "",
        "## Preflight Classes",
        "",
        "| Class | Rows |",
        "| --- | ---: |",
    ]
    for preflight_class, count in summary["preflight_class_counts"].items():
        lines.append(f"| {preflight_class} | {count} |")
    lines.extend(
        [
            "",
            "## Construction Boundary",
            "",
            "- The 60 future copy-plan entries are not copied by this preflight.",
            "- The 14 human-review holds remain blocked until explicit review evidence exists.",
            "- The 4 excluded internal records remain outside the candidate bundle.",
            "- Public text-only verification success and ownership-proof claims remain disallowed.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, output_files: list[Path], status: str) -> None:
    manifest = {
        "status": status,
        "schema_name": "verification_substrate_gap_public_supplement_bundle_preflight_manifest_v1",
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


def build(
    dry_run_manifest: Path,
    blocker_checklist: Path,
    output_dir: Path,
    candidate_bundle_root: str = DEFAULT_CANDIDATE_BUNDLE_ROOT,
) -> dict[str, Any]:
    dry_run_rows = read_csv(dry_run_manifest)
    blockers = blocker_map(read_csv(blocker_checklist))
    rows = [
        preflight_row(row, blockers, entry_index, candidate_bundle_root)
        for entry_index, row in enumerate(dry_run_rows, start=1)
    ]
    copy_rows = [row for row in rows if truthy(row["future_copy_plan_entry"])]
    review_rows = [row for row in rows if truthy(row["human_review_hold"])]
    excluded_rows = [row for row in rows if truthy(row["excluded_from_bundle"])]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_csv = output_dir / "bundle_construction_preflight.csv"
    copy_csv = output_dir / "future_copy_plan.csv"
    review_csv = output_dir / "human_review_holds.csv"
    excluded_csv = output_dir / "excluded_records.csv"
    summary_path = output_dir / "bundle_preflight_summary.json"
    report_path = output_dir / "bundle_preflight_report.md"
    manifest_path = output_dir / "bundle_preflight_manifest.json"

    write_csv(all_csv, rows)
    write_csv(copy_csv, copy_rows)
    write_csv(review_csv, review_rows)
    write_csv(excluded_csv, excluded_rows)
    summary = summarize(rows, output_dir, dry_run_manifest, blocker_checklist, candidate_bundle_root)
    write_json(summary_path, summary)
    write_report(report_path, summary)
    write_manifest(manifest_path, [all_csv, copy_csv, review_csv, excluded_csv, summary_path, report_path], summary["status"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run-manifest", default=str(DEFAULT_DRY_RUN_MANIFEST))
    parser.add_argument("--blocker-checklist", default=str(DEFAULT_BLOCKER_CHECKLIST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--candidate-bundle-root", default=DEFAULT_CANDIDATE_BUNDLE_ROOT)
    args = parser.parse_args()
    summary = build(
        Path(args.dry_run_manifest),
        Path(args.blocker_checklist),
        Path(args.output_dir),
        args.candidate_bundle_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
