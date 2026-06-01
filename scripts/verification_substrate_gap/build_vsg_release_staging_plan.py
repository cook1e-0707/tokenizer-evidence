#!/usr/bin/env python3
"""Build an artifact-only staging/redaction plan for a future VSG supplement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOUNDARY = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "reproducibility_release_boundary_audit_20260601"
    / "release_boundary_decisions.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "reproducibility_release_staging_plan_20260601"
)

OUTPUT_FIELDS = [
    "artifact_group",
    "source_path",
    "boundary_decision",
    "staging_decision",
    "include_in_public_supplement_plan",
    "planned_supplement_path",
    "planned_transform",
    "execution_required",
    "manual_review_required",
    "residual_risk",
    "release_path_reason",
    "release_claim_scope_note",
]


GROUP_ROOTS = {
    "manuscript_source": "manuscript_source",
    "manuscript_figures": "figures",
    "figure_data": "evidence/figure_data",
    "trace_bound_corpus_summary": "evidence/trace_bound_controllability",
    "public_text_verifier_baselines": "evidence/public_text_verifier_baselines",
    "public_predicate_attack_ladder": "evidence/public_predicate_attack_ladder_scope_limited",
    "attack_naturalness_proxy_audit": "evidence/attack_naturalness_proxy",
    "stronger_public_predicate_local_pilot": "evidence/local_pilots/stronger_public_predicate",
    "substrate_matrix": "evidence/substrate_matrix",
    "ownership_stress_test": "evidence/ownership_stress_test",
    "reproducibility_code": "code",
    "reproducibility_config": "configs",
    "reproducibility_tests": "tests",
}


def truthy(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def read_boundary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def target_name(source_path: str) -> str:
    return Path(source_path).name


def planned_path(row: dict[str, str]) -> str:
    group = row["artifact_group"]
    if group == "state_and_scope_records":
        return ""
    source = Path(row["path"])
    if group == "manuscript_source":
        marker = Path("manuscripts") / "69db2644566dcc36c9da320e"
        parts = source.parts
        marker_parts = marker.parts
        for idx in range(0, len(parts) - len(marker_parts) + 1):
            if parts[idx : idx + len(marker_parts)] == marker_parts:
                suffix = Path(*parts[idx + len(marker_parts) :]).as_posix()
                return f"manuscript_source/{suffix}"
    if group == "trace_bound_corpus_summary" and row["boundary_decision"] == "redact_or_summarize_before_release":
        return f"evidence/trace_bound_controllability_redacted/{target_name(row['path'])}"
    root = GROUP_ROOTS.get(group, f"evidence/{group}")
    return f"{root}/{target_name(row['path'])}"


def transform_for(row: dict[str, str]) -> str:
    decision = row["boundary_decision"]
    path = row["path"]
    if decision == "redact_or_summarize_before_release":
        if path.endswith(".csv"):
            return "derive_redacted_csv_drop_source_shard_dir_and_private_path_fields"
        return "derive_redacted_summary_drop_private_path_fields"
    if decision == "security_review_before_release":
        return "review_config_field_names_and_confirm_no_secret_values_before_copy"
    if decision == "scope_review_before_release":
        return "attach_scope_note_or_replace_with_derived_summary_before_copy"
    if decision == "stage_or_copy_to_supplement_before_release":
        return "copy_or_commit_exact_file_after_license_scope_check"
    if decision == "ready_for_reviewed_public_supplement":
        return "copy_exact_file_after_final_license_scope_check"
    if decision == "exclude_from_public_supplement":
        return "do_not_stage_internal_handoff_record"
    return "unclassified_manual_review"


def staging_decision_for(row: dict[str, str]) -> str:
    decision = row["boundary_decision"]
    if decision == "ready_for_reviewed_public_supplement":
        return "direct_include_candidate"
    if decision == "stage_or_copy_to_supplement_before_release":
        return "stage_or_copy_candidate"
    if decision == "redact_or_summarize_before_release":
        return "redacted_derivative_candidate"
    if decision == "security_review_before_release":
        return "security_review_gated_candidate"
    if decision == "scope_review_before_release":
        return "scope_note_gated_candidate"
    if decision == "exclude_from_public_supplement":
        return "excluded_internal_record"
    return "manual_review"


def claim_scope_note_for(row: dict[str, str]) -> str:
    group = row["artifact_group"]
    if group == "public_predicate_attack_ladder":
        return "source-mismatch spoofing evidence only; not protected success and not codeword recovery"
    if group == "stronger_public_predicate_local_pilot":
        return "local non-adopted/historical pilot only; not adopted locked evidence"
    if group == "trace_bound_corpus_summary":
        return "provider-side trace-bound diagnostic summary only; not public text-only verification"
    if group == "state_and_scope_records":
        return "internal handoff/scope record excluded from public supplement"
    return "preserves VSG substrate-gap claim boundary"


def residual_risk_for(row: dict[str, str]) -> str:
    decision = row["boundary_decision"]
    group = row["artifact_group"]
    if decision == "ready_for_reviewed_public_supplement":
        return "final license and scope check still required before publication"
    if decision == "stage_or_copy_to_supplement_before_release":
        return "file is not yet in a reviewed public supplement bundle"
    if decision == "redact_or_summarize_before_release":
        return "private path fields must be removed or summarized before release"
    if decision == "security_review_before_release":
        return "key/HMAC-related field names require human security review"
    if group == "public_predicate_attack_ladder":
        return "source-mismatch examples need scope labeling and may require derived summaries"
    if group == "stronger_public_predicate_local_pilot":
        return "local pilot is not adopted locked evidence and needs a non-claim scope note"
    if decision == "scope_review_before_release":
        return "scope note or derived summary must be reviewed before inclusion"
    if decision == "exclude_from_public_supplement":
        return "internal handoff record is intentionally outside public supplement scope"
    return "manual review required"


def release_path_reason_for(row: dict[str, str]) -> str:
    group = row["artifact_group"]
    decision = row["boundary_decision"]
    if group == "manuscript_source":
        return "preserve active LaTeX source tree under manuscript_source"
    if group == "manuscript_figures":
        return "place rendered manuscript figures under figures"
    if group == "figure_data":
        return "place figure/table inputs under evidence/figure_data"
    if group == "trace_bound_corpus_summary" and decision == "redact_or_summarize_before_release":
        return "place only redacted trace-bound derivatives under trace_bound_controllability_redacted"
    if group == "trace_bound_corpus_summary":
        return "place trace-bound diagnostic summaries under trace_bound_controllability after scope review"
    if group == "public_predicate_attack_ladder":
        return "place attack ladder artifacts in a scope-limited source-mismatch folder"
    if group == "stronger_public_predicate_local_pilot":
        return "place local pilot artifacts outside adopted-locked evidence folders"
    if group == "state_and_scope_records":
        return "exclude internal handoff/scope records from public supplement"
    return f"place {group} artifacts under the mapped supplement group"


def plan_row(row: dict[str, str]) -> dict[str, Any]:
    decision = row["boundary_decision"]
    excluded = decision == "exclude_from_public_supplement"
    execution_required = decision not in {"ready_for_reviewed_public_supplement", "exclude_from_public_supplement"}
    manual_review_required = decision in {
        "redact_or_summarize_before_release",
        "security_review_before_release",
        "scope_review_before_release",
    }
    return {
        "artifact_group": row["artifact_group"],
        "source_path": row["path"],
        "boundary_decision": decision,
        "staging_decision": staging_decision_for(row),
        "include_in_public_supplement_plan": not excluded,
        "planned_supplement_path": "" if excluded else planned_path(row),
        "planned_transform": transform_for(row),
        "execution_required": execution_required,
        "manual_review_required": manual_review_required,
        "residual_risk": residual_risk_for(row),
        "release_path_reason": release_path_reason_for(row),
        "release_claim_scope_note": claim_scope_note_for(row),
    }


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


def summarize(rows: list[dict[str, Any]], output_dir: Path, source_boundary: Path) -> dict[str, Any]:
    staging_counts: dict[str, int] = {}
    transform_counts: dict[str, int] = {}
    target_paths = [row["planned_supplement_path"] for row in rows if row["planned_supplement_path"]]
    duplicate_targets = sorted({path for path in target_paths if target_paths.count(path) > 1})
    for row in rows:
        staging_counts[row["staging_decision"]] = staging_counts.get(row["staging_decision"], 0) + 1
        transform_counts[row["planned_transform"]] = transform_counts.get(row["planned_transform"], 0) + 1
    execution_required = [row for row in rows if truthy(row["execution_required"])]
    manual_review_required = [row for row in rows if truthy(row["manual_review_required"])]
    direct_include = [row for row in rows if row["staging_decision"] == "direct_include_candidate"]
    stage_or_copy = [row for row in rows if row["staging_decision"] == "stage_or_copy_candidate"]
    redaction = [row for row in rows if row["staging_decision"] == "redacted_derivative_candidate"]
    scope_note = [row for row in rows if row["staging_decision"] == "scope_note_gated_candidate"]
    security_review = [row for row in rows if row["staging_decision"] == "security_review_gated_candidate"]
    excluded = [row for row in rows if row["staging_decision"] == "excluded_internal_record"]
    return {
        "status": "PASS_VSG_RELEASE_STAGING_PLAN_RECORDED_PLAN_ONLY",
        "schema_name": "verification_substrate_gap_release_staging_plan_v1",
        "source_boundary_decisions": display_path(source_boundary),
        "output_dir": display_path(output_dir),
        "row_count": len(rows),
        "direct_include_candidate_count": len(direct_include),
        "stage_or_copy_candidate_count": len(stage_or_copy),
        "redacted_derivative_candidate_count": len(redaction),
        "scope_note_gated_candidate_count": len(scope_note),
        "security_review_gated_candidate_count": len(security_review),
        "excluded_internal_record_count": len(excluded),
        "execution_required_count": len(execution_required),
        "manual_review_required_count": len(manual_review_required),
        "duplicate_planned_target_count": len(duplicate_targets),
        "duplicate_planned_targets": duplicate_targets,
        "staging_decision_counts": dict(sorted(staging_counts.items())),
        "planned_transform_counts": dict(sorted(transform_counts.items())),
        "plan_only": True,
        "files_copied": False,
        "public_supplement_copy_performed": False,
        "public_supplement_created": False,
        "publication_performed": False,
        "release_ready_after_plan": False,
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
        "# VSG Release Staging Plan",
        "",
        "This artifact-only plan maps release-boundary decisions to a future",
        "public-supplement layout. It does not copy files, create a supplement",
        "bundle, publish artifacts, start compute, or expand claim scope.",
        "",
        f"Status: `{summary['status']}`",
        f"Rows: `{summary['row_count']}`",
        f"Direct include candidates: `{summary['direct_include_candidate_count']}`",
        f"Stage/copy candidates: `{summary['stage_or_copy_candidate_count']}`",
        f"Redacted derivative candidates: `{summary['redacted_derivative_candidate_count']}`",
        f"Scope-note gated candidates: `{summary['scope_note_gated_candidate_count']}`",
        f"Security-review gated candidates: `{summary['security_review_gated_candidate_count']}`",
        f"Execution-required rows: `{summary['execution_required_count']}`",
        f"Manual-review-required rows: `{summary['manual_review_required_count']}`",
        f"Excluded internal records: `{summary['excluded_internal_record_count']}`",
        f"Duplicate planned targets: `{summary['duplicate_planned_target_count']}`",
        f"Release-ready after this plan: `{summary['release_ready_after_plan']}`",
        "",
        "## Staging Decisions",
        "",
        "| Staging decision | Rows |",
        "| --- | ---: |",
    ]
    for decision, count in summary["staging_decision_counts"].items():
        lines.append(f"| {decision} | {count} |")
    lines.extend(
        [
            "",
            "## Required Before Bundle Construction",
            "",
            "- Execute copy/commit decisions for stage-or-copy candidates.",
            "- Create redacted derivatives for private-path trace summaries.",
            "- Attach scope notes or derived summaries for source-mismatch and local-pilot artifacts.",
            "- Complete security review of key/HMAC-related configuration field names.",
            "- Keep internal handoff and state records outside the public supplement.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(boundary: Path, output_dir: Path) -> dict[str, Any]:
    rows = [plan_row(row) for row in read_boundary(boundary)]
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_csv = output_dir / "release_staging_plan.csv"
    summary_path = output_dir / "release_staging_summary.json"
    report_path = output_dir / "release_staging_report.md"
    manifest_path = output_dir / "release_staging_manifest.json"
    write_csv(plan_csv, rows)
    summary = summarize(rows, output_dir, boundary)
    write_json(summary_path, summary)
    write_report(report_path, summary)
    manifest = {
        "status": summary["status"],
        "schema_name": "verification_substrate_gap_release_staging_manifest_v1",
        "files": [
            {
                "path": display_path(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in [plan_csv, summary_path, report_path]
        ],
        "manifest_self_hash_excluded": True,
    }
    write_json(manifest_path, manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", default=str(DEFAULT_BOUNDARY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = build(Path(args.boundary), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
