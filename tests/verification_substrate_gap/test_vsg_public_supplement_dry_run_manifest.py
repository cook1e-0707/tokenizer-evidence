import csv
import json
from pathlib import Path

from scripts.verification_substrate_gap import build_vsg_public_supplement_dry_run_manifest as manifest


def test_manifest_row_uses_redacted_derivative_as_bundle_source() -> None:
    row = {
        "artifact_group": "trace_bound_corpus_summary",
        "readiness_decision": "redacted_derivative_available_manual_review_required",
        "source_path": "private/source.csv",
        "derivative_path": (
            "results/verification_substrate_gap/public_supplement_review_derivatives_20260601/"
            "evidence/trace_bound_controllability_redacted/qwen_blocks.csv"
        ),
        "planned_supplement_path": "evidence/trace_bound_controllability_redacted/qwen_blocks.csv",
        "manual_review_required": "True",
        "publication_blocker": "True",
        "remaining_action": "human review redacted derivative before bundle inclusion",
        "claim_scope_guard": "provider-side trace-bound diagnostic summary only",
    }

    built = manifest.manifest_row(row)

    assert built["bundle_action"] == "use_redacted_derivative_after_human_review"
    assert built["bundle_source_role"] == "redacted_derivative"
    assert built["bundle_source_path"] == row["derivative_path"]
    assert built["source_exists"] is True
    assert built["manual_review_required"] is True
    assert built["publication_blocker"] is True


def test_manifest_row_attaches_scope_note_without_replacing_source() -> None:
    row = {
        "artifact_group": "public_predicate_attack_ladder",
        "readiness_decision": "scope_note_available_manual_review_required",
        "source_path": (
            "results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/"
            "surrogate_guided_rewrite_summary.json"
        ),
        "derivative_path": "results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv",
        "planned_supplement_path": "evidence/public_predicate_attack_ladder_scope_limited/summary.json",
        "manual_review_required": "True",
        "publication_blocker": "True",
        "remaining_action": "human review scope note before bundle inclusion",
        "claim_scope_guard": "not protected success",
    }

    built = manifest.manifest_row(row)

    assert built["bundle_action"] == "include_source_with_scope_note_after_human_review"
    assert built["bundle_source_role"] == "original_source"
    assert built["bundle_source_path"] == row["source_path"]
    assert built["review_artifact_path"] == row["derivative_path"]
    assert built["review_artifact_exists"] is True
    assert built["publication_blocker"] is True


def test_manifest_row_missing_source_stays_blocked() -> None:
    row = {
        "artifact_group": "figure_data",
        "readiness_decision": "copy_or_commit_required_before_supplement_bundle",
        "source_path": "results/verification_substrate_gap/does_not_exist.csv",
        "derivative_path": "",
        "planned_supplement_path": "evidence/figure_data/does_not_exist.csv",
        "manual_review_required": "False",
        "publication_blocker": "False",
        "remaining_action": "copy into reviewed supplement bundle",
        "claim_scope_guard": "preserves VSG substrate-gap claim boundary",
    }

    built = manifest.manifest_row(row)

    assert built["source_exists"] is False
    assert built["publication_blocker"] is True
    assert "resolve missing dry-run source" in built["remaining_action"]


def test_build_dry_run_manifest_records_bundle_blockers_without_copying(tmp_path: Path) -> None:
    summary = manifest.build(manifest.DEFAULT_READINESS, tmp_path)

    assert summary["status"] == "PASS_VSG_PUBLIC_SUPPLEMENT_DRY_RUN_MANIFEST_RECORDED_NOT_RELEASE_READY"
    assert summary["row_count"] == 78
    assert summary["dry_run_bundle_entry_count"] == 74
    assert summary["excluded_internal_record_count"] == 4
    assert summary["direct_include_entry_count"] == 39
    assert summary["copy_required_entry_count"] == 21
    assert summary["redacted_derivative_entry_count"] == 3
    assert summary["scope_note_review_entry_count"] == 10
    assert summary["security_review_entry_count"] == 1
    assert summary["manual_review_required_count"] == 14
    assert summary["publication_blocker_count"] == 35
    assert summary["missing_source_count"] == 0
    assert summary["missing_review_artifact_count"] == 0
    assert summary["duplicate_planned_target_count"] == 0
    assert summary["release_ready_after_dry_run"] is False
    assert summary["dry_run_only"] is True
    assert summary["files_copied"] is False
    assert summary["public_supplement_created"] is False
    assert summary["publication_performed"] is False
    assert summary["new_slurm_started"] is False
    assert summary["generation_started"] is False
    assert summary["model_scoring_started"] is False
    assert summary["training_started"] is False
    assert summary["allowlist_enabled"] is False
    assert summary["public_text_only_verification_claimed"] is False
    assert summary["ownership_proof_claimed"] is False

    csv_path = tmp_path / "dry_run_bundle_manifest.csv"
    assert csv_path.is_file()
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 78
    assert {row["bundle_action"] for row in rows} == {
        "copy_source_to_bundle_after_review",
        "direct_include_after_final_license_scope_review",
        "exclude_internal_record",
        "include_source_after_security_review",
        "include_source_with_scope_note_after_human_review",
        "use_redacted_derivative_after_human_review",
    }
    included = [row for row in rows if row["include_in_dry_run_bundle"] == "True"]
    assert len(included) == 74
    assert all(row["source_exists"] == "True" for row in included)

    file_manifest = json.loads((tmp_path / "dry_run_bundle_file_manifest.json").read_text(encoding="utf-8"))
    manifest_names = {Path(row["path"]).name for row in file_manifest["files"]}
    assert "dry_run_bundle_file_manifest.json" not in manifest_names
    assert "dry_run_bundle_manifest.csv" in manifest_names
    assert "dry_run_bundle_summary.json" in manifest_names
    assert "dry_run_bundle_report.md" in manifest_names
