import csv
import json
from pathlib import Path

from scripts.verification_substrate_gap import build_vsg_public_supplement_bundle_preflight as preflight


def test_preflight_row_marks_direct_include_as_future_copy_plan() -> None:
    row = {
        "include_in_dry_run_bundle": "True",
        "artifact_group": "manuscript_source",
        "bundle_action": "direct_include_after_final_license_scope_review",
        "bundle_source_path": "manuscripts/69db2644566dcc36c9da320e/main.tex",
        "planned_supplement_path": "manuscript_source/main.tex",
        "review_artifact_path": "",
        "source_exists": "True",
        "publication_blocker": "False",
        "claim_scope_guard": "preserves VSG substrate-gap claim boundary",
    }

    built = preflight.preflight_row(row, {}, 1, "candidate")

    assert built["entry_id"] == "PSP-001"
    assert built["preflight_class"] == "direct_include_final_scope_check"
    assert built["blocker_id"] == ""
    assert built["candidate_bundle_path"] == "candidate/manuscript_source/main.tex"
    assert built["source_exists"] is True
    assert built["future_copy_plan_entry"] is True
    assert built["human_review_hold"] is False
    assert "final license and scope check" in built["required_pre_copy_evidence"]


def test_preflight_row_preserves_human_review_hold_and_blocker_id() -> None:
    row = {
        "include_in_dry_run_bundle": "True",
        "artifact_group": "public_predicate_attack_ladder",
        "bundle_action": "include_source_with_scope_note_after_human_review",
        "bundle_source_path": (
            "results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/"
            "surrogate_guided_rewrite_summary.json"
        ),
        "planned_supplement_path": "evidence/public_predicate_attack_ladder_scope_limited/summary.json",
        "review_artifact_path": "results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv",
        "source_exists": "True",
        "publication_blocker": "True",
        "claim_scope_guard": "public text-only verification success; ownership proof",
    }
    blocker = {
        "blocker_id": "PSB-012",
        "resolution_track": "human_review_required",
        "bundle_source_path": row["bundle_source_path"],
        "planned_supplement_path": row["planned_supplement_path"],
        "bundle_action": row["bundle_action"],
        "required_evidence_before_resolution": "human reviewer approves source with scope note",
    }

    built = preflight.preflight_row(row, {preflight.blocker_key(row): blocker}, 12, "candidate")

    assert built["preflight_class"] == "human_review_hold"
    assert built["blocker_id"] == "PSB-012"
    assert built["source_exists"] is True
    assert built["review_artifact_exists"] is True
    assert built["future_copy_plan_entry"] is False
    assert built["human_review_hold"] is True
    assert built["publication_blocker"] is True
    assert built["required_pre_copy_evidence"] == "human reviewer approves source with scope note"


def test_build_bundle_preflight_records_plan_without_copying(tmp_path: Path) -> None:
    summary = preflight.build(
        preflight.DEFAULT_DRY_RUN_MANIFEST,
        preflight.DEFAULT_BLOCKER_CHECKLIST,
        tmp_path,
        "candidate_root",
    )

    assert summary["status"] == "PASS_VSG_PUBLIC_SUPPLEMENT_BUNDLE_PREFLIGHT_RECORDED_ARTIFACT_ONLY"
    assert summary["row_count"] == 78
    assert summary["included_entry_count"] == 74
    assert summary["future_copy_plan_entry_count"] == 60
    assert summary["human_review_hold_count"] == 14
    assert summary["excluded_internal_record_count"] == 4
    assert summary["direct_include_final_scope_check_count"] == 39
    assert summary["copy_required_preflight_count"] == 21
    assert summary["publication_blocker_count"] == 35
    assert summary["blocker_rows_missing_ids"] == 0
    assert summary["missing_source_count"] == 0
    assert summary["missing_review_artifact_count"] == 0
    assert summary["duplicate_candidate_target_count"] == 0
    assert summary["all_included_sources_present"] is True
    assert summary["all_review_artifacts_present"] is True
    assert summary["all_publication_blockers_linked_to_checklist"] is True
    assert summary["copy_plan_created"] is True
    assert summary["human_review_holds_preserved"] is True
    assert summary["candidate_bundle_created"] is False
    assert summary["files_copied"] is False
    assert summary["human_reviews_performed"] is False
    assert summary["publication_blockers_resolved"] is False
    assert summary["release_ready_after_preflight"] is False
    assert summary["artifact_only"] is True
    assert summary["public_supplement_created"] is False
    assert summary["publication_performed"] is False
    assert summary["new_slurm_started"] is False
    assert summary["generation_started"] is False
    assert summary["model_scoring_started"] is False
    assert summary["training_started"] is False
    assert summary["allowlist_enabled"] is False
    assert summary["public_text_only_verification_claimed"] is False
    assert summary["ownership_proof_claimed"] is False

    all_rows = list(csv.DictReader((tmp_path / "bundle_construction_preflight.csv").open(newline="", encoding="utf-8")))
    copy_rows = list(csv.DictReader((tmp_path / "future_copy_plan.csv").open(newline="", encoding="utf-8")))
    hold_rows = list(csv.DictReader((tmp_path / "human_review_holds.csv").open(newline="", encoding="utf-8")))
    excluded_rows = list(csv.DictReader((tmp_path / "excluded_records.csv").open(newline="", encoding="utf-8")))
    assert len(all_rows) == 78
    assert len(copy_rows) == 60
    assert len(hold_rows) == 14
    assert len(excluded_rows) == 4
    assert all(row["candidate_bundle_path"].startswith("candidate_root/") for row in copy_rows)
    assert all(row["future_copy_plan_entry"] == "False" for row in hold_rows)

    manifest = json.loads((tmp_path / "bundle_preflight_manifest.json").read_text(encoding="utf-8"))
    manifest_names = {Path(row["path"]).name for row in manifest["files"]}
    assert "bundle_preflight_manifest.json" not in manifest_names
    assert "bundle_construction_preflight.csv" in manifest_names
    assert "future_copy_plan.csv" in manifest_names
    assert "human_review_holds.csv" in manifest_names
    assert "excluded_records.csv" in manifest_names
    assert "bundle_preflight_summary.json" in manifest_names
    assert "bundle_preflight_report.md" in manifest_names
