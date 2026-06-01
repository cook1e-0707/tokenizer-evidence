import csv
import json
from pathlib import Path

from scripts.verification_substrate_gap import build_vsg_public_supplement_copy_review_plan as plan


def test_copy_plan_row_records_commands_without_execution() -> None:
    row = {
        "entry_id": "PSP-001",
        "artifact_group": "manuscript_source",
        "bundle_source_path": "manuscripts/69db2644566dcc36c9da320e/main.tex",
        "candidate_bundle_path": "results/verification_substrate_gap/public_supplement_candidate_20260601/main.tex",
        "planned_supplement_path": "main.tex",
        "claim_scope_guard": "preserves VSG substrate-gap claim boundary",
    }

    built = plan.copy_plan_row(row)

    assert built["source_exists"] is True
    assert built["target_exists_now"] is False
    assert built["ready_for_future_copy_after_review"] is True
    assert built["execution_status"] == "not_executed_plan_only"
    assert built["copy_command"].startswith("cp -p ")
    assert "python3" in built["verify_sha256_command"]
    assert built["source_sha256"]


def test_review_row_keeps_human_review_pending() -> None:
    row = {
        "entry_id": "PSP-041",
        "blocker_id": "PSB-017",
        "artifact_group": "public_predicate_attack_ladder",
        "bundle_action": "include_source_with_scope_note_after_human_review",
        "bundle_source_path": (
            "results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/"
            "surrogate_guided_rewrite_summary.json"
        ),
        "candidate_bundle_path": "candidate/summary.json",
        "planned_supplement_path": "summary.json",
        "review_artifact_path": "results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv",
        "required_pre_copy_evidence": "human reviewer approves source with scope note",
        "claim_scope_guard": "public text-only verification success; ownership proof",
    }

    built = plan.review_row(row, 3)

    assert built["review_id"] == "PSR-003"
    assert built["review_type"] == "scope_note_review"
    assert built["review_artifact_exists"] is True
    assert built["approval_status"] == "pending_not_performed"
    assert "source-mismatch" in built["reviewer_assertion_required"]
    assert "ownership proof" in built["claim_scope_guard"]


def test_build_copy_review_plan_is_artifact_only(tmp_path: Path) -> None:
    summary = plan.build(plan.DEFAULT_FUTURE_COPY_PLAN, plan.DEFAULT_HUMAN_REVIEW_HOLDS, tmp_path)

    assert summary["status"] == "PASS_VSG_PUBLIC_SUPPLEMENT_COPY_REVIEW_PLAN_RECORDED_ARTIFACT_ONLY"
    assert summary["copy_command_count"] == 60
    assert summary["review_checklist_count"] == 14
    assert summary["redaction_review_count"] == 3
    assert summary["scope_note_review_count"] == 10
    assert summary["security_review_count"] == 1
    assert summary["missing_copy_source_count"] == 0
    assert summary["existing_target_count"] == 0
    assert summary["missing_review_artifact_count"] == 0
    assert summary["pending_review_count"] == 14
    assert summary["copy_commands_written_as_comments"] is True
    assert summary["all_copy_sources_present"] is True
    assert summary["all_candidate_targets_absent"] is True
    assert summary["all_review_artifacts_present"] is True
    assert summary["all_reviews_pending"] is True
    assert summary["copy_plan_only"] is True
    assert summary["files_copied"] is False
    assert summary["candidate_bundle_created"] is False
    assert summary["human_reviews_performed"] is False
    assert summary["publication_blockers_resolved"] is False
    assert summary["release_ready_after_plan"] is False
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

    copy_rows = list(csv.DictReader((tmp_path / "copy_command_dry_run.csv").open(newline="", encoding="utf-8")))
    review_rows = list(csv.DictReader((tmp_path / "reviewer_facing_checklist.csv").open(newline="", encoding="utf-8")))
    assert len(copy_rows) == 60
    assert len(review_rows) == 14
    assert all(row["execution_status"] == "not_executed_plan_only" for row in copy_rows)
    assert all(row["approval_status"] == "pending_not_performed" for row in review_rows)

    command_text = (tmp_path / "copy_commands_plan.txt").read_text(encoding="utf-8")
    assert "PLAN ONLY" in command_text
    assert "# copy: cp -p " in command_text
    assert not (Path("results/verification_substrate_gap/public_supplement_candidate_20260601")).exists()

    manifest = json.loads((tmp_path / "copy_review_plan_manifest.json").read_text(encoding="utf-8"))
    manifest_names = {Path(row["path"]).name for row in manifest["files"]}
    assert "copy_review_plan_manifest.json" not in manifest_names
    assert "copy_command_dry_run.csv" in manifest_names
    assert "reviewer_facing_checklist.csv" in manifest_names
    assert "copy_commands_plan.txt" in manifest_names
    assert "copy_review_plan_summary.json" in manifest_names
    assert "copy_review_plan_report.md" in manifest_names
