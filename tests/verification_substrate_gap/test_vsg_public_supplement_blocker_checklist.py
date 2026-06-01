import csv
import json
from pathlib import Path

from scripts.verification_substrate_gap import build_vsg_public_supplement_blocker_checklist as checklist


def test_checklist_row_classifies_copy_required_blocker() -> None:
    row = {
        "publication_blocker": "True",
        "manual_review_required": "False",
        "artifact_group": "figure_data",
        "bundle_action": "copy_source_to_bundle_after_review",
        "bundle_source_path": "results/verification_substrate_gap/paper_figure_data_20260530/trace_bound_accepts.csv",
        "planned_supplement_path": "evidence/figure_data/trace_bound_accepts.csv",
        "review_artifact_path": "",
        "source_exists": "True",
        "claim_scope_guard": "preserves VSG substrate-gap claim boundary",
    }

    built = checklist.checklist_row(row, 1)

    assert built["blocker_id"] == "PSB-001"
    assert built["resolution_track"] == "copy_required"
    assert built["source_exists"] is True
    assert built["review_artifact_exists"] is False
    assert "hash verified" in built["required_evidence_before_resolution"]
    assert built["resolution_gate"] == "not_resolved_artifact_only_checklist"


def test_checklist_row_classifies_scope_note_review_blocker() -> None:
    row = {
        "publication_blocker": "True",
        "manual_review_required": "True",
        "artifact_group": "public_predicate_attack_ladder",
        "bundle_action": "include_source_with_scope_note_after_human_review",
        "bundle_source_path": (
            "results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/"
            "surrogate_guided_rewrite_summary.json"
        ),
        "planned_supplement_path": "evidence/public_predicate_attack_ladder_scope_limited/summary.json",
        "review_artifact_path": "results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv",
        "source_exists": "True",
        "claim_scope_guard": "public text-only verification success; ownership proof",
    }

    built = checklist.checklist_row(row, 12)

    assert built["blocker_id"] == "PSB-012"
    assert built["resolution_track"] == "human_review_required"
    assert built["source_exists"] is True
    assert built["review_artifact_exists"] is True
    assert "scope note" in built["required_evidence_before_resolution"]
    assert "ownership proof" in built["claim_scope_guard"]


def test_build_blocker_checklist_splits_remaining_blockers(tmp_path: Path) -> None:
    summary = checklist.build(checklist.DEFAULT_DRY_RUN_MANIFEST, tmp_path)

    assert summary["status"] == "PASS_VSG_PUBLIC_SUPPLEMENT_BLOCKER_CHECKLIST_RECORDED_ARTIFACT_ONLY"
    assert summary["publication_blocker_count"] == 35
    assert summary["copy_required_count"] == 21
    assert summary["human_review_required_count"] == 14
    assert summary["missing_source_count"] == 0
    assert summary["missing_review_artifact_count"] == 0
    assert summary["unclassified_blocker_count"] == 0
    assert summary["all_blockers_have_resolution_track"] is True
    assert summary["all_sources_present"] is True
    assert summary["all_review_artifacts_present"] is True
    assert summary["blockers_resolved"] is False
    assert summary["release_ready_after_checklist"] is False
    assert summary["artifact_only"] is True
    assert summary["files_copied"] is False
    assert summary["human_reviews_performed"] is False
    assert summary["public_supplement_created"] is False
    assert summary["publication_performed"] is False
    assert summary["new_slurm_started"] is False
    assert summary["generation_started"] is False
    assert summary["model_scoring_started"] is False
    assert summary["training_started"] is False
    assert summary["allowlist_enabled"] is False
    assert summary["public_text_only_verification_claimed"] is False
    assert summary["ownership_proof_claimed"] is False

    all_rows = list(csv.DictReader((tmp_path / "blocker_checklist.csv").open(newline="", encoding="utf-8")))
    copy_rows = list(csv.DictReader((tmp_path / "copy_required_checklist.csv").open(newline="", encoding="utf-8")))
    review_rows = list(csv.DictReader((tmp_path / "human_review_checklist.csv").open(newline="", encoding="utf-8")))
    assert len(all_rows) == 35
    assert len(copy_rows) == 21
    assert len(review_rows) == 14
    assert {row["resolution_track"] for row in all_rows} == {"copy_required", "human_review_required"}

    manifest = json.loads((tmp_path / "blocker_checklist_manifest.json").read_text(encoding="utf-8"))
    manifest_names = {Path(row["path"]).name for row in manifest["files"]}
    assert "blocker_checklist_manifest.json" not in manifest_names
    assert "blocker_checklist.csv" in manifest_names
    assert "copy_required_checklist.csv" in manifest_names
    assert "human_review_checklist.csv" in manifest_names
    assert "blocker_checklist_summary.json" in manifest_names
    assert "blocker_checklist_report.md" in manifest_names
