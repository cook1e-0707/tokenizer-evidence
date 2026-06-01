import csv
import json
from pathlib import Path

from scripts.verification_substrate_gap import build_vsg_public_supplement_review_decision_template as template


def test_decision_template_row_requires_pending_review_fields() -> None:
    row = {
        "review_id": "PSR-005",
        "review_type": "scope_note_review",
        "entry_id": "PSP-041",
        "blocker_id": "PSB-017",
        "artifact_group": "public_predicate_attack_ladder",
        "source_path": "source.csv",
        "source_sha256": "abc",
        "review_artifact_path": "scope_notes.csv",
        "review_artifact_sha256": "def",
        "planned_supplement_path": "evidence/attack.csv",
        "candidate_bundle_path": "candidate/evidence/attack.csv",
        "required_evidence": "human reviewer approves source with scope note",
        "reviewer_assertion_required": "approve scope note",
        "claim_scope_guard": "public text-only verification success; ownership proof",
    }

    built = template.decision_template_row(row)

    assert built["review_id"] == "PSR-005"
    assert built["allowed_decisions"] == "approved;rejected;hold"
    assert built["decision_status"] == "pending_not_performed"
    assert built["reviewer_id"] == ""
    assert built["reviewed_at_utc"] == ""
    assert built["source_sha256_verified"] == ""
    assert built["review_artifact_sha256_verified"] == ""
    assert built["reviewer_assertion_confirmed"] == ""
    assert built["claim_scope_guard_preserved"] == ""
    assert built["approval_gate"] == "not_approved_template_only"
    assert "forbidden_claims_not_preserved" in built["failure_condition"]


def test_schema_requires_truthy_fields_for_approval() -> None:
    schema = template.schema()

    assert schema["template_only"] is True
    assert schema["review_approvals_recorded"] is False
    assert schema["human_reviews_performed"] is False
    assert "approved" in schema["allowed_decisions"]
    assert "reviewer_id" in schema["required_fields_for_any_non_pending_decision"]
    assert "claim_scope_guard_preserved" in schema["required_truthy_fields_for_approved"]
    assert "ownership proof" in schema["approval_must_not_claim"]
    assert "scope_note_review" in schema["review_type_specific_required_assertions"]


def test_build_review_decision_template_records_no_approvals(tmp_path: Path) -> None:
    summary = template.build(template.DEFAULT_PACKET_INDEX, tmp_path)

    assert summary["status"] == "PASS_VSG_PUBLIC_SUPPLEMENT_REVIEW_DECISION_TEMPLATE_RECORDED_PENDING_ONLY"
    assert summary["decision_template_row_count"] == 14
    assert summary["pending_decision_count"] == 14
    assert summary["approved_decision_count"] == 0
    assert summary["redaction_review_count"] == 3
    assert summary["scope_note_review_count"] == 10
    assert summary["security_review_count"] == 1
    assert summary["empty_reviewer_id_count"] == 14
    assert summary["empty_reviewed_at_utc_count"] == 14
    assert summary["all_decisions_pending"] is True
    assert summary["schema_written"] is True
    assert summary["decision_records_template_written"] is True
    assert summary["review_approvals_recorded"] is False
    assert summary["human_reviews_performed"] is False
    assert summary["publication_blockers_resolved"] is False
    assert summary["release_ready_after_template"] is False
    assert summary["artifact_only"] is True
    assert summary["files_copied"] is False
    assert summary["candidate_bundle_created"] is False
    assert summary["public_supplement_created"] is False
    assert summary["publication_performed"] is False
    assert summary["new_slurm_started"] is False
    assert summary["generation_started"] is False
    assert summary["model_scoring_started"] is False
    assert summary["training_started"] is False
    assert summary["allowlist_enabled"] is False
    assert summary["public_text_only_verification_claimed"] is False
    assert summary["ownership_proof_claimed"] is False

    rows = list(csv.DictReader((tmp_path / "review_decision_template.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == 14
    assert all(row["decision_status"] == "pending_not_performed" for row in rows)
    assert all(row["approval_gate"] == "not_approved_template_only" for row in rows)
    assert all(not row["reviewer_id"] for row in rows)
    assert all(not row["reviewed_at_utc"] for row in rows)

    schema = json.loads((tmp_path / "review_decision_schema.json").read_text(encoding="utf-8"))
    assert schema["record_type"] == "human_review_decision_record"

    manifest = json.loads((tmp_path / "review_decision_template_manifest.json").read_text(encoding="utf-8"))
    manifest_names = {Path(row["path"]).name for row in manifest["files"]}
    assert "review_decision_template_manifest.json" not in manifest_names
    assert "review_decision_template.csv" in manifest_names
    assert "review_decision_schema.json" in manifest_names
    assert "review_decision_template_summary.json" in manifest_names
    assert "review_decision_template_report.md" in manifest_names
