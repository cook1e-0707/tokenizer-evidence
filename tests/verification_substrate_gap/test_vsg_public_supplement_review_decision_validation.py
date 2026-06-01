import csv
import hashlib
import json
from pathlib import Path

from scripts.verification_substrate_gap import validate_vsg_public_supplement_review_decisions as validator


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_row(tmp_path: Path) -> dict[str, str]:
    source = tmp_path / "source.csv"
    review = tmp_path / "review.csv"
    source.write_text("source\n", encoding="utf-8")
    review.write_text("review\n", encoding="utf-8")
    return {
        "review_id": "PSR-999",
        "review_type": "scope_note_review",
        "entry_id": "PSP-999",
        "blocker_id": "PSB-999",
        "artifact_group": "public_predicate_attack_ladder",
        "source_path": str(source),
        "source_sha256_expected": digest(source),
        "review_artifact_path": str(review),
        "review_artifact_sha256_expected": digest(review),
        "planned_supplement_path": "evidence/example.csv",
        "candidate_bundle_path": "candidate/evidence/example.csv",
        "required_evidence": "human reviewer approves source with scope note",
        "reviewer_assertion_required": "approve scope note",
        "claim_scope_guard": "public text-only verification success; ownership proof",
        "allowed_decisions": "approved;rejected;hold",
        "decision_status": "pending_not_performed",
        "reviewer_id": "",
        "reviewed_at_utc": "",
        "source_sha256_verified": "",
        "review_artifact_sha256_verified": "",
        "reviewer_assertion_confirmed": "",
        "claim_scope_guard_preserved": "",
        "failure_condition": "reject_if_forbidden_claims_not_preserved",
        "approval_gate": "not_approved_template_only",
    }


def write_decisions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_default_decision_validation_records_pending_only(tmp_path: Path) -> None:
    summary = validator.build(validator.DEFAULT_DECISION_RECORDS, validator.DEFAULT_SCHEMA, tmp_path)

    assert summary["status"] == "PASS_VSG_PUBLIC_SUPPLEMENT_REVIEW_DECISIONS_VALIDATED_PENDING_ONLY"
    assert summary["decision_row_count"] == 14
    assert summary["pending_decision_count"] == 14
    assert summary["approved_decision_count"] == 0
    assert summary["invalid_decision_count"] == 0
    assert summary["all_decisions_valid"] is True
    assert summary["all_decisions_pending"] is True
    assert summary["review_approvals_recorded_in_input"] is False
    assert summary["review_approvals_created_by_validator"] is False
    assert summary["human_reviews_performed_by_validator"] is False
    assert summary["release_ready_after_validation"] is False
    assert summary["public_supplement_created"] is False
    assert summary["publication_performed"] is False
    assert summary["new_slurm_started"] is False
    assert summary["generation_started"] is False
    assert summary["model_scoring_started"] is False
    assert summary["training_started"] is False
    assert summary["allowlist_enabled"] is False
    assert summary["public_text_only_verification_claimed"] is False
    assert summary["ownership_proof_claimed"] is False

    rows = list(csv.DictReader((tmp_path / "review_decision_validation.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == 14
    assert all(row["validation_status"] == "valid" for row in rows)

    manifest = json.loads((tmp_path / "review_decision_validation_manifest.json").read_text(encoding="utf-8"))
    manifest_names = {Path(row["path"]).name for row in manifest["files"]}
    assert "review_decision_validation_manifest.json" not in manifest_names
    assert "review_decision_validation.csv" in manifest_names
    assert "review_decision_validation_summary.json" in manifest_names
    assert "review_decision_validation_report.md" in manifest_names


def test_approved_row_requires_reviewer_hash_scope_and_gate(tmp_path: Path) -> None:
    row = base_row(tmp_path)
    row["decision_status"] = "approved"
    decisions = tmp_path / "decisions.csv"
    write_decisions(decisions, [row])

    summary = validator.build(decisions, validator.DEFAULT_SCHEMA, tmp_path / "out")

    assert summary["status"] == "FAIL_VSG_PUBLIC_SUPPLEMENT_REVIEW_DECISIONS_INVALID"
    assert summary["invalid_decision_count"] == 1
    rows = list(csv.DictReader((tmp_path / "out" / "review_decision_validation.csv").open(newline="", encoding="utf-8")))
    errors = rows[0]["validation_errors"]
    assert "missing_required_field:reviewer_id" in errors
    assert "source_sha256_not_marked_verified" in errors
    assert "approved_row_missing_validated_approval_gate" in errors


def test_valid_approved_row_passes_but_does_not_publish(tmp_path: Path) -> None:
    row = base_row(tmp_path)
    row.update(
        {
            "decision_status": "approved",
            "reviewer_id": "reviewer-a",
            "reviewed_at_utc": "2026-06-01T00:00:00Z",
            "source_sha256_verified": "true",
            "review_artifact_sha256_verified": "true",
            "reviewer_assertion_confirmed": "true",
            "claim_scope_guard_preserved": "true",
            "approval_gate": "approved_hash_scope_guard_validated",
        }
    )
    decisions = tmp_path / "decisions.csv"
    write_decisions(decisions, [row])

    summary = validator.build(decisions, validator.DEFAULT_SCHEMA, tmp_path / "out")

    assert summary["status"] == "PASS_VSG_PUBLIC_SUPPLEMENT_REVIEW_DECISIONS_VALIDATED_NON_PENDING_RECORDS"
    assert summary["approved_decision_count"] == 1
    assert summary["invalid_decision_count"] == 0
    assert summary["review_approvals_recorded_in_input"] is True
    assert summary["review_approvals_created_by_validator"] is False
    assert summary["publication_blockers_resolved"] is False
    assert summary["release_ready_after_validation"] is False
    assert summary["files_copied"] is False
    assert summary["public_supplement_created"] is False


def test_hash_mismatch_invalidates_non_pending_decision(tmp_path: Path) -> None:
    row = base_row(tmp_path)
    row.update(
        {
            "decision_status": "hold",
            "reviewer_id": "reviewer-a",
            "reviewed_at_utc": "2026-06-01T00:00:00Z",
            "source_sha256_expected": "bad",
            "source_sha256_verified": "true",
            "review_artifact_sha256_verified": "true",
            "approval_gate": "not_approved_template_only",
        }
    )
    decisions = tmp_path / "decisions.csv"
    write_decisions(decisions, [row])

    summary = validator.build(decisions, validator.DEFAULT_SCHEMA, tmp_path / "out")

    assert summary["status"] == "FAIL_VSG_PUBLIC_SUPPLEMENT_REVIEW_DECISIONS_INVALID"
    assert summary["invalid_review_ids"] == ["PSR-999"]
    rows = list(csv.DictReader((tmp_path / "out" / "review_decision_validation.csv").open(newline="", encoding="utf-8")))
    assert "source_sha256_mismatch" in rows[0]["validation_errors"]


def test_pending_row_with_reviewer_data_is_invalid(tmp_path: Path) -> None:
    row = base_row(tmp_path)
    row["reviewer_id"] = "reviewer-a"
    decisions = tmp_path / "decisions.csv"
    write_decisions(decisions, [row])

    summary = validator.build(decisions, validator.DEFAULT_SCHEMA, tmp_path / "out")

    assert summary["status"] == "FAIL_VSG_PUBLIC_SUPPLEMENT_REVIEW_DECISIONS_INVALID"
    assert summary["invalid_decision_count"] == 1
    rows = list(csv.DictReader((tmp_path / "out" / "review_decision_validation.csv").open(newline="", encoding="utf-8")))
    assert "pending_row_has_reviewer_id" in rows[0]["validation_errors"]
