import csv
import json
from pathlib import Path

from scripts.verification_substrate_gap import build_vsg_public_supplement_human_review_packet as packet


def test_packet_row_adds_redaction_context() -> None:
    context = packet.load_derivative_context(packet.DEFAULT_DERIVATIVES_SUMMARY)
    row = {
        "review_id": "PSR-001",
        "entry_id": "PSP-033",
        "blocker_id": "PSB-009",
        "artifact_group": "trace_bound_corpus_summary",
        "review_type": "redaction_review",
        "source_path": (
            "results/verification_substrate_gap/public_supplement_review_derivatives_20260601/"
            "evidence/trace_bound_controllability_redacted/combined_blocks.csv"
        ),
        "candidate_bundle_path": "candidate/combined_blocks.csv",
        "planned_supplement_path": "evidence/trace_bound_controllability_redacted/combined_blocks.csv",
        "review_artifact_path": "results/verification_substrate_gap/corpora/trace_bound_controllability/combined_blocks.csv",
        "required_evidence": "human reviewer confirms redacted derivative removes private fields",
        "reviewer_assertion_required": "approve redacted derivative",
        "approval_status": "pending_not_performed",
        "claim_scope_guard": "trace-bound-only",
    }

    built = packet.packet_row(row, context)

    assert built["review_id"] == "PSR-001"
    assert built["review_type"] == "redaction_review"
    assert built["source_sha256"] == "e9aaea586d2e019f0a0a85c2f73f43868f577416e4c927d8c9aa804f70e364be"
    assert built["source_row_count"] == 960
    assert built["redaction_dropped_fields"] == "source_shard_dir"
    assert built["redaction_private_marker_hits_after_redaction"] == 0
    assert built["approval_status"] == "pending_not_performed"


def test_packet_row_adds_scope_note_context() -> None:
    context = packet.load_derivative_context(packet.DEFAULT_DERIVATIVES_SUMMARY)
    row = {
        "review_id": "PSR-005",
        "entry_id": "PSP-041",
        "blocker_id": "PSB-017",
        "artifact_group": "public_predicate_attack_ladder",
        "review_type": "scope_note_review",
        "source_path": (
            "results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/"
            "surrogate_guided_rewrite_summary.json"
        ),
        "candidate_bundle_path": "candidate/surrogate_guided_rewrite_summary.json",
        "planned_supplement_path": "evidence/public_predicate_attack_ladder_scope_limited/surrogate_guided_rewrite_summary.json",
        "review_artifact_path": "results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv",
        "required_evidence": "human reviewer approves source with scope note",
        "reviewer_assertion_required": "approve scope note",
        "approval_status": "pending_not_performed",
        "claim_scope_guard": "public text-only verification success; ownership proof",
    }

    built = packet.packet_row(row, context)

    assert built["review_type"] == "scope_note_review"
    assert built["scope_note_type"] == "source_mismatch_spoofing_scope_note"
    assert "source-mismatch spoofing evidence only" in built["allowed_interpretation"]
    assert "ownership proof" in built["forbidden_claims"]
    assert built["review_artifact_row_count"] == 10


def test_build_human_review_packet_keeps_all_reviews_pending(tmp_path: Path) -> None:
    summary = packet.build(packet.DEFAULT_REVIEW_CHECKLIST, packet.DEFAULT_DERIVATIVES_SUMMARY, tmp_path)

    assert summary["status"] == "PASS_VSG_PUBLIC_SUPPLEMENT_HUMAN_REVIEW_PACKET_RECORDED_PENDING_REVIEW"
    assert summary["review_row_count"] == 14
    assert summary["pending_review_count"] == 14
    assert summary["redaction_review_count"] == 3
    assert summary["scope_note_review_count"] == 10
    assert summary["security_review_count"] == 1
    assert summary["missing_source_count"] == 0
    assert summary["missing_review_artifact_count"] == 0
    assert summary["missing_scope_note_context_count"] == 0
    assert summary["redaction_private_marker_hits_after_redaction"] == 0
    assert summary["security_secret_value_hit_count"] == 0
    assert summary["all_reviews_pending"] is True
    assert summary["all_sources_present"] is True
    assert summary["all_review_artifacts_present"] is True
    assert summary["all_scope_notes_resolved"] is True
    assert summary["review_packet_created"] is True
    assert summary["review_approvals_recorded"] is False
    assert summary["human_reviews_performed"] is False
    assert summary["files_copied"] is False
    assert summary["candidate_bundle_created"] is False
    assert summary["publication_blockers_resolved"] is False
    assert summary["release_ready_after_packet"] is False
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

    rows = list(csv.DictReader((tmp_path / "human_review_packet_index.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == 14
    assert {row["review_type"] for row in rows} == {"redaction_review", "scope_note_review", "security_review"}
    assert all(row["approval_status"] == "pending_not_performed" for row in rows)

    cards = (tmp_path / "human_review_cards.md").read_text(encoding="utf-8")
    assert "pending_not_performed" in cards
    assert "Public text-only verification success" not in cards

    manifest = json.loads((tmp_path / "human_review_packet_manifest.json").read_text(encoding="utf-8"))
    manifest_names = {Path(row["path"]).name for row in manifest["files"]}
    assert "human_review_packet_manifest.json" not in manifest_names
    assert "human_review_packet_index.csv" in manifest_names
    assert "human_review_cards.md" in manifest_names
    assert "human_review_packet_summary.json" in manifest_names
    assert "human_review_packet_report.md" in manifest_names
