import csv
import json
from pathlib import Path

from scripts.verification_substrate_gap import build_vsg_public_supplement_readiness_audit as audit


def test_readiness_for_redacted_derivative_is_covered_but_still_review_blocked() -> None:
    row = {
        "artifact_group": "trace_bound_corpus_summary",
        "source_path": "results/verification_substrate_gap/corpora/trace_bound_controllability/qwen_blocks.csv",
        "staging_decision": "redacted_derivative_candidate",
        "planned_supplement_path": "evidence/trace_bound_controllability_redacted/qwen_blocks.csv",
        "release_claim_scope_note": "provider-side trace-bound diagnostic summary only",
    }
    derivatives = {
        "redaction": {
            row["source_path"]: {
                "derivative_path": (
                    "results/verification_substrate_gap/public_supplement_review_derivatives_20260601/"
                    "evidence/trace_bound_controllability_redacted/qwen_blocks.csv"
                ),
                "private_marker_hits_after_redaction": 0,
            }
        },
        "scope": {},
        "security": {},
    }

    readiness = audit.readiness_for(row, derivatives)

    assert readiness["readiness_decision"] == "redacted_derivative_available_manual_review_required"
    assert readiness["derivative_status"] == "covered"
    assert readiness["manual_review_required"] is True
    assert readiness["publication_blocker"] is True


def test_readiness_for_scope_note_uses_forbidden_claim_guard() -> None:
    row = {
        "artifact_group": "public_predicate_attack_ladder",
        "source_path": "attack.csv",
        "staging_decision": "scope_note_gated_candidate",
        "planned_supplement_path": "evidence/public_predicate_attack_ladder_scope_limited/attack.csv",
        "release_claim_scope_note": "source-mismatch spoofing evidence only",
    }
    derivatives = {
        "redaction": {},
        "scope": {
            "attack.csv": {
                "public_label": "source-mismatch public-predicate spoofing artifact only",
                "forbidden_claims": "protected success; codeword recovery",
            }
        },
        "security": {},
    }

    readiness = audit.readiness_for(row, derivatives)

    assert readiness["readiness_decision"] == "scope_note_available_manual_review_required"
    assert readiness["derivative_status"] == "covered"
    assert "protected success" in readiness["claim_scope_guard"]
    assert readiness["publication_blocker"] is True


def test_readiness_for_security_review_requires_no_secret_values() -> None:
    row = {
        "artifact_group": "reproducibility_config",
        "source_path": "configs/verification_substrate_gap/text_only_observability.yaml",
        "staging_decision": "security_review_gated_candidate",
        "planned_supplement_path": "configs/text_only_observability.yaml",
        "release_claim_scope_note": "preserves VSG substrate-gap claim boundary",
    }
    derivatives = {
        "redaction": {},
        "scope": {},
        "security": {
            row["source_path"]: {
                "secret_value_hit_count": 0,
                "security_review_json": (
                    "results/verification_substrate_gap/public_supplement_review_derivatives_20260601/"
                    "security_review_text_only_observability.json"
                ),
            }
        },
    }

    readiness = audit.readiness_for(row, derivatives)

    assert readiness["readiness_decision"] == "security_review_available_manual_review_required"
    assert readiness["derivative_status"] == "covered"
    assert readiness["manual_review_required"] is True
    assert readiness["publication_blocker"] is True


def test_build_readiness_audit_counts_remaining_release_work(tmp_path: Path) -> None:
    summary = audit.build(audit.DEFAULT_STAGING_PLAN, audit.DEFAULT_DERIVATIVES_SUMMARY, tmp_path)

    assert summary["status"] == "PASS_VSG_PUBLIC_SUPPLEMENT_READINESS_AUDIT_RECORDED_REVIEW_REQUIRED"
    assert summary["row_count"] == 78
    assert summary["direct_include_candidate_count"] == 39
    assert summary["stage_or_copy_required_count"] == 21
    assert summary["derivative_required_count"] == 14
    assert summary["derivative_covered_count"] == 14
    assert summary["derivative_uncovered_count"] == 0
    assert summary["manual_review_required_after_derivatives_count"] == 14
    assert summary["publication_blocker_count"] == 35
    assert summary["excluded_internal_record_count"] == 4
    assert summary["release_ready_now"] is False
    assert summary["public_supplement_created"] is False
    assert summary["publication_performed"] is False
    assert summary["new_slurm_started"] is False
    assert summary["generation_started"] is False
    assert summary["model_scoring_started"] is False
    assert summary["training_started"] is False
    assert summary["allowlist_enabled"] is False
    assert summary["public_text_only_verification_claimed"] is False
    assert summary["ownership_proof_claimed"] is False

    decisions_path = tmp_path / "readiness_decisions.csv"
    assert decisions_path.is_file()
    with decisions_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 78
    assert {row["readiness_decision"] for row in rows} == {
        "copy_or_commit_required_before_supplement_bundle",
        "excluded_from_public_supplement",
        "ready_for_final_license_scope_review",
        "redacted_derivative_available_manual_review_required",
        "scope_note_available_manual_review_required",
        "security_review_available_manual_review_required",
    }

    manifest = json.loads((tmp_path / "readiness_manifest.json").read_text(encoding="utf-8"))
    manifest_names = {Path(row["path"]).name for row in manifest["files"]}
    assert "readiness_manifest.json" not in manifest_names
    assert "readiness_decisions.csv" in manifest_names
    assert "readiness_summary.json" in manifest_names
    assert "readiness_report.md" in manifest_names
