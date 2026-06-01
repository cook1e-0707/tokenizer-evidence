import csv
import json
from pathlib import Path

from scripts.verification_substrate_gap import build_vsg_release_staging_plan as plan


def test_plan_row_redacts_private_trace_csv_to_redacted_folder() -> None:
    row = {
        "artifact_group": "trace_bound_corpus_summary",
        "path": "results/verification_substrate_gap/corpora/trace_bound_controllability/qwen_blocks.csv",
        "boundary_decision": "redact_or_summarize_before_release",
    }

    planned = plan.plan_row(row)

    assert planned["staging_decision"] == "redacted_derivative_candidate"
    assert planned["include_in_public_supplement_plan"] is True
    assert planned["planned_supplement_path"] == "evidence/trace_bound_controllability_redacted/qwen_blocks.csv"
    assert planned["planned_transform"] == "derive_redacted_csv_drop_source_shard_dir_and_private_path_fields"
    assert planned["execution_required"] is True
    assert planned["manual_review_required"] is True
    assert "private path" in planned["residual_risk"]
    assert "not public text-only verification" in planned["release_claim_scope_note"]


def test_plan_row_attack_ladder_is_scope_limited_not_protected_success() -> None:
    row = {
        "artifact_group": "public_predicate_attack_ladder",
        "path": "results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/surrogate_guided_rewrite_examples.csv",
        "boundary_decision": "scope_review_before_release",
    }

    planned = plan.plan_row(row)

    assert planned["staging_decision"] == "scope_note_gated_candidate"
    assert planned["planned_supplement_path"] == (
        "evidence/public_predicate_attack_ladder_scope_limited/surrogate_guided_rewrite_examples.csv"
    )
    assert planned["manual_review_required"] is True
    assert "source-mismatch" in planned["release_claim_scope_note"]
    assert "not protected success" in planned["release_claim_scope_note"]
    assert "not codeword recovery" in planned["release_claim_scope_note"]


def test_plan_row_excludes_internal_handoff_record() -> None:
    row = {
        "artifact_group": "state_and_scope_records",
        "path": "results/verification_substrate_gap/VSG_CURRENT_HANDOFF_STATE_20260601.md",
        "boundary_decision": "exclude_from_public_supplement",
    }

    planned = plan.plan_row(row)

    assert planned["staging_decision"] == "excluded_internal_record"
    assert planned["include_in_public_supplement_plan"] is False
    assert planned["planned_supplement_path"] == ""
    assert planned["execution_required"] is False
    assert planned["manual_review_required"] is False


def test_build_release_staging_plan_is_plan_only_and_counted(tmp_path: Path) -> None:
    summary = plan.build(plan.DEFAULT_BOUNDARY, tmp_path)

    assert summary["status"] == "PASS_VSG_RELEASE_STAGING_PLAN_RECORDED_PLAN_ONLY"
    assert summary["row_count"] == 78
    assert summary["direct_include_candidate_count"] == 39
    assert summary["stage_or_copy_candidate_count"] == 21
    assert summary["redacted_derivative_candidate_count"] == 3
    assert summary["scope_note_gated_candidate_count"] == 10
    assert summary["security_review_gated_candidate_count"] == 1
    assert summary["excluded_internal_record_count"] == 4
    assert summary["execution_required_count"] == 35
    assert summary["manual_review_required_count"] == 14
    assert summary["duplicate_planned_target_count"] == 0
    assert summary["plan_only"] is True
    assert summary["files_copied"] is False
    assert summary["public_supplement_copy_performed"] is False
    assert summary["public_supplement_created"] is False
    assert summary["publication_performed"] is False
    assert summary["release_ready_after_plan"] is False
    assert summary["new_slurm_started"] is False
    assert summary["generation_started"] is False
    assert summary["model_scoring_started"] is False
    assert summary["training_started"] is False
    assert summary["allowlist_enabled"] is False
    assert summary["public_text_only_verification_claimed"] is False
    assert summary["ownership_proof_claimed"] is False

    plan_csv = tmp_path / "release_staging_plan.csv"
    assert plan_csv.is_file()
    with plan_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 78
    assert {
        "include_in_public_supplement_plan",
        "residual_risk",
        "release_path_reason",
    }.issubset(rows[0])

    manifest = json.loads((tmp_path / "release_staging_manifest.json").read_text(encoding="utf-8"))
    manifest_names = {Path(row["path"]).name for row in manifest["files"]}
    assert "release_staging_manifest.json" not in manifest_names
    assert "release_staging_plan.csv" in manifest_names
    assert "release_staging_summary.json" in manifest_names
    assert "release_staging_report.md" in manifest_names
