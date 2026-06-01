import csv
import json
from pathlib import Path

from scripts.verification_substrate_gap import build_vsg_release_boundary_audit as audit


def test_classify_ready_release_candidate() -> None:
    row = {
        "artifact_group": "manuscript_source",
        "release_role": "submission_source",
        "path": "paper.tex",
        "exists": "True",
        "tracked_by_git": "True",
        "release_status": "release_candidate",
        "requires_anonymization_review": "False",
        "private_path_hit": "False",
        "secret_term_hit": "False",
    }

    classified = audit.classify(row)

    assert classified["boundary_decision"] == "ready_for_reviewed_public_supplement"
    assert classified["pre_release_review_required"] is False
    assert classified["release_blocker"] is False


def test_classify_private_path_overrides_tracking() -> None:
    row = {
        "artifact_group": "trace_bound_corpus_summary",
        "release_role": "summary_only_not_raw_trace",
        "path": "summary.csv",
        "exists": "True",
        "tracked_by_git": "True",
        "release_status": "release_candidate",
        "requires_anonymization_review": "False",
        "private_path_hit": "True",
        "secret_term_hit": "False",
    }

    classified = audit.classify(row)

    assert classified["boundary_decision"] == "redact_or_summarize_before_release"
    assert classified["pre_release_review_required"] is True
    assert classified["release_blocker"] is True


def test_classify_state_records_as_excluded_not_blocking() -> None:
    row = {
        "artifact_group": "state_and_scope_records",
        "release_role": "scope_record",
        "path": "state.md",
        "exists": "True",
        "tracked_by_git": "True",
        "release_status": "release_candidate_with_caveat",
        "requires_anonymization_review": "True",
        "private_path_hit": "False",
        "secret_term_hit": "False",
    }

    classified = audit.classify(row)

    assert classified["boundary_decision"] == "exclude_from_public_supplement"
    assert classified["pre_release_review_required"] is False
    assert classified["release_blocker"] is False


def test_build_release_boundary_audit_is_artifact_only(tmp_path: Path) -> None:
    summary = audit.build(audit.DEFAULT_INVENTORY, tmp_path)

    assert summary["status"] == "PASS_VSG_RELEASE_BOUNDARY_AUDIT_RECORDED_REVIEW_REQUIRED"
    assert summary["row_count"] == 78
    assert summary["release_ready_now"] is False
    assert summary["release_blocker_count"] > 0
    assert summary["public_supplement_publication_performed"] is False
    assert summary["new_slurm_started"] is False
    assert summary["generation_started"] is False
    assert summary["model_scoring_started"] is False
    assert summary["training_started"] is False
    assert summary["allowlist_enabled"] is False
    assert summary["public_text_only_verification_claimed"] is False
    assert summary["ownership_proof_claimed"] is False

    decisions_csv = tmp_path / "release_boundary_decisions.csv"
    assert decisions_csv.is_file()
    with decisions_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 78
    decisions = {row["boundary_decision"] for row in rows}
    assert "ready_for_reviewed_public_supplement" in decisions
    assert "redact_or_summarize_before_release" in decisions
    assert "exclude_from_public_supplement" in decisions

    manifest = json.loads((tmp_path / "release_boundary_manifest.json").read_text(encoding="utf-8"))
    manifest_names = {Path(row["path"]).name for row in manifest["files"]}
    assert "release_boundary_manifest.json" not in manifest_names
    assert "release_boundary_decisions.csv" in manifest_names
    assert "release_boundary_summary.json" in manifest_names
    assert "release_boundary_report.md" in manifest_names
