import csv
import json
from pathlib import Path

from scripts.verification_substrate_gap import build_vsg_public_supplement_review_derivatives as derivatives


def test_redact_csv_source_removes_source_shard_dir_and_private_paths(tmp_path: Path) -> None:
    row = {
        "source_path": "results/verification_substrate_gap/corpora/trace_bound_controllability/qwen_blocks.csv",
        "planned_supplement_path": "evidence/trace_bound_controllability_redacted/qwen_blocks.csv",
    }

    result = derivatives.redact_csv_source(row, tmp_path)

    target = Path(result["derivative_path"])
    if not target.is_absolute():
        target = derivatives.ROOT / target
    assert target.is_file()
    assert result["row_count"] == 480
    assert result["dropped_fields"] == ["source_shard_dir"]
    assert result["private_marker_hits_after_redaction"] == 0
    with target.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert rows
        assert "source_shard_dir" not in rows[0]
    text = target.read_text(encoding="utf-8")
    assert "/hpcstor" not in text
    assert "guanjie.lin001" not in text


def test_scope_notes_preserve_attack_and_local_pilot_non_claims(tmp_path: Path) -> None:
    rows = [
        {
            "artifact_group": "public_predicate_attack_ladder",
            "source_path": "attack.csv",
            "planned_supplement_path": "evidence/public_predicate_attack_ladder_scope_limited/attack.csv",
            "release_claim_scope_note": "source-mismatch spoofing evidence only; not protected success and not codeword recovery",
            "manual_review_required": "True",
        },
        {
            "artifact_group": "stronger_public_predicate_local_pilot",
            "source_path": "pilot.csv",
            "planned_supplement_path": "evidence/local_pilots/stronger_public_predicate/pilot.csv",
            "release_claim_scope_note": "local non-adopted/historical pilot only; not adopted locked evidence",
            "manual_review_required": "True",
        },
    ]

    result = derivatives.build_scope_notes(rows, tmp_path)

    assert result["scope_note_count"] == 2
    note_text = (tmp_path / "scope_notes.md").read_text(encoding="utf-8")
    assert "source-mismatch public-predicate spoofing artifact only" in note_text
    assert "local non-adopted/historical pilot only" in note_text
    assert "protected success" in note_text
    assert "adopted locked evidence" in note_text


def test_security_review_finds_field_names_not_literal_secrets(tmp_path: Path) -> None:
    row = {
        "source_path": "configs/verification_substrate_gap/text_only_observability.yaml",
        "planned_supplement_path": "configs/text_only_observability.yaml",
    }

    result = derivatives.review_security_config(row, tmp_path)

    assert result["field_name_hit_count"] == 8
    assert result["secret_value_hit_count"] == 0
    assert result["release_recommendation"] == "schema_field_review_required_no_literal_secret_values_detected"
    assert (tmp_path / "security_review_text_only_observability.json").is_file()
    assert (tmp_path / "security_review_text_only_observability.md").is_file()


def test_build_review_derivatives_is_artifact_only_and_counted(tmp_path: Path) -> None:
    summary = derivatives.build(derivatives.DEFAULT_STAGING_PLAN, tmp_path)

    assert summary["status"] == "PASS_VSG_PUBLIC_SUPPLEMENT_REVIEW_DERIVATIVES_RECORDED_ARTIFACT_ONLY"
    assert summary["redacted_csv_written_count"] == 3
    assert summary["redacted_rows_total"] == 1920
    assert summary["private_marker_hits_after_redaction"] == 0
    assert summary["scope_note_count"] == 10
    assert summary["security_review_count"] == 1
    assert summary["security_field_name_hit_count"] == 8
    assert summary["security_secret_value_hit_count"] == 0
    assert summary["review_derivatives_created"] is True
    assert summary["source_files_copied_without_transform"] is False
    assert summary["public_supplement_created"] is False
    assert summary["publication_performed"] is False
    assert summary["new_slurm_started"] is False
    assert summary["generation_started"] is False
    assert summary["model_scoring_started"] is False
    assert summary["training_started"] is False
    assert summary["allowlist_enabled"] is False
    assert summary["public_text_only_verification_claimed"] is False
    assert summary["ownership_proof_claimed"] is False
    assert summary["release_ready_after_derivatives"] is False

    manifest = json.loads((tmp_path / "review_derivatives_manifest.json").read_text(encoding="utf-8"))
    manifest_names = {Path(row["path"]).name for row in manifest["files"]}
    assert "review_derivatives_manifest.json" not in manifest_names
    assert "combined_blocks.csv" in manifest_names
    assert "qwen_blocks.csv" in manifest_names
    assert "llama_blocks.csv" in manifest_names
    assert "scope_notes.csv" in manifest_names
    assert "security_review_text_only_observability.json" in manifest_names
