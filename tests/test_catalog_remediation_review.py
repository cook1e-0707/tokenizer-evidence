import subprocess
import sys
from pathlib import Path

from src.core.bucket_mapping import BucketLayout, FieldBucketSpec
from src.core.catalog_freeze import (
    build_catalog_remediation_review,
    freeze_catalog,
    render_catalog_change_log,
    render_catalog_remediation_review,
    save_audit_report,
    save_catalog_remediation_review,
    save_catalog_remediation_table,
)
from src.core.tokenizer_utils import MockTokenizer
from src.infrastructure.paths import discover_repo_root


def _failed_outcome():
    repo_root = discover_repo_root(Path(__file__).parent)
    layout = BucketLayout(
        catalog_name="review-source",
        fields=(
            FieldBucketSpec(
                field_name="FIELD_A",
                buckets={
                    0: ("safe_a", "bad_a0"),
                    1: ("safe_b", "safe_c"),
                },
            ),
            FieldBucketSpec(
                field_name="FIELD_B",
                buckets={
                    0: ("bad_b0", "bad_b1"),
                    1: ("safe_d", "safe_e"),
                },
            ),
        ),
    )
    tokenizer = MockTokenizer(
        single_token_map={
            "safe_a": 1,
            "safe_b": 2,
            "safe_c": 3,
            "safe_d": 4,
            "safe_e": 5,
        },
        multi_token_map={
            "bad_a0": (10, 11),
            "bad_b0": (12, 13),
            "bad_b1": (14, 15),
        },
    )
    return freeze_catalog(
        source_layout=layout,
        source_catalog_path=repo_root / "configs" / "data" / "real_pilot_catalog.yaml",
        tokenizer=tokenizer,
        tokenizer_name="mock",
        tokenizer_backend="mock",
        tokenizer_revision_source="mock",
        repo_root=repo_root,
    )


def test_build_catalog_remediation_review_reports_bucket_state() -> None:
    outcome = _failed_outcome()
    review = build_catalog_remediation_review(
        report_payload=outcome.to_dict(),
        change_log_text=render_catalog_change_log(outcome),
    )

    assert review.schema_name == "catalog_freeze_remediation_review"
    field_a = next(field for field in review.fields if field.field_name == "FIELD_A")
    field_b = next(field for field in review.fields if field.field_name == "FIELD_B")

    assert field_a.recommended_action == "regroup_with_manual_review"
    assert field_b.recommended_action == "drop_field"

    bucket_a0 = next(bucket for bucket in field_a.buckets if bucket.bucket_id == 0)
    assert bucket_a0.original_members == ("safe_a", "bad_a0")
    assert bucket_a0.surviving_members == ("safe_a",)
    assert bucket_a0.bucket_became_empty is False
    assert bucket_a0.recommended_action == "regroup_with_manual_review"
    assert "multi_token" in bucket_a0.rejection_reasons

    bucket_b0 = next(bucket for bucket in field_b.buckets if bucket.bucket_id == 0)
    assert bucket_b0.bucket_became_empty is True
    assert bucket_b0.recommended_action == "drop_field"
    assert [item.carrier for item in bucket_b0.rejected_members] == ["bad_b0", "bad_b1"]


def test_review_catalog_freeze_cli_outputs_table_and_markdown(tmp_path: Path) -> None:
    outcome = _failed_outcome()
    audit_report_path = tmp_path / "audit.json"
    change_log_path = tmp_path / "change_log.md"
    remediation_table_path = tmp_path / "remediation.json"
    remediation_review_path = tmp_path / "remediation.md"

    save_audit_report(outcome, audit_report_path)
    change_log_path.write_text(render_catalog_change_log(outcome), encoding="utf-8")

    review = build_catalog_remediation_review(
        report_payload=outcome.to_dict(),
        change_log_text=change_log_path.read_text(encoding="utf-8"),
        change_log_path=change_log_path,
    )
    save_catalog_remediation_table(review, remediation_table_path)
    save_catalog_remediation_review(review, remediation_review_path)

    assert remediation_table_path.exists()
    assert remediation_review_path.exists()
    review_text = render_catalog_remediation_review(review)
    assert "## FIELD_A" in review_text
    assert "### Bucket 0" in review_text


def test_review_catalog_freeze_script_runs_end_to_end(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    outcome = _failed_outcome()
    audit_report_path = tmp_path / "audit.json"
    change_log_path = tmp_path / "change_log.md"
    remediation_table_path = tmp_path / "remediation.yaml"
    remediation_review_path = tmp_path / "remediation.md"

    save_audit_report(outcome, audit_report_path)
    change_log_path.write_text(render_catalog_change_log(outcome), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/review_catalog_freeze.py",
            "--audit-report",
            str(audit_report_path),
            "--change-log",
            str(change_log_path),
            "--output-table",
            str(remediation_table_path),
            "--output-review",
            str(remediation_review_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "output_table=" in completed.stdout
    assert remediation_table_path.exists()
    assert remediation_review_path.exists()
