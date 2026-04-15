import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.freeze_catalog import _resolve_tokenizer_settings
from src.core.bucket_mapping import load_bucket_layout
from src.infrastructure.paths import discover_repo_root


def _write_source_catalog(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _success_source_catalog() -> dict[str, object]:
    return {
        "catalog_name": "source-success",
        "fields": [
            {
                "field_name": "FIELD_A",
                "buckets": {
                    "0": ["safea", "dup"],
                    "1": ["betagood", "trail "],
                },
            },
            {
                "field_name": "FIELD_B",
                "buckets": {
                    "0": ["safeb", "dup "],
                    "1": ["gammagood", "two words"],
                },
            },
        ],
    }


def _failure_source_catalog() -> dict[str, object]:
    return {
        "catalog_name": "source-failure",
        "fields": [
            {
                "field_name": "FIELD_A",
                "buckets": {
                    "0": ["dup", "two words"],
                    "1": ["single", "trail "],
                },
            },
            {
                "field_name": "FIELD_B",
                "buckets": {
                    "0": ["safeb", "dup "],
                    "1": ["other", "good"],
                },
            },
        ],
    }


def test_strict_fail_blocks_freeze(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    source_catalog = _write_source_catalog(tmp_path / "source.yaml", _failure_source_catalog())
    frozen_catalog = tmp_path / "carrier_catalog_freeze_v1.yaml"
    audit_report = tmp_path / "tokenizer_audit_report_mock.json"
    change_log = tmp_path / "catalog_changes.md"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/freeze_catalog.py",
            "--source-catalog",
            str(source_catalog),
            "--tokenizer-backend",
            "mock",
            "--frozen-catalog-output",
            str(frozen_catalog),
            "--audit-report-output",
            str(audit_report),
            "--change-log-output",
            str(change_log),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert not frozen_catalog.exists()
    assert audit_report.exists()
    assert change_log.exists()


def test_successful_freeze_emits_artifacts_and_filters_invalid_carriers(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    source_catalog = _write_source_catalog(tmp_path / "source.yaml", _success_source_catalog())
    frozen_catalog = tmp_path / "carrier_catalog_freeze_v1.yaml"
    audit_report = tmp_path / "tokenizer_audit_report_mock.json"
    change_log = tmp_path / "catalog_changes.md"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/freeze_catalog.py",
            "--source-catalog",
            str(source_catalog),
            "--tokenizer-backend",
            "mock",
            "--frozen-catalog-output",
            str(frozen_catalog),
            "--audit-report-output",
            str(audit_report),
            "--change-log-output",
            str(change_log),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "freeze_status=success" in completed.stdout
    assert frozen_catalog.exists()
    assert audit_report.exists()
    assert change_log.exists()

    layout = load_bucket_layout(frozen_catalog)
    assert layout.provenance["tokenizer_name"] == "mock"
    assert layout.provenance["tokenizer_backend"] == "mock"
    assert layout.provenance["tokenizer_revision_source"] == "mock"
    assert layout.provenance["freeze_status"] == "strict_passed"
    assert layout.provenance["source_catalog"] == str(source_catalog.resolve())

    carriers = set(layout.all_carriers())
    assert "dup" not in carriers
    assert "dup " not in carriers
    assert "trail " not in carriers
    assert "two words" not in carriers

    report_payload = json.loads(audit_report.read_text(encoding="utf-8"))
    assert report_payload["success"] is True
    assert report_payload["strict_audit"]["is_alignment_safe"] is True
    removed = {item["carrier"] for item in report_payload["removed_carriers"]}
    assert {"dup", "dup ", "trail ", "two words"}.issubset(removed)


def test_tokenizer_name_can_be_inherited_from_base_experiment_config() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    backend, name = _resolve_tokenizer_settings(
        tokenizer_backend="huggingface",
        tokenizer_name="",
        base_experiment_config=repo_root / "configs" / "experiment" / "exp_recovery.yaml",
    )
    assert backend == "huggingface"
    assert name == "gpt2"
