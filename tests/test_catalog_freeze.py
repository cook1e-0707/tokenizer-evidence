import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.freeze_catalog import _resolve_tokenizer_settings
from src.core.bucket_mapping import load_bucket_layout
from src.core.catalog_freeze import write_frozen_data_config, write_frozen_experiment_config
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


def test_generated_frozen_configs_use_portable_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    frozen_catalog_path = repo_root / "configs" / "data" / "frozen" / "real_pilot_catalog__gpt2__v1.yaml"
    source_catalog_path = repo_root / "configs" / "data" / "source" / "real_pilot_catalog__gpt2__src_v2.yaml"
    base_experiment_config = repo_root / "configs" / "experiment" / "exp_recovery.yaml"

    frozen_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    source_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    base_experiment_config.parent.mkdir(parents=True, exist_ok=True)
    frozen_catalog_path.write_text("catalog_name: frozen\nfields: []\n", encoding="utf-8")
    source_catalog_path.write_text("catalog_name: source\nfields: []\n", encoding="utf-8")
    base_experiment_config.write_text("run:\n  experiment_name: exp_recovery\n", encoding="utf-8")

    data_config_path = repo_root / "configs" / "data" / "generated_data.yaml"
    experiment_config_path = repo_root / "configs" / "experiment" / "generated_experiment.yaml"

    write_frozen_data_config(
        output_path=data_config_path,
        frozen_catalog_path=frozen_catalog_path,
        data_name="real-pilot-frozen",
        source_catalog_path=source_catalog_path,
        repo_root=repo_root,
    )
    write_frozen_experiment_config(
        output_path=experiment_config_path,
        base_experiment_config=base_experiment_config,
        frozen_catalog_path=frozen_catalog_path,
        repo_root=repo_root,
    )

    data_payload = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
    experiment_payload = yaml.safe_load(experiment_config_path.read_text(encoding="utf-8"))

    assert data_payload["data"]["carrier_catalog_path"] == "configs/data/frozen/real_pilot_catalog__gpt2__v1.yaml"
    assert data_payload["data"]["source_carrier_catalog_path"] == (
        "configs/data/source/real_pilot_catalog__gpt2__src_v2.yaml"
    )
    assert experiment_payload["includes"] == ["exp_recovery.yaml"]
    assert experiment_payload["data"]["carrier_catalog_path"] == (
        "configs/data/frozen/real_pilot_catalog__gpt2__v1.yaml"
    )
