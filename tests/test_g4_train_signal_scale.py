from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from src.infrastructure.manifest import build_manifest_from_config, load_manifest
from src.infrastructure.paths import discover_repo_root


def test_g4_configs_emit_qwen_train_and_eval_manifests() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)

    train_manifest = build_manifest_from_config(
        repo_root
        / "configs"
        / "experiment"
        / "scale"
        / "g4"
        / "exp_train__qwen2_5_7b__g4_train_signal_scale_v1.yaml"
    )
    eval_manifest = build_manifest_from_config(
        repo_root
        / "configs"
        / "experiment"
        / "scale"
        / "g4"
        / "exp_eval__qwen2_5_7b__g4_train_signal_scale_v1.yaml"
    )

    train_entry = train_manifest.entries[0]
    eval_entry = eval_manifest.entries[0]

    assert train_entry.entry_point == "scripts/train.py"
    assert eval_entry.entry_point == "scripts/eval.py"
    assert train_entry.model_name == "qwen2.5-7b-instruct"
    assert train_entry.requested_resources.partition == "DGXA100"
    assert train_entry.requested_resources.num_gpus == 1
    assert train_entry.requested_resources.time_limit == "24:00:00"


def test_prepare_g4_manifests_use_sample_count_variants(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_root_base = tmp_path / "scratch" / "tokenizer-evidence" / "g4_train_signal_scale_v1"
    output = tmp_path / "g4_package_dry_run.json"
    train_manifest_path = tmp_path / "train_manifest.json"
    eval_manifest_path = tmp_path / "eval_manifest.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_g4_train_signal_scale.py",
            "--output",
            str(output),
            "--train-manifest-out",
            str(train_manifest_path),
            "--eval-manifest-out",
            str(eval_manifest_path),
            "--output-root-base",
            str(output_root_base),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "wrote G4 dry-run summary" in completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["target_case_count"] == 48
    assert payload["train_manifest_entry_count"] == 48
    assert payload["eval_manifest_entry_count"] == 48
    assert payload["package_config_path"] == "configs/reporting/g4_train_signal_scale_v1.yaml"
    assert payload["environment_setup_present"] is True
    assert payload["environment_setup_contains_zkrfa_activate"] is True
    assert payload["cases"][0]["case_id"] == "S16_U00_s17"
    assert payload["cases"][0]["effective_contract_sample_count"] == 16

    train_manifest = load_manifest(train_manifest_path)
    eval_manifest = load_manifest(eval_manifest_path)
    assert len(train_manifest.entries) == 48
    assert len(eval_manifest.entries) == 48
    first_train = train_manifest.entries[0]
    first_eval = eval_manifest.entries[0]
    assert first_train.primary_config_path == (
        "configs/experiment/scale/g4/exp_train__qwen2_5_7b__g4_train_signal_scale_v1.yaml"
    )
    assert first_eval.primary_config_path == (
        "configs/experiment/scale/g4/exp_eval__qwen2_5_7b__g4_train_signal_scale_v1.yaml"
    )
    assert "/Users/" not in first_train.primary_config_path
    assert "zkrfa_py312/bin/activate" in first_train.requested_resources.environment_setup
    assert 'train.probe_payload_texts=["U00","U03","U12","U15"]' in first_train.overrides
    assert "train.compiled_sample_repeats=1" in first_train.overrides
    s128_entry = next(
        entry
        for entry in train_manifest.entries
        if entry.manifest_id == "g4-train-s128-u00-s17"
    )
    assert "train.compiled_sample_repeats=2" in s128_entry.overrides


def test_build_g4_artifacts_writes_pending_package(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    case_root = tmp_path / "scratch" / "tokenizer-evidence" / "g4_train_signal_scale_v1"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_g4_train_signal_scale_artifacts.py",
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
            "--case-root-base",
            str(case_root),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote G4 summary" in completed.stdout
    summary = json.loads((output_dir / "g4_summary.json").read_text(encoding="utf-8"))
    assert summary["target_count"] == 48
    assert summary["completed_count"] == 0
    assert summary["pending_count"] == 48
    assert summary["paper_ready"] is False
    assert summary["fixed_contract"]["block_count"] == 2
    assert summary["sample_count_variants"][-1]["effective_contract_sample_count"] == 128
    assert summary["sample_count_variants"][-1]["unique_contract_sample_count"] == 64

    rows = list(csv.DictReader((tables_dir / "g4_train_scale.csv").open()))
    assert len(rows) == 48
    assert {row["status"] for row in rows} == {"pending"}
    assert rows[0]["effective_contract_sample_count"] == "16"
    assert rows[-1]["effective_contract_sample_count"] == "128"
    assert (tables_dir / "g4_train_scale.tex").exists()
    assert (tables_dir / "g4_failure_cases.csv").exists()
    assert (output_dir / "g4_run_inclusion_list.json").exists()
    assert (output_dir / "g4_compute_accounting.json").exists()
