from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from src.baselines.base import build_baseline_adapter
from src.infrastructure.manifest import build_manifest_from_config, load_manifest
from src.infrastructure.paths import discover_repo_root


def test_baseline_configs_emit_manifest_entries() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)

    train_manifest = build_manifest_from_config(
        repo_root
        / "configs"
        / "experiment"
        / "prep"
        / "baseline"
        / "exp_train__qwen2_5_7b__matched_budget_baselines_v1.yaml"
    )
    eval_manifest = build_manifest_from_config(
        repo_root
        / "configs"
        / "experiment"
        / "prep"
        / "baseline"
        / "exp_eval__qwen2_5_7b__matched_budget_baselines_v1.yaml"
    )
    calibrate_manifest = build_manifest_from_config(
        repo_root
        / "configs"
        / "experiment"
        / "prep"
        / "baseline"
        / "exp_calibrate__qwen2_5_7b__matched_budget_baselines_v1.yaml"
    )

    assert train_manifest.entries[0].entry_point == "scripts/train.py"
    assert eval_manifest.entries[0].entry_point == "scripts/eval.py"
    assert calibrate_manifest.entries[0].entry_point == "scripts/calibrate.py"
    assert train_manifest.entries[0].requested_resources.partition == "DGXA100"
    assert eval_manifest.entries[0].requested_resources.num_gpus == 1
    train_config_text = (
        repo_root
        / "configs"
        / "experiment"
        / "prep"
        / "baseline"
        / "exp_train__qwen2_5_7b__matched_budget_baselines_v1.yaml"
    ).read_text(encoding="utf-8")
    assert "checkpoint_selection_metric: training_normalized_l_set_mean" in train_config_text
    assert "checkpoint_selection_metric: training_loss" not in train_config_text


def test_english_random_baseline_placeholder_adapter_is_registered() -> None:
    adapter = build_baseline_adapter("baseline_english_random")
    response = adapter.verify({}, Path("."))

    assert response.status == "placeholder"
    assert response.adapter_name == "baseline_english_random"
    assert "English-random active fingerprint" in response.payload["integration_hint"]


def test_prepare_matched_budget_baseline_manifests(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_root_base = tmp_path / "scratch" / "tokenizer-evidence" / "matched_budget_baselines_v1"
    output = tmp_path / "baseline_package_dry_run.json"
    train_manifest_path = tmp_path / "train_manifest.json"
    eval_manifest_path = tmp_path / "eval_manifest.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_matched_budget_baselines.py",
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

    assert "wrote baseline dry-run summary" in completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["target_case_count"] == 48
    assert payload["train_manifest_entry_count"] == 24
    assert payload["eval_manifest_entry_count"] == 48
    assert payload["calibration_method_count"] == 4
    assert payload["fixed_contract"]["query_budget"] == 4
    assert payload["fixed_contract"]["target_far"] == 0.01
    assert payload["cases"][0]["case_id"] == "fixed_representative_U00_s17"

    train_manifest = load_manifest(train_manifest_path)
    eval_manifest = load_manifest(eval_manifest_path)
    assert len(train_manifest.entries) == 24
    assert len(eval_manifest.entries) == 48
    first_train = train_manifest.entries[0]
    first_eval = eval_manifest.entries[0]
    assert first_train.primary_config_path == (
        "configs/experiment/prep/baseline/exp_train__qwen2_5_7b__matched_budget_baselines_v1.yaml"
    )
    assert first_eval.primary_config_path == (
        "configs/experiment/prep/baseline/exp_eval__qwen2_5_7b__matched_budget_baselines_v1.yaml"
    )
    assert "train.objective=fixed_representative" in first_train.overrides
    english_eval = next(
        entry
        for entry in eval_manifest.entries
        if entry.manifest_id == "baseline-eval-english_random_active_fingerprint-u00-s17"
    )
    assert "run.method_name=baseline_english_random" in english_eval.overrides
    assert "results/raw/baseline_placeholder/latest_eval_input.json" in "\n".join(english_eval.overrides)


def test_prepare_matched_budget_baseline_calibration_manifests(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_root_base = tmp_path / "scratch" / "tokenizer-evidence" / "matched_budget_baselines_v1"
    output = tmp_path / "baseline_calibration_package_dry_run.json"
    train_manifest_path = tmp_path / "calibration_train_manifest.json"
    eval_manifest_path = tmp_path / "calibration_eval_manifest.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_matched_budget_baseline_calibration.py",
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

    assert "wrote baseline calibration dry-run summary" in completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["train_manifest_entry_count"] == 8
    assert payload["eval_manifest_entry_count"] == 32
    assert payload["available_negative_sets"] == ["wrong_payload_null"]
    assert payload["missing_negative_sets"] == ["foundation_null", "organic_prompt_null"]
    assert payload["threshold_freeze_allowed"] is False

    train_manifest = load_manifest(train_manifest_path)
    eval_manifest = load_manifest(eval_manifest_path)
    assert len(train_manifest.entries) == 8
    assert len(eval_manifest.entries) == 32
    assert train_manifest.entries[0].manifest_id == (
        "baseline-calibration-train-fixed_representative-u01-s41"
    )
    wrong_payload_eval = next(
        entry
        for entry in eval_manifest.entries
        if entry.manifest_id == "baseline-calibration-eval-fixed_representative-u01-claim-u05-s41"
    )
    assert "eval.payload_text=U05" in wrong_payload_eval.overrides
    assert (
        str(output_root_base / "calibration" / "fixed_representative" / "U01_s41")
        in wrong_payload_eval.output_root
    )


def test_build_matched_budget_baseline_artifacts_writes_pending_package(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    case_root = tmp_path / "scratch" / "tokenizer-evidence" / "matched_budget_baselines_v1"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_matched_budget_baseline_artifacts.py",
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

    assert "wrote baseline summary" in completed.stdout
    summary = json.loads((output_dir / "baseline_summary.json").read_text(encoding="utf-8"))
    assert summary["target_count"] == 48
    assert summary["completed_count"] == 0
    assert summary["valid_completed_count"] == 0
    assert summary["pending_count"] == 48
    assert summary["paper_ready"] is False
    assert summary["fixed_contract"]["query_budget"] == 4
    assert summary["paper_ready_checks"]["calibration_thresholds_frozen_before_final"] is False

    rows = list(csv.DictReader((tables_dir / "matched_budget_baselines.csv").open()))
    assert len(rows) == 48
    assert {row["status"] for row in rows} == {"pending"}
    assert rows[0]["baseline_role"] == "primary_ownership_baseline"
    assert (tables_dir / "matched_budget_baselines.tex").exists()
    assert (tables_dir / "baseline_calibration.csv").exists()
    assert (tables_dir / "baseline_far_summary.csv").exists()
    assert (tables_dir / "baseline_utility_summary.csv").exists()
    assert (output_dir / "baseline_run_inclusion_list.json").exists()
    assert (output_dir / "baseline_calibration_summary.json").exists()
    assert (output_dir / "baseline_compute_accounting.json").exists()


def test_build_matched_budget_baseline_calibration_artifacts_writes_pending_package(
    tmp_path: Path,
) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    case_root = tmp_path / "scratch" / "tokenizer-evidence" / "matched_budget_baselines_v1"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_matched_budget_baseline_calibration_artifacts.py",
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

    assert "wrote baseline calibration summary" in completed.stdout
    summary = json.loads((output_dir / "baseline_calibration_summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 32
    assert summary["completed_count"] == 0
    assert summary["pending_count"] == 32
    assert summary["thresholds_frozen"] is False
    assert summary["missing_negative_sets"] == ["foundation_null", "organic_prompt_null"]

    rows = list(csv.DictReader((tables_dir / "baseline_calibration_cases.csv").open()))
    assert len(rows) == 32
    assert {row["eval_kind"] for row in rows} == {"positive", "wrong_payload_null"}
    assert rows[0]["owner_payload"] == "U01"
    assert (tables_dir / "baseline_far_summary.csv").exists()
    assert (tables_dir / "baseline_utility_summary.csv").exists()
