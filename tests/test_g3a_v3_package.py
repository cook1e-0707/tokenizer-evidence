from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.infrastructure.manifest import build_manifest_from_config, load_manifest
from src.infrastructure.paths import discover_repo_root
from src.training.dataset import TrainingExample
from src.training.hf_causal_lm import _compute_compiled_bucket_loss_with_diagnostics


def test_g3a_v3_configs_emit_qwen_train_and_eval_manifests() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)

    train_manifest = build_manifest_from_config(
        repo_root
        / "configs"
        / "experiment"
        / "scale"
        / "g3a_v3"
        / "exp_train__qwen2_5_7b__g3a_block_scale_v3.yaml"
    )
    eval_manifest = build_manifest_from_config(
        repo_root
        / "configs"
        / "experiment"
        / "scale"
        / "g3a_v3"
        / "exp_eval__qwen2_5_7b__g3a_block_scale_v3.yaml"
    )

    train_entry = train_manifest.entries[0]
    eval_entry = eval_manifest.entries[0]

    assert train_entry.entry_point == "scripts/train.py"
    assert eval_entry.entry_point == "scripts/eval.py"
    assert train_entry.model_name == "qwen2.5-7b-instruct"
    assert train_entry.requested_resources.partition == "DGXA100"
    assert train_entry.requested_resources.num_gpus == 1
    assert train_entry.requested_resources.time_limit == "24:00:00"


def test_prepare_g3a_v3_validation_manifests_and_blocks_unfrozen_final(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_root_base = tmp_path / "scratch" / "tokenizer-evidence" / "g3a_block_scale_v3"

    validation_output = tmp_path / "validation_dry_run.json"
    validation_train_manifest = tmp_path / "validation_train_manifest.json"
    validation_eval_manifest = tmp_path / "validation_eval_manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_g3a_v3_block_scale.py",
            "--phase",
            "validation",
            "--output",
            str(validation_output),
            "--train-manifest-out",
            str(validation_train_manifest),
            "--eval-manifest-out",
            str(validation_eval_manifest),
            "--output-root-base",
            str(output_root_base),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote G3a-v3 validation dry-run summary" in completed.stdout
    validation_payload = json.loads(validation_output.read_text(encoding="utf-8"))
    assert validation_payload["target_case_count"] == 64
    assert validation_payload["train_manifest_entry_count"] == 64
    assert validation_payload["eval_manifest_entry_count"] == 64
    validation_train = load_manifest(validation_train_manifest)
    assert validation_train.entries[0].manifest_id == "g3a-v3-validation-train-hp01-b1-u00-s41"
    assert "train.margin_gamma=0.5" in validation_train.entries[0].overrides
    assert "train.lambda_margin=0.25" in validation_train.entries[0].overrides
    assert "train.checkpoint_selection_metric=training_total_evidence_loss_mean" in validation_train.entries[0].overrides
    assert "train.checkpoint_selection_mode=min" in validation_train.entries[0].overrides
    margin_metric_entry = next(
        entry
        for entry in validation_train.entries
        if "train.checkpoint_selection_metric=training_min_slot_margin" in entry.overrides
    )
    assert "train.checkpoint_selection_mode=max" in margin_metric_entry.overrides

    final_output = tmp_path / "final_dry_run.json"
    final_completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_g3a_v3_block_scale.py",
            "--phase",
            "final",
            "--output",
            str(final_output),
            "--output-root-base",
            str(output_root_base),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert final_completed.returncode != 0
    assert "final manifests are blocked" in final_completed.stderr
    assert not final_output.exists()


def test_build_g3a_v3_artifacts_writes_pending_package(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    case_root = tmp_path / "scratch" / "tokenizer-evidence" / "g3a_block_scale_v3"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_g3a_v3_block_scale_artifacts.py",
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
            "--new-case-root-base",
            str(case_root),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote G3a-v3 summary" in completed.stdout
    summary = json.loads((output_dir / "g3a_v3_summary.json").read_text(encoding="utf-8"))
    assert summary["target_count"] == 144
    assert summary["completed_count"] == 0
    assert summary["pending_count"] == 144
    assert summary["paper_ready"] is False
    assert summary["paper_ready_checks"]["hyperparameters_frozen_before_final_matrix_launch"] is False

    rows = list(csv.DictReader((tables_dir / "g3a_v3_block_scale.csv").open()))
    assert len(rows) == 144
    assert {row["status"] for row in rows} == {"pending"}
    assert "min_bucket_margin" in rows[0]
    assert "exact_gate_success" in rows[0]
    assert (tables_dir / "g3a_v3_block_scale.tex").exists()
    assert (tables_dir / "g3a_v3_slot_margin.csv").exists()
    assert (tables_dir / "g3a_v3_failure_cases.csv").exists()
    assert (output_dir / "g3a_v3_run_inclusion_list.json").exists()
    assert (output_dir / "g3a_v3_compute_accounting.json").exists()


def test_margin_aware_bucket_loss_adds_hinge_margin_term() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.2, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1]], dtype=torch.long)
    example = TrainingExample(
        prompt="Controlled margin objective prompt.",
        target_symbols=(),
        metadata={
            "compiled_allowed_token_ids": [2, 3, 4],
            "compiled_bucket_to_token_ids": {"0": [2], "1": [3], "2": [4]},
            "compiled_target_bucket_id": 1,
            "compiled_target_token_id": 3,
        },
    )

    loss, _max_logit, diagnostics = _compute_compiled_bucket_loss_with_diagnostics(
        torch_module=torch,
        logits=logits,
        attention_mask=attention_mask,
        batch_examples=[example],
        objective_mode="margin_aware_bucket_mass",
        lambda_set=2.0,
        lambda_margin=0.5,
        margin_gamma=1.0,
    )

    assert diagnostics.slot_count == 1
    assert diagnostics.slot_margin_min < 0.0
    assert diagnostics.normalized_l_margin_mean == pytest.approx(1.8)
    assert diagnostics.total_evidence_loss_mean == pytest.approx(
        2.0 * diagnostics.normalized_l_set_mean + 0.5 * diagnostics.normalized_l_margin_mean
    )
    assert float(loss.item()) == pytest.approx(diagnostics.total_evidence_loss_mean)
