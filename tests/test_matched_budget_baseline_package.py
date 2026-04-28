from __future__ import annotations

import csv
import json
import subprocess
import sys
import types
from pathlib import Path

from scripts.eval import _compiled_expected_payload
from scripts.prepare_matched_budget_baseline_calibration import _eval_cases
from src.baselines.base import build_baseline_adapter
from src.core.contract_compiler import CompiledEvalContract
from src.evaluation.canonical_source import load_canonical_evidence_source
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


def test_english_random_baseline_adapter_is_executable(tmp_path: Path) -> None:
    adapter = build_baseline_adapter("baseline_english_random")
    response = adapter.verify(
        {
            "config": {
                "run": {"seed": 17},
                "model": {
                    "name": "qwen2.5-7b-instruct",
                    "tokenizer_name": "Qwen/Qwen2.5-7B-Instruct",
                },
                "eval": {"payload_text": "U00", "min_score": 1.0},
            }
        },
        tmp_path,
    )

    assert response.status == "completed"
    assert response.adapter_name == "baseline_english_random"
    assert response.payload["accepted"] is False
    assert response.payload["baseline_contract_hash"]
    assert (tmp_path / "english_random_fingerprint_result.json").exists()


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
    assert payload["paper_ready_target_case_count"] == 36
    assert payload["train_manifest_entry_count"] == 24
    assert payload["eval_manifest_entry_count"] == 36
    assert payload["calibration_method_count"] == 4
    assert payload["fixed_contract"]["query_budget"] == 4
    assert payload["fixed_contract"]["target_far"] == 0.01
    assert payload["cases"][0]["case_id"] == "fixed_representative_U00_s17"

    train_manifest = load_manifest(train_manifest_path)
    eval_manifest = load_manifest(eval_manifest_path)
    assert len(train_manifest.entries) == 24
    assert len(eval_manifest.entries) == 36
    first_train = train_manifest.entries[0]
    first_eval = eval_manifest.entries[0]
    assert first_train.primary_config_path == (
        "configs/experiment/prep/baseline/exp_train__qwen2_5_7b__matched_budget_baselines_v1.yaml"
    )
    assert first_eval.primary_config_path == (
        "configs/experiment/prep/baseline/exp_eval__qwen2_5_7b__matched_budget_baselines_v1.yaml"
    )
    assert "train.objective=fixed_representative" in first_train.overrides
    assert "eval.min_score=1.0" in first_eval.overrides
    english_eval = next(
        entry
        for entry in eval_manifest.entries
        if entry.manifest_id == "baseline-eval-english_random_active_fingerprint-u00-s17"
    )
    assert "run.method_name=baseline_english_random" in english_eval.overrides
    assert "results/raw/baseline_placeholder/latest_eval_input.json" in "\n".join(english_eval.overrides)
    assert not any("kgw_provenance_control" in entry.manifest_id for entry in eval_manifest.entries)


def test_prepare_matched_budget_baseline_calibration_manifests(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_root_base = tmp_path / "scratch" / "tokenizer-evidence" / "matched_budget_baselines_v1"
    output = tmp_path / "baseline_calibration_package_dry_run.json"
    train_manifest_path = tmp_path / "calibration_train_manifest.json"
    eval_manifest_path = tmp_path / "calibration_eval_manifest.json"
    null_source_manifest_path = tmp_path / "calibration_null_source_manifest.json"

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
            "--null-source-manifest-out",
            str(null_source_manifest_path),
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
    assert payload["null_source_manifest_entry_count"] == 16
    assert payload["eval_manifest_entry_count"] == 48
    assert payload["available_negative_sets"] == [
        "foundation_null",
        "organic_prompt_null",
        "wrong_payload_null",
    ]
    assert payload["missing_negative_sets"] == []
    assert payload["threshold_freeze_allowed"] is False

    train_manifest = load_manifest(train_manifest_path)
    eval_manifest = load_manifest(eval_manifest_path)
    null_source_manifest = load_manifest(null_source_manifest_path)
    assert len(train_manifest.entries) == 8
    assert len(null_source_manifest.entries) == 16
    assert len(eval_manifest.entries) == 48
    assert train_manifest.entries[0].manifest_id == (
        "baseline-calibration-train-fixed_representative-u01-s41"
    )
    null_source = next(
        entry
        for entry in null_source_manifest.entries
        if entry.manifest_id
        == "baseline-calibration-source-fixed_representative-organic_prompt_null-claim-u01-s41"
    )
    assert null_source.entry_point == "scripts/generate_baseline_null_input.py"
    assert "train.objective=organic_prompt_null" in null_source.overrides
    assert "organic_prompt_null" in null_source.tags
    wrong_payload_eval = next(
        entry
        for entry in eval_manifest.entries
        if entry.manifest_id == "baseline-calibration-eval-fixed_representative-u01-claim-u05-s41"
    )
    assert "eval.payload_text=U05" in wrong_payload_eval.overrides
    assert "eval.expected_payload_source=config" in wrong_payload_eval.overrides
    assert (
        str(output_root_base / "calibration" / "fixed_representative" / "U01_s41")
        in wrong_payload_eval.output_root
    )
    organic_eval = next(
        entry
        for entry in eval_manifest.entries
        if entry.manifest_id
        == "baseline-calibration-eval-fixed_representative-organic_prompt_null-claim-u01-s41"
    )
    assert "organic_prompt_null" in organic_eval.tags
    assert (
        str(
            output_root_base
            / "calibration"
            / "fixed_representative"
            / "organic_prompt_null"
            / "U01_s41"
        )
        in organic_eval.output_root
    )


def test_canonical_source_can_use_config_payload_as_false_claim(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    eval_input = tmp_path / "latest_eval_input.json"
    eval_input.write_text(
        json.dumps({"payload_text": "U01"}, sort_keys=True),
        encoding="utf-8",
    )

    source = load_canonical_evidence_source(
        repo_root=repo_root,
        eval_path=str(eval_input),
        default_payload_text="U05",
        prefer_default_payload_text=True,
    )

    assert source.expected_payload_bytes == b"U05"
    assert source.diagnostics["payload_source"] == "config.eval.payload_text_override"


def test_compiled_gate_false_claim_uses_payload_units_from_train_contract(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    train_contract_path = tmp_path / "compiled_train_contract.json"
    eval_contract = {
        "payload_label": "U01",
        "payload_units": [1, 14],
        "block_count": 2,
        "fields_per_block": 2,
        "slot_field_names": ["SECTION", "TOPIC", "SECTION", "TOPIC"],
        "expected_slot_values": ["news", "market", "report", "travel"],
        "exact_slot_prefixes": ["a", "b", "c", "d"],
        "prompt_contract_name": "compiled_slot_request_v1",
        "render_format": "canonical_v1",
        "artifact_format": "compiled_slot_values",
    }
    train_contract_path.write_text(
        json.dumps(
            {
                "schema_name": "compiled_train_contract",
                "model_name": "qwen2.5-7b-instruct",
                "tokenizer_name": "Qwen/Qwen2.5-7B-Instruct",
                "tokenizer_backend": "huggingface",
                "tokenizer_contract_hash": "tok",
                "catalog_path": "catalog.yaml",
                "catalog_sha256": "catalog",
                "catalog_name": "test",
                "prompt_contract_name": "compiled_slot_request_v1",
                "prompt_contract_hash": "prompt",
                "dataset_hash": "dataset",
                "contract_hash": "contract",
                "payload_label_to_units": {"U01": [1, 14], "U05": [5, 10]},
                "fields_per_block": 2,
                "block_count": 2,
                "render_format": "canonical_v1",
                "sample_count": 0,
                "samples": [],
                "eval_contract": eval_contract,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    expected_payload, label, units = _compiled_expected_payload(
        config=types.SimpleNamespace(
            eval=types.SimpleNamespace(payload_text="U05", expected_payload_source="config")
        ),
        diagnostics={"compiled_train_contract_path": str(train_contract_path)},
        compiled_eval_contract=CompiledEvalContract.from_dict(eval_contract),
        repo_root=repo_root,
    )

    assert expected_payload == (5, 10)
    assert label == "U05"
    assert units == (5, 10)


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
            "--audit-doc",
            str(tmp_path / "baseline_artifact_audit.md"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "wrote baseline summary" in completed.stdout
    summary = json.loads((output_dir / "baseline_summary.json").read_text(encoding="utf-8"))
    assert summary["target_count"] == 36
    assert summary["reporting_row_count"] == 48
    assert summary["completed_count"] == 0
    assert summary["valid_completed_count"] == 0
    assert summary["pending_count"] == 36
    assert summary["unavailable_count"] == 12
    assert summary["control_unavailable_count"] == 12
    assert summary["paper_ready"] is False
    assert summary["fixed_contract"]["query_budget"] == 4
    assert summary["paper_ready_checks"]["calibration_thresholds_frozen_before_final"] is True

    rows = list(csv.DictReader((tables_dir / "matched_budget_baselines.csv").open()))
    assert len(rows) == 48
    assert {row["status"] for row in rows} == {"pending", "unavailable"}
    assert rows[0]["baseline_role"] == "primary_ownership_baseline"
    assert rows[0]["frozen_threshold"] == "1.0"
    kgw = next(row for row in rows if row["method_slug"] == "kgw_provenance_control")
    assert kgw["paper_ready_denominator"] == "False"
    assert kgw["result_class"] == "task_mismatched_control_unavailable"
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
    assert summary["case_count"] == 48
    assert summary["completed_count"] == 0
    assert summary["pending_count"] == 48
    assert summary["thresholds_frozen"] is False
    assert summary["missing_negative_sets"] == []

    rows = list(csv.DictReader((tables_dir / "baseline_calibration_cases.csv").open()))
    assert len(rows) == 48
    assert {row["eval_kind"] for row in rows} == {
        "foundation_null",
        "organic_prompt_null",
        "positive",
        "wrong_payload_null",
    }
    assert rows[0]["owner_payload"] == "U01"
    assert (tables_dir / "baseline_far_summary.csv").exists()
    assert (tables_dir / "baseline_utility_summary.csv").exists()


def test_build_matched_budget_baseline_calibration_maps_claim_payload_runs(
    tmp_path: Path,
) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    case_root_base = tmp_path / "scratch" / "tokenizer-evidence" / "matched_budget_baselines_v1"
    owner_root = case_root_base / "calibration" / "fixed_representative" / "U01_s41"

    for claim_payload, accepted in {"U01": True, "U05": False}.items():
        run_dir = owner_root / "runs" / "exp_eval" / f"run_claim_{claim_payload}"
        run_dir.mkdir(parents=True)
        (run_dir / "config.resolved.yaml").write_text(
            f"eval:\n  payload_text: {claim_payload}\n  expected_payload_source: config\n",
            encoding="utf-8",
        )
        (run_dir / "eval_summary.json").write_text(
            json.dumps(
                {
                    "schema_name": "eval_run_summary",
                    "schema_version": 3,
                    "run_id": f"run_claim_{claim_payload}",
                    "experiment_name": "exp_eval",
                    "method_name": "our_method",
                    "model_name": "qwen2.5-7b-instruct",
                    "seed": 41,
                    "git_commit": "test",
                    "timestamp": "20260428T000000Z",
                    "hostname": "test",
                    "slurm_job_id": None,
                    "status": "completed" if accepted else "failed",
                    "dataset_name": "matched-budget-baselines-v1",
                    "sample_count": 1,
                    "accepted": accepted,
                    "match_ratio": 1.0 if accepted else 0.0,
                    "threshold": 0.5,
                    "verification_mode": "compiled_gate",
                    "verifier_success": accepted,
                    "decoded_payload": claim_payload if accepted else "",
                    "utility_acceptance_rate": 1.0 if accepted else 0.0,
                    "run_dir": str(run_dir),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_matched_budget_baseline_calibration_artifacts.py",
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
            "--case-root-base",
            str(case_root_base),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    rows = list(csv.DictReader((tables_dir / "baseline_calibration_cases.csv").open()))
    positive = next(
        row
        for row in rows
        if row["method_slug"] == "fixed_representative"
        and row["owner_payload"] == "U01"
        and row["claim_payload"] == "U01"
    )
    wrong_claim = next(
        row
        for row in rows
        if row["method_slug"] == "fixed_representative"
        and row["owner_payload"] == "U01"
        and row["claim_payload"] == "U05"
    )

    assert positive["eval_summary_path"].endswith("run_claim_U01/eval_summary.json")
    assert wrong_claim["eval_summary_path"].endswith("run_claim_U05/eval_summary.json")
    assert wrong_claim["accepted"] == "False"
    assert wrong_claim["result_class"] == "valid_completed"
    assert wrong_claim["score_name"] == "claim_conditioned_match_ratio"
    assert wrong_claim["ownership_score"] == "0.0"


def test_build_matched_budget_baseline_calibration_freezes_thresholds_when_complete(
    tmp_path: Path,
) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    case_root_base = tmp_path / "scratch" / "tokenizer-evidence" / "matched_budget_baselines_v1"
    package_config = json.loads(
        json.dumps(
            {
                "calibration_split": {
                    "payloads": ["U01", "U05", "U09", "U13"],
                    "seed": 41,
                    "negative_sets": [
                        "foundation_null",
                        "wrong_payload_null",
                        "organic_prompt_null",
                    ],
                },
                "fixed_contract": {
                    "block_count": 2,
                    "query_budget": 4,
                    "target_far": 0.01,
                },
                "baseline_methods": [
                    {
                        "id": "fixed_representative",
                        "slug": "fixed_representative",
                        "method_name": "our_method",
                        "baseline_family": "fixed_representative",
                        "baseline_role": "primary_ownership_baseline",
                        "train_objective": "fixed_representative",
                        "requires_training": True,
                        "requires_external_integration": False,
                    },
                    {
                        "id": "uniform_bucket",
                        "slug": "uniform_bucket",
                        "method_name": "our_method",
                        "baseline_family": "uniform_bucket",
                        "baseline_role": "primary_ownership_baseline",
                        "train_objective": "uniform_bucket",
                        "requires_training": True,
                        "requires_external_integration": False,
                    },
                ],
            }
        )
    )

    for case in _eval_cases(package_config, str(case_root_base)):
        run_dir = (
            Path(str(case["case_root"]))
            / "runs"
            / "exp_eval"
            / f"synthetic_complete_{case['case_id']}"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "config.resolved.yaml").write_text(
            f"eval:\n  payload_text: {case['claim_payload']}\n  expected_payload_source: config\n",
            encoding="utf-8",
        )
        accepted = bool(case["label"])
        (run_dir / "eval_summary.json").write_text(
            json.dumps(
                {
                    "schema_name": "eval_run_summary",
                    "schema_version": 3,
                    "run_id": str(case["case_id"]),
                    "experiment_name": "exp_eval",
                    "method_name": "our_method",
                    "model_name": "qwen2.5-7b-instruct",
                    "seed": 41,
                    "git_commit": "test",
                    "timestamp": "20260428T000000Z",
                    "hostname": "test",
                    "slurm_job_id": None,
                    "status": "completed" if accepted else "failed",
                    "dataset_name": "matched-budget-baselines-v1",
                    "sample_count": 1,
                    "accepted": accepted,
                    "match_ratio": 1.0 if accepted else 0.0,
                    "threshold": 0.5,
                    "verification_mode": "compiled_gate",
                    "verifier_success": accepted,
                    "decoded_payload": case["claim_payload"] if accepted else "",
                    "utility_acceptance_rate": 1.0 if accepted else 0.0,
                    "run_dir": str(run_dir),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_matched_budget_baseline_calibration_artifacts.py",
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
            "--case-root-base",
            str(case_root_base),
            "--eval-registry",
            str(tmp_path / "empty_registry.jsonl"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads((output_dir / "baseline_calibration_summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 48
    assert summary["completed_count"] == 48
    assert summary["pending_count"] == 0
    assert summary["thresholds_frozen"] is True
    assert summary["missing_negative_sets"] == []
    assert {row["threshold_status"] for row in summary["method_rows"]} == {"frozen"}
    assert {row["frozen_threshold"] for row in summary["method_rows"]} == {1.0}


def test_build_matched_budget_baseline_calibration_blocks_zero_sensitivity_threshold(
    tmp_path: Path,
) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    case_root_base = tmp_path / "scratch" / "tokenizer-evidence" / "matched_budget_baselines_v1"
    package_config = json.loads(
        json.dumps(
            {
                "calibration_split": {
                    "payloads": ["U01", "U05", "U09", "U13"],
                    "seed": 41,
                    "negative_sets": [
                        "foundation_null",
                        "wrong_payload_null",
                        "organic_prompt_null",
                    ],
                },
                "fixed_contract": {
                    "block_count": 2,
                    "query_budget": 4,
                    "target_far": 0.01,
                },
                "baseline_methods": [
                    {
                        "id": "fixed_representative",
                        "slug": "fixed_representative",
                        "method_name": "our_method",
                        "baseline_family": "fixed_representative",
                        "baseline_role": "primary_ownership_baseline",
                        "train_objective": "fixed_representative",
                        "requires_training": True,
                        "requires_external_integration": False,
                    },
                    {
                        "id": "uniform_bucket",
                        "slug": "uniform_bucket",
                        "method_name": "our_method",
                        "baseline_family": "uniform_bucket",
                        "baseline_role": "primary_ownership_baseline",
                        "train_objective": "uniform_bucket",
                        "requires_training": True,
                        "requires_external_integration": False,
                    },
                ],
            }
        )
    )

    for case in _eval_cases(package_config, str(case_root_base)):
        run_dir = (
            Path(str(case["case_root"]))
            / "runs"
            / "exp_eval"
            / f"synthetic_overlap_{case['case_id']}"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "config.resolved.yaml").write_text(
            f"eval:\n  payload_text: {case['claim_payload']}\n  expected_payload_source: config\n",
            encoding="utf-8",
        )
        accepted = bool(case["label"]) or (
            case["method_slug"] == "fixed_representative" and bool(case["negative_set"])
        )
        score = 1.0 if accepted else 0.0
        (run_dir / "eval_summary.json").write_text(
            json.dumps(
                {
                    "schema_name": "eval_run_summary",
                    "schema_version": 3,
                    "run_id": str(case["case_id"]),
                    "experiment_name": "exp_eval",
                    "method_name": "our_method",
                    "model_name": "qwen2.5-7b-instruct",
                    "seed": 41,
                    "git_commit": "test",
                    "timestamp": "20260428T000000Z",
                    "hostname": "test",
                    "slurm_job_id": None,
                    "status": "completed" if accepted else "failed",
                    "dataset_name": "matched-budget-baselines-v1",
                    "sample_count": 1,
                    "accepted": accepted,
                    "match_ratio": score,
                    "threshold": 0.5,
                    "verification_mode": "compiled_gate",
                    "verifier_success": accepted,
                    "decoded_payload": case["claim_payload"] if accepted else "",
                    "utility_acceptance_rate": 1.0 if accepted else 0.0,
                    "run_dir": str(run_dir),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_matched_budget_baseline_calibration_artifacts.py",
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
            "--case-root-base",
            str(case_root_base),
            "--eval-registry",
            str(tmp_path / "empty_registry.jsonl"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads((output_dir / "baseline_calibration_summary.json").read_text(encoding="utf-8"))
    fixed = next(row for row in summary["method_rows"] if row["method_slug"] == "fixed_representative")
    uniform = next(row for row in summary["method_rows"] if row["method_slug"] == "uniform_bucket")
    assert summary["thresholds_frozen"] is False
    assert fixed["threshold_status"] == "far_unmatched"
    assert fixed["frozen_threshold"] == ""
    assert fixed["true_accept_count"] == 4
    assert uniform["threshold_status"] == "frozen"


def test_build_matched_budget_baseline_artifacts_requires_real_contract_hash_match(
    tmp_path: Path,
) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    case_root_base = tmp_path / "scratch" / "tokenizer-evidence" / "matched_budget_baselines_v1"
    case_root = case_root_base / "final" / "fixed_representative" / "U00_s17"
    train_run_dir = case_root / "runs" / "exp_train" / "synthetic_train"
    eval_run_dir = case_root / "runs" / "exp_eval" / "synthetic_eval"
    train_run_dir.mkdir(parents=True)
    eval_run_dir.mkdir(parents=True)
    checkpoint_path = train_run_dir / "adapter" / "adapter.bin"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_text("adapter", encoding="utf-8")
    train_eval_contract = {
        "payload_label": "U00",
        "payload_units": [0, 0],
        "block_count": 2,
        "fields_per_block": 2,
        "slot_field_names": ["SECTION", "TOPIC", "SECTION", "TOPIC"],
        "expected_slot_values": ["news", "market", "report", "travel"],
        "exact_slot_prefixes": ["SECTION=", "TOPIC=", "SECTION=", "TOPIC="],
        "prompt_contract_name": "compiled_slot_request_v1",
        "render_format": "canonical_v1",
        "artifact_format": "compiled_slot_values",
    }
    train_contract = {
        "schema_name": "compiled_train_contract",
        "model_name": "qwen2.5-7b-instruct",
        "tokenizer_name": "Qwen/Qwen2.5-7B-Instruct",
        "tokenizer_backend": "huggingface",
        "tokenizer_contract_hash": "tok",
        "catalog_path": "catalog.yaml",
        "catalog_sha256": "catalog",
        "catalog_name": "test",
        "prompt_contract_name": "compiled_slot_request_v1",
        "prompt_contract_hash": "prompt",
        "dataset_hash": "dataset",
        "contract_hash": "trainhash",
        "payload_label_to_units": {"U00": [0, 0]},
        "fields_per_block": 2,
        "block_count": 2,
        "render_format": "canonical_v1",
        "sample_count": 1,
        "samples": [
            {
                "field_name": "SECTION",
                "bucket_to_token_ids": {"0": [1], "1": [2]},
            }
        ],
        "eval_contract": train_eval_contract,
    }
    (train_run_dir / "compiled_train_contract.json").write_text(
        json.dumps(train_contract, sort_keys=True),
        encoding="utf-8",
    )
    (train_run_dir / "compiled_eval_contract.json").write_text(
        json.dumps(train_eval_contract, sort_keys=True),
        encoding="utf-8",
    )
    (train_run_dir / "config.resolved.yaml").write_text(
        "train:\n  objective: fixed_representative\n  generation_prompt: Select.\n  generation_do_sample: false\n  generation_max_new_tokens: 1\n",
        encoding="utf-8",
    )
    latest_eval_input = {
        "compiled_train_contract_hash": "trainhash",
        "compiled_train_contract_path": str(train_run_dir / "compiled_train_contract.json"),
        "compiled_eval_contract": train_eval_contract,
        "checkpoint_path": str(checkpoint_path.parent),
    }
    (case_root / "runs" / "exp_train" / "latest_eval_input.json").write_text(
        json.dumps(latest_eval_input, sort_keys=True),
        encoding="utf-8",
    )
    (train_run_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "schema_name": "train_run_summary",
                "schema_version": 3,
                "run_id": "synthetic_train",
                "experiment_name": "exp_train",
                "method_name": "our_method",
                "model_name": "qwen2.5-7b-instruct",
                "seed": 17,
                "git_commit": "test",
                "timestamp": "20260428T000000Z",
                "hostname": "test",
                "slurm_job_id": None,
                "status": "completed",
                "objective": "fixed_representative",
                "dataset_name": "matched-budget-baselines-v1",
                "dataset_size": 1,
                "steps": 1,
                "final_loss": 0.0,
                "run_dir": str(train_run_dir),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (eval_run_dir / "eval_summary.json").write_text(
        json.dumps(
            {
                "schema_name": "eval_run_summary",
                "schema_version": 3,
                "run_id": "synthetic_eval",
                "experiment_name": "exp_eval",
                "method_name": "our_method",
                "model_name": "qwen2.5-7b-instruct",
                "seed": 17,
                "git_commit": "test",
                "timestamp": "20260428T000000Z",
                "hostname": "test",
                "slurm_job_id": None,
                "status": "completed",
                "dataset_name": "matched-budget-baselines-v1",
                "sample_count": 1,
                "accepted": True,
                "match_ratio": 1.0,
                "threshold": 1.0,
                "verification_mode": "compiled_gate",
                "render_format": "canonical_v1",
                "verifier_success": True,
                "decoded_payload": "U00",
                "utility_acceptance_rate": 1.0,
                "diagnostics": {
                    "compiled_train_contract_hash": "trainhash",
                    "compiled_eval_contract": train_eval_contract,
                },
                "run_dir": str(eval_run_dir),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_matched_budget_baseline_artifacts.py",
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
            "--case-root-base",
            str(case_root_base),
            "--audit-doc",
            str(tmp_path / "baseline_artifact_audit.md"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    rows = list(csv.DictReader((tables_dir / "matched_budget_baselines.csv").open()))
    row = next(item for item in rows if item["case_id"] == "fixed_representative_U00_s17")
    assert row["valid_completed"] == "True"
    assert row["success"] == "True"
    assert row["contract_hash_status"] == "match"
    assert row["contract_hash_missing_fields"] == ""
