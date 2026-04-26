import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.core.compiled_repair_diagnostics import build_compiled_verification_report
from src.core.contract_compiler import CompiledEvalContract
from src.core.scaffolded_completion import FoundationGateResult, FoundationSlotDiagnostic
from src.evaluation.report import EvalRunSummary, TrainRunSummary
from src.infrastructure.manifest import build_manifest_from_config, load_manifest
from src.infrastructure.paths import discover_repo_root
from src.training.dataset import TrainingExample
from src.training.hf_causal_lm import _compute_compiled_bucket_loss_with_diagnostics


def test_g3a_v2_configs_emit_qwen_train_and_eval_manifests() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)

    train_manifest = build_manifest_from_config(
        repo_root
        / "configs"
        / "experiment"
        / "scale"
        / "g3a_v2"
        / "exp_train__qwen2_5_7b__g3a_block_scale_v2.yaml"
    )
    eval_manifest = build_manifest_from_config(
        repo_root
        / "configs"
        / "experiment"
        / "scale"
        / "g3a_v2"
        / "exp_eval__qwen2_5_7b__g3a_block_scale_v2.yaml"
    )

    train_entry = train_manifest.entries[0]
    eval_entry = eval_manifest.entries[0]

    assert train_entry.entry_point == "scripts/train.py"
    assert eval_entry.entry_point == "scripts/eval.py"
    assert train_entry.model_name == "qwen2.5-7b-instruct"
    assert train_entry.requested_resources.partition == "DGXA100"
    assert train_entry.requested_resources.num_gpus == 1
    assert train_entry.requested_resources.time_limit == "24:00:00"


def test_prepare_g3a_v2_final_and_pilot_manifests(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_root_base = tmp_path / "scratch" / "tokenizer-evidence" / "g3a_block_scale_v2"

    final_output = tmp_path / "final_dry_run.json"
    final_train_manifest = tmp_path / "final_train_manifest.json"
    final_eval_manifest = tmp_path / "final_eval_manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_g3a_v2_block_scale.py",
            "--phase",
            "final",
            "--output",
            str(final_output),
            "--train-manifest-out",
            str(final_train_manifest),
            "--eval-manifest-out",
            str(final_eval_manifest),
            "--output-root-base",
            str(output_root_base),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote G3a-v2 final dry-run summary" in completed.stdout
    final_payload = json.loads(final_output.read_text(encoding="utf-8"))
    assert final_payload["target_case_count"] == 36
    assert final_payload["train_manifest_entry_count"] == 36
    assert final_payload["eval_manifest_entry_count"] == 36
    final_train = load_manifest(final_train_manifest)
    assert final_train.entries[0].manifest_id == "g3a-v2-final-train-b1-u00-s17"
    assert f"runtime.output_root={output_root_base / 'final' / 'b1' / 'U00_s17' / 'runs'}" in final_train.entries[0].overrides
    assert "train.lambda_set=2.0" not in final_train.entries[0].overrides

    pilot_output = tmp_path / "pilot_dry_run.json"
    pilot_train_manifest = tmp_path / "pilot_train_manifest.json"
    pilot_eval_manifest = tmp_path / "pilot_eval_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_g3a_v2_block_scale.py",
            "--phase",
            "pilot",
            "--output",
            str(pilot_output),
            "--train-manifest-out",
            str(pilot_train_manifest),
            "--eval-manifest-out",
            str(pilot_eval_manifest),
            "--output-root-base",
            str(output_root_base),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    pilot_payload = json.loads(pilot_output.read_text(encoding="utf-8"))
    assert pilot_payload["target_case_count"] == 128
    pilot_train = load_manifest(pilot_train_manifest)
    assert len(pilot_train.entries) == 128
    assert pilot_train.entries[0].manifest_id == "g3a-v2-pilot-train-hp01-b1-u01-s41"
    assert "train.lora_r=16" in pilot_train.entries[0].overrides
    assert "train.learning_rate=5e-05" in pilot_train.entries[0].overrides
    assert "train.epochs=64" in pilot_train.entries[0].overrides
    assert "train.lambda_set=1.0" in pilot_train.entries[0].overrides


def test_build_g3a_v2_artifacts_writes_pending_package(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    case_root = tmp_path / "scratch" / "tokenizer-evidence" / "g3a_block_scale_v2"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_g3a_v2_block_scale_artifacts.py",
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
            "--audit-doc",
            str(tmp_path / "g3a_v2_artifact_audit.md"),
            "--new-case-root-base",
            str(case_root),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote G3a-v2 summary" in completed.stdout
    summary = json.loads((output_dir / "g3a_v2_summary.json").read_text(encoding="utf-8"))
    assert summary["target_count"] == 36
    assert summary["completed_count"] == 0
    assert summary["valid_completed_count"] == 0
    assert summary["success_count"] == 0
    assert summary["method_failure_count"] == 0
    assert summary["invalid_excluded_count"] == 0
    assert summary["pending_count"] == 36
    assert summary["paper_ready"] is False
    assert summary["paper_ready_checks"]["no_pending_runs"] is False

    rows = list(csv.DictReader((tables_dir / "g3a_v2_block_scale.csv").open()))
    assert len(rows) == 36
    assert {row["status"] for row in rows} == {"pending"}
    assert (tables_dir / "g3a_v2_slot_diagnostics.csv").exists()
    assert (tables_dir / "g3a_v2_symbol_diagnostics.csv").exists()
    assert (tables_dir / "g3a_v2_failure_cases.csv").exists()
    assert (tables_dir / "g3a_v2_block_scale.tex").exists()
    assert (output_dir / "g3a_v2_run_inclusion_list.json").exists()
    assert (output_dir / "g3a_v2_compute_accounting.json").exists()


def _write_g3a_v2_case_artifacts(
    case_root: Path,
    *,
    payload: str,
    seed: int,
    block_count: int,
    accepted: bool,
    decoded_payload: str,
) -> None:
    train_run_dir = case_root / "runs" / "exp_train" / f"exp_train__mock__s{seed}"
    eval_run_dir = case_root / "runs" / "exp_eval" / f"exp_eval__mock__s{seed}"
    checkpoint_dir = train_run_dir / "checkpoints" / "hf_best"
    train_run_dir.mkdir(parents=True, exist_ok=True)
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "adapter_config.json").write_text('{"r":16}\n', encoding="utf-8")
    (checkpoint_dir / "adapter_model.safetensors").write_text("tiny-adapter\n", encoding="utf-8")

    payload_units = [int(payload[1:])]
    eval_contract = {
        "payload_label": payload,
        "payload_units": payload_units,
        "expected_slot_values": ["news", "market"],
        "slot_field_names": ["SECTION", "TOPIC"],
        "exact_slot_prefixes": ["SECTION=", "TOPIC="],
        "fields_per_block": 2,
        "block_count": block_count,
        "render_format": "canonical_v1",
        "prompt_contract_name": "compiled_slot_request_v1",
        "artifact_format": "compiled_slot_values",
    }
    train_contract = {
        "schema_name": "compiled_train_contract",
        "model_name": "qwen2.5-7b-instruct",
        "tokenizer_name": "Qwen/Qwen2.5-7B-Instruct",
        "tokenizer_backend": "transformers",
        "tokenizer_contract_hash": "tokenizer-hash",
        "catalog_path": "configs/data/frozen/real_pilot_catalog__qwen2_5_7b_compiled__v1.yaml",
        "catalog_sha256": "catalog-hash",
        "catalog_name": "real-pilot-compiled-c3-g3a-v2",
        "prompt_contract_name": "compiled_slot_request_v1",
        "prompt_contract_hash": "prompt-hash",
        "dataset_hash": "dataset-hash",
        "contract_hash": f"train-contract-{payload}-{block_count}",
        "payload_label_to_units": {"U00": [0], "U03": [3]},
        "fields_per_block": 2,
        "block_count": block_count,
        "render_format": "canonical_v1",
        "sample_count": 2,
        "samples": [
            {
                "field_name": "SECTION",
                "bucket_to_token_ids": {"0": [1], "1": [2], "2": [3], "3": [4]},
            },
            {
                "field_name": "TOPIC",
                "bucket_to_token_ids": {"0": [5], "1": [6], "2": [7], "3": [8]},
            },
        ],
        "eval_contract": eval_contract,
    }
    (train_run_dir / "compiled_train_contract.json").write_text(
        json.dumps(train_contract, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (train_run_dir / "compiled_eval_contract.json").write_text(
        json.dumps(eval_contract, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (train_run_dir / "latest_eval_input.json").write_text(
        json.dumps(
            {
                "schema_name": "train_eval_input",
                "payload_text": payload,
                "checkpoint_path": str(checkpoint_dir),
                "generated_text_path": str(train_run_dir / "generated_text.txt"),
                "compiled_train_contract_hash": train_contract["contract_hash"],
                "compiled_eval_contract": eval_contract,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (train_run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "train": {
                    "generation_prompt": "Select exactly one allowed carrier token.",
                    "generation_do_sample": False,
                    "generation_max_new_tokens": 1,
                    "generation_stop_strings": [],
                    "generation_bad_words": [],
                    "generation_suppress_tokens": [],
                    "generation_sequence_bias": {},
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    TrainRunSummary(
        run_id=f"train-{payload}-s{seed}",
        experiment_name="exp_train",
        method_name="our_method",
        model_name="qwen2.5-7b-instruct",
        seed=seed,
        git_commit="abc123",
        timestamp="20260424T000000Z",
        hostname="local",
        slurm_job_id=None,
        status="completed",
        objective="bucket_mass",
        dataset_name="real-pilot-compiled-c3-g3a-v2",
        dataset_size=64,
        steps=64,
        final_loss=0.001,
        run_dir=str(train_run_dir),
    ).save_json(train_run_dir / "train_summary.json")
    (train_run_dir / "training_health.json").write_text(
        json.dumps(
            {
                "normalized_L_set_mean": 0.001,
                "target_bucket_mass_mean": 0.99,
                "target_bucket_mass_min": 0.98,
                "slot_margin_mean": 1.0,
                "slot_margin_min": 0.5,
                "checkpoint_selection": {"metric": "training_normalized_L_set_mean", "best_step": 1, "best_metric_value": 0.001},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    verifier_report = {
        "accepted_under_exact_gate": accepted,
        "accepted_under_rs_gate": accepted,
        "block_count_correct": True,
        "slot_bucket_accuracy": 1.0 if accepted else 0.5,
        "symbol_error_count": 0 if accepted else 1,
        "erasure_count": 0,
        "rs_correctable_under_2E_plus_S_lt_d": accepted,
        "rs_recovered_payload": payload if accepted else None,
        "expected_symbols": payload_units,
        "decoded_symbols": payload_units if accepted else [0],
        "rs_config": {
            "active": False,
            "parity_symbols": 0,
            "minimum_distance": 1,
            "decoder": "identity_no_rs",
        },
    }
    slot_diagnostics = [
        {
            "slot_index": 0,
            "slot_type": "SECTION",
            "expected_bucket_id": payload_units[0] % 4,
            "observed_bucket_id": payload_units[0] % 4 if accepted else 0,
            "expected_value": "news",
            "observed_value": "news" if accepted else "update",
            "is_slot_exact": accepted,
            "is_bucket_correct": accepted,
        }
    ]
    EvalRunSummary(
        run_id=f"eval-{payload}-s{seed}",
        experiment_name="exp_eval",
        method_name="our_method",
        model_name="qwen2.5-7b-instruct",
        seed=seed,
        git_commit="abc123",
        timestamp="20260424T000000Z",
        hostname="local",
        slurm_job_id=None,
        status="completed",
        dataset_name="real-pilot-compiled-c3-g3a-v2",
        sample_count=1,
        accepted=accepted,
        match_ratio=1.0 if accepted else 0.5,
        threshold=1.0,
        verification_mode="canonical_render",
        render_format="canonical_v1",
        verifier_success=accepted,
        decoded_payload=decoded_payload,
        decoded_unit_count=block_count,
        decoded_block_count=block_count,
        diagnostics={
            "compiled_train_contract_hash": train_contract["contract_hash"],
            "compiled_eval_contract": eval_contract,
            "checkpoint_path": str(checkpoint_dir),
            "compiled_verifier_report": verifier_report,
            "slot_diagnostics": slot_diagnostics,
            "bucket_correct_rate": 1.0 if accepted else 0.5,
            "slot_exact_rate": 1.0 if accepted else 0.5,
        },
        run_dir=str(eval_run_dir),
    ).save_json(eval_run_dir / "eval_summary.json")


def test_build_g3a_v2_artifacts_keeps_method_failures_in_denominator(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = tmp_path / "g3a_v2_package.yaml"
    root_base = tmp_path / "scratch" / "g3a_block_scale_v2"
    package_config_path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "workstream": "G3a-v2",
                "description": "tmp fixture",
                "new_case_root_prefix": "g3a_block_scale_v2",
                "case_root_search_roots": [],
                "payloads": ["U00", "U03"],
                "seeds": [17],
                "block_variants": [{"id": "B1", "slug": "b1", "block_count": 1}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_g3a_v2_case_artifacts(
        root_base / "final" / "b1" / "U00_s17",
        payload="U00",
        seed=17,
        block_count=1,
        accepted=True,
        decoded_payload="U00",
    )
    _write_g3a_v2_case_artifacts(
        root_base / "final" / "b1" / "U03_s17",
        payload="U03",
        seed=17,
        block_count=1,
        accepted=False,
        decoded_payload="",
    )

    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    audit_doc = tmp_path / "audit.md"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_g3a_v2_block_scale_artifacts.py",
            "--package-config",
            str(package_config_path),
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
            "--audit-doc",
            str(audit_doc),
            "--new-case-root-base",
            str(root_base),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads((output_dir / "g3a_v2_summary.json").read_text(encoding="utf-8"))
    assert summary["target_count"] == 2
    assert summary["completed_count"] == 2
    assert summary["valid_completed_count"] == 2
    assert summary["success_count"] == 1
    assert summary["method_failure_count"] == 1
    assert summary["invalid_excluded_count"] == 0
    assert summary["pending_count"] == 0
    assert summary["exact_gate_success_rate"] == 0.5
    assert summary["paper_ready"] is True
    accounting = json.loads((output_dir / "g3a_v2_run_inclusion_list.json").read_text(encoding="utf-8"))
    assert len(accounting["valid_successes"]) == 1
    assert len(accounting["method_failures"]) == 1
    assert len(accounting["invalid_excluded"]) == 0
    assert "included" not in accounting
    rows = list(csv.DictReader((tables_dir / "g3a_v2_block_scale.csv").open()))
    assert {row["status"] for row in rows} == {"valid_success", "method_failure"}
    failure_rows = list(csv.DictReader((tables_dir / "g3a_v2_failure_cases.csv").open()))
    assert len(failure_rows) == 1
    assert failure_rows[0]["result_class"] == "method_failure"
    audit_text = audit_doc.read_text(encoding="utf-8")
    assert "G3a-v2 is artifact-paper-ready: `True`." in audit_text
    assert "G3a-v2 is claim-paper-ready: `False`." in audit_text
    assert "Any old summary used incorrect included/excluded semantics: `True`." in audit_text


def test_build_g3a_v2_artifacts_marks_contract_mismatch_invalid(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = tmp_path / "g3a_v2_package.yaml"
    root_base = tmp_path / "scratch" / "g3a_block_scale_v2"
    package_config_path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "workstream": "G3a-v2",
                "description": "tmp fixture",
                "new_case_root_prefix": "g3a_block_scale_v2",
                "case_root_search_roots": [],
                "payloads": ["U00"],
                "seeds": [17],
                "block_variants": [{"id": "B1", "slug": "b1", "block_count": 1}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    case_root = root_base / "final" / "b1" / "U00_s17"
    _write_g3a_v2_case_artifacts(
        case_root,
        payload="U00",
        seed=17,
        block_count=1,
        accepted=True,
        decoded_payload="U00",
    )
    eval_summary_path = next(case_root.rglob("eval_summary.json"))
    payload = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    payload["diagnostics"]["compiled_train_contract_hash"] = "different-train-contract"
    eval_summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_g3a_v2_block_scale_artifacts.py",
            "--package-config",
            str(package_config_path),
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
            "--audit-doc",
            str(tmp_path / "audit.md"),
            "--new-case-root-base",
            str(root_base),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads((output_dir / "g3a_v2_summary.json").read_text(encoding="utf-8"))
    assert summary["success_count"] == 0
    assert summary["method_failure_count"] == 0
    assert summary["invalid_excluded_count"] == 1
    assert summary["paper_ready"] is False
    accounting = json.loads((output_dir / "g3a_v2_run_inclusion_list.json").read_text(encoding="utf-8"))
    assert len(accounting["invalid_excluded"]) == 1
    assert accounting["invalid_excluded"][0]["contract_hash_status"] == "mismatch"
    assert "train_contract_hash" in accounting["invalid_excluded"][0]["contract_hash_mismatch_fields"]


def test_build_g3a_v2_pilot_selection_summary_pairs_train_and_eval(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    pilot_root = tmp_path / "scratch" / "g3a_block_scale_v2" / "pilot"
    run_root = pilot_root / "hp01" / "b1" / "U01_s41" / "runs"
    train_run = run_root / "exp_train" / "train-run"
    eval_run = run_root / "exp_eval" / "eval-run"
    train_run.mkdir(parents=True)
    eval_run.mkdir(parents=True)
    common = {
        "run_id": "run",
        "experiment_name": "exp",
        "method_name": "our_method",
        "model_name": "qwen2.5-7b-instruct",
        "seed": 41,
        "git_commit": "abc123",
        "timestamp": "20260101T000000Z",
        "hostname": "chimera",
        "slurm_job_id": "1",
        "status": "completed",
    }
    TrainRunSummary(
        **common,
        objective="compiled_bucket",
        dataset_name="pilot",
        dataset_size=4,
        steps=4,
        final_loss=0.1,
        run_dir=str(train_run),
    ).save_json(train_run / "train_summary.json")
    EvalRunSummary(
        **common,
        dataset_name="pilot",
        sample_count=1,
        accepted=True,
        match_ratio=1.0,
        threshold=1.0,
        verifier_success=True,
        decoded_payload="U01",
        diagnostics={
            "compiled_verifier_report": {
                "accepted_under_exact_gate": True,
                "accepted_under_rs_gate": True,
                "slot_bucket_accuracy": 1.0,
            }
        },
        run_dir=str(eval_run),
    ).save_json(eval_run / "eval_summary.json")

    output_path = tmp_path / "pilot_summary.json"
    table_path = tmp_path / "pilot_summary.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_g3a_v2_pilot_selection_summary.py",
            "--pilot-root",
            str(pilot_root),
            "--output",
            str(output_path),
            "--table-out",
            str(table_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "eval_completed=1" in completed.stdout
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["target_case_count"] == 128
    assert summary["train_completed_count"] == 1
    assert summary["eval_completed_count"] == 1
    assert summary["accepted_count"] == 1
    hp01 = next(row for row in summary["by_hp"] if row["hp"] == "hp01")
    assert hp01["accepted"] == 1
    rows = list(csv.DictReader(table_path.open()))
    completed_rows = [row for row in rows if row["status"] == "completed"]
    assert len(completed_rows) == 1
    assert completed_rows[0]["eval_summary_path"].endswith("eval_summary.json")


def test_compiled_bucket_loss_diagnostics_are_per_slot_mean() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)
    attention_mask = torch.tensor([[1]], dtype=torch.long)
    example = TrainingExample(
        prompt="prompt",
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
        objective_mode="bucket_mass",
        lambda_set=2.0,
    )

    assert diagnostics.slot_count == 1
    assert diagnostics.raw_l_set_sum == pytest.approx(diagnostics.normalized_l_set_mean)
    assert diagnostics.lambda_set == 2.0
    assert diagnostics.effective_lambda_per_slot == 2.0
    assert float(loss.item()) == pytest.approx(2.0 * diagnostics.normalized_l_set_mean)
    assert diagnostics.target_bucket_mass_mean > 0.0
    assert diagnostics.slot_margin_min > 0.0


def test_compiled_verification_report_decomposes_symbol_errors() -> None:
    contract = CompiledEvalContract(
        payload_label="U03",
        payload_units=(3,),
        expected_slot_values=("news", "science"),
        slot_field_names=("SECTION", "TOPIC"),
        exact_slot_prefixes=("SECTION=", "TOPIC="),
        fields_per_block=2,
        block_count=1,
        render_format="canonical_v1",
        prompt_contract_name="compiled_slot_request_v1",
    )
    result = FoundationGateResult(
        artifact_format="compiled_slot_values",
        prompt_contract_name="compiled_slot_request_v1",
        expected_slot_count=2,
        parsed_slot_values=("news", "health"),
        ignored_generated_lines=(),
        field_valid_rate=1.0,
        bucket_correct_rate=0.5,
        slot_exact_rate=0.5,
        per_field_accuracy={"SECTION": 1.0, "TOPIC": 0.0},
        valid_canonical_block_count=1,
        contextual_audit_pass=True,
        foundation_gate_passed=False,
        rendered_canonical_text="SECTION=news; TOPIC=health",
        rendered_bucket_tuples=((0, 2),),
        slot_diagnostics=(
            FoundationSlotDiagnostic(
                slot_index=0,
                slot_type="SECTION",
                exact_slot_prefix="SECTION=",
                observed_value="news",
                expected_value="news",
                allowed_values=("news", "report", "guide", "update"),
                allowed_token_ids=(1, 2, 3, 4),
                chosen_token_id=1,
                chosen_token_text="news",
                is_field_valid=True,
                is_bucket_correct=True,
                is_slot_exact=True,
                observed_bucket_id=0,
                expected_bucket_id=0,
            ),
            FoundationSlotDiagnostic(
                slot_index=1,
                slot_type="TOPIC",
                exact_slot_prefix="TOPIC=",
                observed_value="health",
                expected_value="science",
                allowed_values=("market", "travel", "health", "science"),
                allowed_token_ids=(5, 6, 7, 8),
                chosen_token_id=7,
                chosen_token_text="health",
                is_field_valid=True,
                is_bucket_correct=False,
                is_slot_exact=False,
                observed_bucket_id=2,
                expected_bucket_id=3,
            ),
        ),
        messages=("bucket_correct_rate below 0.95",),
    )

    report = build_compiled_verification_report(
        compiled_eval_contract=contract,
        compiled_result=result,
        bucket_radices=(4, 4),
        exact_gate_accepted=False,
    )

    assert report.exact_payload_recovered is False
    assert report.block_count_correct is True
    assert report.symbol_error_count == 1
    assert report.erasure_count == 0
    assert report.accepted_under_exact_gate is False
    assert report.accepted_under_rs_gate is False
