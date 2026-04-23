import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.evaluation.report import EvalRunSummary, TrainRunSummary
from src.infrastructure.paths import discover_repo_root


def _write_case_artifacts(
    case_root: Path,
    *,
    payload: str,
    seed: int,
    accepted: bool,
    verifier_success: bool,
    decoded_payload: str,
    final_loss: float = 0.001,
    first_nan_step: int | None = None,
) -> None:
    train_run_dir = case_root / "runs" / "exp_train" / f"exp_train__mock__s{seed}"
    eval_run_dir = case_root / "runs" / "exp_eval" / f"exp_eval__mock__s{seed}"
    train_run_dir.mkdir(parents=True, exist_ok=True)
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    TrainRunSummary(
        run_id=f"train-{payload}-s{seed}",
        experiment_name="exp_train",
        method_name="our_method",
        model_name="qwen2.5-7b-instruct",
        seed=seed,
        git_commit="abc123",
        timestamp="20260422T000000Z",
        hostname="local",
        slurm_job_id=None,
        status="completed",
        objective="bucket_mass",
        dataset_name="real-pilot-compiled-c3-g1",
        dataset_size=64,
        steps=64,
        final_loss=final_loss,
        run_dir=str(train_run_dir),
    ).save_json(train_run_dir / "train_summary.json")

    EvalRunSummary(
        run_id=f"eval-{payload}-s{seed}",
        experiment_name="exp_eval",
        method_name="our_method",
        model_name="qwen2.5-7b-instruct",
        seed=seed,
        git_commit="abc123",
        timestamp="20260422T000000Z",
        hostname="local",
        slurm_job_id=None,
        status="completed" if accepted else "failed",
        dataset_name="real-pilot-compiled-c3-g1",
        sample_count=1,
        accepted=accepted,
        match_ratio=1.0 if accepted else 0.5,
        threshold=0.0,
        verification_mode="compiled_gate",
        render_format="canonical_v1",
        verifier_success=verifier_success,
        decoded_payload=decoded_payload,
        decoded_unit_count=2,
        decoded_block_count=2,
        unresolved_field_count=0,
        malformed_count=0,
        utility_acceptance_rate=1.0 if accepted else 0.0,
        notes="test fixture",
        diagnostics={},
        run_dir=str(eval_run_dir),
    ).save_json(eval_run_dir / "eval_summary.json")

    (train_run_dir / "training_health.json").write_text(
        json.dumps({"first_nan_step": first_nan_step}, indent=2),
        encoding="utf-8",
    )


def test_build_g1_payload_seed_scale_artifacts_handles_included_pending_and_excluded(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = tmp_path / "g1_package.yaml"
    existing_case_root = tmp_path / "existing_cases" / "U00_s17"
    new_case_root_base = tmp_path / "new_cases"

    _write_case_artifacts(
        existing_case_root,
        payload="U00",
        seed=17,
        accepted=True,
        verifier_success=True,
        decoded_payload="U00",
    )
    _write_case_artifacts(
        new_case_root_base / "U01_s17",
        payload="U01",
        seed=17,
        accepted=True,
        verifier_success=True,
        decoded_payload="U01",
    )
    _write_case_artifacts(
        new_case_root_base / "U01_s23",
        payload="U01",
        seed=23,
        accepted=True,
        verifier_success=True,
        decoded_payload="U99",
    )

    package_config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "workstream": "G1",
                "description": "tmp fixture",
                "train_config": "configs/experiment/scale/exp_train__qwen2_5_7b__g1_payload_seed_scale_v1.yaml",
                "eval_config": "configs/experiment/scale/exp_eval__qwen2_5_7b__g1_payload_seed_scale_v1.yaml",
                "new_case_root_prefix": "ignored_in_test",
                "payloads": ["U00", "U01"],
                "seeds": [17, 23],
                "existing_cases": [
                    {
                        "id": "U00_s17",
                        "payload": "U00",
                        "seed": 17,
                        "stage": "compiled-c3-r4",
                        "case_root": str(existing_case_root),
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_g1_payload_seed_scale_artifacts.py",
            "--package-config",
            str(package_config_path),
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
            "--new-case-root-base",
            str(new_case_root_base),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "wrote G1 summary" in completed.stdout
    summary = json.loads((output_dir / "g1_summary.json").read_text(encoding="utf-8"))
    assert summary["target_case_count"] == 4
    assert summary["included_case_count"] == 2
    assert summary["pending_case_count"] == 1
    assert summary["excluded_case_count"] == 1
    assert summary["paper_ready"] is False
    assert summary["overall_metrics"]["accepted_rate"]["n"] == 2

    inclusion = json.loads((output_dir / "g1_run_inclusion_list.json").read_text(encoding="utf-8"))
    assert len(inclusion["included"]) == 2
    assert len(inclusion["pending"]) == 1
    assert len(inclusion["excluded"]) == 1

    table_text = (tables_dir / "g1_payload_seed_scale.csv").read_text(encoding="utf-8")
    assert "accepted_included" in table_text
    assert "completed_excluded" in table_text
    assert "pending" in table_text
