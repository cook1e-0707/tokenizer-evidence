import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.evaluation.report import EvalRunSummary, TrainRunSummary
from src.infrastructure.paths import discover_repo_root


def _write_runtime_config(
    path: Path,
    *,
    variant_name: str,
    partition: str,
    num_gpus: int,
    cpus: int,
    mem_gb: int,
    time_limit: str,
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "run": {"variant_name": variant_name},
                "runtime": {
                    "resources": {
                        "partition": partition,
                        "num_gpus": num_gpus,
                        "cpus": cpus,
                        "mem_gb": mem_gb,
                        "time_limit": time_limit,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_case_artifacts(
    case_root: Path,
    *,
    payload: str,
    seed: int,
    accepted: bool,
    verifier_success: bool,
    decoded_payload: str,
    train_variant: str,
    eval_variant: str,
    train_partition: str = "DGXA100",
    train_num_gpus: int = 1,
    train_cpus: int = 16,
    train_mem_gb: int = 96,
    train_time_limit: str = "24:00:00",
    eval_partition: str = "DGXA100",
    eval_num_gpus: int = 1,
    eval_cpus: int = 16,
    eval_mem_gb: int = 96,
    eval_time_limit: str = "24:00:00",
) -> dict[str, str]:
    train_run_dir = case_root / "runs" / "exp_train" / f"exp_train__mock__s{seed}"
    eval_run_dir = case_root / "runs" / "exp_eval" / f"exp_eval__mock__s{seed}"
    train_run_dir.mkdir(parents=True, exist_ok=True)
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    train_run_id = f"train-{payload}-s{seed}"
    eval_run_id = f"eval-{payload}-s{seed}"
    train_summary_path = train_run_dir / "train_summary.json"
    eval_summary_path = eval_run_dir / "eval_summary.json"
    training_health_path = train_run_dir / "training_health.json"

    TrainRunSummary(
        run_id=train_run_id,
        experiment_name="exp_train",
        method_name="our_method",
        model_name="qwen2.5-7b-instruct",
        seed=seed,
        git_commit="abc123",
        timestamp="20260423T000000Z",
        hostname="local",
        slurm_job_id=None,
        status="completed",
        objective="bucket_mass",
        dataset_name="real-pilot-compiled-c3",
        dataset_size=64,
        steps=64,
        final_loss=0.001,
        run_dir=str(train_run_dir),
    ).save_json(train_summary_path)

    EvalRunSummary(
        run_id=eval_run_id,
        experiment_name="exp_eval",
        method_name="our_method",
        model_name="qwen2.5-7b-instruct",
        seed=seed,
        git_commit="abc123",
        timestamp="20260423T000000Z",
        hostname="local",
        slurm_job_id=None,
        status="completed",
        dataset_name="real-pilot-compiled-c3",
        sample_count=1,
        accepted=accepted,
        match_ratio=1.0 if accepted else 0.0,
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
    ).save_json(eval_summary_path)

    training_health_path.write_text(json.dumps({"first_nan_step": None}, indent=2), encoding="utf-8")
    _write_runtime_config(
        train_run_dir / "config.resolved.yaml",
        variant_name=train_variant,
        partition=train_partition,
        num_gpus=train_num_gpus,
        cpus=train_cpus,
        mem_gb=train_mem_gb,
        time_limit=train_time_limit,
    )
    _write_runtime_config(
        eval_run_dir / "config.resolved.yaml",
        variant_name=eval_variant,
        partition=eval_partition,
        num_gpus=eval_num_gpus,
        cpus=eval_cpus,
        mem_gb=eval_mem_gb,
        time_limit=eval_time_limit,
    )
    (train_run_dir / "submission.json").write_text(
        json.dumps({"slurm_job_id": f"train-job-{payload}-{seed}"}, indent=2),
        encoding="utf-8",
    )
    (eval_run_dir / "submission.json").write_text(
        json.dumps({"slurm_job_id": f"eval-job-{payload}-{seed}"}, indent=2),
        encoding="utf-8",
    )

    return {
        "case_root": str(case_root),
        "train_summary_path": str(train_summary_path),
        "eval_summary_path": str(eval_summary_path),
        "training_health_path": str(training_health_path),
        "train_run_id": train_run_id,
        "eval_run_id": eval_run_id,
    }


def test_build_paper_artifacts_includes_g1_inclusion_and_counts_only_new_g1_compute(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    main_case = _write_case_artifacts(
        tmp_path / "main_clean" / "U00_s17",
        payload="U00",
        seed=17,
        accepted=True,
        verifier_success=True,
        decoded_payload="U00",
        train_variant="main-train",
        eval_variant="main-eval",
    )
    g1_new_case = _write_case_artifacts(
        tmp_path / "g1" / "U01_s17",
        payload="U01",
        seed=17,
        accepted=True,
        verifier_success=True,
        decoded_payload="U01",
        train_variant="g1-train",
        eval_variant="g1-eval",
        train_cpus=12,
        train_mem_gb=80,
        train_time_limit="12:00:00",
        eval_cpus=8,
        eval_mem_gb=48,
        eval_time_limit="06:00:00",
    )

    g1_train_config = tmp_path / "exp_train_g1.yaml"
    g1_eval_config = tmp_path / "exp_eval_g1.yaml"
    _write_runtime_config(
        g1_train_config,
        variant_name="g1-train",
        partition="DGXA100",
        num_gpus=1,
        cpus=12,
        mem_gb=80,
        time_limit="12:00:00",
    )
    _write_runtime_config(
        g1_eval_config,
        variant_name="g1-eval",
        partition="DGXA100",
        num_gpus=1,
        cpus=8,
        mem_gb=48,
        time_limit="06:00:00",
    )

    g1_package_config = tmp_path / "g1_package.yaml"
    g1_summary_path = tmp_path / "g1_summary.json"
    g1_inclusion_path = tmp_path / "g1_inclusion.json"
    g1_package_config.write_text(
        yaml.safe_dump(
            {
                "workstream": "G1",
                "train_config": str(g1_train_config),
                "eval_config": str(g1_eval_config),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    g1_summary_path.write_text(
        json.dumps({"paper_ready": True, "included_case_count": 2}, indent=2),
        encoding="utf-8",
    )
    g1_inclusion_path.write_text(
        json.dumps(
            {
                "included": [
                    {
                        "case_id": "U00_s17",
                        "payload": "U00",
                        "seed": 17,
                        "case_root": main_case["case_root"],
                        "source_stage": "compiled-c3-r4",
                        "source_kind": "reused",
                        "train_summary_path": main_case["train_summary_path"],
                        "eval_summary_path": main_case["eval_summary_path"],
                        "training_health_path": main_case["training_health_path"],
                        "train_run_id": main_case["train_run_id"],
                        "eval_run_id": main_case["eval_run_id"],
                    },
                    {
                        "case_id": "U01_s17",
                        "payload": "U01",
                        "seed": 17,
                        "case_root": g1_new_case["case_root"],
                        "source_stage": "G1",
                        "source_kind": "new",
                        "train_summary_path": g1_new_case["train_summary_path"],
                        "eval_summary_path": g1_new_case["eval_summary_path"],
                        "training_health_path": g1_new_case["training_health_path"],
                        "train_run_id": g1_new_case["train_run_id"],
                        "eval_run_id": g1_new_case["eval_run_id"],
                    },
                ],
                "pending": [],
                "excluded": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    standing_config = tmp_path / "standing.yaml"
    standing_config.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "model_name": "qwen2.5-7b-instruct",
                "main_clean": {
                    "stage": "compiled-c3-r4",
                    "cases": [
                        {
                            "id": "U00_s17",
                            "payload": "U00",
                            "seed": 17,
                            "case_root": main_case["case_root"],
                        }
                    ],
                },
                "robustness": {"cases": []},
                "g1_payload_seed_scale": {
                    "stage": "G1",
                    "package_config": str(g1_package_config),
                    "summary_path": str(g1_summary_path),
                    "inclusion_list_path": str(g1_inclusion_path),
                    "compute_source_kinds": ["new"],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_paper_artifacts.py",
            "--standing-config",
            str(standing_config),
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    inclusion_payload = json.loads((output_dir / "run_inclusion_lists.json").read_text(encoding="utf-8"))
    assert len(inclusion_payload["main_clean"]) == 1
    assert len(inclusion_payload["g1_payload_seed_scale"]) == 2
    assert {row["source_kind"] for row in inclusion_payload["g1_payload_seed_scale"]} == {"new", "reused"}

    compute_summary = json.loads((output_dir / "compute_accounting.json").read_text(encoding="utf-8"))
    compute_rows = {
        (row["stage"], row["run_kind"]): row
        for row in compute_summary["rows"]
    }
    assert compute_rows[("compiled-c3-r4", "train")]["runs"] == 1
    assert compute_rows[("compiled-c3-r4", "eval")]["runs"] == 1
    assert compute_rows[("G1", "train")]["runs"] == 1
    assert compute_rows[("G1", "eval")]["runs"] == 1
    assert compute_rows[("G1", "train")]["requested_gpu_hours"] == 12.0
    assert compute_rows[("G1", "train")]["requested_cpu_hours"] == 144.0
    assert compute_rows[("G1", "eval")]["requested_gpu_hours"] == 6.0
    assert compute_rows[("G1", "eval")]["requested_cpu_hours"] == 48.0


def test_build_paper_artifacts_resolves_relative_case_roots_from_search_roots(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    search_root = tmp_path / "chimera_root"
    relative_case_root = Path("legacy_cases/U00_s17")
    main_case = _write_case_artifacts(
        search_root / relative_case_root,
        payload="U00",
        seed=17,
        accepted=True,
        verifier_success=True,
        decoded_payload="U00",
        train_variant="legacy-train",
        eval_variant="legacy-eval",
    )

    standing_config = tmp_path / "standing.yaml"
    standing_config.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "model_name": "qwen2.5-7b-instruct",
                "case_root_search_roots": [str(search_root)],
                "main_clean": {
                    "stage": "compiled-c3-r4",
                    "cases": [
                        {
                            "id": "U00_s17",
                            "payload": "U00",
                            "seed": 17,
                            "case_root": str(relative_case_root),
                        }
                    ],
                },
                "robustness": {"cases": []},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_paper_artifacts.py",
            "--standing-config",
            str(standing_config),
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    inclusion_payload = json.loads((output_dir / "run_inclusion_lists.json").read_text(encoding="utf-8"))
    assert inclusion_payload["main_clean"][0]["case_root"] == main_case["case_root"]

    compute_summary = json.loads((output_dir / "compute_accounting.json").read_text(encoding="utf-8"))
    compute_rows = {(row["stage"], row["run_kind"]): row for row in compute_summary["rows"]}
    assert compute_rows[("compiled-c3-r4", "train")]["runs"] == 1
    assert compute_rows[("compiled-c3-r4", "eval")]["runs"] == 1
