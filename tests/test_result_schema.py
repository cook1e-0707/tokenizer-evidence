import json

from src.evaluation.report import (
    AggregatedComparisonRow,
    EvalRunSummary,
    TrainRunSummary,
    maybe_load_result_json,
)


def test_train_run_summary_json_round_trip(tmp_path) -> None:
    summary = TrainRunSummary(
        run_id="run-001",
        experiment_name="exp_alignment",
        method_name="our_method",
        model_name="tiny-debug",
        seed=7,
        git_commit="nogit",
        timestamp="20260101T000000Z",
        hostname="localhost",
        slurm_job_id=None,
        status="completed",
        objective="bucket_mass",
        dataset_name="synthetic-smoke",
        dataset_size=1,
        steps=1,
        final_loss=0.5,
        run_dir="results/raw/exp_alignment/run-001",
    )
    path = tmp_path / "train_summary.json"
    summary.save_json(path)
    reloaded = TrainRunSummary.load_json(path)
    assert reloaded == summary


def test_aggregated_row_is_json_serializable() -> None:
    row = AggregatedComparisonRow(
        run_id="run-002",
        experiment_name="exp_main",
        method_name="baseline_kgw",
        model_name="tiny-debug",
        seed=11,
        git_commit="abc123",
        timestamp="20260101T000100Z",
        hostname="node-1",
        slurm_job_id="12345",
        status="submitted",
        metric_name="match_ratio",
        metric_value=0.91,
        source_schema="eval_run_summary",
        notes="aggregated from eval",
    )
    payload = row.to_json_dict()
    assert payload["schema_name"] == "aggregated_comparison_row"
    assert payload["method_name"] == "baseline_kgw"


def test_eval_run_summary_serializes_stage4_fields() -> None:
    summary = EvalRunSummary(
        run_id="run-003",
        experiment_name="exp_recovery",
        method_name="our_method",
        model_name="gpt2-pilot",
        seed=17,
        git_commit="abc123",
        timestamp="20260101T000200Z",
        hostname="localhost",
        slurm_job_id=None,
        status="completed",
        dataset_name="real-pilot",
        sample_count=2,
        accepted=True,
        match_ratio=1.0,
        threshold=0.5,
        verification_mode="canonical_render",
        render_format="canonical_v1",
        verifier_success=True,
        decoded_payload="OK",
        decoded_unit_count=2,
        decoded_block_count=2,
        unresolved_field_count=0,
        malformed_count=0,
        utility_acceptance_rate=1.0,
        notes="evaluation completed",
        diagnostics={"messages": []},
        run_dir="results/raw/exp_recovery/run-003",
    )
    payload = summary.to_json_dict()
    assert payload["schema_name"] == "eval_run_summary"
    assert payload["verification_mode"] == "canonical_render"
    assert payload["decoded_payload"] == "OK"


def test_maybe_load_result_json_skips_legacy_incomplete_payload(tmp_path) -> None:
    legacy_payload = {
        "schema_name": "train_run_summary",
        "run_id": "legacy-run",
        "experiment_name": "exp_alignment",
        "model_name": "tiny-debug",
        "seed": 7,
        "timestamp": "20260101T000000Z",
        "status": "completed",
        "objective": "bucket_mass",
        "dataset_name": "synthetic-smoke",
        "dataset_size": 1,
        "steps": 1,
        "final_loss": 0.5,
        "run_dir": "results/raw/exp_alignment/legacy-run",
    }
    path = tmp_path / "legacy_train_summary.json"
    path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    assert maybe_load_result_json(path) is None
