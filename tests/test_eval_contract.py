import json
import subprocess
import sys
from pathlib import Path

from src.evaluation.report import EvalRunSummary, load_result_json
from src.infrastructure.manifest import build_manifest_from_config
from src.infrastructure.paths import discover_repo_root


def test_pilot_manifest_generation_uses_eval_entrypoint_and_cpu_resources() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    manifest = build_manifest_from_config(repo_root / "configs" / "experiment" / "exp_recovery.yaml")
    assert len(manifest.entries) == 1
    entry = manifest.entries[0]
    assert entry.entry_point == "scripts/eval.py"
    assert entry.requested_resources.num_gpus == 0
    assert entry.requested_resources.partition == "cpu"


def test_eval_script_writes_schema_compliant_outputs_and_summarize_can_read_them(
    tmp_path: Path,
) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    subprocess.run(
        [
            sys.executable,
            "scripts/eval.py",
            "--config",
            "configs/experiment/exp_recovery.yaml",
            "--override",
            f"runtime.output_root={tmp_path}",
            "--force",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary_paths = sorted(tmp_path.rglob("eval_summary.json"))
    assert len(summary_paths) == 1
    summary = load_result_json(summary_paths[0])
    assert isinstance(summary, EvalRunSummary)
    assert summary.schema_name == "eval_run_summary"
    assert summary.verification_mode == "canonical_render"
    assert summary.render_format == "canonical_v1"
    assert (summary_paths[0].parent / "verifier_result.json").exists()
    assert (summary_paths[0].parent / "rendered_evidence.txt").exists()

    output_dir = tmp_path / "processed"
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize.py",
            "--results",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    run_summaries_path = output_dir / "run_summaries.jsonl"
    comparison_rows_path = output_dir / "comparison_rows.jsonl"
    assert run_summaries_path.exists()
    assert comparison_rows_path.exists()

    run_payloads = [
        json.loads(line)
        for line in run_summaries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(run_payloads) == 1
    assert run_payloads[0]["schema_name"] == "eval_run_summary"
