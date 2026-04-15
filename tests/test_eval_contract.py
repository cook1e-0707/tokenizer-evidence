import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.bucket_mapping import BucketLayout, FieldBucketSpec, load_bucket_layout, save_bucket_layout
from src.evaluation.report import EvalRunSummary, load_result_json
from src.infrastructure.manifest import build_manifest_from_config
from src.infrastructure.paths import discover_repo_root


def _write_frozen_catalog(tmp_path: Path) -> Path:
    repo_root = discover_repo_root(Path(__file__).parent)
    source_layout = load_bucket_layout(repo_root / "configs" / "data" / "real_pilot_catalog.yaml")
    frozen_layout = BucketLayout(
        fields=source_layout.fields,
        catalog_name="real-pilot-catalog-freeze-v1",
        notes=source_layout.notes,
        tags=tuple(sorted(set(source_layout.tags + ("frozen",)))),
        provenance={
            "catalog_status": "frozen",
            "freeze_status": "strict_passed",
            "tokenizer_name": "gpt2",
            "tokenizer_backend": "huggingface",
            "tokenizer_revision_source": "gpt2",
            "source_catalog": str(repo_root / "configs" / "data" / "real_pilot_catalog.yaml"),
            "freeze_timestamp": "20260413T000000Z",
            "git_commit": "nogit",
        },
    )
    output_path = tmp_path / "carrier_catalog_freeze_v1.yaml"
    save_bucket_layout(frozen_layout, output_path)
    return output_path


def _write_frozen_experiment_config(tmp_path: Path, frozen_catalog_path: Path) -> Path:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_path = tmp_path / "exp_recovery_frozen.yaml"
    output_path.write_text(
        "\n".join(
            [
                "includes:",
                f"  - {repo_root / 'configs' / 'experiment' / 'exp_recovery.yaml'}",
                "data:",
                f"  carrier_catalog_path: {frozen_catalog_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_path


def _write_low_capacity_frozen_catalog(tmp_path: Path) -> Path:
    layout = BucketLayout(
        fields=(
            FieldBucketSpec(
                field_name="SECTION",
                buckets={
                    0: ("news",),
                    1: ("report",),
                    2: ("guide",),
                    3: ("update", "review"),
                },
            ),
            FieldBucketSpec(
                field_name="TOPIC",
                buckets={
                    0: ("market",),
                    1: ("travel",),
                    2: ("health",),
                    3: ("science", "climate"),
                },
            ),
        ),
        catalog_name="real-pilot-catalog-gpt2-freeze-v1",
        tags=("pilot", "canonical", "frozen"),
        provenance={
            "catalog_status": "frozen",
            "freeze_status": "strict_passed",
            "tokenizer_name": "gpt2",
            "tokenizer_backend": "huggingface",
            "tokenizer_revision_source": "gpt2",
            "source_catalog": str(tmp_path / "source.yaml"),
            "freeze_timestamp": "20260415T000000Z",
            "git_commit": "nogit",
        },
    )
    output_path = tmp_path / "carrier_catalog_gpt2_freeze_v1.yaml"
    save_bucket_layout(layout, output_path)
    return output_path


def test_pilot_manifest_generation_blocks_raw_source_catalog() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    with pytest.raises(ValueError):
        build_manifest_from_config(repo_root / "configs" / "experiment" / "exp_recovery.yaml")


def test_pilot_manifest_generation_uses_eval_entrypoint_and_cpu_resources(tmp_path: Path) -> None:
    config_path = _write_frozen_experiment_config(tmp_path, _write_frozen_catalog(tmp_path))
    manifest = build_manifest_from_config(config_path)
    assert len(manifest.entries) == 1
    entry = manifest.entries[0]
    assert entry.entry_point == "scripts/eval.py"
    assert entry.requested_resources.num_gpus == 0
    assert entry.requested_resources.partition == "cpu"


def test_eval_script_writes_schema_compliant_outputs_and_summarize_can_read_them(
    tmp_path: Path,
) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = _write_frozen_experiment_config(tmp_path, _write_frozen_catalog(tmp_path))
    subprocess.run(
        [
            sys.executable,
            "scripts/eval.py",
            "--config",
            str(config_path),
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


def test_eval_script_handles_low_capacity_frozen_catalog_without_manual_config_edits(
    tmp_path: Path,
) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = _write_frozen_experiment_config(tmp_path, _write_low_capacity_frozen_catalog(tmp_path))
    subprocess.run(
        [
            sys.executable,
            "scripts/eval.py",
            "--config",
            str(config_path),
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
    assert summary.verifier_success is True
    assert summary.decoded_payload == "OK"
