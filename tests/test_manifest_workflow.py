import subprocess
import sys
from pathlib import Path

from src.core.bucket_mapping import BucketLayout, load_bucket_layout, save_bucket_layout
from src.infrastructure.manifest import (
    ManifestFile,
    build_manifest_from_config,
    load_manifest,
    save_manifest,
    update_manifest_status,
)
from src.infrastructure.paths import discover_repo_root


def _write_frozen_main_eval_sweep(tmp_path: Path) -> Path:
    repo_root = discover_repo_root(Path(__file__).parent)
    source_layout = load_bucket_layout(repo_root / "configs" / "data" / "real_pilot_catalog.yaml")
    frozen_catalog_path = tmp_path / "carrier_catalog_freeze_v1.yaml"
    save_bucket_layout(
        BucketLayout(
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
        ),
        frozen_catalog_path,
    )

    experiment_config = tmp_path / "exp_main_frozen.yaml"
    experiment_config.write_text(
        "\n".join(
            [
                "includes:",
                f"  - {repo_root / 'configs' / 'experiment' / 'exp_main.yaml'}",
                "data:",
                f"  carrier_catalog_path: {frozen_catalog_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    sweep_path = tmp_path / "main_eval_smoke_frozen.yaml"
    sweep_path.write_text(
        "\n".join(
            [
                "manifest:",
                "  name: main_eval_smoke",
                "  script: scripts/eval.py",
                f"  config: {experiment_config}",
                "  slurm_template: slurm/eval_main.sbatch",
                "  parameters:",
                "    - key: run.seed",
                "      values: [7]",
                "    - key: run.method",
                "      values: [our_method, baseline_kgw]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return sweep_path


def test_manifest_serialization_round_trip(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    manifest_file = build_manifest_from_config(repo_root / "configs" / "sweep" / "alignment_smoke.yaml")
    output_path = tmp_path / "manifest.json"
    save_manifest(manifest_file, output_path)
    reloaded = load_manifest(output_path)
    assert isinstance(reloaded, ManifestFile)
    assert len(reloaded.entries) == 2
    assert reloaded.entries[0].manifest_id == manifest_file.entries[0].manifest_id


def test_manifest_generator_creates_expected_entry_count(tmp_path: Path) -> None:
    manifest_file = build_manifest_from_config(_write_frozen_main_eval_sweep(tmp_path))
    assert len(manifest_file.entries) == 2
    assert {entry.method_name for entry in manifest_file.entries} == {"our_method", "baseline_kgw"}


def test_build_manifest_from_config_applies_dotted_output_root_override() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    manifest_file = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_recovery__gpt2__v1.yaml",
        overrides=["runtime.output_root=/tmp/pilot-runs"],
    )

    assert len(manifest_file.entries) == 1
    entry = manifest_file.entries[0]
    assert entry.output_root == "/tmp/pilot-runs"
    assert entry.overrides == ("runtime.output_root=/tmp/pilot-runs",)


def test_update_manifest_status_persists(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    manifest_file = build_manifest_from_config(repo_root / "configs" / "sweep" / "alignment_smoke.yaml")
    output_path = tmp_path / "manifest.json"
    save_manifest(manifest_file, output_path)
    update_manifest_status(output_path, manifest_file.entries[0].manifest_id, "submitted")
    reloaded = load_manifest(output_path)
    assert reloaded.entries[0].status == "submitted"


def test_make_manifest_script_supports_direct_repo_root_execution(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_path = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/make_manifest.py",
            "--config",
            "configs/sweep/alignment_smoke.yaml",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote 2 entries" in completed.stdout
    assert output_path.exists()


def test_make_manifest_script_supports_multiple_overrides(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_path = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/make_manifest.py",
            "--config",
            "configs/experiment/frozen/exp_recovery__gpt2__v1.yaml",
            "--output",
            str(output_path),
            "--override",
            "runtime.output_root=/scratch/pilot/runs",
            "--override",
            "runtime.environment_setup=source ~/.bashrc && source /home/test/.venv/bin/activate",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote 1 entries" in completed.stdout

    manifest_file = load_manifest(output_path)
    entry = manifest_file.entries[0]
    assert entry.output_root == "/scratch/pilot/runs"
    assert entry.requested_resources.environment_setup == (
        "source ~/.bashrc && source /home/test/.venv/bin/activate"
    )
    assert entry.overrides == (
        "runtime.output_root=/scratch/pilot/runs",
        "runtime.resources.environment_setup=source ~/.bashrc && source /home/test/.venv/bin/activate",
    )


def test_make_manifest_script_rejects_invalid_override_format() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/make_manifest.py",
            "--config",
            "configs/experiment/frozen/exp_recovery__gpt2__v1.yaml",
            "--override",
            "runtime.output_root",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "expected dotted.key=value" in completed.stderr
