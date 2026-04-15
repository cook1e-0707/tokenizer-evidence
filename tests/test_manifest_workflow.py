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
