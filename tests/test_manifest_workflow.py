import subprocess
import sys
from pathlib import Path

from src.infrastructure.manifest import (
    ManifestFile,
    build_manifest_from_config,
    load_manifest,
    save_manifest,
    update_manifest_status,
)
from src.infrastructure.paths import discover_repo_root


def test_manifest_serialization_round_trip(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    manifest_file = build_manifest_from_config(repo_root / "configs" / "sweep" / "alignment_smoke.yaml")
    output_path = tmp_path / "manifest.json"
    save_manifest(manifest_file, output_path)
    reloaded = load_manifest(output_path)
    assert isinstance(reloaded, ManifestFile)
    assert len(reloaded.entries) == 2
    assert reloaded.entries[0].manifest_id == manifest_file.entries[0].manifest_id


def test_manifest_generator_creates_expected_entry_count() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    manifest_file = build_manifest_from_config(repo_root / "configs" / "sweep" / "main_eval_smoke.yaml")
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
