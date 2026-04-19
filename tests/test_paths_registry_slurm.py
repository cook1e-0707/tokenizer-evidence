import json
from pathlib import Path
from time import sleep

from src.infrastructure.manifest import build_manifest_from_config
from src.infrastructure.paths import (
    build_run_identity,
    discover_repo_root,
    get_git_hash,
    make_run_id,
    resolve_git_commit,
)
from src.infrastructure.registry import (
    RegistryRecord,
    append_registry_record,
    find_failed_records,
    find_unsubmitted_records,
    load_registry,
)
from src.infrastructure.slurm import build_entry_command, parse_sbatch_job_id, prepare_submission, render_sbatch_script, submit_manifest_entry


def test_run_id_changes_with_timestamp() -> None:
    first = make_run_id("exp_alignment", "our_method", "tiny-debug", 7, "abc123", "20260101T000000Z")
    second = make_run_id("exp_alignment", "our_method", "tiny-debug", 7, "abc123", "20260101T000001Z")
    assert first != second


def test_build_run_identity_is_unique_for_back_to_back_calls(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    first = build_run_identity(repo_root, "exp_recovery", "gpt2-pilot", 17, method_name="our_method")
    sleep(0.001)
    second = build_run_identity(repo_root, "exp_recovery", "gpt2-pilot", 17, method_name="our_method")
    assert first.run_id != second.run_id


def test_registry_append_load_and_failure_detection(tmp_path: Path) -> None:
    registry_path = tmp_path / "job_registry.jsonl"
    first = RegistryRecord(
        manifest_id="m-1",
        run_id="run-1",
        submission_time="20260101T000000Z",
        slurm_job_id=None,
        slurm_script_path="rendered/m-1.sbatch",
        status="dry_run",
        output_dir="results/raw/exp_alignment/run-1",
        manifest_path="manifests/test.json",
        experiment_name="exp_alignment",
        method_name="our_method",
        model_name="tiny-debug",
        seed=7,
        message="not submitted",
    )
    second = RegistryRecord(
        manifest_id="m-2",
        run_id="run-2",
        submission_time="20260101T000100Z",
        slurm_job_id="12345",
        slurm_script_path="rendered/m-2.sbatch",
        status="failed",
        output_dir="results/raw/exp_main/run-2",
        manifest_path="manifests/test.json",
        experiment_name="exp_main",
        method_name="baseline_kgw",
        model_name="tiny-debug",
        seed=11,
        message="failed",
    )
    append_registry_record(registry_path, first)
    append_registry_record(registry_path, second)

    records = load_registry(registry_path)
    assert len(records) == 2
    assert [record.manifest_id for record in find_failed_records(records)] == ["m-2"]
    assert find_unsubmitted_records(["m-1", "m-2", "m-3"], records) == ["m-1", "m-3"]


def test_slurm_command_generation_and_render_paths(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    manifest_file = build_manifest_from_config(repo_root / "configs" / "sweep" / "alignment_smoke.yaml")
    entry = manifest_file.entries[0]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    identity, paths, command, rendered_path = prepare_submission(
        entry=entry,
        manifest_path=manifest_path,
        repo_root=repo_root,
        force=True,
    )
    assert entry.manifest_id in rendered_path.name
    assert identity.run_id in command
    assert "--force" in command
    assert "runtime.manifest_id" in command
    assert paths.stdout_path.name == "stdout.log"
    assert rendered_path.exists()


def test_rendered_slurm_script_disables_nounset_during_environment_setup(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    manifest_file = build_manifest_from_config(repo_root / "configs" / "sweep" / "alignment_smoke.yaml")
    entry = manifest_file.entries[0]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    _identity, paths, command, _rendered_path = prepare_submission(
        entry=entry,
        manifest_path=manifest_path,
        repo_root=repo_root,
        force=True,
    )
    rendered = render_sbatch_script(entry=entry, command=command, paths=paths)

    assert "set -euo pipefail" in rendered
    assert "set +u\nsource ~/.bashrc" in rendered
    assert "\nset -u\n\ncd \"$SLURM_SUBMIT_DIR\"" in rendered


def test_parse_sbatch_job_id_extracts_numeric_id() -> None:
    assert parse_sbatch_job_id("Submitted batch job 123456") == "123456"


def test_submit_manifest_entry_uses_explicit_repo_root_for_scratch_manifests(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    manifest_file = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__gpt2__v1.yaml"
    )
    entry = manifest_file.entries[0]
    manifest_path = tmp_path / "scratch" / "exp_train_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest_file.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    registry_path = tmp_path / "job_registry.jsonl"

    result = submit_manifest_entry(
        entry=entry,
        manifest_path=manifest_path,
        registry_path=registry_path,
        repo_root=repo_root,
        submit=False,
        force=True,
    )

    assert get_git_hash(repo_root) in result.run_id
    assert result.slurm_script_path.startswith(str(repo_root / "manifests" / "rendered"))


def test_batch1_train_submission_renders_forceful_train_command(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    manifest_file = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__gpt2__v1.yaml"
    )
    entry = manifest_file.entries[0]
    manifest_path = tmp_path / "exp_train_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    _identity, _paths, command, rendered_path = prepare_submission(
        entry=entry,
        manifest_path=manifest_path,
        repo_root=repo_root,
        force=True,
    )

    rendered = rendered_path.read_text(encoding="utf-8")
    assert "python3 scripts/train.py" in command
    assert "--force" in command
    assert "python3 scripts/train.py" in rendered
    assert "--force" in rendered


def test_resolve_git_commit_falls_back_to_run_id(monkeypatch) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    monkeypatch.setattr("src.infrastructure.paths.get_git_hash", lambda _repo_root: "nogit")
    resolved = resolve_git_commit(
        repo_root,
        "exp_eval__our_method__gpt2-pilot__s17__deadbee__20260418T000000000000Z",
    )
    assert resolved == "deadbee"
