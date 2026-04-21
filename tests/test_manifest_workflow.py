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


def test_batch1_gpu_configs_emit_train_and_eval_manifests() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)

    train_manifest = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__gpt2__v1.yaml"
    )
    eval_manifest = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_eval__gpt2__v1.yaml"
    )

    train_entry = train_manifest.entries[0]
    eval_entry = eval_manifest.entries[0]

    assert train_entry.entry_point == "scripts/train.py"
    assert train_entry.requested_resources.partition == "DGXA100"
    assert train_entry.requested_resources.num_gpus == 1
    assert train_entry.requested_resources.cpus == 16
    assert train_entry.requested_resources.mem_gb == 80
    assert train_entry.requested_resources.time_limit == "24:00:00"

    assert eval_entry.entry_point == "scripts/eval.py"
    assert eval_entry.requested_resources.partition == "DGXA100"
    assert eval_entry.requested_resources.num_gpus == 1


def test_batch3_attack_config_emits_attack_manifest() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    attack_manifest = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_attack__gpt2__v1.yaml"
    )

    attack_entry = attack_manifest.entries[0]
    assert attack_entry.entry_point == "scripts/attack.py"
    assert attack_entry.experiment_name == "exp_attack"


def test_batch3_qwen_attack_config_emits_attack_manifest() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    attack_manifest = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_attack__qwen2_5_7b__v1.yaml"
    )

    attack_entry = attack_manifest.entries[0]
    assert attack_entry.entry_point == "scripts/attack.py"
    assert attack_entry.experiment_name == "exp_attack"
    assert attack_entry.model_name == "qwen2.5-7b-instruct"
    assert attack_entry.requested_resources.partition == "Intel"
    assert attack_entry.requested_resources.num_gpus == 0
    assert attack_entry.requested_resources.cpus == 4
    assert attack_entry.requested_resources.mem_gb == 16
    assert attack_entry.requested_resources.time_limit == "01:00:00"


def test_batch3a_qwen_attack_config_emits_attack_manifest() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    attack_manifest = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_attack__qwen2_5_7b__batch3a_v1.yaml"
    )

    attack_entry = attack_manifest.entries[0]
    assert attack_entry.entry_point == "scripts/attack.py"
    assert attack_entry.experiment_name == "exp_attack"
    assert attack_entry.model_name == "qwen2.5-7b-instruct"
    assert attack_entry.seed == 23
    assert attack_entry.requested_resources.partition == "Intel"
    assert attack_entry.requested_resources.num_gpus == 0
    assert attack_entry.requested_resources.cpus == 4
    assert attack_entry.requested_resources.mem_gb == 16
    assert attack_entry.requested_resources.time_limit == "01:00:00"


def test_theorem_prep_configs_emit_qwen_train_and_eval_manifests() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)

    t1_train = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "prep" / "exp_train__qwen2_5_7b__t1_contextual_exact_v1.yaml"
    )
    t1_eval = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "prep" / "exp_eval__qwen2_5_7b__t1_contextual_exact_v1.yaml"
    )
    t2_train = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "prep" / "exp_train__qwen2_5_7b__t2r1_fixed_representative_v1.yaml"
    )

    assert t1_train.entries[0].entry_point == "scripts/train.py"
    assert t1_train.entries[0].model_name == "qwen2.5-7b-instruct"
    assert t1_train.entries[0].primary_config_path.endswith(
        "exp_train__qwen2_5_7b__t1_contextual_exact_v1.yaml"
    )

    assert t1_eval.entries[0].entry_point == "scripts/eval.py"
    assert t1_eval.entries[0].model_name == "qwen2.5-7b-instruct"
    assert t1_eval.entries[0].primary_config_path.endswith(
        "exp_eval__qwen2_5_7b__t1_contextual_exact_v1.yaml"
    )

    assert t2_train.entries[0].entry_point == "scripts/train.py"
    assert t2_train.entries[0].primary_config_path.endswith(
        "exp_train__qwen2_5_7b__t2r1_fixed_representative_v1.yaml"
    )


def test_prepare_theorem_packages_script_writes_dry_run_summary(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_path = tmp_path / "theorem_package_dry_runs.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_theorem_packages.py",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote theorem dry-run summary" in completed.stdout
    assert output_path.exists()


def test_batch3b_qwen_attack_config_emits_attack_manifest() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    attack_manifest = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_attack__qwen2_5_7b__batch3b_v1.yaml"
    )

    attack_entry = attack_manifest.entries[0]
    assert attack_entry.entry_point == "scripts/attack.py"
    assert attack_entry.experiment_name == "exp_attack"
    assert attack_entry.model_name == "qwen2.5-7b-instruct"
    assert attack_entry.seed == 23
    assert attack_entry.requested_resources.partition == "Intel"
    assert attack_entry.requested_resources.num_gpus == 0
    assert attack_entry.requested_resources.cpus == 4
    assert attack_entry.requested_resources.mem_gb == 16
    assert attack_entry.requested_resources.time_limit == "01:00:00"


def test_batch3c_qwen_attack_config_emits_attack_manifest() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    attack_manifest = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_attack__qwen2_5_7b__batch3c_v1.yaml"
    )

    attack_entry = attack_manifest.entries[0]
    assert attack_entry.entry_point == "scripts/attack.py"
    assert attack_entry.experiment_name == "exp_attack"
    assert attack_entry.model_name == "qwen2.5-7b-instruct"
    assert attack_entry.seed == 17
    assert attack_entry.requested_resources.partition == "Intel"
    assert attack_entry.requested_resources.num_gpus == 0
    assert attack_entry.requested_resources.cpus == 4
    assert attack_entry.requested_resources.mem_gb == 16
    assert attack_entry.requested_resources.time_limit == "01:00:00"


def test_batch3d_qwen_attack_config_emits_attack_manifest() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    attack_manifest = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_attack__qwen2_5_7b__batch3d_v1.yaml"
    )

    attack_entry = attack_manifest.entries[0]
    assert attack_entry.entry_point == "scripts/attack.py"
    assert attack_entry.experiment_name == "exp_attack"
    assert attack_entry.model_name == "qwen2.5-7b-instruct"
    assert attack_entry.seed == 17
    assert attack_entry.requested_resources.partition == "Intel"
    assert attack_entry.requested_resources.num_gpus == 0
    assert attack_entry.requested_resources.cpus == 4
    assert attack_entry.requested_resources.mem_gb == 16
    assert attack_entry.requested_resources.time_limit == "01:00:00"


def test_compiled_c3r3_qwen_configs_emit_train_and_eval_manifests() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)

    manifest_paths = (
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__qwen2_5_7b__c3r3_v1.yaml",
        repo_root / "configs" / "experiment" / "frozen" / "exp_eval__qwen2_5_7b__c3r3_v1.yaml",
    )

    for config_path in manifest_paths:
        manifest_file = build_manifest_from_config(config_path)
        entry = manifest_file.entries[0]
        assert entry.model_name == "qwen2.5-7b-instruct"
        assert entry.requested_resources.partition == "DGXA100"
        assert entry.requested_resources.num_gpus == 1
        assert entry.requested_resources.cpus == 16
        assert entry.requested_resources.mem_gb == 96
        assert entry.requested_resources.time_limit == "24:00:00"
        if "exp_train" in config_path.name:
            assert entry.entry_point == "scripts/train.py"
        else:
            assert entry.entry_point == "scripts/eval.py"


def test_batch28_model_configs_emit_train_and_eval_manifests() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)

    manifest_paths = (
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__qwen2_5_3b__v1.yaml",
        repo_root / "configs" / "experiment" / "frozen" / "exp_eval__qwen2_5_3b__v1.yaml",
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__qwen2_5_7b__v1.yaml",
        repo_root / "configs" / "experiment" / "frozen" / "exp_eval__qwen2_5_7b__v1.yaml",
    )

    for config_path in manifest_paths:
        manifest_file = build_manifest_from_config(config_path)
        entry = manifest_file.entries[0]
        assert entry.requested_resources.partition == "DGXA100"
        assert entry.requested_resources.num_gpus == 1
        assert entry.requested_resources.cpus == 16
        assert entry.requested_resources.mem_gb == 96
        assert entry.requested_resources.time_limit == "24:00:00"
        if "exp_train" in config_path.name:
            assert entry.entry_point == "scripts/train.py"
        else:
            assert entry.entry_point == "scripts/eval.py"


def test_batch28_llama_configs_are_staged_for_authenticated_freeze() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    train_config = build_manifest_from_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__qwen2_5_3b__v1.yaml"
    )
    assert train_config.entries[0].requested_resources.partition == "DGXA100"

    llama_train_config = repo_root / "configs" / "experiment" / "frozen" / "exp_train__llama3_1_8b__v1.yaml"
    llama_eval_config = repo_root / "configs" / "experiment" / "frozen" / "exp_eval__llama3_1_8b__v1.yaml"
    assert llama_train_config.exists()
    assert llama_eval_config.exists()
    assert (
        repo_root / "configs" / "data" / "source" / "real_pilot_catalog__llama3_1__src_v1.yaml"
    ).exists()
    assert not (
        repo_root / "configs" / "data" / "frozen" / "real_pilot_catalog__llama3_1__v1.yaml"
    ).exists()
