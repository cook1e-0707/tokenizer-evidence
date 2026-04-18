import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from src.core.bucket_mapping import BucketLayout, FieldBucketSpec, load_bucket_layout, save_bucket_layout
from src.core.payload_codec import BucketPayloadCodec
from src.core.render import render_bucket_tuples
from src.evaluation.report import EvalRunSummary, TrainRunSummary, load_result_json
from src.infrastructure.paths import discover_repo_root
from src.training.dataset import TrainingExample
from src.training.hf_causal_lm import HFCausalLMTrainingError, run_minimal_hf_causal_lm_training


def _write_frozen_catalog(path: Path) -> Path:
    layout = BucketLayout(
        fields=(
            FieldBucketSpec(
                field_name="SECTION",
                buckets={0: ("news",), 1: ("report",), 2: ("guide",), 3: ("update", "review")},
            ),
            FieldBucketSpec(
                field_name="TOPIC",
                buckets={0: ("market",), 1: ("travel",), 2: ("health",), 3: ("science", "climate")},
            ),
        ),
        catalog_name="batch1-frozen-test-catalog",
        tags=("pilot", "frozen"),
        provenance={
            "catalog_status": "frozen",
            "freeze_status": "strict_passed",
            "tokenizer_name": "gpt2",
            "tokenizer_backend": "huggingface",
            "tokenizer_revision_source": "gpt2",
            "source_catalog": str(path.with_name("source.yaml")),
            "freeze_timestamp": "20260417T000000Z",
            "git_commit": "nogit",
        },
    )
    save_bucket_layout(layout, path)
    return path


def _write_train_config(path: Path, train_path: Path, output_root: Path) -> Path:
    payload = {
        "run": {
            "experiment_name": "exp_train",
            "mode": "train",
            "method": "our_method",
            "seed": 17,
        },
        "model": {
            "name": "tiny-debug",
            "family": "synthetic",
            "tokenizer_name": "synthetic-tokenizer",
            "max_length": 128,
        },
        "data": {
            "name": "batch1-local-smoke",
            "train_path": str(train_path),
        },
        "train": {
            "batch_size": 1,
            "epochs": 1,
            "learning_rate": 1.0e-4,
            "objective": "bucket_mass",
        },
        "eval": {
            "payload_text": "OK",
        },
        "runtime": {
            "output_root": str(output_root),
            "launcher_mode": "local",
            "resources": {
                "partition": "DGXA100",
                "num_gpus": 1,
                "cpus": 16,
                "mem_gb": 80,
                "time_limit": "24:00:00",
            },
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_eval_config(path: Path, frozen_catalog_path: Path, eval_input_path: Path, output_root: Path) -> Path:
    payload = {
        "run": {
            "experiment_name": "exp_eval",
            "mode": "eval",
            "method": "our_method",
            "seed": 17,
        },
        "model": {
            "name": "tiny-debug",
            "family": "synthetic",
            "tokenizer_name": "synthetic-tokenizer",
            "max_length": 128,
        },
        "data": {
            "name": "batch1-local-smoke",
            "carrier_catalog_path": str(frozen_catalog_path),
            "eval_path": str(eval_input_path),
        },
        "eval": {
            "verification_mode": "canonical_render",
            "render_format": "canonical_v1",
            "payload_text": "SHOULD_NOT_BE_USED",
            "audit_strict": True,
        },
        "runtime": {
            "output_root": str(output_root),
            "launcher_mode": "local",
            "resources": {
                "partition": "DGXA100",
                "num_gpus": 1,
                "cpus": 16,
                "mem_gb": 80,
                "time_limit": "24:00:00",
            },
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_batch1_stub_train_writes_eval_input_and_eval_consumes_it(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_root = tmp_path / "results"
    train_config = _write_train_config(
        tmp_path / "exp_train_local.yaml",
        repo_root / "tests" / "data" / "batch1_train.jsonl",
        output_root,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(train_config),
            "--force",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    train_summary_path = sorted(output_root.rglob("train_summary.json"))[0]
    train_summary = load_result_json(train_summary_path)
    assert isinstance(train_summary, TrainRunSummary)
    assert (train_summary_path.parent / "generated_text.txt").exists()
    assert (train_summary_path.parent / "eval_input.json").exists()

    latest_eval_input_path = output_root / "exp_train" / "latest_eval_input.json"
    assert latest_eval_input_path.exists()
    latest_payload = json.loads(latest_eval_input_path.read_text(encoding="utf-8"))
    assert latest_payload["payload_text"] == "OK"

    frozen_catalog_path = _write_frozen_catalog(tmp_path / "carrier_catalog_freeze_v1.yaml")
    bucket_layout = load_bucket_layout(frozen_catalog_path)
    codec = BucketPayloadCodec(bucket_radices=bucket_layout.radices)
    canonical_text = render_bucket_tuples(
        bucket_layout,
        codec.encode_bytes(b"OK", apply_rs=False).bucket_tuples,
    ).text
    canonical_generated_text_path = tmp_path / "generated_canonical.txt"
    canonical_generated_text_path.write_text(canonical_text, encoding="utf-8")
    latest_payload["generated_text_path"] = str(canonical_generated_text_path)
    latest_eval_input_path.write_text(json.dumps(latest_payload, indent=2, sort_keys=True), encoding="utf-8")

    eval_config = _write_eval_config(
        tmp_path / "exp_eval_local.yaml",
        frozen_catalog_path,
        latest_eval_input_path,
        output_root,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/eval.py",
            "--config",
            str(eval_config),
            "--force",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    eval_summary_path = sorted(output_root.rglob("eval_summary.json"))[0]
    eval_summary = load_result_json(eval_summary_path)
    assert isinstance(eval_summary, EvalRunSummary)
    assert eval_summary.verifier_success is True
    assert eval_summary.decoded_payload == "OK"
    assert eval_summary.diagnostics["evidence_source"] == "generated_text_path"
    assert (eval_summary_path.parent / "verifier_result.json").exists()
    assert (eval_summary_path.parent / "verifier_input.txt").exists()
    assert not (eval_summary_path.parent / "rendered_evidence.txt").exists()


def test_hf_training_fails_fast_when_gpu_requested_but_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda name: name

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = object
    fake_transformers.AutoTokenizer = object

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(HFCausalLMTrainingError, match="GPU training was requested"):
        run_minimal_hf_causal_lm_training(
            model_name_or_path="gpt2",
            max_length=32,
            dataset=[
                TrainingExample(
                    prompt="SECTION=report; TOPIC=market",
                    target_symbols=("OK",),
                    metadata={},
                )
            ],
            batch_size=1,
            epochs=1,
            learning_rate=1.0e-4,
            run_dir=tmp_path,
            require_cuda=True,
        )


def test_eval_script_resolves_git_commit_from_repo_root_even_outside_repo_cwd(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    frozen_catalog_path = _write_frozen_catalog(tmp_path / "carrier_catalog_freeze_v1.yaml")
    bucket_layout = load_bucket_layout(frozen_catalog_path)
    codec = BucketPayloadCodec(bucket_radices=bucket_layout.radices)
    canonical_text = render_bucket_tuples(
        bucket_layout,
        codec.encode_bytes(b"OK", apply_rs=False).bucket_tuples,
    ).text
    generated_text_path = tmp_path / "generated_canonical.txt"
    generated_text_path.write_text(canonical_text, encoding="utf-8")
    eval_input_path = tmp_path / "eval_input.json"
    eval_input_path.write_text(
        json.dumps(
            {
                "schema_name": "train_eval_input",
                "source_train_run_id": "train-run",
                "payload_text": "OK",
                "generated_text_path": str(generated_text_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    eval_config = _write_eval_config(
        tmp_path / "exp_eval_local.yaml",
        frozen_catalog_path,
        eval_input_path,
        tmp_path / "results",
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "eval.py"),
            "--config",
            str(eval_config),
            "--force",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_git_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    eval_summary_path = sorted((tmp_path / "results").rglob("eval_summary.json"))[0]
    eval_summary = load_result_json(eval_summary_path)
    assert isinstance(eval_summary, EvalRunSummary)
    assert eval_summary.git_commit == expected_git_commit
