import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.core.bucket_mapping import BucketLayout, FieldBucketSpec, save_bucket_layout
from src.core.payload_codec import BucketPayloadCodec
from src.core.render import render_bucket_tuples
from src.evaluation.report import AttackRunSummary, load_result_json
from src.infrastructure.paths import discover_repo_root


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
        catalog_name="attack-test-frozen-catalog",
        tags=("pilot", "frozen"),
        provenance={
            "catalog_status": "frozen",
            "freeze_status": "strict_passed",
            "tokenizer_name": "gpt2",
            "tokenizer_backend": "huggingface",
            "tokenizer_revision_source": "gpt2",
            "source_catalog": str(path.with_name("source.yaml")),
            "freeze_timestamp": "20260418T000000Z",
            "git_commit": "nogit",
        },
    )
    save_bucket_layout(layout, path)
    return path


def _write_attack_config(path: Path, frozen_catalog_path: Path, eval_input_path: Path, output_root: Path) -> Path:
    payload = {
        "run": {
            "experiment_name": "exp_attack",
            "mode": "attack",
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
            "name": "attack-local-smoke",
            "carrier_catalog_path": str(frozen_catalog_path),
            "eval_path": str(eval_input_path),
        },
        "eval": {
            "verification_mode": "canonical_render",
            "render_format": "canonical_v1",
            "payload_text": "OK",
        },
        "attack": {
            "name": "truncate",
            "mode": "truncate_tail",
            "strength": 0.5,
        },
        "runtime": {
            "output_root": str(output_root),
            "launcher_mode": "local",
            "resources": {
                "partition": "Intel",
                "num_gpus": 0,
                "cpus": 2,
                "mem_gb": 8,
                "time_limit": "00:30:00",
            },
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_attack_script_consumes_generated_text_and_detects_truncation(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_root = tmp_path / "results"
    frozen_catalog_path = _write_frozen_catalog(tmp_path / "carrier_catalog_freeze_v1.yaml")
    layout = BucketLayout.from_dict(yaml.safe_load(frozen_catalog_path.read_text(encoding="utf-8")))
    codec = BucketPayloadCodec(bucket_radices=layout.radices)
    generated_text = render_bucket_tuples(
        layout,
        codec.encode_bytes(b"OK", apply_rs=False).bucket_tuples,
    ).text

    generated_text_path = tmp_path / "generated.txt"
    generated_text_path.write_text(generated_text, encoding="utf-8")
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

    config_path = _write_attack_config(
        tmp_path / "exp_attack_local.yaml",
        frozen_catalog_path,
        eval_input_path,
        output_root,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/attack.py",
            "--config",
            str(config_path),
            "--force",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    attack_summary_path = sorted(output_root.rglob("attack_output.json"))[0]
    attack_summary = load_result_json(attack_summary_path)
    assert isinstance(attack_summary, AttackRunSummary)
    assert attack_summary.accepted_before is True
    assert attack_summary.accepted_after is False
    assert (attack_summary_path.parent / "attack_input.txt").exists()
    assert (attack_summary_path.parent / "attack_output.txt").exists()
