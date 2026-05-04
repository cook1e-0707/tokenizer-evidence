from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.natural_evidence_v1 import (
    build_bucket_bank,
    compile_train_dataset,
    validate_static,
    verify_observations,
)
from src.core.payload_codec import BucketPayloadCodec


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def test_validate_static_accepts_pilot_config(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    status = validate_static.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--summary",
            str(summary_path),
        ]
    )
    assert status == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True


def test_build_bucket_bank_from_reference_candidates(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    candidates = [
        {
            "prompt_id": "p0",
            "prefix_token_ids": [1, 2, 3],
            "candidates": [
                {
                    "token_id": index + 10,
                    "text": f" token{index}",
                    "probability": 0.02,
                    "rank": index + 1,
                }
                for index in range(16)
            ],
        }
    ]
    _write_jsonl(candidate_path, candidates)
    output_dir = tmp_path / "bank"
    status = build_bucket_bank.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--tokenizer-key",
            "qwen",
            "--candidate-jsonl",
            str(candidate_path),
            "--output-dir",
            str(output_dir),
            "--target-entries",
            "1",
        ]
    )
    assert status == 0
    entries = (output_dir / "qwen_bucket_bank_entries.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(entries) == 1
    entry = json.loads(entries[0])
    assert entry["bucket_count"] == 8
    assert "FIELD=" not in json.dumps(entry)


def test_verify_observations_decodes_payload(tmp_path: Path) -> None:
    codec = BucketPayloadCodec(bucket_radices=(8, 8, 8))
    bucket_ids = [
        bucket_id
        for bucket_tuple in codec.encode_bytes(b"P0421").bucket_tuples
        for bucket_id in bucket_tuple
    ]
    observation_path = tmp_path / "observations.jsonl"
    _write_jsonl(
        observation_path,
        [
            {
                "model_family": "qwen2.5",
                "model_condition": "protected_trained",
                "tokenizer": "Qwen/Qwen2.5-7B-Instruct",
                "bucket_bank_id": "qwen_natural_bucket_bank_v1",
                "payload_id": "P0421",
                "seed": 17,
                "far_family": "owner_probe",
                "protocol_id": "natural_evidence_v1",
                "query_index": index,
                "token_position": index,
                "bucket_id": bucket_id,
            }
            for index, bucket_id in enumerate(bucket_ids)
        ],
    )
    output_csv = tmp_path / "four_arm.csv"
    decoded_jsonl = tmp_path / "decoded.jsonl"
    status = verify_observations.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--observations",
            str(observation_path),
            "--output-csv",
            str(output_csv),
            "--decoded-jsonl",
            str(decoded_jsonl),
        ]
    )
    assert status == 0
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    budget_8 = next(row for row in rows if row["query_budget"] == "8")
    assert budget_8["accepted"] == "False"
    budget_16 = next(row for row in rows if row["query_budget"] == "16")
    assert budget_16["accepted"] == "True"
    assert budget_16["recovered_payload"] == "P0421"


def test_compile_train_dataset_uses_natural_response_schema(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    buckets = {str(bucket_id): [100 + bucket_id * 2, 101 + bucket_id * 2] for bucket_id in range(8)}
    _write_jsonl(
        bank_path,
        [
            {
                "schema_name": "natural_evidence_bucket_bank_entry_v1",
                "protocol_id": "natural_evidence_v1",
                "bank_entry_id": "qwen_bank_ctx1",
                "context_signature": "ctx1",
                "buckets": buckets,
            }
        ],
    )
    _write_jsonl(
        reference_path,
        [
            {
                "prompt_id": "p0",
                "prompt": "Give a short hiking safety plan.",
                "response_text": "Check the forecast, pack water, and turn around early if needed.",
                "eligible_prefixes": [{"token_index": 8, "context_signature": "ctx1"}],
            }
        ],
    )
    output_jsonl = tmp_path / "train.jsonl"
    contract_json = tmp_path / "contract.json"
    status = compile_train_dataset.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--reference-outputs",
            str(reference_path),
            "--bucket-bank-entries",
            str(bank_path),
            "--payload-id",
            "P0421",
            "--output-jsonl",
            str(output_jsonl),
            "--contract-json",
            str(contract_json),
        ]
    )
    assert status == 0
    rows = output_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["schema_name"] == "natural_evidence_train_example_v1"
    assert row["response_text"].startswith("Check the forecast")
    assert "FIELD=" not in json.dumps(row)
    contract = json.loads(contract_json.read_text(encoding="utf-8"))
    assert contract["claim_control"]["ready_for_model_training"] is False
