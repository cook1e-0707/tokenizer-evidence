from __future__ import annotations

import json
from pathlib import Path

from scripts.natural_evidence_v2 import review_r4_after_868151_quality_repaired_generation as review


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_quality_repaired_generation_review_fixture(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "job"
    shard = input_dir / "shards" / "shard_00"
    output_dir = tmp_path / "review"

    _write_jsonl(
        shard / "r4_generated_outputs.jsonl",
        [
            {"response_text_sha256": "h0"},
            {"response_text_sha256": "h1"},
        ],
    )
    _write_json(
        shard / "first_token_event_decode" / "first_token_event_decode_summary.json",
        {
            "summary_by_arm": {
                "protected": {
                    "blocks": 4,
                    "accepts": 3,
                    "accepts_ignoring_quality": 3,
                    "forbidden_public_surface_count": 0,
                    "duplicate_response_hash_count": 0,
                },
                "raw": {
                    "blocks": 4,
                    "accepts": 0,
                    "accepts_ignoring_quality": 0,
                    "forbidden_public_surface_count": 0,
                    "duplicate_response_hash_count": 0,
                },
                "task_only": {
                    "blocks": 4,
                    "accepts": 0,
                    "accepts_ignoring_quality": 0,
                    "forbidden_public_surface_count": 0,
                    "duplicate_response_hash_count": 0,
                },
                "wrong_key": {
                    "blocks": 4,
                    "accepts": 0,
                    "accepts_ignoring_quality": 0,
                    "forbidden_public_surface_count": 0,
                    "duplicate_response_hash_count": 0,
                },
                "wrong_payload": {
                    "blocks": 4,
                    "accepts": 0,
                    "accepts_ignoring_quality": 0,
                    "forbidden_public_surface_count": 0,
                    "duplicate_response_hash_count": 0,
                },
            }
        },
    )
    _write_jsonl(
        shard / "first_token_event_decode" / "first_token_event_rows.jsonl",
        [
            {"event_source": "token_id_trace", "event_status": "target", "condition": "protected"},
            {"event_source": "token_id_trace", "event_status": "erasure", "condition": "raw"},
        ],
    )
    _write_jsonl(
        shard / "first_token_event_decode" / "first_token_event_decode_rows.jsonl",
        [
            {
                "arm": "protected",
                "block_id": "shard_00_block_00",
                "accept": True,
                "decoded_bits": "10100101",
                "expected_bits": "10100101",
                "complete_pairs": 8,
                "required_pairs": 8,
                "checksum_valid": True,
                "bits_match_condition": True,
                "forbidden_public_surface_count": 0,
                "duplicate_response_hash_count": 0,
                "pair_trace": [{"bit_index": 0, "decoded_bit": 1, "support": 2}],
            }
        ],
    )
    for mode in ("decode_all", "decode_none"):
        _write_json(
            shard / mode / "decode_summary.json",
            {
                "summary_by_arm": {
                    "protected": {
                        "blocks": 1,
                        "accepts": 0,
                        "accepts_ignoring_quality": 0,
                        "forbidden_public_surface_count": 0,
                        "duplicate_response_hash_count": 0,
                    }
                }
            },
        )

    monkeypatch.setattr(
        "sys.argv",
        [
            "review",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--expected-shards",
            "1",
            "--expected-event-trace-rows",
            "2",
        ],
    )

    assert review.main() == 0
    summary = json.loads((output_dir / "quality_repaired_generation_review_summary.json").read_text())
    assert summary["first_token_block_diagnostic_gate_pass"] is True
    assert summary["claim_policy"] == "diagnostic_only_not_locked_positive_not_paper_claim"
