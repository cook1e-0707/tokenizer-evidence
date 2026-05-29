from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/natural_evidence_v2/aggregate_r4_after_870987_same_family_raw_null_generation.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_shard(root: Path, shard: int, *, text: str = "Use a bucket to catch dripping water.") -> None:
    shard_dir = root / "shards" / f"shard_{shard:02d}"
    ft_dir = shard_dir / "first_token_event_decode"
    all_dir = shard_dir / "decode_all"
    none_dir = shard_dir / "decode_none"
    ft_dir.mkdir(parents=True)
    all_dir.mkdir(parents=True)
    none_dir.mkdir(parents=True)
    _write_json(ft_dir / "first_token_event_decode_summary.json", {"status": "recorded"})
    _write_json(all_dir / "decode_summary.json", {"status": "report_only"})
    _write_json(none_dir / "decode_summary.json", {"status": "report_only"})
    _write_json(
        shard_dir / "trace_binding_validation.json",
        {"status": "PASS_R4_FIRST_TOKEN_EVENT_TRACE_BINDING", "checked_rows": 2, "invalid_rows": 0},
    )
    with (ft_dir / "first_token_event_per_block.csv").open("w", encoding="utf-8") as handle:
        handle.write(
            "block_id,arm,source_condition,accept,accept_ignoring_quality,complete_pairs,required_pairs,"
            "decoded_bits,expected_bits,bits_match_condition,checksum_valid,forbidden_public_surface_count,"
            "duplicate_response_hash_count\n"
        )
        for arm in ("protected", "raw", "task_only", "wrong_key", "wrong_payload"):
            handle.write(
                f"shard_{shard:02d}_block_00,{arm},{arm},False,False,0,8,--------,10100101,"
                "False,False,0,0\n"
            )
    for mode_dir, mode in ((all_dir, "all"), (none_dir, "none")):
        with (mode_dir / "per_block_decode.csv").open("w", encoding="utf-8") as handle:
            handle.write(
                "block_id,arm,accept,complete_pairs,required_pairs,selected_coordinates_observed,"
                "selected_coordinates_total,min_pair_support,matched_surface_count,selected_surface_count,"
                "checksum_valid,bits_match_condition,forbidden_public_surface_count,format_scrub_mode\n"
            )
            for arm in ("protected", "raw", "task_only", "wrong_key", "wrong_payload"):
                handle.write(
                    f"shard_{shard:02d}_block_00,{arm},False,0,8,0,16,0,0,0,False,False,0,{mode}\n"
                )
    with (shard_dir / "r4_generated_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(2):
            handle.write(
                json.dumps(
                    {
                        "arm": "raw",
                        "generation_condition": "raw",
                        "generation_id": f"gen-{root.name}-{shard}-{index}",
                        "output_text_sha256": f"{root.name}-{shard}-{index}",
                        "response_text": text,
                        "wrong_key_replay_accept": False,
                        "wrong_payload_replay_accept": False,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def test_same_family_raw_null_aggregate_passes_toy_with_ordinary_bucket(tmp_path: Path) -> None:
    roots = []
    for model in ("qwen2_5_3b_instruct_raw", "qwen2_5_7b_instruct_raw", "qwen2_5_14b_instruct_raw"):
        root = tmp_path / model
        _write_shard(root, 0)
        _write_shard(root, 1)
        roots.append(root)
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(out),
            "--model-roots",
            *[str(root) for root in roots],
            "--expected-shards-per-model",
            "2",
            "--expected-generated-rows-per-shard",
            "2",
        ],
        check=True,
    )
    summary = json.loads((out / "same_family_raw_null_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE"
    assert summary["same_family_raw_null_gate_pass"] is True
    assert summary["aggregate"]["ordinary_domain_literal_count"] > 0
    assert summary["aggregate"]["technical_forbidden_public_surface_count"] == 0


def test_same_family_raw_null_aggregate_fails_technical_literal(tmp_path: Path) -> None:
    roots = []
    for model in ("qwen2_5_3b_instruct_raw", "qwen2_5_7b_instruct_raw", "qwen2_5_14b_instruct_raw"):
        root = tmp_path / model
        text = "The payload watermark is a hidden signal for the decoder."
        _write_shard(root, 0, text=text)
        roots.append(root)
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(out),
            "--model-roots",
            *[str(root) for root in roots],
            "--expected-shards-per-model",
            "1",
            "--expected-generated-rows-per-shard",
            "2",
        ],
        check=False,
    )
    assert result.returncode == 1
    summary = json.loads((out / "same_family_raw_null_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE"
    assert summary["aggregate"]["technical_forbidden_public_surface_count"] > 0
