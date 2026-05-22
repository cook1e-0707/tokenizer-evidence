from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/natural_evidence_v2/aggregate_r4_after_870987_prefar_organic_null_generation.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_standard_summary(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_GENERATION_GATE",
            "combined_standard_control_summary_by_arm": {
                arm: {"total_blocks": 256, "total_accepts": 0, "total_accepts_ignoring_quality": 0}
                for arm in ("raw", "task_only", "wrong_key", "wrong_payload")
            },
        },
    )


def _write_shard(root: Path, shard: int) -> None:
    shard_dir = root / f"shard_{shard:02d}"
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
        {"status": "PASS_R4_FIRST_TOKEN_EVENT_TRACE_BINDING", "checked_rows": 3, "invalid_rows": 0},
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
        for index in range(3):
            handle.write(
                json.dumps(
                    {
                        "generation_id": f"gen-{shard}-raw-{index}",
                        "generation_condition": "raw",
                        "output_text_sha256": f"{shard}-raw-{index}",
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def test_prefar_organic_null_aggregate_passes_toy(tmp_path: Path) -> None:
    root = tmp_path / "shards"
    _write_shard(root, 0)
    _write_shard(root, 1)
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(out),
            "--shard-roots",
            str(root),
            "--expected-shards",
            "2",
            "--expected-generated-rows-per-shard",
            "3",
        ],
        check=True,
    )
    summary = json.loads((out / "prefar_organic_null_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_GENERATION_GATE"
    assert summary["organic_null_raw_summary"]["blocks"] == 2
    assert summary["organic_null_raw_summary"]["accepts"] == 0
