from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (
    match_surfaces,
    read_json,
    read_jsonl,
    resolve,
    segment_units,
    write_csv_new,
    write_json_new,
    write_text_new,
)


DEFAULT_INPUT = ROOT / "results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524"
DEFAULT_PRECOMMIT = ROOT / "results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512"
DEFAULT_OUTPUT = ROOT / "results/natural_evidence_v2/status/r4_artifact_only_oracle_recoverability_20260512"

PHRASE_ORACLE_BUCKETS = {
    0: {
        "set",
        "plan",
        "schedule",
        "organize",
        "assign",
        "establish",
        "arrange",
        "track",
        "record",
    },
    1: {
        "create",
        "prepare",
        "develop",
        "make",
        "build",
        "draft",
        "write",
        "clarify",
        "explain",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run artifact-only R4 oracle recoverability audits.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_PRECOMMIT / "surface_bank.json")
    parser.add_argument("--codebook", type=Path, default=DEFAULT_PRECOMMIT / "codebook.json")
    parser.add_argument("--decoder-spec", type=Path, default=DEFAULT_PRECOMMIT / "decoder_spec.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def block_id(row: Mapping[str, Any]) -> str:
    shard = str(row.get("replicate_group_id", ""))
    frame_index = int(row.get("frame_index", 0))
    return f"{shard}_block_{frame_index // 64:02d}"


def oracle_decode_from_observations(rows: list[Mapping[str, Any]], *, phrase_oracle: bool) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[int, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
    target_bits: dict[tuple[str, int], int] = {}
    for row in rows:
        if str(row.get("decode_condition", "")) != "protected":
            continue
        word = str(row.get("first_word", "")).lower()
        coord = int(row.get("frame_index", 0)) % 16
        bid = block_id(row)
        target_bits.setdefault((bid, coord), int(row.get("target_bit", 0)))
        bit = row.get("observed_bucket_id")
        if phrase_oracle and bit is None:
            for bucket_id, words in PHRASE_ORACLE_BUCKETS.items():
                if word in words:
                    bit = bucket_id
                    break
        if bit is None:
            continue
        grouped[bid][coord][int(bit)] += 1
    out: dict[str, dict[str, Any]] = {}
    for bid, coord_votes in grouped.items():
        support = 0
        correct = 0
        min_margin = 10**9
        for coord in range(16):
            votes = coord_votes.get(coord, Counter())
            if not votes:
                continue
            support += 1
            zero = votes.get(0, 0)
            one = votes.get(1, 0)
            margin = abs(one - zero)
            min_margin = min(min_margin, margin)
            majority = 1 if one > zero else 0
            target = target_bits.get((bid, coord), majority)
            correct += int(majority == target)
        out[bid] = {
            "block_id": bid,
            "support": support,
            "correct_coordinates": correct,
            "min_margin": 0 if min_margin == 10**9 else min_margin,
            "oracle_accept": support >= 16 and correct == support and (0 if min_margin == 10**9 else min_margin) >= 3,
        }
    return out


def structure_scrub_oracle(generated: list[Mapping[str, Any]], surface_bank: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, Counter[int]] = defaultdict(Counter)
    for row in generated:
        if str(row.get("generation_condition", "")) != "protected":
            continue
        shard = str(row.get("replicate_group_id", ""))
        prompt_index = int(row.get("prompt_index", 0))
        bid = f"{shard}_block_{prompt_index // 64:02d}"
        for match in match_surfaces(str(row.get("response_text", "")), surface_bank, scrub_mode="all"):
            grouped[bid][int(match["coordinate_id"])] += 1
    out: dict[str, dict[str, Any]] = {}
    for bid, counts in grouped.items():
        support = len(counts)
        out[bid] = {
            "block_id": bid,
            "support": support,
            "oracle_accept": support >= 22,
        }
    return out


def main() -> int:
    args = parse_args()
    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    observations = read_jsonl(input_dir / "r3_2_slot_observations.jsonl")
    generated = read_jsonl(input_dir / "r3_2_generated_outputs.jsonl")
    surface_bank = read_json(resolve(args.surface_bank)) if resolve(args.surface_bank).exists() else {"entries": []}

    segment_unit_counts = [len(segment_units(str(row.get("response_text", "")))) for row in generated if row.get("generation_condition") == "protected"]
    segment_oracle_upper_bound = sum(1 for count in segment_unit_counts if count >= 8)
    exact = oracle_decode_from_observations(observations, phrase_oracle=False)
    phrase = oracle_decode_from_observations(observations, phrase_oracle=True)
    scrub = structure_scrub_oracle(generated, surface_bank)

    all_block_ids = sorted(set(exact) | set(phrase) | set(scrub))
    per_block = []
    for bid in all_block_ids:
        per_block.append(
            {
                "block_id": bid,
                "exact_oracle_accept": bool(exact.get(bid, {}).get("oracle_accept", False)),
                "phrase_surface_oracle_accept": bool(phrase.get(bid, {}).get("oracle_accept", False)),
                "structure_scrub_oracle_accept": bool(scrub.get(bid, {}).get("oracle_accept", False)),
                "exact_support": exact.get(bid, {}).get("support", 0),
                "phrase_support": phrase.get(bid, {}).get("support", 0),
                "structure_scrub_support": scrub.get(bid, {}).get("support", 0),
            }
        )
    erasure_counts = Counter()
    for row in observations:
        if row.get("decode_condition") == "protected" and not row.get("resolved_bucket_hit"):
            erasure_counts["observed_first_word_not_in_primary_bucket_set"] += 1
    summary = {
        "schema_name": "natural_evidence_v2_r4_artifact_only_oracle_recoverability_v1",
        "base_negative_artifact": "853524",
        "block_count": len(all_block_ids),
        "segment_oracle_blocks_with_at_least_8_units": segment_oracle_upper_bound,
        "exact_oracle_accepts": sum(1 for row in per_block if row["exact_oracle_accept"]),
        "phrase_surface_oracle_accepts": sum(1 for row in per_block if row["phrase_surface_oracle_accept"]),
        "structure_scrub_oracle_accepts": sum(1 for row in per_block if row["structure_scrub_oracle_accept"]),
        "go_no_go": {
            "structure_scrub_oracle_upper_bound_required": ">=70/96",
            "phrase_surface_oracle_upper_bound_required": ">=80/96",
            "null_oracle_accepts_required": "0/96",
            "status": "DIAGNOSTIC_ONLY_NO_GO_FOR_853524_RECLASSIFICATION",
        },
        "erasure_taxonomy": dict(erasure_counts),
        "does_not_reclassify_853524": True,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
    }
    write_json_new(output_dir / "oracle_recoverability_summary.json", summary)
    write_csv_new(
        output_dir / "per_block_oracle.csv",
        per_block,
        [
            "block_id",
            "exact_oracle_accept",
            "phrase_surface_oracle_accept",
            "structure_scrub_oracle_accept",
            "exact_support",
            "phrase_support",
            "structure_scrub_support",
        ],
    )
    per_coordinate = [
        {
            "coordinate_id": coord,
            "protected_rows": sum(1 for row in observations if row.get("decode_condition") == "protected" and int(row.get("frame_index", 0)) % 16 == coord),
            "resolved_rows": sum(1 for row in observations if row.get("decode_condition") == "protected" and int(row.get("frame_index", 0)) % 16 == coord and row.get("resolved_bucket_hit")),
        }
        for coord in range(16)
    ]
    write_csv_new(output_dir / "per_coordinate_oracle.csv", per_coordinate, ["coordinate_id", "protected_rows", "resolved_rows"])
    write_csv_new(output_dir / "erasure_taxonomy.csv", [{"reason": k, "count": v} for k, v in erasure_counts.items()], ["reason", "count"])
    report = "\n".join(
        [
            "# R4 artifact-only oracle recoverability audit",
            "",
            "This audit uses `853524` only as failure taxonomy and upper-bound evidence.",
            "",
            f"Exact oracle accepts: `{summary['exact_oracle_accepts']}/{summary['block_count']}`",
            f"Phrase-surface oracle accepts: `{summary['phrase_surface_oracle_accepts']}/{summary['block_count']}`",
            f"Structure-scrub oracle accepts: `{summary['structure_scrub_oracle_accepts']}/{summary['block_count']}`",
            "",
            "The audit does not reclassify `853524` and does not authorize Slurm.",
            "",
        ]
    )
    write_text_new(output_dir / "oracle_recoverability_report.md", report)
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
