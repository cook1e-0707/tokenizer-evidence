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
    technical_literal_hits,
    write_csv_new,
    write_json_new,
    write_jsonl_new,
)


DEFAULT_PRECOMMIT = ROOT / "results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode generated outputs with the R4 cover-natural erasure-aware "
            "phrase ECC decoder. Artifact-only when used with existing transcripts."
        )
    )
    parser.add_argument("--generated-outputs", type=Path, required=True)
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_PRECOMMIT / "surface_bank.json")
    parser.add_argument("--codebook", type=Path, default=DEFAULT_PRECOMMIT / "codebook.json")
    parser.add_argument("--decoder-spec", type=Path, default=DEFAULT_PRECOMMIT / "decoder_spec.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format-scrub", default="all")
    parser.add_argument("--prompts-per-block", type=int, default=64)
    parser.add_argument("--split-label", default="dev")
    parser.add_argument("--include-protected-controls", action="store_true")
    return parser.parse_args()


def block_id_for(row: Mapping[str, Any], prompts_per_block: int) -> str:
    shard = str(row.get("replicate_group_id", "local"))
    prompt_index = int(row.get("prompt_index", 0))
    return f"{shard}_block_{prompt_index // prompts_per_block:02d}"


def expected_bits_for_condition(condition: str, codebook: Mapping[str, Any]) -> list[int]:
    if condition == "wrong_payload":
        return [int(bit) for bit in codebook["wrong_payload_codeword_bits"]]
    if condition == "wrong_key":
        return [int(bit) for bit in codebook["wrong_key_codeword_bits"]]
    return [int(bit) for bit in codebook["protected_codeword_bits"]]


def decode_block(
    *,
    block_id: str,
    condition: str,
    rows: list[Mapping[str, Any]],
    surface_bank: Mapping[str, Any],
    codebook: Mapping[str, Any],
    decoder_spec: Mapping[str, Any],
    scrub_mode: str,
    split_label: str,
) -> dict[str, Any]:
    votes: dict[int, Counter[int]] = defaultdict(Counter)
    matched_surface_count = 0
    forbidden_hits: Counter[str] = Counter()
    for row in rows:
        text = str(row.get("response_text", ""))
        forbidden_hits.update(technical_literal_hits(text))
        for match in match_surfaces(text, surface_bank, scrub_mode=scrub_mode):
            coord = int(match["coordinate_id"])
            bit = int(match["polarity_or_code_symbol"])
            weight = float(match["weight"])
            votes[coord][bit] += weight
            matched_surface_count += 1
    expected = expected_bits_for_condition(condition, codebook)
    observed_bits: list[int | str] = []
    margins: list[float] = []
    support_count = 0
    correct_count = 0
    for coord, expected_bit in enumerate(expected):
        counter = votes.get(coord, Counter())
        if not counter:
            observed_bits.append("")
            continue
        zero = float(counter.get(0, 0.0))
        one = float(counter.get(1, 0.0))
        bit = 1 if one > zero else 0
        observed_bits.append(bit)
        support_count += 1
        margins.append(abs(one - zero))
        if bit == int(expected_bit):
            correct_count += 1
    min_support = int(decoder_spec.get(f"min_required_observed_coordinates_{split_label}", 22))
    if f"min_required_observed_coordinates_{split_label}" not in decoder_spec:
        min_support = int(decoder_spec.get("min_required_observed_coordinates_locked", 22))
    min_margin = float(decoder_spec.get(f"min_weighted_margin_{split_label}", 3))
    if f"min_weighted_margin_{split_label}" not in decoder_spec:
        min_margin = float(decoder_spec.get("min_weighted_margin_locked", 3))
    weighted_margin = min(margins) if margins else 0.0
    payload_matches = support_count > 0 and correct_count == support_count
    accepted = (
        payload_matches
        and support_count >= min_support
        and weighted_margin >= min_margin
        and not forbidden_hits
    )
    return {
        "schema_name": "natural_evidence_v2_r4_cover_natural_decode_row_v1",
        "block_id": block_id,
        "arm": condition,
        "accept": accepted,
        "support": support_count,
        "weighted_margin": weighted_margin,
        "observed_coordinates": support_count,
        "erased_coordinates": len(expected) - support_count,
        "matched_surface_count": matched_surface_count,
        "format_scrub_mode": scrub_mode,
        "payload_matches": payload_matches,
        "correct_observed_coordinates": correct_count,
        "wrong_key_margin": 0.0 if condition == "wrong_key" else None,
        "wrong_payload_margin": 0.0 if condition == "wrong_payload" else None,
        "forbidden_public_surface_count": sum(forbidden_hits.values()),
        "forbidden_public_surfaces": dict(forbidden_hits),
        "majority_bits": observed_bits,
    }


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    generated = read_jsonl(resolve(args.generated_outputs))
    surface_bank = read_json(resolve(args.surface_bank))
    codebook = read_json(resolve(args.codebook))
    decoder_spec = read_json(resolve(args.decoder_spec))
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in generated:
        condition = str(row.get("generation_condition", "unknown"))
        grouped[(condition, block_id_for(row, args.prompts_per_block))].append(row)

    decode_rows: list[dict[str, Any]] = []
    for (condition, block_id), rows in sorted(grouped.items()):
        decode_rows.append(
            decode_block(
                block_id=block_id,
                condition=condition,
                rows=rows,
                surface_bank=surface_bank,
                codebook=codebook,
                decoder_spec=decoder_spec,
                scrub_mode=args.format_scrub,
                split_label=args.split_label,
            )
        )
        if args.include_protected_controls and condition == "protected":
            for control_condition in ("wrong_key", "wrong_payload"):
                decode_rows.append(
                    decode_block(
                        block_id=block_id,
                        condition=control_condition,
                        rows=rows,
                        surface_bank=surface_bank,
                        codebook=codebook,
                        decoder_spec=decoder_spec,
                        scrub_mode=args.format_scrub,
                        split_label=args.split_label,
                    )
                )
    summary_by_arm: dict[str, Counter[str]] = defaultdict(Counter)
    for row in decode_rows:
        arm = str(row["arm"])
        summary_by_arm[arm]["blocks"] += 1
        summary_by_arm[arm]["accepts"] += int(bool(row["accept"]))
    summary = {
        "schema_name": "natural_evidence_v2_r4_cover_natural_decode_summary_v1",
        "generated_outputs": str(args.generated_outputs),
        "surface_bank": str(args.surface_bank),
        "codebook": str(args.codebook),
        "decoder_spec": str(args.decoder_spec),
        "format_scrub_mode": args.format_scrub,
        "summary_by_arm": {arm: dict(counter) for arm, counter in sorted(summary_by_arm.items())},
        "generation_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
    }
    write_jsonl_new(output_dir / "decode_rows.jsonl", decode_rows)
    write_json_new(output_dir / "decode_summary.json", summary)
    write_csv_new(
        output_dir / "per_block_decode.csv",
        decode_rows,
        [
            "block_id",
            "arm",
            "accept",
            "support",
            "weighted_margin",
            "observed_coordinates",
            "erased_coordinates",
            "matched_surface_count",
            "format_scrub_mode",
            "payload_matches",
            "forbidden_public_surface_count",
        ],
    )
    per_coordinate = [
        {
            "coordinate_id": index,
            "observed_blocks": sum(1 for row in decode_rows if row["majority_bits"][index] != ""),
            "total_blocks": len(decode_rows),
        }
        for index in range(32)
    ]
    write_csv_new(output_dir / "per_coordinate_decode.csv", per_coordinate, ["coordinate_id", "observed_blocks", "total_blocks"])
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
