from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    match_surfaces,
    read_json,
    read_jsonl,
    resolve,
    technical_literal_hits,
    write_csv_new,
    write_json_new,
    write_jsonl_new,
)


DEFAULT_SURFACE_BANK = ROOT / "results/natural_evidence_v2/precommit/r4_after_864832_two_sided_cover_bank_20260516/surface_bank.json"
DEFAULT_PRECOMMIT = ROOT / "results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516"
DEFAULT_ROUTE_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_864832_reliability_codebook_oracle_route.yaml"


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def as_bits(value: Any, *, name: str, length: int = 8) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of length {length}")
    bits = [int(item) for item in value]
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError(f"{name} must contain only 0/1 bits")
    return bits


def block_id_for(row: Mapping[str, Any], prompts_per_block: int) -> str:
    shard = str(row.get("replicate_group_id", "local"))
    prompt_index = int(row.get("prompt_index", 0))
    return f"{shard}_block_{prompt_index // prompts_per_block:02d}"


def expected_bits_for_condition(condition: str, route_config: Mapping[str, Any]) -> list[int]:
    expected = as_bits(route_config["expected_codeword_bits"], name="expected_codeword_bits")
    if condition == "wrong_payload":
        return as_bits(route_config["wrong_payload_bits"], name="wrong_payload_bits", length=4) + as_bits(
            route_config["wrong_payload_checksum_bits"], name="wrong_payload_checksum_bits", length=4
        )
    if condition == "wrong_key":
        mask = as_bits(route_config["wrong_key_xor_mask"], name="wrong_key_xor_mask")
        return [bit ^ mask_bit for bit, mask_bit in zip(expected, mask, strict=True)]
    return expected


def complement(bits: Sequence[int]) -> list[int]:
    return [1 - int(bit) for bit in bits]


def selected_coordinate_set(codebook: Mapping[str, Any]) -> set[int]:
    return {int(item) for item in codebook.get("selected_coordinates", [])}


def decode_pair_bits(
    votes: Mapping[int, Counter[int]],
    pair_mapping: Sequence[Mapping[str, Any]],
) -> tuple[list[int | str], list[dict[str, Any]]]:
    decoded_bits: list[int | str] = [""] * 8
    pair_rows: list[dict[str, Any]] = []
    for item in sorted(pair_mapping, key=lambda row: int(row["bit_index"])):
        bit_index = int(item["bit_index"])
        coordinates = [int(coordinate) for coordinate in item["coordinates"]]
        pair_counter: Counter[int] = Counter()
        support = 0
        coordinate_votes: list[dict[str, Any]] = []
        for coordinate in coordinates:
            counter = votes.get(coordinate, Counter())
            zero = float(counter.get(0, 0.0))
            one = float(counter.get(1, 0.0))
            coordinate_votes.append({"coordinate_id": coordinate, "zero": zero, "one": one})
            if zero == 0.0 and one == 0.0:
                continue
            support += 1
            if one > zero:
                pair_counter[1] += 1
            elif zero > one:
                pair_counter[0] += 1
        if support == 0:
            decoded = ""
            failure = "missing_pair"
        elif pair_counter[0] == pair_counter[1]:
            decoded = ""
            failure = "pair_tie"
        else:
            decoded = 1 if pair_counter[1] > pair_counter[0] else 0
            decoded_bits[bit_index] = decoded
            failure = ""
        pair_rows.append(
            {
                "bit_index": bit_index,
                "coordinates": coordinates,
                "support": support,
                "pair_zero_votes": int(pair_counter[0]),
                "pair_one_votes": int(pair_counter[1]),
                "decoded_bit": decoded,
                "failure_reason": failure,
                "coordinate_votes": coordinate_votes,
            }
        )
    return decoded_bits, pair_rows


def decode_block(
    *,
    block_id: str,
    condition: str,
    rows: Sequence[Mapping[str, Any]],
    surface_bank: Mapping[str, Any],
    codebook: Mapping[str, Any],
    route_config: Mapping[str, Any],
    scrub_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    votes: dict[int, Counter[int]] = defaultdict(Counter)
    selected_coordinates = selected_coordinate_set(codebook)
    matched_surface_count = 0
    selected_surface_count = 0
    forbidden_hits: Counter[str] = Counter()
    event_rows: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("response_text", ""))
        generation_id = str(row.get("generation_id", ""))
        forbidden_hits.update(technical_literal_hits(text))
        for match in match_surfaces(text, surface_bank, scrub_mode=scrub_mode):
            matched_surface_count += 1
            coordinate = int(match["coordinate_id"])
            bit = int(match["polarity_or_code_symbol"])
            is_selected = coordinate in selected_coordinates
            if is_selected:
                selected_surface_count += 1
                votes[coordinate][bit] += float(match["weight"])
            event_rows.append(
                {
                    "block_id": block_id,
                    "arm": condition,
                    "generation_id": generation_id,
                    "coordinate_id": coordinate,
                    "bit": bit,
                    "selected_coordinate": is_selected,
                    "matched_phrase": str(match["matched_phrase"]),
                    "surface_id": str(match["surface_id"]),
                    "format_scrub_mode": scrub_mode,
                }
            )

    pair_mapping = codebook["pair_to_bit_mapping"]
    decoded_bits, pair_rows = decode_pair_bits(votes, pair_mapping)
    expected_bits = expected_bits_for_condition(condition, route_config)
    complete = all(bit != "" for bit in decoded_bits)
    payload_bits = [int(bit) for bit in decoded_bits[:4]] if complete else []
    checksum_bits = [int(bit) for bit in decoded_bits[4:]] if complete else []
    checksum_expected = complement(payload_bits) if complete else []
    checksum_valid = checksum_bits == checksum_expected if complete else False
    bits_match_condition = decoded_bits == expected_bits if complete else False
    min_pair_support = min((int(row["support"]) for row in pair_rows), default=0)
    complete_pairs = sum(1 for bit in decoded_bits if bit != "")
    selected_coordinates_observed = sum(
        1 for coordinate in selected_coordinates if votes.get(coordinate, Counter())
    )
    rejected_reasons: list[str] = []
    if not complete:
        rejected_reasons.append("incomplete_pairs")
    if complete and not checksum_valid:
        rejected_reasons.append("checksum_mismatch")
    if complete and not bits_match_condition:
        rejected_reasons.append("condition_codeword_mismatch")
    if forbidden_hits:
        rejected_reasons.append("forbidden_public_surface")
    accepted = bool(complete and checksum_valid and bits_match_condition and not forbidden_hits)
    decode_row = {
        "schema_name": "natural_evidence_v2_r4_after_864832_reliability_decode_row_v1",
        "block_id": block_id,
        "arm": condition,
        "accept": accepted,
        "format_scrub_mode": scrub_mode,
        "complete_pairs": complete_pairs,
        "required_pairs": 8,
        "selected_coordinates_observed": selected_coordinates_observed,
        "selected_coordinates_total": len(selected_coordinates),
        "min_pair_support": min_pair_support,
        "matched_surface_count": matched_surface_count,
        "selected_surface_count": selected_surface_count,
        "decoded_bits": decoded_bits,
        "expected_bits": expected_bits,
        "payload_bits": payload_bits,
        "checksum_bits": checksum_bits,
        "checksum_expected": checksum_expected,
        "checksum_valid": checksum_valid,
        "bits_match_condition": bits_match_condition,
        "forbidden_public_surface_count": sum(forbidden_hits.values()),
        "forbidden_public_surfaces": dict(forbidden_hits),
        "rejected_reasons": rejected_reasons,
        "pair_trace": pair_rows,
        "generation_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }
    return decode_row, event_rows


def summarize(decode_rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_arm: dict[str, Counter[str]] = defaultdict(Counter)
    for row in decode_rows:
        arm = str(row["arm"])
        by_arm[arm]["blocks"] += 1
        by_arm[arm]["accepts"] += int(bool(row["accept"]))
        by_arm[arm]["forbidden_public_surface_count"] += int(row["forbidden_public_surface_count"])
    return {
        "schema_name": "natural_evidence_v2_r4_after_864832_reliability_decode_summary_v1",
        "generated_outputs": str(args.generated_outputs),
        "surface_bank": str(args.surface_bank),
        "codebook": str(args.codebook),
        "decoder_spec": str(args.decoder_spec),
        "route_config": str(args.route_config),
        "format_scrub_mode": str(args.format_scrub),
        "summary_by_arm": {arm: dict(counter) for arm, counter in sorted(by_arm.items())},
        "decode_rows": len(decode_rows),
        "generation_started": False,
        "slurm_submitted": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode R4 generated outputs with the after-864832 reliability codebook.")
    parser.add_argument("--generated-outputs", type=Path, required=True)
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_SURFACE_BANK)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_PRECOMMIT / "codebook.json")
    parser.add_argument("--decoder-spec", type=Path, default=DEFAULT_PRECOMMIT / "decoder_spec.json")
    parser.add_argument("--route-config", type=Path, default=DEFAULT_ROUTE_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format-scrub", default="all")
    parser.add_argument("--prompts-per-block", type=int, default=64)
    parser.add_argument("--include-protected-controls", action="store_true")
    parser.add_argument(
        "--control-source-condition",
        default="protected",
        help=(
            "When --include-protected-controls is set, decode wrong-key and "
            "wrong-payload controls from this generated condition. Historical "
            "adapter routes use protected; controller-only routes use controlled_base."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated = read_jsonl(resolve(args.generated_outputs))
    surface_bank = read_json(resolve(args.surface_bank))
    codebook = read_json(resolve(args.codebook))
    decoder_spec = read_json(resolve(args.decoder_spec))
    route_config = read_yaml(resolve(args.route_config))
    if decoder_spec.get("decoder") != "pair_majority_then_checksum":
        raise ValueError("decoder spec must be pair_majority_then_checksum")
    output_dir = resolve(args.output_dir)

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in generated:
        condition = str(row.get("generation_condition", "unknown"))
        grouped[(condition, block_id_for(row, args.prompts_per_block))].append(row)

    decode_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for (condition, block_id), rows in sorted(grouped.items()):
        decode_row, block_events = decode_block(
            block_id=block_id,
            condition=condition,
            rows=rows,
            surface_bank=surface_bank,
            codebook=codebook,
            route_config=route_config,
            scrub_mode=str(args.format_scrub),
        )
        decode_rows.append(decode_row)
        event_rows.extend(block_events)
        if args.include_protected_controls and condition == str(args.control_source_condition):
            for control_condition in ("wrong_key", "wrong_payload"):
                control_row, control_events = decode_block(
                    block_id=block_id,
                    condition=control_condition,
                    rows=rows,
                    surface_bank=surface_bank,
                    codebook=codebook,
                    route_config=route_config,
                    scrub_mode=str(args.format_scrub),
                )
                decode_rows.append(control_row)
                event_rows.extend(control_events)

    write_jsonl_new(output_dir / "decode_rows.jsonl", decode_rows)
    write_jsonl_new(output_dir / "matched_surface_events.jsonl", event_rows)
    write_json_new(output_dir / "decode_summary.json", summarize(decode_rows, args))
    write_csv_new(
        output_dir / "per_block_decode.csv",
        decode_rows,
        [
            "block_id",
            "arm",
            "accept",
            "complete_pairs",
            "required_pairs",
            "selected_coordinates_observed",
            "selected_coordinates_total",
            "min_pair_support",
            "matched_surface_count",
            "selected_surface_count",
            "checksum_valid",
            "bits_match_condition",
            "forbidden_public_surface_count",
            "format_scrub_mode",
        ],
    )
    per_pair_rows: list[dict[str, Any]] = []
    for row in decode_rows:
        for pair in row["pair_trace"]:
            per_pair_rows.append(
                {
                    "block_id": row["block_id"],
                    "arm": row["arm"],
                    "bit_index": pair["bit_index"],
                    "coordinates": json.dumps(pair["coordinates"]),
                    "support": pair["support"],
                    "pair_zero_votes": pair["pair_zero_votes"],
                    "pair_one_votes": pair["pair_one_votes"],
                    "decoded_bit": pair["decoded_bit"],
                    "failure_reason": pair["failure_reason"],
                }
            )
    write_csv_new(
        output_dir / "per_pair_decode.csv",
        per_pair_rows,
        [
            "block_id",
            "arm",
            "bit_index",
            "coordinates",
            "support",
            "pair_zero_votes",
            "pair_one_votes",
            "decoded_bit",
            "failure_reason",
        ],
    )
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "decode_rows": len(decode_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
