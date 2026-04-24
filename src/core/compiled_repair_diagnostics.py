from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from src.core.payload_codec import BucketPayloadCodec, PayloadCodecError
from src.core.scaffolded_completion import FoundationGateResult


@dataclass(frozen=True)
class CompiledVerificationReport:
    exact_payload_recovered: bool
    block_count_correct: bool
    slot_bucket_accuracy: float
    symbol_error_count: int
    erasure_count: int
    rs_correctable_under_2e_plus_s_lt_d: bool
    rs_recovered_payload: str | None
    accepted_under_exact_gate: bool
    accepted_under_rs_gate: bool
    expected_symbols: tuple[int, ...]
    decoded_symbols: tuple[int | None, ...]
    rs_config: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rs_correctable_under_2E_plus_S_lt_d"] = payload.pop(
            "rs_correctable_under_2e_plus_s_lt_d"
        )
        return payload


def _decode_block_symbol(
    codec: BucketPayloadCodec,
    bucket_tuple: Sequence[int | None],
) -> int | None:
    if any(value is None for value in bucket_tuple):
        return None
    try:
        return codec.mixed_radix_codec.decode_bucket_tuple_to_int(
            tuple(int(value) for value in bucket_tuple if value is not None)
        )
    except PayloadCodecError:
        return None


def build_compiled_verification_report(
    *,
    compiled_eval_contract: Any,
    compiled_result: FoundationGateResult,
    bucket_radices: Sequence[int],
    exact_gate_accepted: bool,
    rs_parity_symbols: int = 0,
) -> CompiledVerificationReport:
    """Decompose compiled-gate results without changing the strict exact gate."""

    codec = BucketPayloadCodec(bucket_radices=tuple(int(value) for value in bucket_radices))
    fields_per_block = int(compiled_eval_contract.fields_per_block)
    expected_symbols = tuple(int(value) for value in compiled_eval_contract.payload_units)
    expected_block_count = int(compiled_eval_contract.block_count)
    slot_diagnostics = tuple(compiled_result.slot_diagnostics)
    slot_bucket_accuracy = (
        sum(1 for item in slot_diagnostics if item.is_bucket_correct) / len(slot_diagnostics)
        if slot_diagnostics
        else 0.0
    )
    decoded_symbols: list[int | None] = []
    for block_index in range(expected_block_count):
        block_slots = slot_diagnostics[
            block_index * fields_per_block : (block_index + 1) * fields_per_block
        ]
        if len(block_slots) != fields_per_block:
            decoded_symbols.append(None)
            continue
        decoded_symbols.append(
            _decode_block_symbol(
                codec,
                tuple(item.observed_bucket_id for item in block_slots),
            )
        )

    erasure_count = sum(1 for symbol in decoded_symbols if symbol is None)
    symbol_error_count = sum(
        1
        for observed, expected in zip(decoded_symbols, expected_symbols, strict=False)
        if observed is not None and observed != expected
    )
    min_distance = int(rs_parity_symbols) + 1 if rs_parity_symbols > 0 else 1
    rs_correctable = (2 * symbol_error_count + erasure_count) < min_distance
    exact_payload_recovered = (
        tuple(symbol for symbol in decoded_symbols if symbol is not None) == expected_symbols
        and erasure_count == 0
    )
    rs_recovered_payload = (
        str(compiled_eval_contract.payload_label)
        if exact_payload_recovered
        else None
    )
    accepted_under_rs_gate = bool(
        rs_correctable and rs_recovered_payload == str(compiled_eval_contract.payload_label)
    )
    return CompiledVerificationReport(
        exact_payload_recovered=exact_payload_recovered,
        block_count_correct=compiled_result.valid_canonical_block_count == expected_block_count,
        slot_bucket_accuracy=slot_bucket_accuracy,
        symbol_error_count=symbol_error_count,
        erasure_count=erasure_count,
        rs_correctable_under_2e_plus_s_lt_d=rs_correctable,
        rs_recovered_payload=rs_recovered_payload,
        accepted_under_exact_gate=bool(exact_gate_accepted),
        accepted_under_rs_gate=accepted_under_rs_gate,
        expected_symbols=expected_symbols,
        decoded_symbols=tuple(decoded_symbols),
        rs_config={
            "active": False,
            "parity_symbols": int(rs_parity_symbols),
            "minimum_distance": min_distance,
            "decoder": "identity_no_rs" if rs_parity_symbols == 0 else "not_configured_for_g3a_v2",
            "note": "No RS decoder is active for G3a-v2; recovery is not faked.",
        },
    )


def build_majority_decoder_report(
    *,
    compiled_eval_contract: Any,
    decoded_symbols: Sequence[int | None],
) -> dict[str, object]:
    expected_symbols = tuple(int(value) for value in compiled_eval_contract.payload_units)
    block_count = int(compiled_eval_contract.block_count)
    if block_count != 4:
        return {
            "applicable": False,
            "reason": "majority diagnostic is only defined for B4 block_count=4",
        }

    votes_by_coordinate: dict[str, list[int | None]] = defaultdict(list)
    for index, symbol in enumerate(decoded_symbols):
        votes_by_coordinate[str(index)].append(symbol)
    coordinate_reports: list[dict[str, object]] = []
    conflicting_votes = 0
    for coordinate, votes in sorted(votes_by_coordinate.items(), key=lambda item: int(item[0])):
        non_erased = [vote for vote in votes if vote is not None]
        counts = Counter(non_erased)
        if counts:
            top_count = counts.most_common(1)[0][1]
            winners = sorted(symbol for symbol, count in counts.items() if count == top_count)
            majority_symbol = winners[0]
            tie = len(winners) > 1
        else:
            majority_symbol = None
            tie = True
        if len(set(non_erased)) > 1:
            conflicting_votes += 1
        coordinate_reports.append(
            {
                "coordinate": int(coordinate),
                "votes": votes,
                "majority_symbol": majority_symbol,
                "tie": tie,
                "expected_symbol": expected_symbols[int(coordinate)] if int(coordinate) < len(expected_symbols) else None,
            }
        )

    return {
        "applicable": False,
        "reason": (
            "G3a-v2 B4 uses fixed-width codeword positions, not a registered repeated-copy "
            "redundancy scheme; majority output is diagnostic only."
        ),
        "per_coordinate_votes": coordinate_reports,
        "majority_recovered_payload": None,
        "number_of_conflicting_votes": conflicting_votes,
        "official_verifier": False,
    }
