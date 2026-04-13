from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from src.core.bucket_mapping import BucketLayout
from src.core.parser import (
    EvidenceRecord,
    ParsedCarrier,
    ParsedCarrierBlock,
    ParserConfig,
    candidate_symbols,
    extract_candidates,
    load_evidence_records,
    load_expected_symbols,
    parse_structured_carrier_text,
    scan_candidate_windows,
)
from src.core.payload_codec import BucketPayloadCodec
from src.core.render import render_config_from_name
from src.core.synthetic_examples import build_synthetic_smoke_example


@dataclass(frozen=True)
class VerificationConfig:
    verification_mode: str = "synthetic_fixture"
    render_format: str = "canonical_v1"
    min_score: float = 0.0
    max_candidates: int | None = None
    min_match_ratio: float = 1.0
    scan_windows: bool = True
    require_all_fields: bool = True
    decode_as_bytes: bool = True
    apply_rs: bool = False


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    verification_mode: str
    render_format: str | None
    decoded_units: tuple[int, ...]
    decoded_payload: str | None
    decoded_bucket_tuples: tuple[tuple[int, ...], ...]
    parsed_blocks: tuple[ParsedCarrierBlock, ...]
    parsed_carriers: tuple[ParsedCarrier, ...]
    unresolved_fields: tuple[str, ...]
    bucket_mismatches: tuple[str, ...]
    messages: tuple[str, ...]
    expected_payload_units: tuple[int, ...] = ()
    details: dict[str, object] = field(default_factory=dict)
    recovered_symbols: tuple[str, ...] = ()
    expected_symbols: tuple[str, ...] = ()
    match_ratio: float = 0.0
    observed_count: int = 0
    malformed_count: int = 0

    @property
    def accepted(self) -> bool:
        return self.success

    def to_dict(self) -> dict[str, object]:
        def normalize(value: object) -> object:
            if is_dataclass(value):
                return asdict(value)
            if isinstance(value, tuple):
                return [normalize(item) for item in value]
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            return value

        return normalize(self)

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path


def compute_match_ratio(recovered_symbols: Iterable[str], expected_symbols: Iterable[str]) -> float:
    recovered = tuple(recovered_symbols)
    expected = tuple(expected_symbols)
    if not expected:
        return 0.0
    matches = sum(1 for left, right in zip(recovered, expected) if left == right)
    return matches / len(expected)


def _parser_config_for_verification(config: VerificationConfig) -> ParserConfig:
    if config.verification_mode == "canonical_render":
        render_config = render_config_from_name(config.render_format)
        return ParserConfig(
            min_score=config.min_score,
            max_candidates=config.max_candidates,
            block_separator=render_config.block_separator,
            field_separator=render_config.field_separator.rstrip(),
        )
    if config.verification_mode == "synthetic_fixture":
        return ParserConfig(
            min_score=config.min_score,
            max_candidates=config.max_candidates,
        )
    raise ValueError(f"Unsupported verification mode: {config.verification_mode}")


def verify_records(
    records: list[EvidenceRecord],
    expected_symbols: Iterable[str],
    config: VerificationConfig | None = None,
) -> VerificationResult:
    verify_config = config or VerificationConfig()
    expected = tuple(expected_symbols)
    parser_config = ParserConfig(
        min_score=verify_config.min_score,
        max_candidates=verify_config.max_candidates,
    )

    candidate_sets: list[list[EvidenceRecord]]
    if verify_config.scan_windows and expected:
        candidate_sets = scan_candidate_windows(records, len(expected), parser_config)
    else:
        candidate_sets = [extract_candidates(records, parser_config)]

    best_symbols: tuple[str, ...] = ()
    best_ratio = -1.0
    best_index = 0
    for index, candidate_set in enumerate(candidate_sets):
        recovered = candidate_symbols(candidate_set)
        ratio = compute_match_ratio(recovered, expected)
        if ratio > best_ratio:
            best_symbols = recovered
            best_ratio = ratio
            best_index = index

    accepted = best_ratio >= verify_config.min_match_ratio and best_symbols[: len(expected)] == expected
    return VerificationResult(
        success=accepted,
        verification_mode=verify_config.verification_mode,
        render_format=verify_config.render_format if verify_config.verification_mode == "canonical_render" else None,
        decoded_units=(),
        decoded_payload=None,
        decoded_bucket_tuples=(),
        parsed_blocks=(),
        parsed_carriers=(),
        unresolved_fields=(),
        bucket_mismatches=(),
        messages=(f"selected legacy candidate window {best_index}",),
        details={"selected_window_index": best_index},
        recovered_symbols=best_symbols,
        expected_symbols=expected,
        match_ratio=max(best_ratio, 0.0),
        observed_count=len(best_symbols),
        malformed_count=0,
    )


def verify_structured_text(
    text: str,
    bucket_layout: BucketLayout,
    payload_codec: BucketPayloadCodec,
    expected_payload: bytes | Sequence[int] | None = None,
    config: VerificationConfig | None = None,
) -> VerificationResult:
    verify_config = config or VerificationConfig()
    parser_config = _parser_config_for_verification(verify_config)
    parsed_blocks = tuple(parse_structured_carrier_text(text, bucket_layout, parser_config))
    parsed_carriers = tuple(carrier for block in parsed_blocks for carrier in block.carriers)

    unresolved_fields: list[str] = []
    bucket_tuples: list[tuple[int, ...]] = []
    messages: list[str] = []
    bucket_mismatches: list[str] = []

    for block in parsed_blocks:
        unresolved = block.unresolved_fields(bucket_layout.field_names)
        if unresolved:
            unresolved_fields.extend(f"block_{block.block_index}:{field}" for field in unresolved)
            messages.append(f"block {block.block_index} missing or unresolved fields: {unresolved}")
        bucket_tuple = block.bucket_tuple(bucket_layout.field_names)
        if bucket_tuple is not None:
            bucket_tuples.append(bucket_tuple)

    decoded_units: tuple[int, ...] = ()
    decoded_payload: str | None = None
    if bucket_tuples:
        decoded_units = payload_codec.decode_units(bucket_tuples, apply_rs=verify_config.apply_rs)
        if verify_config.decode_as_bytes:
            try:
                decoded_bytes = payload_codec.decode_bytes(bucket_tuples, apply_rs=verify_config.apply_rs)
                decoded_payload = decoded_bytes.decode("utf-8", errors="replace")
            except Exception:
                decoded_payload = None

    expected_payload_units: tuple[int, ...] = ()
    if expected_payload is not None:
        if isinstance(expected_payload, bytes):
            expected_payload_units = tuple(expected_payload)
        else:
            expected_payload_units = tuple(int(item) for item in expected_payload)
        if decoded_units and decoded_units != expected_payload_units:
            bucket_mismatches.append("decoded payload units differ from expected payload units")

    match_ratio = 0.0
    if expected_payload_units:
        match_ratio = compute_match_ratio(
            tuple(str(item) for item in decoded_units),
            tuple(str(item) for item in expected_payload_units),
        )

    malformed = [carrier for carrier in parsed_carriers if carrier.parse_status == "malformed"]
    if malformed:
        messages.append(f"encountered {len(malformed)} malformed field segments")

    success = bool(bucket_tuples)
    if verify_config.require_all_fields and unresolved_fields:
        success = False
    if expected_payload_units and decoded_units != expected_payload_units:
        success = False
    if malformed:
        success = False

    return VerificationResult(
        success=success,
        verification_mode=verify_config.verification_mode,
        render_format=verify_config.render_format if verify_config.verification_mode == "canonical_render" else None,
        decoded_units=decoded_units,
        decoded_payload=decoded_payload,
        decoded_bucket_tuples=tuple(bucket_tuples),
        parsed_blocks=parsed_blocks,
        parsed_carriers=parsed_carriers,
        unresolved_fields=tuple(unresolved_fields),
        bucket_mismatches=tuple(bucket_mismatches),
        messages=tuple(messages),
        expected_payload_units=expected_payload_units,
        details={"num_blocks": len(parsed_blocks)},
        match_ratio=match_ratio,
        observed_count=len(parsed_carriers),
        malformed_count=len(malformed),
    )


def verify_canonical_rendered_text(
    text: str,
    bucket_layout: BucketLayout,
    payload_codec: BucketPayloadCodec,
    expected_payload: bytes | Sequence[int] | None = None,
    config: VerificationConfig | None = None,
) -> VerificationResult:
    verify_config = config or VerificationConfig(verification_mode="canonical_render")
    if verify_config.verification_mode != "canonical_render":
        raise ValueError("verify_canonical_rendered_text requires verification_mode='canonical_render'")
    return verify_structured_text(
        text=text,
        bucket_layout=bucket_layout,
        payload_codec=payload_codec,
        expected_payload=expected_payload,
        config=verify_config,
    )


def verify_fixture(path: Path, config: VerificationConfig | None = None) -> VerificationResult:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "records" in payload:
            records = load_evidence_records(path)
            expected = load_expected_symbols(path)
            return verify_records(records, expected, config=config)

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported verification fixture format: {path}")
    bucket_layout = BucketLayout.from_dict(payload["layout"])
    payload_codec = BucketPayloadCodec(bucket_radices=bucket_layout.radices)
    expected_payload = payload.get("expected_payload_bytes")
    if isinstance(expected_payload, str):
        expected_payload_bytes = expected_payload.encode("utf-8")
    else:
        expected_payload_bytes = None
    return verify_structured_text(
        text=str(payload["text"]),
        bucket_layout=bucket_layout,
        payload_codec=payload_codec,
        expected_payload=expected_payload_bytes,
        config=config or VerificationConfig(verification_mode="synthetic_fixture"),
    )


def run_synthetic_smoke_verification() -> VerificationResult:
    example = build_synthetic_smoke_example()
    return verify_structured_text(
        text=example.rendered_text,
        bucket_layout=example.layout,
        payload_codec=example.codec,
        expected_payload=example.payload,
        config=VerificationConfig(
            verification_mode="canonical_render",
            require_all_fields=True,
            decode_as_bytes=True,
        ),
    )
