from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.core.bucket_mapping import BucketLayout, FieldBucketSpec
from src.core.render import render_config_from_name


@dataclass(frozen=True)
class EvidenceRecord:
    position: int
    bucket_id: str
    symbol: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EvidenceRecord":
        return cls(
            position=int(data["position"]),
            bucket_id=str(data["bucket_id"]),
            symbol=str(data["symbol"]),
            score=float(data["score"]),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParserConfig:
    min_score: float = 0.0
    max_candidates: int | None = None
    block_separator: str = "\n"
    field_separator: str = ";"


@dataclass(frozen=True)
class ParsedCarrier:
    raw_text: str
    field_name: str | None
    carrier_value: str | None
    bucket_id: int | None
    span: tuple[int, int]
    parse_status: str
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedCarrierBlock:
    block_index: int
    raw_text: str
    span: tuple[int, int]
    carriers: tuple[ParsedCarrier, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "raw_text": self.raw_text,
            "span": self.span,
            "carriers": [carrier.to_dict() for carrier in self.carriers],
        }

    def carrier_map(self) -> dict[str, ParsedCarrier]:
        return {
            carrier.field_name: carrier
            for carrier in self.carriers
            if carrier.field_name is not None
        }

    def unresolved_fields(self, expected_fields: Sequence[str]) -> tuple[str, ...]:
        carrier_map = self.carrier_map()
        unresolved: list[str] = []
        for field_name in expected_fields:
            candidate = carrier_map.get(field_name)
            if candidate is None or candidate.bucket_id is None or candidate.parse_status != "resolved":
                unresolved.append(field_name)
        return tuple(unresolved)

    def bucket_tuple(self, field_order: Sequence[str]) -> tuple[int, ...] | None:
        carrier_map = self.carrier_map()
        bucket_ids: list[int] = []
        for field_name in field_order:
            candidate = carrier_map.get(field_name)
            if candidate is None or candidate.bucket_id is None or candidate.parse_status != "resolved":
                return None
            bucket_ids.append(candidate.bucket_id)
        return tuple(bucket_ids)


def load_evidence_records(path: Path) -> list[EvidenceRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    return [EvidenceRecord.from_mapping(item) for item in records]


def load_expected_symbols(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return tuple(payload.get("expected_sequence", []))
    return ()


def extract_candidates(
    records: Iterable[EvidenceRecord],
    config: ParserConfig | None = None,
) -> list[EvidenceRecord]:
    parser_config = config or ParserConfig()
    grouped: dict[int, list[EvidenceRecord]] = {}
    for record in records:
        if record.score < parser_config.min_score:
            continue
        grouped.setdefault(record.position, []).append(record)

    selected: list[EvidenceRecord] = []
    for position in sorted(grouped):
        best = sorted(grouped[position], key=lambda item: (-item.score, item.symbol))[0]
        selected.append(best)

    if parser_config.max_candidates is not None:
        selected = selected[: parser_config.max_candidates]
    return selected


def candidate_symbols(records: Iterable[EvidenceRecord]) -> tuple[str, ...]:
    return tuple(record.symbol for record in records)


def scan_candidate_windows(
    records: Iterable[EvidenceRecord],
    window_size: int,
    config: ParserConfig | None = None,
) -> list[list[EvidenceRecord]]:
    candidates = extract_candidates(records, config=config)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(candidates) <= window_size:
        return [candidates]
    return [candidates[index : index + window_size] for index in range(len(candidates) - window_size + 1)]


def _resolve_bucket_id(field_spec: FieldBucketSpec | None, carrier_value: str) -> tuple[int | None, str, str]:
    if field_spec is None:
        return None, "unknown_field", "field is not defined in the bucket layout"
    bucket_id = field_spec.lookup_bucket_id(carrier_value)
    if bucket_id is None:
        return None, "unresolved_carrier", "carrier not present in bucket specification"
    return bucket_id, "resolved", ""


def parse_structured_carrier_text(
    text: str,
    bucket_layout: BucketLayout,
    config: ParserConfig | None = None,
) -> list[ParsedCarrierBlock]:
    parser_config = config or ParserConfig()
    field_pattern = re.compile(r"^\s*(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.+?)\s*$")
    blocks: list[ParsedCarrierBlock] = []

    offset = 0
    block_index = 0
    for raw_line in text.split(parser_config.block_separator):
        line = raw_line.strip()
        line_start = offset
        offset += len(raw_line) + len(parser_config.block_separator)
        if not line:
            continue

        carriers: list[ParsedCarrier] = []
        local_cursor = 0
        for raw_segment in raw_line.split(parser_config.field_separator):
            segment = raw_segment.strip()
            segment_start = line_start + local_cursor
            local_cursor += len(raw_segment) + len(parser_config.field_separator)
            if not segment:
                continue

            match = field_pattern.match(segment)
            if match is None:
                carriers.append(
                    ParsedCarrier(
                        raw_text=segment,
                        field_name=None,
                        carrier_value=None,
                        bucket_id=None,
                        span=(segment_start, segment_start + len(segment)),
                        parse_status="malformed",
                        message="segment does not match FIELD=value format",
                    )
                )
                continue

            field_name = match.group("field")
            carrier_value = match.group("value")
            field_spec = bucket_layout.field_map.get(field_name)
            bucket_id, parse_status, message = _resolve_bucket_id(field_spec, carrier_value)
            carriers.append(
                ParsedCarrier(
                    raw_text=segment,
                    field_name=field_name,
                    carrier_value=carrier_value,
                    bucket_id=bucket_id,
                    span=(segment_start, segment_start + len(segment)),
                    parse_status=parse_status,
                    message=message,
                )
            )

        blocks.append(
            ParsedCarrierBlock(
                block_index=block_index,
                raw_text=raw_line.strip(),
                span=(line_start, line_start + len(raw_line)),
                carriers=tuple(carriers),
            )
        )
        block_index += 1

    return blocks


def parse_canonical_rendered_text(
    text: str,
    bucket_layout: BucketLayout,
    format_name: str = "canonical_v1",
) -> list[ParsedCarrierBlock]:
    render_config = render_config_from_name(format_name)
    return parse_structured_carrier_text(
        text=text,
        bucket_layout=bucket_layout,
        config=ParserConfig(
            block_separator=render_config.block_separator,
            field_separator=render_config.field_separator.rstrip(),
        ),
    )
