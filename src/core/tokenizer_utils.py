from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml

from src.core.bucket_mapping import BucketLayout, FieldBucketSpec, load_bucket_layout


class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        ...


@dataclass(frozen=True)
class CarrierDiagnostic:
    carrier: str
    normalized_carrier: str
    token_ids: tuple[int, ...]
    is_single_token: bool
    is_invalid: bool
    duplicate_normalized_form: bool
    token_collision: bool
    detokenized_text: str | None
    detokenization_matches: bool | None
    reasons: tuple[str, ...]
    colliding_carriers: tuple[str, ...] = ()
    bucket_locations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CarrierAuditResult:
    num_total: int
    num_single_token: int
    num_multi_token: int
    num_invalid: int
    num_duplicates: int
    diagnostics: tuple[CarrierDiagnostic, ...]
    num_token_collisions: int = 0
    field_summaries: dict[str, dict[str, object]] = field(default_factory=dict)
    rejected_carriers: tuple[CarrierDiagnostic, ...] = ()

    @property
    def is_alignment_safe(self) -> bool:
        return (
            self.num_total > 0
            and self.num_multi_token == 0
            and self.num_invalid == 0
            and self.num_duplicates == 0
            and self.num_token_collisions == 0
            and not self.rejected_carriers
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "num_total": self.num_total,
            "num_single_token": self.num_single_token,
            "num_multi_token": self.num_multi_token,
            "num_invalid": self.num_invalid,
            "num_duplicates": self.num_duplicates,
            "num_token_collisions": self.num_token_collisions,
            "is_alignment_safe": self.is_alignment_safe,
            "field_summaries": self.field_summaries,
            "rejected_carriers": [diagnostic.to_dict() for diagnostic in self.rejected_carriers],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path


@dataclass(frozen=True)
class AlignmentReport:
    required_tokens: tuple[str, ...]
    present_tokens: tuple[str, ...]
    missing_tokens: tuple[str, ...]
    vocabulary_size: int
    is_aligned: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MockTokenizer:
    """Lightweight tokenizer for tests and smoke audits."""

    def __init__(
        self,
        single_token_map: dict[str, int] | None = None,
        multi_token_map: dict[str, Sequence[int]] | None = None,
    ) -> None:
        self.single_token_map = dict(single_token_map or {})
        self.multi_token_map = {
            text: tuple(int(token_id) for token_id in token_ids)
            for text, token_ids in (multi_token_map or {}).items()
        }
        self.id_to_text = {token_id: text for text, token_id in self.single_token_map.items()}
        self.text_to_token_id = dict(self.single_token_map)
        self.next_token_id = max(self.id_to_text.keys(), default=1_000) + 1

    def encode(self, text: str) -> list[int]:
        if text in self.multi_token_map:
            return list(self.multi_token_map[text])
        if text in self.single_token_map:
            return [self.single_token_map[text]]
        parts = [part for part in text.split(" ") if part]
        if not parts:
            return []
        token_ids: list[int] = []
        for part in parts:
            if part not in self.text_to_token_id:
                token_id = self.next_token_id
                self.next_token_id += 1
                self.text_to_token_id[part] = token_id
                self.id_to_text[token_id] = part
            token_ids.append(self.text_to_token_id[part])
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        if len(token_ids) == 1 and token_ids[0] in self.id_to_text:
            return self.id_to_text[token_ids[0]]
        decoded = [self.id_to_text.get(token_id, f"<tok:{token_id}>") for token_id in token_ids]
        return " ".join(decoded)


class HuggingFaceTokenizerAdapter:
    """Thin adapter over a transformers tokenizer instance."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(
            self.tokenizer.decode(
                list(token_ids),
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
        )


def load_tokenizer(
    backend: str,
    tokenizer_name_or_path: str = "",
) -> TokenizerProtocol:
    normalized_backend = backend.strip().lower() or "mock"
    if normalized_backend == "mock":
        return MockTokenizer()
    if normalized_backend in {"huggingface", "hf"}:
        if not tokenizer_name_or_path.strip():
            raise ValueError("tokenizer_name_or_path is required for tokenizer_backend=huggingface")
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "transformers is required for tokenizer_backend=huggingface"
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
        return HuggingFaceTokenizerAdapter(tokenizer)
    raise ValueError(f"Unsupported tokenizer backend: {backend}")


def normalize_token(token: str) -> str:
    return token.strip()


def normalize_carrier(carrier: str) -> str:
    return re.sub(r"\s+", " ", carrier.strip())


def audit_token_alignment(vocabulary: Sequence[str], required_tokens: Sequence[str]) -> AlignmentReport:
    normalized_vocabulary = {normalize_token(token) for token in vocabulary}
    normalized_required = tuple(normalize_token(token) for token in required_tokens)
    present = tuple(token for token in normalized_required if token in normalized_vocabulary)
    missing = tuple(token for token in normalized_required if token not in normalized_vocabulary)
    return AlignmentReport(
        required_tokens=normalized_required,
        present_tokens=present,
        missing_tokens=missing,
        vocabulary_size=len(normalized_vocabulary),
        is_aligned=not missing,
    )


def require_alignment(vocabulary: Sequence[str], required_tokens: Sequence[str]) -> AlignmentReport:
    report = audit_token_alignment(vocabulary, required_tokens)
    if not report.is_aligned:
        raise ValueError(f"Tokenizer missing required tokens: {report.missing_tokens}")
    return report


def _bucket_locations(bucket_layout: BucketLayout | None, carrier: str) -> tuple[str, ...]:
    if bucket_layout is None:
        return ()
    locations: list[str] = []
    for field_spec in bucket_layout.fields:
        bucket_id = field_spec.lookup_bucket_id(carrier)
        if bucket_id is not None:
            locations.append(f"{field_spec.field_name}:{bucket_id}")
    return tuple(locations)


def _field_names_for_carrier(bucket_layout: BucketLayout | None, carrier: str) -> tuple[str, ...]:
    if bucket_layout is None:
        return ()
    field_names: list[str] = []
    for field_spec in bucket_layout.fields:
        if field_spec.lookup_bucket_id(carrier) is not None or carrier in field_spec.disallowed_carriers:
            field_names.append(field_spec.field_name)
    return tuple(field_names)


def _is_disallowed(field_spec: FieldBucketSpec | None, carrier: str) -> bool:
    return field_spec is not None and carrier in field_spec.disallowed_carriers


def audit_carriers(
    carriers: Sequence[str],
    tokenizer: TokenizerProtocol,
    bucket_layout: BucketLayout | None = None,
) -> CarrierAuditResult:
    normalized_groups: dict[str, list[str]] = {}
    token_groups: dict[int, list[str]] = {}
    for carrier in carriers:
        normalized_groups.setdefault(normalize_carrier(carrier), []).append(carrier)
        encoded = tokenizer.encode(carrier)
        if len(encoded) == 1:
            token_groups.setdefault(encoded[0], []).append(carrier)

    diagnostics: list[CarrierDiagnostic] = []
    for carrier in carriers:
        normalized = normalize_carrier(carrier)
        token_ids = tuple(tokenizer.encode(carrier))
        detokenized_text: str | None = None
        detokenization_matches: bool | None = None
        reasons: list[str] = []
        field_names = _field_names_for_carrier(bucket_layout, carrier)

        if carrier == "":
            reasons.append("empty_string")
        if carrier.isspace():
            reasons.append("whitespace_only")
        if carrier != carrier.strip():
            reasons.append("leading_or_trailing_whitespace")
        if any(character in carrier for character in ("\n", "\r", "\t")):
            reasons.append("unstable_control_whitespace")
        if bucket_layout is not None and any(
            _is_disallowed(bucket_layout.get_field_spec(field_name), carrier) for field_name in field_names
        ):
            reasons.append("disallowed_carrier")

        if hasattr(tokenizer, "decode"):
            detokenized_text = tokenizer.decode(token_ids)
            detokenization_matches = detokenized_text == carrier
            if detokenization_matches is False:
                reasons.append("detokenization_mismatch")

        if len(token_ids) == 0:
            reasons.append("tokenizer_returned_no_tokens")
        elif len(token_ids) > 1:
            reasons.append("multi_token")

        duplicate_normalized = len(normalized_groups[normalized]) > 1
        if duplicate_normalized:
            reasons.append("duplicate_normalized_form")

        colliding_carriers: tuple[str, ...] = ()
        token_collision = False
        if len(token_ids) == 1:
            colliding = token_groups[token_ids[0]]
            distinct_normalized = {normalize_carrier(item) for item in colliding}
            token_collision = len(distinct_normalized) > 1
            if token_collision:
                colliding_carriers = tuple(item for item in colliding if item != carrier)
                reasons.append("token_collision")

        diagnostics.append(
            CarrierDiagnostic(
                carrier=carrier,
                normalized_carrier=normalized,
                token_ids=token_ids,
                is_single_token=len(token_ids) == 1,
                is_invalid=bool(reasons and any(reason not in {"multi_token", "duplicate_normalized_form", "token_collision"} for reason in reasons)),
                duplicate_normalized_form=duplicate_normalized,
                token_collision=token_collision,
                detokenized_text=detokenized_text,
                detokenization_matches=detokenization_matches,
                reasons=tuple(reasons),
                colliding_carriers=colliding_carriers,
                bucket_locations=_bucket_locations(bucket_layout, carrier),
            )
        )

    return CarrierAuditResult(
        num_total=len(diagnostics),
        num_single_token=sum(1 for item in diagnostics if item.is_single_token),
        num_multi_token=sum(1 for item in diagnostics if not item.is_single_token and item.token_ids),
        num_invalid=sum(1 for item in diagnostics if item.is_invalid),
        num_duplicates=sum(1 for item in diagnostics if item.duplicate_normalized_form),
        num_token_collisions=sum(1 for item in diagnostics if item.token_collision),
        field_summaries={
            field_spec.field_name: {
                "field_type": field_spec.field_type,
                "bucket_count": field_spec.bucket_count,
                "num_total": len(relevant := [
                    item
                    for item in diagnostics
                    if field_spec.lookup_bucket_id(item.carrier) is not None
                    or item.carrier in field_spec.disallowed_carriers
                ]),
                "num_single_token": sum(1 for item in relevant if item.is_single_token),
                "num_multi_token": sum(
                    1 for item in relevant if not item.is_single_token and item.token_ids
                ),
                "num_invalid": sum(1 for item in relevant if item.is_invalid),
                "num_duplicates": sum(1 for item in relevant if item.duplicate_normalized_form),
                "num_token_collisions": sum(1 for item in relevant if item.token_collision),
                "rejected_count": sum(1 for item in relevant if item.reasons),
                "passed": all(not item.reasons for item in relevant),
            }
            for field_spec in (bucket_layout.fields if bucket_layout is not None else ())
        },
        rejected_carriers=tuple(item for item in diagnostics if item.reasons),
        diagnostics=tuple(diagnostics),
    )


def load_carrier_candidates(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    elif suffix == ".json":
        payload = json.loads(text)
    else:
        return [line.rstrip("\n") for line in text.splitlines() if line.strip()]

    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        if "carriers" in payload and isinstance(payload["carriers"], list):
            return [str(item) for item in payload["carriers"]]
        if "fields" in payload:
            return list(BucketLayout.from_dict(payload).all_carriers())
    raise ValueError(f"Unsupported carrier file format: {path}")


def load_carriers_and_layout(
    carrier_path: Path | None = None,
    bucket_spec_path: Path | None = None,
    include_disallowed: bool = True,
) -> tuple[list[str], BucketLayout | None]:
    bucket_layout = load_bucket_layout(bucket_spec_path) if bucket_spec_path is not None else None
    carriers: list[str] = []
    if carrier_path is not None:
        carriers.extend(load_carrier_candidates(carrier_path))
    elif bucket_layout is not None:
        carriers.extend(bucket_layout.all_carriers())

    if include_disallowed and bucket_layout is not None:
        for field_spec in bucket_layout.fields:
            carriers.extend(field_spec.disallowed_carriers)

    if not carriers and bucket_layout is None:
        raise ValueError("Provide a carrier file or a bucket spec")
    return carriers, bucket_layout
