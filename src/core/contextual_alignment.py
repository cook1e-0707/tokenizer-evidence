from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from src.core.bucket_mapping import BucketLayout
from src.core.tokenizer_utils import TokenizerProtocol


@dataclass(frozen=True)
class ContextualSlotTarget:
    field_name: str
    exact_slot_prefix: str
    allowed_values: tuple[str, ...]


@dataclass(frozen=True)
class ContextualCarrierDiagnostic:
    field_name: str
    exact_slot_prefix: str
    carrier: str
    matching_token_ids: tuple[int, ...]
    matching_token_texts: tuple[str, ...]
    is_valid_next_token: bool
    is_ambiguous: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ContextualCarrierAuditResult:
    prompt_contract_name: str
    field_summaries: dict[str, dict[str, object]]
    valid_token_map: dict[str, dict[str, dict[str, int]]]
    diagnostics: tuple[ContextualCarrierDiagnostic, ...]

    @property
    def is_context_safe(self) -> bool:
        return all(item.is_valid_next_token for item in self.diagnostics)

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_contract_name": self.prompt_contract_name,
            "field_summaries": self.field_summaries,
            "valid_token_map": self.valid_token_map,
            "is_context_safe": self.is_context_safe,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def _iter_token_ids(tokenizer: TokenizerProtocol) -> Iterable[int]:
    raw_tokenizer = getattr(tokenizer, "tokenizer", None)
    if raw_tokenizer is not None:
        vocab_size = getattr(raw_tokenizer, "vocab_size", None)
        if vocab_size is not None:
            return range(int(vocab_size))
        get_vocab = getattr(raw_tokenizer, "get_vocab", None)
        if callable(get_vocab):
            return sorted(int(token_id) for token_id in get_vocab().values())

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return range(int(vocab_size))

    id_to_text = getattr(tokenizer, "id_to_text", None)
    if isinstance(id_to_text, Mapping):
        return sorted(int(token_id) for token_id in id_to_text)

    raise ValueError(
        "Tokenizer does not expose a usable token-id iterator for contextual carrier audit"
    )


def _build_suffix_index(
    *,
    tokenizer: TokenizerProtocol,
    exact_slot_prefix: str,
    token_ids: Sequence[int],
) -> dict[str, list[int]]:
    prompt_token_ids = list(tokenizer.encode(exact_slot_prefix))
    suffix_to_token_ids: dict[str, list[int]] = {}
    for token_id in token_ids:
        decoded = tokenizer.decode([*prompt_token_ids, int(token_id)])
        if not decoded.startswith(exact_slot_prefix):
            continue
        suffix = decoded[len(exact_slot_prefix) :]
        if not suffix:
            continue
        suffix_to_token_ids.setdefault(suffix, []).append(int(token_id))
    return suffix_to_token_ids


def audit_contextual_field_values(
    *,
    field_allowed_values: Mapping[str, Sequence[str]],
    exact_slot_prefixes: Mapping[str, str],
    tokenizer: TokenizerProtocol,
    prompt_contract_name: str,
) -> ContextualCarrierAuditResult:
    slot_targets = [
        ContextualSlotTarget(
            field_name=field_name,
            exact_slot_prefix=exact_slot_prefixes[field_name],
            allowed_values=tuple(allowed_values),
        )
        for field_name, allowed_values in field_allowed_values.items()
    ]
    return audit_contextual_slot_targets(
        slot_targets=slot_targets,
        tokenizer=tokenizer,
        prompt_contract_name=prompt_contract_name,
    )


def audit_contextual_slot_targets(
    *,
    slot_targets: Sequence[ContextualSlotTarget],
    tokenizer: TokenizerProtocol,
    prompt_contract_name: str,
) -> ContextualCarrierAuditResult:
    token_ids = tuple(_iter_token_ids(tokenizer))
    prefix_indices: dict[str, dict[str, list[int]]] = {}
    for slot_target in slot_targets:
        exact_slot_prefix = slot_target.exact_slot_prefix
        if exact_slot_prefix in prefix_indices:
            continue
        prefix_indices[exact_slot_prefix] = _build_suffix_index(
            tokenizer=tokenizer,
            exact_slot_prefix=exact_slot_prefix,
            token_ids=token_ids,
        )

    diagnostics: list[ContextualCarrierDiagnostic] = []
    valid_token_map: dict[str, dict[str, dict[str, int]]] = {}
    field_summaries: dict[str, dict[str, object]] = {}
    for slot_target in slot_targets:
        field_name = slot_target.field_name
        allowed_values = slot_target.allowed_values
        exact_slot_prefix = slot_target.exact_slot_prefix
        suffix_index = prefix_indices[exact_slot_prefix]
        field_valid: dict[str, int] = {}
        field_diagnostics: list[ContextualCarrierDiagnostic] = []
        for carrier in allowed_values:
            matching_token_ids = tuple(int(token_id) for token_id in suffix_index.get(carrier, []))
            matching_token_texts = tuple(tokenizer.decode([token_id]) for token_id in matching_token_ids)
            reasons: list[str] = []
            if not matching_token_ids:
                reasons.append("not_single_next_token_in_context")
            elif len(matching_token_ids) > 1:
                reasons.append("ambiguous_contextual_token")
            diagnostic = ContextualCarrierDiagnostic(
                field_name=field_name,
                exact_slot_prefix=exact_slot_prefix,
                carrier=carrier,
                matching_token_ids=matching_token_ids,
                matching_token_texts=matching_token_texts,
                is_valid_next_token=len(matching_token_ids) == 1,
                is_ambiguous=len(matching_token_ids) > 1,
                reasons=tuple(reasons),
            )
            diagnostics.append(diagnostic)
            field_diagnostics.append(diagnostic)
            if diagnostic.is_valid_next_token:
                field_valid[carrier] = matching_token_ids[0]
        valid_token_map.setdefault(field_name, {})[exact_slot_prefix] = field_valid
        field_summary_key = f"{field_name}@{exact_slot_prefix}"
        field_summaries[field_summary_key] = {
            "field_name": field_name,
            "exact_slot_prefix": exact_slot_prefix,
            "num_total": len(field_diagnostics),
            "num_valid": sum(1 for item in field_diagnostics if item.is_valid_next_token),
            "num_invalid": sum(1 for item in field_diagnostics if not item.is_valid_next_token),
            "num_ambiguous": sum(1 for item in field_diagnostics if item.is_ambiguous),
            "passed": all(item.is_valid_next_token for item in field_diagnostics),
        }
    return ContextualCarrierAuditResult(
        prompt_contract_name=prompt_contract_name,
        field_summaries=field_summaries,
        valid_token_map=valid_token_map,
        diagnostics=tuple(diagnostics),
    )


def audit_contextual_carriers(
    *,
    bucket_layout: BucketLayout,
    exact_slot_prefixes: Mapping[str, str],
    tokenizer: TokenizerProtocol,
    prompt_contract_name: str,
) -> ContextualCarrierAuditResult:
    field_allowed_values = {
        field_spec.field_name: tuple(
            carrier
            for bucket_id in field_spec.bucket_ids
            for carrier in field_spec.bucket_members(bucket_id)
        )
        for field_spec in bucket_layout.fields
    }
    return audit_contextual_field_values(
        field_allowed_values=field_allowed_values,
        exact_slot_prefixes=exact_slot_prefixes,
        tokenizer=tokenizer,
        prompt_contract_name=prompt_contract_name,
    )
