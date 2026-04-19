from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from src.core.bucket_mapping import BucketLayout
from src.core.canonical_contract import CanonicalEvidenceBundle
from src.core.catalog_freeze import load_required_frozen_catalog
from src.core.contextual_alignment import audit_contextual_field_values
from src.core.render import render_bucket_tuples, render_config_from_name


SCAFFOLDED_ARTIFACT_FORMAT = "scaffolded_slot_values"
FOUNDATION_ARTIFACT_FORMAT = "foundation_slot_values"
COMPILED_ARTIFACT_FORMAT = "compiled_slot_values"
DEFAULT_FIELDWISE_PROMPT_CONTRACT = "slot_request_v2"
FOUNDATION_FIELDWISE_PROMPT_CONTRACT = "foundation_v1"
COMPILED_FIELDWISE_PROMPT_CONTRACT = "compiled_slot_request_v1"


@dataclass(frozen=True)
class ScaffoldedCompletionTarget:
    prompt: str
    slot_field_names: tuple[str, ...]
    expected_slot_values: tuple[str, ...]
    artifact_format: str = SCAFFOLDED_ARTIFACT_FORMAT

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FieldwiseSlotTarget:
    slot_index: int
    block_index: int
    field_name: str
    prompt: str
    exact_slot_prefix: str
    allowed_values: tuple[str, ...]
    allowed_value_bucket_ids: dict[str, int]
    expected_value: str
    expected_bucket_id: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FieldwiseGenerationPlan:
    payload_text: str
    slot_targets: tuple[FieldwiseSlotTarget, ...]
    expected_slot_values: tuple[str, ...]
    fields_per_block: int
    prompt_contract_name: str = DEFAULT_FIELDWISE_PROMPT_CONTRACT
    artifact_format: str = SCAFFOLDED_ARTIFACT_FORMAT

    @property
    def slot_field_names(self) -> tuple[str, ...]:
        return tuple(target.field_name for target in self.slot_targets)

    @property
    def exact_slot_prefixes(self) -> dict[str, str]:
        prefixes: dict[str, str] = {}
        for target in self.slot_targets:
            prefixes.setdefault(target.field_name, target.exact_slot_prefix)
        return prefixes

    def to_dict(self) -> dict[str, object]:
        return {
            "payload_text": self.payload_text,
            "slot_targets": [target.to_dict() for target in self.slot_targets],
            "expected_slot_values": list(self.expected_slot_values),
            "fields_per_block": self.fields_per_block,
            "prompt_contract_name": self.prompt_contract_name,
            "artifact_format": self.artifact_format,
        }


@dataclass(frozen=True)
class ScaffoldedCompletionParseResult:
    artifact_format: str
    expected_slot_count: int
    parsed_slot_values: tuple[str, ...]
    valid_slot_values: tuple[str, ...]
    malformed_slot_values: tuple[str, ...]
    ignored_generated_lines: tuple[str, ...]
    reconstructed_text: str
    valid_canonical_block_count: int
    first_field_prefix_hit_rate: float
    field_order_exact_rate: float
    value_slot_exact_rate: float
    per_slot_exact_rate: float
    parse_success_rate: float
    per_field_accuracy: dict[str, float]
    slot_diagnostics: tuple[dict[str, object], ...]
    first_divergence_position: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FoundationSlotDiagnostic:
    slot_index: int
    slot_type: str
    exact_slot_prefix: str
    observed_value: str
    expected_value: str
    allowed_values: tuple[str, ...]
    allowed_token_ids: tuple[int, ...]
    chosen_token_id: int | None
    chosen_token_text: str | None
    is_field_valid: bool
    is_bucket_correct: bool
    is_slot_exact: bool
    observed_bucket_id: int | None
    expected_bucket_id: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FoundationGateResult:
    artifact_format: str
    prompt_contract_name: str
    expected_slot_count: int
    parsed_slot_values: tuple[str, ...]
    ignored_generated_lines: tuple[str, ...]
    field_valid_rate: float
    bucket_correct_rate: float
    slot_exact_rate: float
    per_field_accuracy: dict[str, float]
    valid_canonical_block_count: int
    contextual_audit_pass: bool
    foundation_gate_passed: bool
    rendered_canonical_text: str
    rendered_bucket_tuples: tuple[tuple[int, ...], ...]
    slot_diagnostics: tuple[FoundationSlotDiagnostic, ...]
    messages: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["slot_diagnostics"] = [item.to_dict() for item in self.slot_diagnostics]
        return payload


def _extract_slot_values_from_rendered_text(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for segment in line.split(";"):
            assignment = segment.strip()
            if not assignment or "=" not in assignment:
                continue
            _field_name, value = assignment.split("=", 1)
            values.append(value.strip())
    return tuple(values)


def build_scaffolded_completion_target(
    bundle: CanonicalEvidenceBundle,
    instruction: str,
) -> ScaffoldedCompletionTarget:
    render_config = render_config_from_name(bundle.contract.render_format)
    slot_field_names = bundle.contract.field_names * bundle.contract.block_count
    expected_slot_values = _extract_slot_values_from_rendered_text(bundle.rendered.text)
    if len(expected_slot_values) != len(slot_field_names):
        raise ValueError(
            "Rendered canonical evidence does not match expected slot layout: "
            f"slots={len(slot_field_names)}, values={len(expected_slot_values)}"
        )

    prompt_lines = [instruction.strip() or "Output one carrier value per line and nothing else.", "Slots:"]
    for index, field_name in enumerate(slot_field_names, start=1):
        prompt_lines.append(f"{index}. {field_name}")
    prompt_lines.append("Values:")
    prompt = render_config.block_separator.join(prompt_lines)
    return ScaffoldedCompletionTarget(
        prompt=prompt,
        slot_field_names=slot_field_names,
        expected_slot_values=expected_slot_values,
    )


def _ordered_field_values(layout: BucketLayout, field_name: str) -> tuple[str, ...]:
    field_spec = layout.get_field_spec(field_name)
    ordered: list[str] = []
    for bucket_id in field_spec.bucket_ids:
        ordered.extend(field_spec.bucket_members(bucket_id))
    return tuple(ordered)


def _slot_field_names_from_metadata(
    *,
    layout: BucketLayout,
    expected_slot_values: Sequence[str],
    slot_field_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if slot_field_names:
        return tuple(str(item) for item in slot_field_names)
    expected_slot_count = len(expected_slot_values)
    repeated = layout.field_names * max(1, (expected_slot_count + len(layout.field_names) - 1) // len(layout.field_names))
    return tuple(repeated[:expected_slot_count])


def _build_fieldwise_slot_prompt(
    *,
    field_name: str,
    allowed_values: Sequence[str],
    prompt_contract_name: str,
    instruction: str = "",
    payload_text: str = "",
    slot_index: int = 0,
    total_slots: int = 0,
    block_index: int = 0,
) -> str:
    if prompt_contract_name == FOUNDATION_FIELDWISE_PROMPT_CONTRACT:
        return f"{field_name}="
    prompt_lines = [
        instruction.strip() or "Output exactly one allowed carrier value for the requested slot.",
        f"Payload: {payload_text}",
        f"Slot index: {slot_index + 1}/{total_slots}",
        f"Block index: {block_index + 1}",
        f"Field: {field_name}",
        f"Allowed carriers: {', '.join(allowed_values)}",
        "Value: ",
    ]
    return "\n".join(prompt_lines)


def build_fieldwise_generation_plan(
    bundle: CanonicalEvidenceBundle,
    instruction: str,
    *,
    prompt_contract_name: str = DEFAULT_FIELDWISE_PROMPT_CONTRACT,
    max_blocks: int | None = None,
) -> FieldwiseGenerationPlan:
    layout = load_required_frozen_catalog(Path(bundle.contract.catalog_path))
    effective_block_count = bundle.contract.block_count
    if max_blocks is not None:
        effective_block_count = max(1, min(int(max_blocks), bundle.contract.block_count))
    slot_field_names = bundle.contract.field_names * effective_block_count
    expected_slot_values = _extract_slot_values_from_rendered_text(bundle.rendered.text)
    expected_slot_values = expected_slot_values[: len(slot_field_names)]
    if len(expected_slot_values) != len(slot_field_names):
        raise ValueError(
            "Rendered canonical evidence does not match expected slot layout: "
            f"slots={len(slot_field_names)}, values={len(expected_slot_values)}"
        )

    slot_targets: list[FieldwiseSlotTarget] = []
    fields_per_block = len(bundle.contract.field_names)
    for slot_index, (field_name, expected_value) in enumerate(
        zip(slot_field_names, expected_slot_values, strict=True)
    ):
        field_spec = layout.get_field_spec(field_name)
        expected_bucket_id = field_spec.lookup_bucket_id(expected_value)
        if expected_bucket_id is None:
            raise ValueError(
                f"Expected value {expected_value!r} is not valid for field {field_name!r}"
            )
        allowed_values = _ordered_field_values(layout, field_name)
        prompt_text = _build_fieldwise_slot_prompt(
            field_name=field_name,
            allowed_values=allowed_values,
            prompt_contract_name=prompt_contract_name,
            instruction=instruction,
            payload_text=bundle.contract.payload_text,
            slot_index=slot_index,
            total_slots=len(slot_field_names),
            block_index=slot_index // fields_per_block,
        )
        slot_targets.append(
            FieldwiseSlotTarget(
                slot_index=slot_index,
                block_index=slot_index // fields_per_block,
                field_name=field_name,
                prompt=prompt_text,
                exact_slot_prefix=prompt_text,
                allowed_values=allowed_values,
                allowed_value_bucket_ids={
                    value: field_spec.lookup_bucket_id(value)
                    for value in allowed_values
                    if field_spec.lookup_bucket_id(value) is not None
                },
                expected_value=expected_value,
                expected_bucket_id=expected_bucket_id,
            )
        )
    return FieldwiseGenerationPlan(
        payload_text=bundle.contract.payload_text,
        slot_targets=tuple(slot_targets),
        expected_slot_values=expected_slot_values,
        fields_per_block=fields_per_block,
        prompt_contract_name=prompt_contract_name,
        artifact_format=(
            FOUNDATION_ARTIFACT_FORMAT
            if prompt_contract_name == FOUNDATION_FIELDWISE_PROMPT_CONTRACT
            else SCAFFOLDED_ARTIFACT_FORMAT
        ),
    )


def render_foundation_slot_values(
    *,
    slot_values: Sequence[str],
    layout: BucketLayout,
    slot_field_names: Sequence[str],
    render_format: str = "canonical_v1",
) -> tuple[str, tuple[tuple[int, ...], ...]]:
    fields_per_block = len(layout.field_names)
    bucket_tuples: list[tuple[int, ...]] = []
    member_indices_per_block: list[dict[str, int]] = []
    for start in range(0, len(slot_field_names), fields_per_block):
        block_field_names = tuple(slot_field_names[start : start + fields_per_block])
        block_values = tuple(slot_values[start : start + fields_per_block])
        if len(block_field_names) != fields_per_block or len(block_values) != fields_per_block:
            break
        if block_field_names != layout.field_names:
            continue
        bucket_ids: list[int] = []
        member_indices: dict[str, int] = {}
        block_valid = True
        for field_name, value in zip(block_field_names, block_values, strict=True):
            field_spec = layout.get_field_spec(field_name)
            bucket_id = field_spec.lookup_bucket_id(value)
            if bucket_id is None:
                block_valid = False
                break
            members = field_spec.bucket_members(bucket_id)
            try:
                member_indices[field_name] = members.index(value)
            except ValueError:
                block_valid = False
                break
            bucket_ids.append(bucket_id)
        if block_valid:
            bucket_tuples.append(tuple(bucket_ids))
            member_indices_per_block.append(member_indices)

    if not bucket_tuples:
        return "", ()

    rendered = render_bucket_tuples(
        layout,
        bucket_tuples,
        config=render_config_from_name(render_format),
        member_indices_per_block=member_indices_per_block,
    )
    return rendered.text, rendered.bucket_tuples


def evaluate_foundation_completion(
    raw_text: str,
    *,
    layout: BucketLayout,
    expected_slot_values: Sequence[str],
    exact_slot_prefixes: dict[str, str],
    tokenizer: object,
    prompt_contract_name: str,
    render_format: str = "canonical_v1",
    slot_field_names: Sequence[str] | None = None,
    artifact_format: str = FOUNDATION_ARTIFACT_FORMAT,
) -> FoundationGateResult:
    normalized_slot_field_names = _slot_field_names_from_metadata(
        layout=layout,
        expected_slot_values=expected_slot_values,
        slot_field_names=slot_field_names,
    )
    parsed_slot_values = tuple(line.strip() for line in raw_text.splitlines() if line.strip())
    expected_slot_count = len(expected_slot_values)
    used_values = tuple(parsed_slot_values[:expected_slot_count])
    ignored_generated_lines = tuple(parsed_slot_values[expected_slot_count:])

    field_allowed_values = {
        field_name: _ordered_field_values(layout, field_name)
        for field_name in dict.fromkeys(normalized_slot_field_names)
    }
    contextual_audit = audit_contextual_field_values(
        field_allowed_values=field_allowed_values,
        exact_slot_prefixes=exact_slot_prefixes,
        tokenizer=tokenizer,
        prompt_contract_name=prompt_contract_name,
    )

    exact_matches = 0
    field_valid_matches = 0
    bucket_correct_matches = 0
    field_totals: dict[str, int] = {}
    field_exact_matches: dict[str, int] = {}
    slot_diagnostics: list[FoundationSlotDiagnostic] = []
    for slot_index, field_name in enumerate(normalized_slot_field_names):
        observed_value = used_values[slot_index] if slot_index < len(used_values) else ""
        expected_value = str(expected_slot_values[slot_index]) if slot_index < len(expected_slot_values) else ""
        exact_slot_prefix = exact_slot_prefixes[field_name]
        allowed_values = field_allowed_values[field_name]
        field_token_map = contextual_audit.valid_token_map.get(field_name, {}).get(exact_slot_prefix, {})
        chosen_token_id = field_token_map.get(observed_value)
        chosen_token_text = tokenizer.decode([chosen_token_id]) if chosen_token_id is not None else None
        field_spec = layout.get_field_spec(field_name)
        observed_bucket_id = field_spec.lookup_bucket_id(observed_value) if observed_value else None
        expected_bucket_id = field_spec.lookup_bucket_id(expected_value) if expected_value else None
        is_field_valid = chosen_token_id is not None
        is_bucket_correct = is_field_valid and observed_bucket_id == expected_bucket_id
        is_slot_exact = is_field_valid and observed_value == expected_value
        if is_field_valid:
            field_valid_matches += 1
        if is_bucket_correct:
            bucket_correct_matches += 1
        if is_slot_exact:
            exact_matches += 1
        field_totals[field_name] = field_totals.get(field_name, 0) + 1
        if is_slot_exact:
            field_exact_matches[field_name] = field_exact_matches.get(field_name, 0) + 1
        slot_diagnostics.append(
            FoundationSlotDiagnostic(
                slot_index=slot_index,
                slot_type=field_name,
                exact_slot_prefix=exact_slot_prefix,
                observed_value=observed_value,
                expected_value=expected_value,
                allowed_values=allowed_values,
                allowed_token_ids=tuple(field_token_map[value] for value in allowed_values if value in field_token_map),
                chosen_token_id=chosen_token_id,
                chosen_token_text=chosen_token_text,
                is_field_valid=is_field_valid,
                is_bucket_correct=is_bucket_correct,
                is_slot_exact=is_slot_exact,
                observed_bucket_id=observed_bucket_id,
                expected_bucket_id=expected_bucket_id,
            )
        )

    field_valid_rate = field_valid_matches / expected_slot_count if expected_slot_count else 0.0
    bucket_correct_rate = bucket_correct_matches / expected_slot_count if expected_slot_count else 0.0
    slot_exact_rate = exact_matches / expected_slot_count if expected_slot_count else 0.0
    per_field_accuracy = {
        field_name: field_exact_matches.get(field_name, 0) / total
        for field_name, total in field_totals.items()
        if total
    }
    rendered_canonical_text, rendered_bucket_tuples = render_foundation_slot_values(
        slot_values=used_values,
        layout=layout,
        slot_field_names=normalized_slot_field_names,
        render_format=render_format,
    )
    fields_per_block = len(layout.field_names)
    expected_block_count = expected_slot_count // fields_per_block if fields_per_block else 0
    valid_canonical_block_count = len(rendered_bucket_tuples)

    foundation_gate_passed = (
        contextual_audit.is_context_safe
        and field_valid_rate == 1.0
        and bucket_correct_rate >= 0.95
        and slot_exact_rate >= 0.95
        and valid_canonical_block_count == expected_block_count
    )
    messages: list[str] = []
    if not contextual_audit.is_context_safe:
        messages.append("contextual carrier audit failed")
    if field_valid_rate < 1.0:
        messages.append("field_valid_rate below 1.0")
    if bucket_correct_rate < 0.95:
        messages.append("bucket_correct_rate below 0.95")
    if slot_exact_rate < 0.95:
        messages.append("slot_exact_rate below 0.95")
    if valid_canonical_block_count != expected_block_count:
        messages.append("deterministic canonical render did not cover every expected block")

    return FoundationGateResult(
        artifact_format=artifact_format,
        prompt_contract_name=prompt_contract_name,
        expected_slot_count=expected_slot_count,
        parsed_slot_values=used_values,
        ignored_generated_lines=ignored_generated_lines,
        field_valid_rate=field_valid_rate,
        bucket_correct_rate=bucket_correct_rate,
        slot_exact_rate=slot_exact_rate,
        per_field_accuracy=per_field_accuracy,
        valid_canonical_block_count=valid_canonical_block_count,
        contextual_audit_pass=contextual_audit.is_context_safe,
        foundation_gate_passed=foundation_gate_passed,
        rendered_canonical_text=rendered_canonical_text,
        rendered_bucket_tuples=rendered_bucket_tuples,
        slot_diagnostics=tuple(slot_diagnostics),
        messages=tuple(messages),
    )


def parse_scaffolded_completion(
    raw_text: str,
    *,
    layout: BucketLayout,
    slot_field_names: Sequence[str],
    expected_slot_values: Sequence[str],
) -> ScaffoldedCompletionParseResult:
    parsed_slot_values = tuple(line.strip() for line in raw_text.splitlines() if line.strip())
    expected_slot_count = len(slot_field_names)
    used_values = parsed_slot_values[:expected_slot_count]
    ignored_generated_lines = parsed_slot_values[expected_slot_count:]

    valid_slot_values: list[str] = []
    malformed_slot_values: list[str] = []
    exact_matches = 0
    first_divergence_position: int | None = None
    slot_diagnostics: list[dict[str, object]] = []
    field_totals: dict[str, int] = {}
    field_exact_matches: dict[str, int] = {}

    for index, field_name in enumerate(slot_field_names):
        observed_value = used_values[index] if index < len(used_values) else ""
        expected_value = expected_slot_values[index] if index < len(expected_slot_values) else ""
        field_spec = layout.get_field_spec(field_name)
        observed_bucket_id = field_spec.lookup_bucket_id(observed_value) if observed_value else None
        expected_bucket_id = field_spec.lookup_bucket_id(expected_value) if expected_value else None
        field_valid = observed_bucket_id is not None
        bucket_correct = field_valid and observed_bucket_id == expected_bucket_id
        if field_valid:
            valid_slot_values.append(observed_value)
        else:
            malformed_slot_values.append(observed_value)

        if observed_value == expected_value:
            exact_matches += 1
        elif first_divergence_position is None:
            first_divergence_position = index
        field_totals[field_name] = field_totals.get(field_name, 0) + 1
        if observed_value == expected_value:
            field_exact_matches[field_name] = field_exact_matches.get(field_name, 0) + 1
        slot_diagnostics.append(
            {
                "slot_index": index,
                "slot_type": field_name,
                "observed_value": observed_value,
                "expected_value": expected_value,
                "field_valid": field_valid,
                "bucket_correct": bucket_correct,
                "observed_bucket_id": observed_bucket_id,
                "expected_bucket_id": expected_bucket_id,
            }
        )

    fields_per_block = len(layout.field_names)
    valid_blocks: list[str] = []
    for start in range(0, expected_slot_count, fields_per_block):
        block_values = used_values[start : start + fields_per_block]
        if len(block_values) != fields_per_block:
            break
        block_field_names = slot_field_names[start : start + fields_per_block]
        assignments: list[str] = []
        block_valid = True
        for field_name, value in zip(block_field_names, block_values):
            if not value or layout.get_field_spec(field_name).lookup_bucket_id(value) is None:
                block_valid = False
                break
            assignments.append(f"{field_name}={value}")
        if block_valid:
            valid_blocks.append("; ".join(assignments))

    reconstructed_text = "\n".join(valid_blocks)
    block_count = max(1, expected_slot_count // fields_per_block)
    valid_canonical_block_count = len(valid_blocks)
    first_field_prefix_hit_rate = (
        1.0 if reconstructed_text.startswith(f"{layout.field_names[0]}=") else 0.0
    )
    field_order_exact_rate = valid_canonical_block_count / block_count
    value_slot_exact_rate = exact_matches / expected_slot_count if expected_slot_count else 0.0
    per_slot_exact_rate = value_slot_exact_rate
    parse_success_rate = len(valid_slot_values) / expected_slot_count if expected_slot_count else 0.0
    per_field_accuracy = {
        field_name: field_exact_matches.get(field_name, 0) / total
        for field_name, total in field_totals.items()
        if total
    }

    return ScaffoldedCompletionParseResult(
        artifact_format=SCAFFOLDED_ARTIFACT_FORMAT,
        expected_slot_count=expected_slot_count,
        parsed_slot_values=used_values,
        valid_slot_values=tuple(valid_slot_values),
        malformed_slot_values=tuple(malformed_slot_values),
        ignored_generated_lines=tuple(ignored_generated_lines),
        reconstructed_text=reconstructed_text,
        valid_canonical_block_count=valid_canonical_block_count,
        first_field_prefix_hit_rate=first_field_prefix_hit_rate,
        field_order_exact_rate=field_order_exact_rate,
        value_slot_exact_rate=value_slot_exact_rate,
        per_slot_exact_rate=per_slot_exact_rate,
        parse_success_rate=parse_success_rate,
        per_field_accuracy=per_field_accuracy,
        slot_diagnostics=tuple(slot_diagnostics),
        first_divergence_position=first_divergence_position,
    )
