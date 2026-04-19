from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from src.core.bucket_mapping import BucketLayout
from src.core.canonical_contract import CanonicalEvidenceBundle
from src.core.render import render_config_from_name


SCAFFOLDED_ARTIFACT_FORMAT = "scaffolded_slot_values"


@dataclass(frozen=True)
class ScaffoldedCompletionTarget:
    prompt: str
    slot_field_names: tuple[str, ...]
    expected_slot_values: tuple[str, ...]
    artifact_format: str = SCAFFOLDED_ARTIFACT_FORMAT

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
    parse_success_rate: float
    first_divergence_position: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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

    for index, field_name in enumerate(slot_field_names):
        observed_value = used_values[index] if index < len(used_values) else ""
        expected_value = expected_slot_values[index] if index < len(expected_slot_values) else ""
        field_spec = layout.get_field_spec(field_name)
        if observed_value and field_spec.lookup_bucket_id(observed_value) is not None:
            valid_slot_values.append(observed_value)
        else:
            malformed_slot_values.append(observed_value)

        if observed_value == expected_value:
            exact_matches += 1
        elif first_divergence_position is None:
            first_divergence_position = index

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
    parse_success_rate = len(valid_slot_values) / expected_slot_count if expected_slot_count else 0.0

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
        parse_success_rate=parse_success_rate,
        first_divergence_position=first_divergence_position,
    )
