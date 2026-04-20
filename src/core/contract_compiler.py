from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

from src.core.bucket_mapping import BucketLayout
from src.core.catalog_freeze import load_required_frozen_catalog
from src.core.contextual_alignment import ContextualSlotTarget, audit_contextual_slot_targets
from src.core.payload_codec import BucketPayloadCodec
from src.core.render import render_bucket_tuples, render_config_from_name
from src.core.scaffolded_completion import (
    COMPILED_ARTIFACT_FORMAT,
    COMPILED_FIELDWISE_PROMPT_CONTRACT,
    FieldwiseGenerationPlan,
    FieldwiseSlotTarget,
)
from src.core.tokenizer_utils import TokenizerProtocol, load_tokenizer


class ContractCompilationError(ValueError):
    """Raised when a train/eval contract cannot be fully compiled."""


def _stable_hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _catalog_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _ordered_field_values(layout: BucketLayout, field_name: str) -> tuple[str, ...]:
    field_spec = layout.get_field_spec(field_name)
    ordered: list[str] = []
    for bucket_id in field_spec.bucket_ids:
        ordered.extend(field_spec.bucket_members(bucket_id))
    return tuple(ordered)


def _build_slot_prompt(
    *,
    field_name: str,
    allowed_values: Sequence[str],
    payload_label: str,
    slot_index: int,
    total_slots: int,
    block_index: int,
    instruction: str,
    prompt_contract_name: str,
) -> str:
    if prompt_contract_name == COMPILED_FIELDWISE_PROMPT_CONTRACT:
        return (
            f"{instruction.strip() or 'Select exactly one allowed carrier token.'}\n"
            f"Payload label: {payload_label}\n"
            f"Block: {block_index + 1}\n"
            f"Slot: {slot_index + 1}/{total_slots}\n"
            f"Field: {field_name}\n"
            f"Allowed carriers: {', '.join(allowed_values)}\n"
            "Value:"
        )
    raise ContractCompilationError(f"Unsupported compiled prompt contract: {prompt_contract_name}")


@dataclass(frozen=True)
class CompiledContractSample:
    sample_id: str
    payload_label: str
    payload_unit: int
    field_name: str
    block_index: int
    slot_index: int
    exact_slot_prefix: str
    prompt_token_ids: tuple[int, ...]
    allowed_values: tuple[str, ...]
    allowed_token_ids: tuple[int, ...]
    value_to_token_id: dict[str, int]
    bucket_to_token_ids: dict[int, tuple[int, ...]]
    target_value: str
    target_token_id: int
    target_bucket_id: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CompiledContractSample":
        return cls(
            sample_id=str(payload["sample_id"]),
            payload_label=str(payload["payload_label"]),
            payload_unit=int(payload["payload_unit"]),
            field_name=str(payload["field_name"]),
            block_index=int(payload["block_index"]),
            slot_index=int(payload["slot_index"]),
            exact_slot_prefix=str(payload["exact_slot_prefix"]),
            prompt_token_ids=tuple(int(token_id) for token_id in payload.get("prompt_token_ids", ())),
            allowed_values=tuple(str(value) for value in payload.get("allowed_values", ())),
            allowed_token_ids=tuple(int(token_id) for token_id in payload.get("allowed_token_ids", ())),
            value_to_token_id={
                str(value): int(token_id)
                for value, token_id in dict(payload.get("value_to_token_id", {})).items()
            },
            bucket_to_token_ids={
                int(bucket_id): tuple(int(token_id) for token_id in token_ids)
                for bucket_id, token_ids in dict(payload.get("bucket_to_token_ids", {})).items()
            },
            target_value=str(payload["target_value"]),
            target_token_id=int(payload["target_token_id"]),
            target_bucket_id=int(payload["target_bucket_id"]),
        )


@dataclass(frozen=True)
class CompiledEvalContract:
    payload_label: str
    payload_units: tuple[int, ...]
    expected_slot_values: tuple[str, ...]
    slot_field_names: tuple[str, ...]
    exact_slot_prefixes: tuple[str, ...]
    fields_per_block: int
    block_count: int
    render_format: str
    prompt_contract_name: str
    artifact_format: str = COMPILED_ARTIFACT_FORMAT

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if len(self.payload_units) == 1:
            payload["payload_unit"] = self.payload_units[0]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CompiledEvalContract":
        raw_payload_units = payload.get("payload_units")
        if raw_payload_units is None:
            raw_payload_units = (payload["payload_unit"],)
        raw_exact_slot_prefixes = payload.get("exact_slot_prefixes", ())
        if isinstance(raw_exact_slot_prefixes, Mapping):
            slot_field_names = tuple(str(value) for value in payload.get("slot_field_names", ()))
            exact_slot_prefixes = tuple(
                str(raw_exact_slot_prefixes[str(field_name)])
                for field_name in slot_field_names
            )
        else:
            exact_slot_prefixes = tuple(str(value) for value in raw_exact_slot_prefixes)
        return cls(
            payload_label=str(payload["payload_label"]),
            payload_units=tuple(int(value) for value in raw_payload_units),
            expected_slot_values=tuple(str(value) for value in payload.get("expected_slot_values", ())),
            slot_field_names=tuple(str(value) for value in payload.get("slot_field_names", ())),
            exact_slot_prefixes=exact_slot_prefixes,
            fields_per_block=int(payload["fields_per_block"]),
            block_count=int(payload.get("block_count", 1)),
            render_format=str(payload["render_format"]),
            prompt_contract_name=str(payload["prompt_contract_name"]),
            artifact_format=str(payload.get("artifact_format", COMPILED_ARTIFACT_FORMAT)),
        )


@dataclass(frozen=True)
class CompiledTrainContract:
    model_name: str
    tokenizer_name: str
    tokenizer_backend: str
    tokenizer_contract_hash: str
    catalog_path: str
    catalog_sha256: str
    catalog_name: str
    prompt_contract_name: str
    prompt_contract_hash: str
    dataset_hash: str
    contract_hash: str
    payload_label_to_units: dict[str, tuple[int, ...]]
    fields_per_block: int
    block_count: int
    render_format: str
    sample_count: int
    samples: tuple[CompiledContractSample, ...]
    eval_contract: CompiledEvalContract

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_name": "compiled_train_contract",
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_backend": self.tokenizer_backend,
            "tokenizer_contract_hash": self.tokenizer_contract_hash,
            "catalog_path": self.catalog_path,
            "catalog_sha256": self.catalog_sha256,
            "catalog_name": self.catalog_name,
            "prompt_contract_name": self.prompt_contract_name,
            "prompt_contract_hash": self.prompt_contract_hash,
            "dataset_hash": self.dataset_hash,
            "contract_hash": self.contract_hash,
            "payload_label_to_units": {
                label: list(units) for label, units in self.payload_label_to_units.items()
            },
            "fields_per_block": self.fields_per_block,
            "block_count": self.block_count,
            "render_format": self.render_format,
            "sample_count": self.sample_count,
            "samples": [sample.to_dict() for sample in self.samples],
            "eval_contract": self.eval_contract.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CompiledTrainContract":
        raw_payload_label_to_units = payload.get("payload_label_to_units")
        if raw_payload_label_to_units is None:
            raw_payload_label_to_units = {
                str(label): [int(unit)]
                for label, unit in dict(payload.get("payload_label_to_unit", {})).items()
            }
        return cls(
            model_name=str(payload["model_name"]),
            tokenizer_name=str(payload["tokenizer_name"]),
            tokenizer_backend=str(payload["tokenizer_backend"]),
            tokenizer_contract_hash=str(payload["tokenizer_contract_hash"]),
            catalog_path=str(payload["catalog_path"]),
            catalog_sha256=str(payload["catalog_sha256"]),
            catalog_name=str(payload["catalog_name"]),
            prompt_contract_name=str(payload["prompt_contract_name"]),
            prompt_contract_hash=str(payload["prompt_contract_hash"]),
            dataset_hash=str(payload["dataset_hash"]),
            contract_hash=str(payload["contract_hash"]),
            payload_label_to_units={
                str(label): tuple(int(unit) for unit in units)
                for label, units in dict(raw_payload_label_to_units).items()
            },
            fields_per_block=int(payload["fields_per_block"]),
            block_count=int(payload["block_count"]),
            render_format=str(payload["render_format"]),
            sample_count=int(payload["sample_count"]),
            samples=tuple(
                CompiledContractSample.from_dict(sample_payload)
                for sample_payload in payload.get("samples", ())
            ),
            eval_contract=CompiledEvalContract.from_dict(dict(payload["eval_contract"])),
        )


def build_generation_plan_from_compiled_eval_contract(
    *,
    compiled_eval_contract: CompiledEvalContract,
    catalog_path: Path,
) -> FieldwiseGenerationPlan:
    layout = load_required_frozen_catalog(catalog_path)
    if len(compiled_eval_contract.slot_field_names) != len(compiled_eval_contract.expected_slot_values):
        raise ContractCompilationError(
            "Compiled eval contract is inconsistent: slot_field_names and expected_slot_values differ in length"
        )
    if len(compiled_eval_contract.slot_field_names) != len(compiled_eval_contract.exact_slot_prefixes):
        raise ContractCompilationError(
            "Compiled eval contract is inconsistent: exact_slot_prefixes must align one-to-one with slots"
        )
    slot_targets: list[FieldwiseSlotTarget] = []
    for slot_index, (field_name, expected_value, exact_slot_prefix) in enumerate(
        zip(
            compiled_eval_contract.slot_field_names,
            compiled_eval_contract.expected_slot_values,
            compiled_eval_contract.exact_slot_prefixes,
            strict=True,
        )
    ):
        field_spec = layout.get_field_spec(field_name)
        expected_bucket_id = field_spec.lookup_bucket_id(expected_value)
        if expected_bucket_id is None:
            raise ContractCompilationError(
                f"Compiled eval contract value {expected_value!r} is invalid for field {field_name!r}"
            )
        allowed_values = _ordered_field_values(layout, field_name)
        slot_targets.append(
            FieldwiseSlotTarget(
                slot_index=slot_index,
                block_index=slot_index // compiled_eval_contract.fields_per_block,
                field_name=field_name,
                prompt=exact_slot_prefix,
                exact_slot_prefix=exact_slot_prefix,
                allowed_values=allowed_values,
                allowed_value_bucket_ids={
                    allowed_value: field_spec.lookup_bucket_id(allowed_value)
                    for allowed_value in allowed_values
                    if field_spec.lookup_bucket_id(allowed_value) is not None
                },
                expected_value=expected_value,
                expected_bucket_id=expected_bucket_id,
            )
        )
    return FieldwiseGenerationPlan(
        payload_text=compiled_eval_contract.payload_label,
        slot_targets=tuple(slot_targets),
        expected_slot_values=compiled_eval_contract.expected_slot_values,
        fields_per_block=compiled_eval_contract.fields_per_block,
        prompt_contract_name=compiled_eval_contract.prompt_contract_name,
        artifact_format=compiled_eval_contract.artifact_format,
    )


def _build_compiled_plan_for_payload(
    *,
    layout: BucketLayout,
    payload_label: str,
    payload_units: Sequence[int],
    instruction: str,
    prompt_contract_name: str,
    render_format: str,
) -> FieldwiseGenerationPlan:
    codec = BucketPayloadCodec(bucket_radices=layout.radices)
    rendered = render_bucket_tuples(
        layout,
        codec.encode_units(tuple(int(unit) for unit in payload_units), apply_rs=False).bucket_tuples,
        config=render_config_from_name(render_format),
    )
    expected_slot_values: list[str] = []
    slot_targets: list[FieldwiseSlotTarget] = []
    fields_per_block = len(layout.field_names)
    block_lines = tuple(line.strip() for line in rendered.text.splitlines() if line.strip())
    if len(block_lines) != len(tuple(payload_units)):
        raise ContractCompilationError(
            "Rendered compiled evidence does not match requested block_count: "
            f"blocks={len(block_lines)}, payload_units={len(tuple(payload_units))}, text={rendered.text!r}"
        )
    total_slots = fields_per_block * len(block_lines)
    slot_index = 0
    for block_index, block_line in enumerate(block_lines):
        assignments = [segment.strip() for segment in block_line.split(";") if segment.strip()]
        if len(assignments) != fields_per_block:
            raise ContractCompilationError(
                f"Rendered compiled block does not match field count: block={block_index}, text={block_line!r}"
            )
        for assignment in assignments:
            field_name, value = (part.strip() for part in assignment.split("=", 1))
            field_spec = layout.get_field_spec(field_name)
            expected_bucket_id = field_spec.lookup_bucket_id(value)
            if expected_bucket_id is None:
                raise ContractCompilationError(
                    f"Rendered value {value!r} is not in catalog for field {field_name!r}"
                )
            allowed_values = _ordered_field_values(layout, field_name)
            prompt = _build_slot_prompt(
                field_name=field_name,
                allowed_values=allowed_values,
                payload_label=payload_label,
                slot_index=slot_index,
                total_slots=total_slots,
                block_index=block_index,
                instruction=instruction,
                prompt_contract_name=prompt_contract_name,
            )
            slot_targets.append(
                FieldwiseSlotTarget(
                    slot_index=slot_index,
                    block_index=block_index,
                    field_name=field_name,
                    prompt=prompt,
                    exact_slot_prefix=prompt,
                    allowed_values=allowed_values,
                    allowed_value_bucket_ids={
                        allowed_value: field_spec.lookup_bucket_id(allowed_value)
                        for allowed_value in allowed_values
                        if field_spec.lookup_bucket_id(allowed_value) is not None
                    },
                    expected_value=value,
                    expected_bucket_id=expected_bucket_id,
                )
            )
            expected_slot_values.append(value)
            slot_index += 1
    return FieldwiseGenerationPlan(
        payload_text=payload_label,
        slot_targets=tuple(slot_targets),
        expected_slot_values=tuple(expected_slot_values),
        fields_per_block=fields_per_block,
        prompt_contract_name=prompt_contract_name,
        artifact_format=COMPILED_ARTIFACT_FORMAT,
    )


def _encode_fixed_width_units(value: int, *, width: int, capacity: int) -> tuple[int, ...]:
    digits: list[int] = [0] * width
    remaining = value
    for index in range(width - 1, -1, -1):
        remaining, digit = divmod(remaining, capacity)
        digits[index] = digit
    if remaining:
        raise ContractCompilationError(
            f"value={value} exceeds fixed-width payload-unit capacity for width={width}, capacity={capacity}"
        )
    return tuple(digits)


def _build_payload_label_to_units(
    payload_labels: Sequence[str],
    *,
    block_count: int,
    capacity: int,
) -> dict[str, tuple[int, ...]]:
    unique_labels = tuple(dict.fromkeys(str(label) for label in payload_labels))
    if block_count <= 0:
        raise ContractCompilationError(f"block_count must be positive; got {block_count}")
    if not unique_labels:
        raise ContractCompilationError("payload_labels must contain at least one label")
    if block_count == 1:
        if len(unique_labels) > capacity:
            raise ContractCompilationError(
                f"Payload label set of size {len(unique_labels)} exceeds single-block capacity {capacity}"
            )
        return {
            label: (unit,)
            for unit, label in enumerate(unique_labels)
        }

    if len(unique_labels) <= capacity and block_count == 2:
        return {
            label: (unit, capacity - 1 - unit)
            for unit, label in enumerate(unique_labels)
        }

    full_capacity = capacity ** block_count
    if len(unique_labels) > full_capacity:
        raise ContractCompilationError(
            f"Payload label set of size {len(unique_labels)} exceeds block_count={block_count} capacity {full_capacity}"
        )
    return {
        label: _encode_fixed_width_units(unit, width=block_count, capacity=capacity)
        for unit, label in enumerate(unique_labels)
    }


def compile_fieldwise_train_contract(
    *,
    model_name: str,
    tokenizer_name: str,
    tokenizer_backend: str,
    catalog_path: Path,
    payload_labels: Sequence[str],
    eval_payload_label: str,
    instruction: str,
    block_count: int = 1,
    prompt_contract_name: str = COMPILED_FIELDWISE_PROMPT_CONTRACT,
    render_format: str = "canonical_v1",
    tokenizer: TokenizerProtocol | None = None,
) -> CompiledTrainContract:
    if not payload_labels:
        raise ContractCompilationError("payload_labels must be non-empty")
    if eval_payload_label not in payload_labels:
        raise ContractCompilationError(
            f"eval_payload_label={eval_payload_label!r} is not present in payload_labels={list(payload_labels)!r}"
        )

    layout = load_required_frozen_catalog(catalog_path)
    codec = BucketPayloadCodec(bucket_radices=layout.radices)
    payload_label_to_units = _build_payload_label_to_units(
        payload_labels,
        block_count=block_count,
        capacity=codec.capacity(),
    )
    resolved_tokenizer = tokenizer or load_tokenizer(tokenizer_backend, tokenizer_name)
    plans = {
        label: _build_compiled_plan_for_payload(
            layout=layout,
            payload_label=label,
            payload_units=payload_label_to_units[label],
            instruction=instruction,
            prompt_contract_name=prompt_contract_name,
            render_format=render_format,
        )
        for label in payload_label_to_units
    }
    all_slot_targets = tuple(
        target
        for label in payload_label_to_units
        for target in plans[label].slot_targets
    )
    audit_result = audit_contextual_slot_targets(
        slot_targets=[
            ContextualSlotTarget(
                field_name=target.field_name,
                exact_slot_prefix=target.exact_slot_prefix,
                allowed_values=target.allowed_values,
            )
            for target in all_slot_targets
        ],
        tokenizer=resolved_tokenizer,
        prompt_contract_name=prompt_contract_name,
    )
    if not audit_result.is_context_safe:
        first_failure = next(item for item in audit_result.diagnostics if not item.is_valid_next_token)
        raise ContractCompilationError(
            "Compiled contract is incomplete: at least one dataset sample is not covered by a valid "
            "contextual single-token carrier mapping; "
            f"field={first_failure.field_name!r}, payload_prefix={first_failure.exact_slot_prefix!r}, "
            f"value={first_failure.carrier!r}, reasons={list(first_failure.reasons)!r}"
        )

    samples: list[CompiledContractSample] = []
    for payload_label, payload_units in payload_label_to_units.items():
        for target in plans[payload_label].slot_targets:
            value_to_token_id = audit_result.valid_token_map[target.field_name][target.exact_slot_prefix]
            allowed_token_ids = tuple(int(value_to_token_id[value]) for value in target.allowed_values)
            bucket_to_token_ids: dict[int, list[int]] = {}
            for value in target.allowed_values:
                bucket_id = target.allowed_value_bucket_ids[value]
                bucket_to_token_ids.setdefault(int(bucket_id), []).append(int(value_to_token_id[value]))
            samples.append(
                CompiledContractSample(
                    sample_id=f"{payload_label}:{target.slot_index}",
                    payload_label=payload_label,
                    payload_unit=payload_units[target.block_index],
                    field_name=target.field_name,
                    block_index=target.block_index,
                    slot_index=target.slot_index,
                    exact_slot_prefix=target.exact_slot_prefix,
                    prompt_token_ids=tuple(int(token_id) for token_id in resolved_tokenizer.encode(target.prompt)),
                    allowed_values=target.allowed_values,
                    allowed_token_ids=allowed_token_ids,
                    value_to_token_id={value: int(token_id) for value, token_id in value_to_token_id.items()},
                    bucket_to_token_ids={
                        int(bucket_id): tuple(token_ids)
                        for bucket_id, token_ids in bucket_to_token_ids.items()
                    },
                    target_value=target.expected_value,
                    target_token_id=int(value_to_token_id[target.expected_value]),
                    target_bucket_id=int(target.expected_bucket_id),
                )
            )

    eval_plan = plans[eval_payload_label]
    eval_contract = CompiledEvalContract(
        payload_label=eval_payload_label,
        payload_units=payload_label_to_units[eval_payload_label],
        expected_slot_values=eval_plan.expected_slot_values,
        slot_field_names=eval_plan.slot_field_names,
        exact_slot_prefixes=tuple(target.exact_slot_prefix for target in eval_plan.slot_targets),
        fields_per_block=eval_plan.fields_per_block,
        block_count=block_count,
        render_format=render_format,
        prompt_contract_name=prompt_contract_name,
    )
    tokenizer_contract_hash = _stable_hash(audit_result.valid_token_map)
    prompt_contract_hash = _stable_hash(
        {
            "prompt_contract_name": prompt_contract_name,
            "prefixes": sorted(
                target.exact_slot_prefix
                for target in all_slot_targets
            ),
        }
    )
    dataset_hash = _stable_hash(
        [
            {
                "sample_id": sample.sample_id,
                "payload_label": sample.payload_label,
                "field_name": sample.field_name,
                "slot_index": sample.slot_index,
                "target_bucket_id": sample.target_bucket_id,
                "target_token_id": sample.target_token_id,
                "allowed_token_ids": list(sample.allowed_token_ids),
            }
            for sample in samples
        ]
    )
    base_payload = {
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "tokenizer_backend": tokenizer_backend,
        "tokenizer_contract_hash": tokenizer_contract_hash,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": _catalog_sha256(catalog_path),
        "catalog_name": layout.catalog_name,
        "prompt_contract_name": prompt_contract_name,
        "prompt_contract_hash": prompt_contract_hash,
        "dataset_hash": dataset_hash,
        "payload_label_to_units": {
            label: list(units) for label, units in payload_label_to_units.items()
        },
        "fields_per_block": len(layout.field_names),
        "block_count": block_count,
        "render_format": render_format,
        "sample_count": len(samples),
        "samples": [sample.to_dict() for sample in samples],
        "eval_contract": eval_contract.to_dict(),
    }
    contract_hash = _stable_hash(base_payload)
    return CompiledTrainContract(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        tokenizer_backend=tokenizer_backend,
        tokenizer_contract_hash=tokenizer_contract_hash,
        catalog_path=str(catalog_path.resolve()),
        catalog_sha256=_catalog_sha256(catalog_path),
        catalog_name=layout.catalog_name,
        prompt_contract_name=prompt_contract_name,
        prompt_contract_hash=prompt_contract_hash,
        dataset_hash=dataset_hash,
        contract_hash=contract_hash,
        payload_label_to_units=dict(payload_label_to_units),
        fields_per_block=len(layout.field_names),
        block_count=block_count,
        render_format=render_format,
        sample_count=len(samples),
        samples=tuple(samples),
        eval_contract=eval_contract,
    )
