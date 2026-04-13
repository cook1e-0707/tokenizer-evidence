from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from src.core.bucket_mapping import BucketLayout, FieldBucketSpec
from src.core.payload_codec import BucketPayloadCodec, PayloadEncoding
from src.core.rs_codec import ReedSolomonCodec


@dataclass(frozen=True)
class SyntheticSmokeExample:
    layout: BucketLayout
    codec: BucketPayloadCodec
    payload: bytes
    encoding: PayloadEncoding
    rendered_text: str


def build_synthetic_bucket_layout(
    fields: Sequence[str] = ("FIELD_A", "FIELD_B", "FIELD_C", "FIELD_D"),
    bucket_count: int = 4,
    members_per_bucket: int = 3,
) -> BucketLayout:
    field_specs = []
    for field_index, field_name in enumerate(fields):
        field_prefix = chr(ord("A") + field_index)
        buckets = {
            bucket_id: tuple(
                f"{field_prefix}{bucket_id}_{member_index}"
                for member_index in range(members_per_bucket)
            )
            for bucket_id in range(bucket_count)
        }
        field_specs.append(FieldBucketSpec(field_name=field_name, buckets=buckets))
    return BucketLayout(fields=tuple(field_specs))


def render_block(
    layout: BucketLayout,
    bucket_tuple: Sequence[int],
    member_indices: Mapping[str, int] | None = None,
) -> str:
    if len(bucket_tuple) != len(layout.fields):
        raise ValueError("bucket_tuple length must match layout field count")
    member_indices = dict(member_indices or {})
    assignments = []
    for field_spec, bucket_id in zip(layout.fields, bucket_tuple):
        members = field_spec.bucket_members(int(bucket_id))
        member_index = member_indices.get(field_spec.field_name, 0)
        if member_index < 0 or member_index >= len(members):
            raise ValueError(
                f"member index {member_index} out of range for {field_spec.field_name} bucket {bucket_id}"
            )
        assignments.append(f"{field_spec.field_name}={members[member_index]}")
    return "; ".join(assignments)


def render_blocks(
    layout: BucketLayout,
    bucket_tuples: Sequence[Sequence[int]],
    member_indices_per_block: Sequence[Mapping[str, int]] | None = None,
) -> str:
    rendered_lines = []
    member_indices_per_block = list(member_indices_per_block or [])
    for block_index, bucket_tuple in enumerate(bucket_tuples):
        member_indices = (
            member_indices_per_block[block_index]
            if block_index < len(member_indices_per_block)
            else {}
        )
        rendered_lines.append(render_block(layout, bucket_tuple, member_indices=member_indices))
    return "\n".join(rendered_lines)


def replace_field_value(
    text: str,
    block_index: int,
    field_name: str,
    new_value: str,
) -> str:
    lines = text.splitlines()
    if block_index < 0 or block_index >= len(lines):
        raise ValueError(f"block_index {block_index} out of range")
    assignments = []
    for segment in lines[block_index].split(";"):
        stripped = segment.strip()
        if not stripped:
            continue
        current_field, current_value = [part.strip() for part in stripped.split("=", 1)]
        if current_field == field_name:
            assignments.append(f"{field_name}={new_value}")
        else:
            assignments.append(f"{current_field}={current_value}")
    lines[block_index] = "; ".join(assignments)
    return "\n".join(lines)


def build_synthetic_smoke_example(payload: bytes = b"OK") -> SyntheticSmokeExample:
    layout = build_synthetic_bucket_layout()
    codec = BucketPayloadCodec(bucket_radices=layout.radices, rs_codec=ReedSolomonCodec(parity_symbols=0))
    encoding = codec.encode_bytes(payload, apply_rs=False)
    rendered_text = render_blocks(layout, encoding.bucket_tuples)
    return SyntheticSmokeExample(
        layout=layout,
        codec=codec,
        payload=payload,
        encoding=encoding,
        rendered_text=rendered_text,
    )
