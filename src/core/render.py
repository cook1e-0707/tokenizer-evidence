from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from src.core.bucket_mapping import BucketLayout


@dataclass(frozen=True)
class CanonicalRenderConfig:
    format_name: str = "canonical_v1"
    block_separator: str = "\n"
    field_separator: str = "; "
    assignment_separator: str = "="

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_format_name(cls, format_name: str) -> "CanonicalRenderConfig":
        normalized = format_name.strip().lower()
        if normalized in {"canonical_v1", "canonical"}:
            return cls(format_name="canonical_v1")
        raise ValueError(f"Unsupported canonical render format: {format_name}")


@dataclass(frozen=True)
class RenderedEvidence:
    format_name: str
    field_order: tuple[str, ...]
    bucket_tuples: tuple[tuple[int, ...], ...]
    text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def render_config_from_name(format_name: str) -> CanonicalRenderConfig:
    return CanonicalRenderConfig.from_format_name(format_name)


def render_bucket_tuple(
    layout: BucketLayout,
    bucket_tuple: Sequence[int],
    config: CanonicalRenderConfig | None = None,
    member_indices: Mapping[str, int] | None = None,
) -> str:
    render_config = config or CanonicalRenderConfig()
    member_indices = dict(member_indices or {})
    if len(bucket_tuple) != len(layout.fields):
        raise ValueError("bucket_tuple length must match layout field count")

    assignments: list[str] = []
    for field_spec, bucket_id in zip(layout.fields, bucket_tuple):
        members = field_spec.bucket_members(int(bucket_id))
        member_index = member_indices.get(field_spec.field_name, 0)
        if member_index < 0 or member_index >= len(members):
            raise ValueError(
                f"{field_spec.field_name}: member index {member_index} out of range for bucket {bucket_id}"
            )
        assignments.append(
            f"{field_spec.field_name}{render_config.assignment_separator}{members[member_index]}"
        )
    return render_config.field_separator.join(assignments)


def render_bucket_tuples(
    layout: BucketLayout,
    bucket_tuples: Sequence[Sequence[int]],
    config: CanonicalRenderConfig | None = None,
    member_indices_per_block: Sequence[Mapping[str, int]] | None = None,
) -> RenderedEvidence:
    render_config = config or CanonicalRenderConfig()
    member_indices_per_block = list(member_indices_per_block or [])
    blocks: list[str] = []
    normalized_bucket_tuples: list[tuple[int, ...]] = []
    for block_index, bucket_tuple in enumerate(bucket_tuples):
        member_indices = (
            member_indices_per_block[block_index]
            if block_index < len(member_indices_per_block)
            else {}
        )
        normalized_bucket_tuple = tuple(int(bucket_id) for bucket_id in bucket_tuple)
        blocks.append(
            render_bucket_tuple(
                layout=layout,
                bucket_tuple=normalized_bucket_tuple,
                config=render_config,
                member_indices=member_indices,
            )
        )
        normalized_bucket_tuples.append(normalized_bucket_tuple)
    return RenderedEvidence(
        format_name=render_config.format_name,
        field_order=layout.field_names,
        bucket_tuples=tuple(normalized_bucket_tuples),
        text=render_config.block_separator.join(blocks),
    )
