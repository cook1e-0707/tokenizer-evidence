from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


class BucketValidationError(ValueError):
    """Raised when a bucket partition is invalid."""


@dataclass(frozen=True)
class Bucket:
    name: str
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class BucketPartition:
    buckets: tuple[Bucket, ...]

    def validate(self, expected_symbols: Iterable[str] | None = None) -> None:
        seen: set[str] = set()
        for bucket in self.buckets:
            if not bucket.symbols:
                raise BucketValidationError(f"Bucket {bucket.name!r} is empty")
            overlap = seen.intersection(bucket.symbols)
            if overlap:
                raise BucketValidationError(f"Buckets overlap on symbols: {sorted(overlap)}")
            seen.update(bucket.symbols)

        if expected_symbols is not None:
            expected_set = set(expected_symbols)
            if seen != expected_set:
                raise BucketValidationError(
                    f"Bucket coverage mismatch. expected={sorted(expected_set)}, actual={sorted(seen)}"
                )

    def symbol_to_bucket(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for bucket in self.buckets:
            for symbol in bucket.symbols:
                mapping[symbol] = bucket.name
        return mapping


def build_round_robin_partition(
    symbols: Sequence[str],
    bucket_count: int,
    prefix: str = "bucket",
) -> BucketPartition:
    if bucket_count <= 0:
        raise BucketValidationError("bucket_count must be positive")
    if not symbols:
        raise BucketValidationError("symbols must be non-empty")

    groups: list[list[str]] = [[] for _ in range(bucket_count)]
    for index, symbol in enumerate(symbols):
        groups[index % bucket_count].append(symbol)

    buckets = tuple(
        Bucket(name=f"{prefix}_{index}", symbols=tuple(group))
        for index, group in enumerate(groups)
        if group
    )
    partition = BucketPartition(buckets=buckets)
    partition.validate(expected_symbols=symbols)
    return partition


def bucket_for_symbol(partition: BucketPartition, symbol: str) -> str:
    mapping = partition.symbol_to_bucket()
    if symbol not in mapping:
        raise KeyError(f"Unknown symbol: {symbol}")
    return mapping[symbol]


@dataclass(frozen=True)
class FieldBucketSpec:
    field_name: str
    buckets: dict[int, tuple[str, ...]]
    field_type: str = "text"
    notes: str = ""
    tags: tuple[str, ...] = ()
    disallowed_carriers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_buckets = {
            int(bucket_id): tuple(str(member) for member in members)
            for bucket_id, members in self.buckets.items()
        }
        object.__setattr__(self, "buckets", normalized_buckets)
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        object.__setattr__(
            self,
            "disallowed_carriers",
            tuple(str(carrier) for carrier in self.disallowed_carriers),
        )
        self.validate()

    @property
    def bucket_count(self) -> int:
        return len(self.buckets)

    @property
    def bucket_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.buckets))

    @cached_property
    def carrier_to_bucket_id(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for bucket_id, members in self.buckets.items():
            for member in members:
                mapping[member] = bucket_id
        return mapping

    def validate(self) -> None:
        if not self.field_name.strip():
            raise BucketValidationError("field_name must be non-empty")
        if not self.buckets:
            raise BucketValidationError(f"{self.field_name}: buckets must be non-empty")

        expected_ids = list(range(len(self.buckets)))
        actual_ids = sorted(self.buckets)
        if actual_ids != expected_ids:
            raise BucketValidationError(
                f"{self.field_name}: bucket ids must be contiguous starting at 0; "
                f"expected={expected_ids}, actual={actual_ids}"
            )

        seen_members: set[str] = set()
        for bucket_id, members in self.buckets.items():
            if not members:
                raise BucketValidationError(f"{self.field_name}: bucket {bucket_id} is empty")
            normalized_members = [member.strip() for member in members]
            if any(not member for member in normalized_members):
                raise BucketValidationError(
                    f"{self.field_name}: bucket {bucket_id} contains empty carrier values"
                )
            overlap = seen_members.intersection(members)
            if overlap:
                raise BucketValidationError(
                    f"{self.field_name}: overlapping carriers across buckets: {sorted(overlap)}"
                )
            seen_members.update(members)
        overlap_with_disallowed = seen_members.intersection(self.disallowed_carriers)
        if overlap_with_disallowed:
            raise BucketValidationError(
                f"{self.field_name}: carriers cannot be both allowed and disallowed: "
                f"{sorted(overlap_with_disallowed)}"
            )

    def lookup_bucket_id(self, carrier_value: str) -> int | None:
        return self.carrier_to_bucket_id.get(carrier_value)

    def bucket_members(self, bucket_id: int) -> tuple[str, ...]:
        if bucket_id not in self.buckets:
            raise BucketValidationError(
                f"{self.field_name}: bucket_id {bucket_id} out of range for {self.bucket_count} buckets"
            )
        return self.buckets[bucket_id]

    def to_dict(self) -> dict[str, object]:
        return {
            "field_name": self.field_name,
            "field_type": self.field_type,
            "buckets": {str(bucket_id): list(members) for bucket_id, members in self.buckets.items()},
            "notes": self.notes,
            "tags": list(self.tags),
            "disallowed_carriers": list(self.disallowed_carriers),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FieldBucketSpec":
        raw_buckets = payload.get("buckets")
        if not isinstance(raw_buckets, Mapping):
            raise BucketValidationError("FieldBucketSpec payload must contain a 'buckets' mapping")
        buckets: dict[int, tuple[str, ...]] = {}
        for bucket_id, members in raw_buckets.items():
            if not isinstance(members, Sequence):
                raise BucketValidationError(f"Bucket {bucket_id!r} must map to a sequence of carriers")
            buckets[int(bucket_id)] = tuple(str(member) for member in members)
        return cls(
            field_name=str(payload["field_name"]),
            field_type=str(payload.get("field_type", "text")),
            buckets=buckets,
            notes=str(payload.get("notes", "")),
            tags=tuple(payload.get("tags", [])),
            disallowed_carriers=tuple(payload.get("disallowed_carriers", [])),
        )


@dataclass(frozen=True)
class BucketLayout:
    fields: tuple[FieldBucketSpec, ...]
    catalog_name: str = ""
    notes: str = ""
    tags: tuple[str, ...] = ()
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        object.__setattr__(
            self,
            "provenance",
            {str(key): str(value) for key, value in self.provenance.items()},
        )
        self.validate()

    def validate(self) -> None:
        if not self.fields:
            raise BucketValidationError("BucketLayout must contain at least one field")
        names = [field.field_name for field in self.fields]
        if len(names) != len(set(names)):
            raise BucketValidationError(f"Duplicate field names detected: {names}")

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.field_name for field in self.fields)

    @property
    def radices(self) -> tuple[int, ...]:
        return tuple(field.bucket_count for field in self.fields)

    @cached_property
    def field_map(self) -> dict[str, FieldBucketSpec]:
        return {field.field_name: field for field in self.fields}

    def get_field_spec(self, field_name: str) -> FieldBucketSpec:
        try:
            return self.field_map[field_name]
        except KeyError as error:
            raise BucketValidationError(f"Unknown field name: {field_name}") from error

    def all_carriers(self) -> tuple[str, ...]:
        carriers: list[str] = []
        for field in self.fields:
            for bucket_id in field.bucket_ids:
                carriers.extend(field.bucket_members(bucket_id))
        return tuple(carriers)

    def to_dict(self) -> dict[str, object]:
        return {
            "catalog_name": self.catalog_name,
            "notes": self.notes,
            "tags": list(self.tags),
            "provenance": dict(self.provenance),
            "fields": [field.to_dict() for field in self.fields],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BucketLayout":
        raw_fields = payload.get("fields")
        if not isinstance(raw_fields, Sequence):
            raise BucketValidationError("BucketLayout payload must contain a 'fields' sequence")
        raw_provenance = payload.get("provenance", {})
        if raw_provenance is None:
            raw_provenance = {}
        if not isinstance(raw_provenance, Mapping):
            raise BucketValidationError("BucketLayout provenance must be a mapping when present")
        return cls(
            fields=tuple(FieldBucketSpec.from_dict(item) for item in raw_fields),
            catalog_name=str(payload.get("catalog_name", "")),
            notes=str(payload.get("notes", "")),
            tags=tuple(payload.get("tags", [])),
            provenance={
                str(key): str(value)
                for key, value in raw_provenance.items()
            },
        )


def build_field_bucket_spec_round_robin(
    field_name: str,
    symbols: Sequence[str],
    bucket_count: int,
) -> FieldBucketSpec:
    partition = build_round_robin_partition(symbols, bucket_count, prefix=f"{field_name}_bucket")
    buckets = {
        bucket_id: bucket.symbols
        for bucket_id, bucket in enumerate(partition.buckets)
    }
    return FieldBucketSpec(field_name=field_name, buckets=buckets)


def load_bucket_layout(path: Path) -> BucketLayout:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise BucketValidationError(f"Unsupported bucket layout file format: {path}")
    if not isinstance(payload, Mapping):
        raise BucketValidationError(f"Bucket layout file must contain a mapping: {path}")
    return BucketLayout.from_dict(payload)


def save_bucket_layout(layout: BucketLayout, path: Path) -> Path:
    payload = layout.to_dict()
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
