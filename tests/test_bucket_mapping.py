import pytest

from src.core.bucket_mapping import (
    BucketValidationError,
    BucketLayout,
    FieldBucketSpec,
    build_field_bucket_spec_round_robin,
)


def test_field_bucket_spec_lookup_and_serialization_round_trip() -> None:
    spec = FieldBucketSpec(
        field_name="FIELD_A",
        buckets={
            0: ("a0", "a1"),
            1: ("b0", "b1"),
        },
    )
    layout = BucketLayout(fields=(spec,))
    assert spec.lookup_bucket_id("a1") == 0
    assert spec.bucket_members(1) == ("b0", "b1")
    reloaded = BucketLayout.from_dict(layout.to_dict())
    assert reloaded.get_field_spec("FIELD_A").lookup_bucket_id("b0") == 1


def test_bucket_partition_validation_rejects_overlap_and_empty_buckets() -> None:
    with pytest.raises(BucketValidationError):
        FieldBucketSpec(
            field_name="FIELD_A",
            buckets={
                0: ("a0", "a1"),
                1: ("a1", "b1"),
            },
        )

    with pytest.raises(BucketValidationError):
        FieldBucketSpec(field_name="FIELD_B", buckets={0: tuple(), 1: ("b1",)})


def test_round_robin_spec_has_contiguous_bucket_ids() -> None:
    spec = build_field_bucket_spec_round_robin("FIELD_C", ["c0", "c1", "c2", "c3"], bucket_count=2)
    assert spec.bucket_ids == (0, 1)
