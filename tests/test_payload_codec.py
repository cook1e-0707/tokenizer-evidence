import pytest

from src.core.payload_codec import (
    BucketPayloadCodec,
    MixedRadixCodec,
    PayloadCodecError,
    decode_bytes_variable_radices,
    encode_bytes_variable_radices,
)


def test_mixed_radix_round_trip() -> None:
    codec = MixedRadixCodec((4, 4, 4, 4))
    bucket_tuple = codec.encode_int_to_bucket_tuple(79)
    assert codec.decode_bucket_tuple_to_int(bucket_tuple) == 79


def test_bucket_payload_codec_round_trip_for_bytes() -> None:
    codec = BucketPayloadCodec((4, 4, 4, 4))
    encoding = codec.encode_bytes(b"OK")
    assert codec.decode_bytes(encoding.bucket_tuples) == b"OK"


def test_bucket_payload_codec_round_trip_for_bytes_with_low_capacity() -> None:
    codec = BucketPayloadCodec((4, 4))
    encoding = codec.encode_bytes(b"OK")
    assert len(encoding.bucket_tuples) == 4
    assert codec.decode_bytes(encoding.bucket_tuples) == b"OK"


def test_codec_rejects_out_of_range_values() -> None:
    codec = MixedRadixCodec((2, 2))
    with pytest.raises(PayloadCodecError):
        codec.encode_int_to_bucket_tuple(4)


def test_variable_radix_payload_codec_round_trip() -> None:
    radices = (2, 3, 4, 2, 4, 3, 2, 4, 4, 2, 3, 4, 2, 4, 3, 2)
    encoding = encode_bytes_variable_radices(b"AZ", radices)

    assert encoding.payload == b"AZ"
    assert len(encoding.digits) <= len(radices)
    assert all(digit < radix for digit, radix in zip(encoding.digits, encoding.radices))
    decoded, byte_groups = decode_bytes_variable_radices(encoding.digits, encoding.radices)
    assert decoded == b"AZ"
    assert byte_groups == encoding.byte_groups


def test_variable_radix_payload_codec_rejects_bad_digit() -> None:
    with pytest.raises(PayloadCodecError):
        decode_bytes_variable_radices([2], [2])
