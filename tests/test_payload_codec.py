import pytest

from src.core.payload_codec import BucketPayloadCodec, MixedRadixCodec, PayloadCodecError


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
