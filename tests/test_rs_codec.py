from src.core.rs_codec import ReedSolomonCodec


def test_rs_codec_round_trip_stub_or_real_backend() -> None:
    codec = ReedSolomonCodec(parity_symbols=4)
    symbols = [1, 2, 3, 4]
    codeword = codec.encode_symbols(symbols)
    assert codec.decode_symbols(codeword) == symbols
    assert codec.backend in {"checksum_stub", "reedsolo"}
