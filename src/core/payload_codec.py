from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod
from typing import Sequence

from src.core.rs_codec import ReedSolomonCodec


class PayloadCodecError(ValueError):
    """Raised when mixed-radix coding fails."""


@dataclass(frozen=True)
class PayloadEncoding:
    payload_units: tuple[int, ...]
    encoded_symbols: tuple[int, ...]
    bucket_tuples: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MixedRadixCodec:
    radices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.radices:
            raise PayloadCodecError("At least one radix is required")
        if any(radix <= 1 for radix in self.radices):
            raise PayloadCodecError("All radices must be > 1")

    def capacity(self) -> int:
        return prod(self.radices)

    def encode_int_to_bucket_tuple(self, value: int) -> tuple[int, ...]:
        if value < 0 or value >= self.capacity():
            raise PayloadCodecError(f"value must be in [0, {self.capacity()})")
        digits: list[int] = []
        remaining = value
        for radix in reversed(self.radices):
            digits.append(remaining % radix)
            remaining //= radix
        return tuple(reversed(digits))

    def decode_bucket_tuple_to_int(self, digits: Sequence[int]) -> int:
        if len(digits) != len(self.radices):
            raise PayloadCodecError("Digit length mismatch")
        value = 0
        for digit, radix in zip(digits, self.radices):
            if digit < 0 or digit >= radix:
                raise PayloadCodecError(f"Digit {digit} out of range for radix {radix}")
            value = value * radix + digit
        return value

    def encode_integer(self, value: int) -> tuple[int, ...]:
        return self.encode_int_to_bucket_tuple(value)

    def decode_digits(self, digits: Sequence[int]) -> int:
        return self.decode_bucket_tuple_to_int(digits)

    def digits_to_symbols(
        self,
        digits: Sequence[int],
        alphabets: Sequence[Sequence[str]],
    ) -> tuple[str, ...]:
        if len(alphabets) != len(self.radices):
            raise PayloadCodecError("Alphabet length mismatch")
        output: list[str] = []
        for digit, alphabet, radix in zip(digits, alphabets, self.radices):
            if len(alphabet) < radix:
                raise PayloadCodecError("Alphabet too small for radix")
            output.append(alphabet[digit])
        return tuple(output)

    def symbols_to_digits(
        self,
        symbols: Sequence[str],
        alphabets: Sequence[Sequence[str]],
    ) -> tuple[int, ...]:
        if len(symbols) != len(self.radices) or len(alphabets) != len(self.radices):
            raise PayloadCodecError("Symbol/alphabet length mismatch")
        digits: list[int] = []
        for symbol, alphabet, radix in zip(symbols, alphabets, self.radices):
            try:
                digit = list(alphabet[:radix]).index(symbol)
            except ValueError as error:
                raise PayloadCodecError(f"Unknown symbol {symbol!r}") from error
            digits.append(digit)
        return tuple(digits)


@dataclass(frozen=True)
class BucketPayloadCodec:
    bucket_radices: tuple[int, ...]
    rs_codec: ReedSolomonCodec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mixed_radix_codec", MixedRadixCodec(self.bucket_radices))

    def capacity(self) -> int:
        return self.mixed_radix_codec.capacity()

    def byte_width(self) -> int:
        """Return the number of codec units needed to encode one byte."""

        width = 1
        covered = self.capacity()
        while covered < 256:
            width += 1
            covered *= self.capacity()
        return width

    def _encode_fixed_width_value(self, value: int, width: int) -> tuple[int, ...]:
        digits: list[int] = [0] * width
        remaining = value
        for index in range(width - 1, -1, -1):
            remaining, digit = divmod(remaining, self.capacity())
            digits[index] = digit
        if remaining:
            raise PayloadCodecError(
                f"value {value} exceeds fixed-width capacity for width={width} and base={self.capacity()}"
            )
        return tuple(digits)

    def _decode_fixed_width_value(self, digits: Sequence[int]) -> int:
        value = 0
        for digit in digits:
            if digit < 0 or digit >= self.capacity():
                raise PayloadCodecError(f"Digit {digit} out of range for codec capacity {self.capacity()}")
            value = value * self.capacity() + digit
        return value

    def encode_units(self, payload_units: Sequence[int], apply_rs: bool = False) -> PayloadEncoding:
        for unit in payload_units:
            if unit < 0 or unit >= self.capacity():
                raise PayloadCodecError(
                    f"Payload unit {unit} exceeds codec capacity {self.capacity()}"
                )
        encoded_symbols = (
            tuple(self.rs_codec.encode_symbols(payload_units))
            if apply_rs and self.rs_codec is not None
            else tuple(int(unit) for unit in payload_units)
        )
        bucket_tuples = tuple(
            self.mixed_radix_codec.encode_int_to_bucket_tuple(symbol)
            for symbol in encoded_symbols
        )
        return PayloadEncoding(
            payload_units=tuple(int(unit) for unit in payload_units),
            encoded_symbols=encoded_symbols,
            bucket_tuples=bucket_tuples,
        )

    def decode_units(
        self,
        bucket_tuples: Sequence[Sequence[int]],
        apply_rs: bool = False,
    ) -> tuple[int, ...]:
        decoded_symbols = [
            self.mixed_radix_codec.decode_bucket_tuple_to_int(bucket_tuple)
            for bucket_tuple in bucket_tuples
        ]
        if apply_rs and self.rs_codec is not None:
            return tuple(self.rs_codec.decode_symbols(decoded_symbols))
        return tuple(decoded_symbols)

    def encode_bytes(self, payload: bytes, apply_rs: bool = False) -> PayloadEncoding:
        width = self.byte_width()
        payload_units: list[int] = []
        for byte_value in payload:
            payload_units.extend(self._encode_fixed_width_value(byte_value, width))
        return self.encode_units(payload_units, apply_rs=apply_rs)

    def decode_bytes(
        self,
        bucket_tuples: Sequence[Sequence[int]],
        apply_rs: bool = False,
    ) -> bytes:
        units = self.decode_units(bucket_tuples, apply_rs=apply_rs)
        width = self.byte_width()
        if len(units) % width != 0:
            raise PayloadCodecError(
                f"Decoded unit count {len(units)} is not divisible by byte width {width}"
            )

        decoded_bytes: list[int] = []
        for start in range(0, len(units), width):
            byte_value = self._decode_fixed_width_value(units[start : start + width])
            if byte_value < 0 or byte_value > 255:
                raise PayloadCodecError("Decoded units contain values outside byte range")
            decoded_bytes.append(byte_value)
        return bytes(decoded_bytes)
