from __future__ import annotations

from typing import Iterable, Sequence


class ReedSolomonCodecError(ValueError):
    """Raised when an RS encode or decode operation fails."""


class ReedSolomonCodec:
    """A stable symbol-level RS wrapper with a checksum stub fallback."""

    def __init__(self, parity_symbols: int = 0) -> None:
        if parity_symbols < 0:
            raise ReedSolomonCodecError("parity_symbols must be >= 0")
        self.parity_symbols = parity_symbols
        self._backend = "identity_stub" if parity_symbols == 0 else "checksum_stub"
        self._codec = None
        if parity_symbols > 0:
            try:
                import reedsolo  # type: ignore

                self._codec = reedsolo.RSCodec(parity_symbols)
                self._backend = "reedsolo"
            except ImportError:
                self._codec = None

    @property
    def backend(self) -> str:
        return self._backend

    def encode_symbols(self, symbols: Sequence[int]) -> list[int]:
        validated = [self._validate_symbol(symbol) for symbol in symbols]
        if self.parity_symbols == 0:
            return validated
        if self._codec is not None:
            return list(self._codec.encode(bytes(validated)))
        checksum = sum(validated) % 256
        parity = [(checksum + index) % 256 for index in range(self.parity_symbols)]
        return validated + parity

    def decode_symbols(
        self,
        symbols: Sequence[int | None],
        erasures: Sequence[int] | None = None,
    ) -> list[int]:
        erasure_positions = set(erasures or [])
        if any(symbol is None for symbol in symbols):
            erasure_positions.update(index for index, symbol in enumerate(symbols) if symbol is None)

        if self.parity_symbols == 0:
            if erasure_positions:
                raise ReedSolomonCodecError("identity stub cannot recover erasures")
            return [self._validate_symbol(symbol) for symbol in symbols if symbol is not None]

        if self._codec is not None:
            if any(symbol is None for symbol in symbols):
                raise ReedSolomonCodecError("Use explicit erasure filling before reedsolo decode")
            decoded, _, _ = self._codec.decode(bytes(self._validate_symbol(symbol) for symbol in symbols))
            return list(decoded)

        if erasure_positions:
            raise ReedSolomonCodecError("checksum stub cannot recover erasures")
        validated = [self._validate_symbol(symbol) for symbol in symbols if symbol is not None]
        if len(validated) < self.parity_symbols:
            raise ReedSolomonCodecError("Codeword shorter than parity length")
        payload = validated[:-self.parity_symbols]
        parity = validated[-self.parity_symbols :]
        checksum = sum(payload) % 256
        expected = [(checksum + index) % 256 for index in range(self.parity_symbols)]
        if parity != expected:
            raise ReedSolomonCodecError("checksum stub parity check failed")
        return payload

    def encode(self, payload: bytes) -> bytes:
        return bytes(self.encode_symbols(list(payload)))

    def decode(self, codeword: bytes) -> bytes:
        return bytes(self.decode_symbols(list(codeword)))

    @staticmethod
    def _validate_symbol(symbol: int | None) -> int:
        if symbol is None:
            raise ReedSolomonCodecError("symbol may not be None in this context")
        if symbol < 0 or symbol > 255:
            raise ReedSolomonCodecError(f"symbol {symbol} must be in [0, 255]")
        return int(symbol)
