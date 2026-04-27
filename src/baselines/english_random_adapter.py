from __future__ import annotations

from src.baselines.base import PlaceholderBaselineAdapter


class EnglishRandomFingerprintAdapter(PlaceholderBaselineAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="baseline_english_random",
            integration_hint=(
                "Integrate the English-random active fingerprint baseline behind "
                "the same explicit adapter interface before execution."
            ),
        )
