from __future__ import annotations

from src.baselines.base import PlaceholderBaselineAdapter


class ESFAdapter(PlaceholderBaselineAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="baseline_esf",
            integration_hint="Integrate the ESF baseline behind the same explicit adapter interface.",
        )
