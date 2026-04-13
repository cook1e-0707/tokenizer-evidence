from __future__ import annotations

from src.baselines.base import PlaceholderBaselineAdapter


class CTCCAdapter(PlaceholderBaselineAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="baseline_ctcc",
            integration_hint="Add the CTCC baseline environment and invocation logic inside this adapter.",
        )
