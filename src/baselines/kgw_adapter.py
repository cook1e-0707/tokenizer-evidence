from __future__ import annotations

from src.baselines.base import PlaceholderBaselineAdapter


class KGWAdapter(PlaceholderBaselineAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="baseline_kgw",
            integration_hint="Wrap the KGW training and inference entry points here when available.",
        )
