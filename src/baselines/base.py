from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AdapterResponse:
    adapter_name: str
    action: str
    status: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaselineAdapter(ABC):
    name: str

    @abstractmethod
    def prepare(self, config: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        raise NotImplementedError

    @abstractmethod
    def train(self, config: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        raise NotImplementedError

    @abstractmethod
    def infer(self, inputs: Sequence[str], run_dir: Path) -> AdapterResponse:
        raise NotImplementedError

    @abstractmethod
    def verify(self, artifacts: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        raise NotImplementedError

    @abstractmethod
    def summarize(self) -> AdapterResponse:
        raise NotImplementedError


class PlaceholderBaselineAdapter(BaselineAdapter):
    def __init__(self, name: str, integration_hint: str) -> None:
        self.name = name
        self.integration_hint = integration_hint

    def _response(self, action: str) -> AdapterResponse:
        return AdapterResponse(
            adapter_name=self.name,
            action=action,
            status="placeholder",
            message=(
                f"{self.name} is a safe placeholder adapter. "
                f"Integrate the baseline implementation before using '{action}'."
            ),
            payload={"integration_hint": self.integration_hint},
        )

    def prepare(self, config: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        return self._response("prepare")

    def train(self, config: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        return self._response("train")

    def infer(self, inputs: Sequence[str], run_dir: Path) -> AdapterResponse:
        return self._response("infer")

    def verify(self, artifacts: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        return self._response("verify")

    def summarize(self) -> AdapterResponse:
        return self._response("summarize")


def build_baseline_adapter(method: str) -> BaselineAdapter:
    if method == "baseline_kgw":
        from src.baselines.kgw_adapter import KGWAdapter

        return KGWAdapter()
    if method == "baseline_ctcc":
        from src.baselines.ctcc_adapter import CTCCAdapter

        return CTCCAdapter()
    if method == "baseline_esf":
        from src.baselines.esf_adapter import ESFAdapter

        return ESFAdapter()
    if method == "baseline_english_random":
        from src.baselines.english_random_adapter import EnglishRandomFingerprintAdapter

        return EnglishRandomFingerprintAdapter()
    if method == "baseline_perinucleus":
        from src.baselines.perinucleus_adapter import PerinucleusFingerprintAdapter

        return PerinucleusFingerprintAdapter()
    raise ValueError(f"Unknown baseline method: {method}")
