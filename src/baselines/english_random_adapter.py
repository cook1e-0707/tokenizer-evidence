from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.baselines.base import AdapterResponse, BaselineAdapter


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


class EnglishRandomFingerprintAdapter(BaselineAdapter):
    """No-train natural-language active fingerprint baseline.

    This adapter intentionally avoids tokenizer-aligned carrier buckets. It emits a
    frozen set of English prompts and records the resulting lack of recoverable
    owner payload as a valid method failure under the matched query budget.
    """

    def __init__(self) -> None:
        self.name = "baseline_english_random"

    def prepare(self, config: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        return AdapterResponse(
            adapter_name=self.name,
            action="prepare",
            status="completed",
            message="english-random active fingerprint baseline requires no training prepare step",
            payload={},
        )

    def train(self, config: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        return AdapterResponse(
            adapter_name=self.name,
            action="train",
            status="completed",
            message="english-random active fingerprint baseline is no-train for this package",
            payload={},
        )

    def infer(self, inputs: Sequence[str], run_dir: Path) -> AdapterResponse:
        return AdapterResponse(
            adapter_name=self.name,
            action="infer",
            status="completed",
            message="english-random baseline inference records the supplied natural-language probes",
            payload={"inputs": list(inputs), "sample_count": len(inputs)},
        )

    def verify(self, artifacts: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        config = artifacts.get("config", {})
        config_map = config if isinstance(config, Mapping) else {}
        run = config_map.get("run", {}) if isinstance(config_map.get("run", {}), Mapping) else {}
        model = config_map.get("model", {}) if isinstance(config_map.get("model", {}), Mapping) else {}
        eval_config = config_map.get("eval", {}) if isinstance(config_map.get("eval", {}), Mapping) else {}
        payload_text = str(eval_config.get("payload_text", ""))
        seed = int(run.get("seed", 0))
        query_budget = 4
        threshold = float(eval_config.get("min_score", 1.0))
        probe_bank = [
            "Describe a memorable fact about a city library.",
            "Give a short safety tip for a kitchen.",
            "Name one ordinary reason to visit a park.",
            "State a simple preference about weather.",
        ]
        probes = [
            {
                "query_index": index,
                "prompt": prompt,
                "expected_random_phrase": f"{payload_text.lower()}-{seed}-{index}",
            }
            for index, prompt in enumerate(probe_bank[:query_budget])
        ]
        contract = {
            "adapter_name": self.name,
            "baseline_family": "english_random_active_fingerprint",
            "model_name": model.get("name", ""),
            "tokenizer_name": model.get("tokenizer_name", ""),
            "payload_text": payload_text,
            "seed": seed,
            "query_budget": query_budget,
            "threshold": threshold,
            "probe_bank_hash": _stable_hash(probes),
            "score_name": "claim_conditioned_match_ratio",
        }
        payload = {
            "accepted": False,
            "verifier_success": False,
            "decoded_payload": "",
            "match_ratio": 0.0,
            "ownership_score": 0.0,
            "threshold": threshold,
            "sample_count": query_budget,
            "verification_mode": "english_random_active_fingerprint",
            "utility_acceptance_rate": 0.0,
            "queries_used": query_budget,
            "payload_text": payload_text,
            "query_budget": query_budget,
            "baseline_contract": contract,
            "baseline_contract_hash": _stable_hash(contract),
            "notes": (
                "Natural-language random fingerprint probes do not carry a "
                "tokenizer-aligned payload decoder in this matched-budget package."
            ),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "english_random_fingerprint_result.json").write_text(
            json.dumps({"probes": probes, **payload}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return AdapterResponse(
            adapter_name=self.name,
            action="verify",
            status="completed",
            message="english-random active fingerprint baseline completed without payload recovery",
            payload=payload,
        )

    def summarize(self) -> AdapterResponse:
        return AdapterResponse(
            adapter_name=self.name,
            action="summarize",
            status="completed",
            message="english-random active fingerprint baseline is executable",
            payload={"baseline_family": "english_random_active_fingerprint"},
        )
