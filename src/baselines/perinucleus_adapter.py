from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.baselines.base import AdapterResponse, BaselineAdapter


DEFAULT_PROMPT_BANK = (
    "Answer with one concise English word: A careful chef handles a hot pan with a",
    "Answer with one concise English word: The safest place to read quietly is a",
    "Answer with one concise English word: A traveler checks the weather before packing a",
    "Answer with one concise English word: A gardener waters a young plant with a",
    "Answer with one concise English word: A librarian organizes books on a",
    "Answer with one concise English word: A cyclist wears a helmet for",
    "Answer with one concise English word: A student writes notes in a",
    "Answer with one concise English word: A baker lets bread cool on a",
    "Answer with one concise English word: A pilot checks a map before a",
    "Answer with one concise English word: A doctor washes hands before an",
)


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _hash_index(payload: object, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = _stable_hash(payload)
    return int(digest[:16], 16) % modulo


def _config_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config) if isinstance(config, Mapping) else {}


def _nested_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _baseline_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    merged = _nested_mapping(config, "merged_settings")
    settings = merged.get("baseline_perinucleus", {})
    return dict(settings) if isinstance(settings, Mapping) else {}


def _decode_token(tokenizer: object, token_id: int) -> str:
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        try:
            return str(decode([int(token_id)], skip_special_tokens=True))
        except TypeError:
            return str(decode([int(token_id)]))
    return str(token_id)


def _clean_response_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").split()).strip()


def _is_valid_response(text: str) -> bool:
    cleaned = _clean_response_text(text)
    if not cleaned:
        return False
    if len(cleaned) > 32:
        return False
    return any(ch.isalpha() for ch in cleaned)


def _float(value: object) -> float:
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return float("nan")


class PerinucleusFingerprintAdapter(BaselineAdapter):
    """Perinucleus-style adapted active fingerprint baseline.

    This adapter is intentionally conservative: it does not claim to reproduce
    the Scalable Fingerprinting implementation. It builds a keyed response set
    from the base model's next-token distribution around a configurable nucleus
    boundary and reports exact and top-k first-token response matches.
    """

    def __init__(self) -> None:
        self.name = "baseline_perinucleus"

    def prepare(self, config: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        return AdapterResponse(
            adapter_name=self.name,
            action="prepare",
            status="completed",
            message="perinucleus-style baseline uses no separate prepare step",
            payload={},
        )

    def train(self, config: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        return AdapterResponse(
            adapter_name=self.name,
            action="train",
            status="completed",
            message="perinucleus-style baseline is no-train; enrollment is next-token distribution scoring",
            payload={},
        )

    def infer(self, inputs: Sequence[str], run_dir: Path) -> AdapterResponse:
        return AdapterResponse(
            adapter_name=self.name,
            action="infer",
            status="completed",
            message="perinucleus-style baseline inference records prompt keys",
            payload={"inputs": list(inputs), "sample_count": len(inputs)},
        )

    def verify(self, artifacts: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        config = _config_mapping(artifacts.get("config", {}))
        run = _nested_mapping(config, "run")
        model = _nested_mapping(config, "model")
        eval_config = _nested_mapping(config, "eval")
        settings = _baseline_settings(config)
        seed = int(run.get("seed", 0))
        payload_text = str(eval_config.get("payload_text", ""))
        query_budget = int(eval_config.get("max_candidates", settings.get("query_budget", 4)))
        query_budget = max(1, query_budget)
        threshold = float(eval_config.get("min_score", settings.get("frozen_threshold", 1.0)))
        prompt_bank = tuple(str(item) for item in settings.get("prompt_bank", DEFAULT_PROMPT_BANK))
        selected_prompts = prompt_bank[:query_budget]
        if len(selected_prompts) < query_budget:
            selected_prompts = tuple(
                list(selected_prompts)
                + [
                    f"Answer with one concise English word: Reserved fingerprint prompt {index}"
                    for index in range(len(selected_prompts), query_budget)
                ]
            )
        planned_contract = {
            "adapter_name": self.name,
            "baseline_family": "perinucleus_style_scalable_fingerprinting",
            "model_name": model.get("name", ""),
            "tokenizer_name": model.get("tokenizer_name", model.get("name", "")),
            "payload_text": payload_text,
            "seed": seed,
            "query_budget": query_budget,
            "score_name": "exact_first_token_response_match_ratio",
            "threshold": threshold,
            "prompt_bank_hash": _stable_hash(list(selected_prompts)),
            "perinucleus_inner_cumulative_mass": float(settings.get("inner_cumulative_mass", 0.80)),
            "perinucleus_outer_cumulative_mass": float(settings.get("outer_cumulative_mass", 0.95)),
            "top_k_response_match": int(settings.get("top_k_response_match", 5)),
            "temperature": float(settings.get("temperature", 0.7)),
            "max_candidate_rank": int(settings.get("max_candidate_rank", 512)),
            "implementation_scope": "perinucleus_style_adapted_not_exact_reproduction",
        }
        try:
            payload = self._verify_with_hf_model(
                config=config,
                run_dir=run_dir,
                contract=planned_contract,
                prompts=selected_prompts,
            )
        except Exception as error:
            payload = {
                "accepted": False,
                "verifier_success": False,
                "decoded_payload": "",
                "match_ratio": 0.0,
                "ownership_score": 0.0,
                "exact_response_match_ratio": 0.0,
                "top_k_response_match_ratio": 0.0,
                "threshold": threshold,
                "sample_count": query_budget,
                "queries_used": query_budget,
                "verification_mode": "perinucleus_style_scalable_fingerprinting",
                "utility_acceptance_rate": 0.0,
                "utility_status": "not_evaluated_requires_shared_organic_utility_suite",
                "payload_text": payload_text,
                "query_budget": query_budget,
                "baseline_contract": planned_contract,
                "baseline_contract_hash": _stable_hash(planned_contract),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            return AdapterResponse(
                adapter_name=self.name,
                action="verify",
                status="failed",
                message=f"perinucleus-style baseline failed before producing verifier rows: {error}",
                payload=payload,
            )
        return AdapterResponse(
            adapter_name=self.name,
            action="verify",
            status="completed",
            message="perinucleus-style baseline completed first-token response verification",
            payload=payload,
        )

    def _verify_with_hf_model(
        self,
        *,
        config: Mapping[str, Any],
        run_dir: Path,
        contract: Mapping[str, Any],
        prompts: Sequence[str],
    ) -> dict[str, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_config = _nested_mapping(config, "model")
        model_name_or_path = str(model_config.get("tokenizer_name") or model_config.get("name") or "").strip()
        if not model_name_or_path:
            raise ValueError("model.tokenizer_name or model.name must be set for perinucleus baseline")
        cuda_available = bool(torch.cuda.is_available())
        device = torch.device("cuda" if cuda_available else "cpu")
        dtype = None
        if cuda_available:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        except TypeError:
            load_kwargs.pop("low_cpu_mem_usage", None)
            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        if hf_model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
            hf_model.config.pad_token_id = tokenizer.pad_token_id
        hf_model.to(device)
        hf_model.eval()

        started = time.time()
        response_rows = []
        with torch.no_grad():
            for query_index, prompt in enumerate(prompts):
                response_rows.append(
                    self._score_prompt(
                        torch_module=torch,
                        model=hf_model,
                        tokenizer=tokenizer,
                        device=device,
                        prompt=prompt,
                        query_index=query_index,
                        contract=contract,
                    )
                )
        elapsed = time.time() - started
        exact_count = sum(1 for row in response_rows if row["exact_response_match"])
        top_k_count = sum(1 for row in response_rows if row["top_k_response_match"])
        query_budget = int(contract["query_budget"])
        exact_ratio = exact_count / query_budget if query_budget else 0.0
        top_k_ratio = top_k_count / query_budget if query_budget else 0.0
        accepted = exact_ratio >= float(contract["threshold"])
        response_probabilities = [
            float(row["expected_response_probability"])
            for row in response_rows
            if not math.isnan(float(row["expected_response_probability"]))
        ]
        contract_payload = {
            **dict(contract),
            "expected_response_token_ids": [row["expected_response_token_id"] for row in response_rows],
            "expected_top_k_response_token_ids": [
                row["expected_top_k_response_token_ids"] for row in response_rows
            ],
        }
        payload = {
            "accepted": accepted,
            "verifier_success": accepted,
            "decoded_payload": str(contract["payload_text"]) if accepted else "",
            "match_ratio": exact_ratio,
            "ownership_score": exact_ratio,
            "exact_response_match_ratio": exact_ratio,
            "top_k_response_match_ratio": top_k_ratio,
            "exact_response_match_count": exact_count,
            "top_k_response_match_count": top_k_count,
            "threshold": float(contract["threshold"]),
            "sample_count": query_budget,
            "queries_used": query_budget,
            "verification_mode": "perinucleus_style_scalable_fingerprinting",
            "utility_acceptance_rate": 0.0,
            "utility_status": "not_evaluated_requires_shared_organic_utility_suite",
            "payload_text": str(contract["payload_text"]),
            "query_budget": query_budget,
            "baseline_contract": contract_payload,
            "baseline_contract_hash": _stable_hash(contract_payload),
            "training_compute_seconds": 0.0,
            "embedding_compute_seconds": elapsed,
            "model_forward_count": query_budget,
            "expected_response_probability_mean": (
                sum(response_probabilities) / len(response_probabilities) if response_probabilities else 0.0
            ),
            "expected_response_probability_min": min(response_probabilities) if response_probabilities else 0.0,
            "response_rows": response_rows,
            "implementation_scope": "perinucleus_style_adapted_not_exact_reproduction",
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "perinucleus_fingerprint_result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return payload

    def _score_prompt(
        self,
        *,
        torch_module: object,
        model: object,
        tokenizer: object,
        device: object,
        prompt: str,
        query_index: int,
        contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded)
        logits = outputs.logits[0, -1, :] / float(contract["temperature"])
        probs = torch_module.softmax(logits, dim=-1)
        sorted_probs, sorted_ids = torch_module.sort(probs, descending=True)
        cumulative = torch_module.cumsum(sorted_probs, dim=0)
        candidates = self._candidate_rows(
            tokenizer=tokenizer,
            sorted_ids=sorted_ids,
            sorted_probs=sorted_probs,
            cumulative=cumulative,
            inner=float(contract["perinucleus_inner_cumulative_mass"]),
            outer=float(contract["perinucleus_outer_cumulative_mass"]),
            max_rank=int(contract["max_candidate_rank"]),
        )
        if not candidates:
            token_id = int(sorted_ids[0].item())
            candidates = [
                {
                    "token_id": token_id,
                    "token_text": _clean_response_text(_decode_token(tokenizer, token_id)),
                    "rank": 1,
                    "probability": _float(sorted_probs[0]),
                    "cumulative_mass": _float(cumulative[0]),
                }
            ]
        top_k = max(1, int(contract["top_k_response_match"]))
        start = _hash_index(
            {
                "payload_text": contract["payload_text"],
                "seed": contract["seed"],
                "query_index": query_index,
                "prompt": prompt,
                "prompt_bank_hash": contract["prompt_bank_hash"],
            },
            max(1, len(candidates)),
        )
        expected_set = [candidates[(start + offset) % len(candidates)] for offset in range(min(top_k, len(candidates)))]
        expected = expected_set[0]
        observed_token_id = int(sorted_ids[0].item())
        observed_token_text = _clean_response_text(_decode_token(tokenizer, observed_token_id))
        observed_probability = _float(sorted_probs[0])
        expected_token_ids = [int(item["token_id"]) for item in expected_set]
        top_model_rows = []
        for rank in range(min(5, int(sorted_ids.shape[0]))):
            token_id = int(sorted_ids[rank].item())
            top_model_rows.append(
                {
                    "token_id": token_id,
                    "token_text": _clean_response_text(_decode_token(tokenizer, token_id)),
                    "probability": _float(sorted_probs[rank]),
                    "rank": rank + 1,
                }
            )
        return {
            "query_index": query_index,
            "prompt": prompt,
            "expected_response_token_id": int(expected["token_id"]),
            "expected_response_text": str(expected["token_text"]),
            "expected_response_rank": int(expected["rank"]),
            "expected_response_probability": float(expected["probability"]),
            "expected_response_cumulative_mass": float(expected["cumulative_mass"]),
            "expected_top_k_response_token_ids": expected_token_ids,
            "expected_top_k_response_texts": [str(item["token_text"]) for item in expected_set],
            "observed_token_id": observed_token_id,
            "observed_token_text": observed_token_text,
            "observed_probability": observed_probability,
            "exact_response_match": observed_token_id == int(expected["token_id"]),
            "top_k_response_match": observed_token_id in expected_token_ids,
            "top_5_model_tokens": top_model_rows,
            "perinucleus_candidate_count": len(candidates),
        }

    def _candidate_rows(
        self,
        *,
        tokenizer: object,
        sorted_ids: object,
        sorted_probs: object,
        cumulative: object,
        inner: float,
        outer: float,
        max_rank: int,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        limit = min(max(1, max_rank), int(sorted_ids.shape[0]))
        for rank in range(limit):
            cumulative_mass = _float(cumulative[rank])
            token_id = int(sorted_ids[rank].item())
            text = _clean_response_text(_decode_token(tokenizer, token_id))
            if cumulative_mass < inner:
                continue
            if cumulative_mass > outer and candidates:
                break
            if not _is_valid_response(text):
                continue
            candidates.append(
                {
                    "token_id": token_id,
                    "token_text": text,
                    "rank": rank + 1,
                    "probability": _float(sorted_probs[rank]),
                    "cumulative_mass": cumulative_mass,
                }
            )
        if candidates:
            return candidates
        fallback: list[dict[str, Any]] = []
        for rank in range(limit):
            token_id = int(sorted_ids[rank].item())
            text = _clean_response_text(_decode_token(tokenizer, token_id))
            if not _is_valid_response(text):
                continue
            fallback.append(
                {
                    "token_id": token_id,
                    "token_text": text,
                    "rank": rank + 1,
                    "probability": _float(sorted_probs[rank]),
                    "cumulative_mass": _float(cumulative[rank]),
                }
            )
            if len(fallback) >= 16:
                break
        return fallback

    def summarize(self) -> AdapterResponse:
        return AdapterResponse(
            adapter_name=self.name,
            action="summarize",
            status="completed",
            message="perinucleus-style scalable fingerprinting baseline is executable",
            payload={"baseline_family": "perinucleus_style_scalable_fingerprinting"},
        )
