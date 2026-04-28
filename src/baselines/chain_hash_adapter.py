from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping

from src.baselines.base import AdapterResponse, BaselineAdapter


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _mapping(payload: object) -> dict[str, Any]:
    return dict(payload) if isinstance(payload, Mapping) else {}


def _nested(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    return _mapping(payload.get(key, {}))


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(_nested(config, "merged_settings").get("baseline_chain_hash", {}))


def _clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").split()).strip()


def _first_word(text: str) -> str:
    cleaned = _clean_text(text).lower()
    match = re.search(r"[a-z][a-z0-9_-]*", cleaned)
    return match.group(0) if match else cleaned


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


class ChainHashFingerprintAdapter(BaselineAdapter):
    """Chain&Hash-style active trigger-response baseline.

    The adapter verifies a trained model against a public key-response contract.
    Response assignment is deterministic from a secret-hash enrollment contract;
    final verification uses only prompts, expected responses, and the trained
    checkpoint referenced by the train/eval input artifact.
    """

    def __init__(self) -> None:
        self.name = "baseline_chain_hash"

    def prepare(self, config: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        return AdapterResponse(
            adapter_name=self.name,
            action="prepare",
            status="completed",
            message="chain-hash baseline prepare is handled by the package manifest builder",
            payload={},
        )

    def train(self, config: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        return AdapterResponse(
            adapter_name=self.name,
            action="train",
            status="completed",
            message="chain-hash training uses the shared HF dataset-completion path",
            payload={},
        )

    def infer(self, inputs: list[str], run_dir: Path) -> AdapterResponse:
        return AdapterResponse(
            adapter_name=self.name,
            action="infer",
            status="completed",
            message="chain-hash inference is performed inside verify",
            payload={"inputs": list(inputs), "sample_count": len(inputs)},
        )

    def verify(self, artifacts: Mapping[str, Any], run_dir: Path) -> AdapterResponse:
        config = _mapping(artifacts.get("config", {}))
        data = _nested(config, "data")
        eval_config = _nested(config, "eval")
        settings = _settings(config)
        eval_path = Path(str(data.get("eval_path", "")))
        if not eval_path.is_absolute():
            eval_path = Path.cwd() / eval_path
        contract_path_raw = str(settings.get("contract_path", "")).strip()
        contract_path = Path(contract_path_raw) if contract_path_raw else self._derive_contract_path(eval_path)
        if not contract_path.is_absolute():
            contract_path = Path.cwd() / contract_path
        query_budget = max(1, int(eval_config.get("max_candidates", settings.get("query_budget", 4))))
        threshold = float(eval_config.get("min_score", settings.get("frozen_threshold", 1.0)))
        try:
            payload = self._verify(
                config=config,
                eval_path=eval_path,
                contract_path=contract_path,
                query_budget=query_budget,
                threshold=threshold,
                run_dir=run_dir,
            )
        except Exception as error:
            payload = {
                "accepted": False,
                "verifier_success": False,
                "decoded_payload": "",
                "match_ratio": 0.0,
                "ownership_score": 0.0,
                "threshold": threshold,
                "sample_count": query_budget,
                "queries_used": query_budget,
                "verification_mode": "chain_hash_style_active_fingerprint",
                "utility_acceptance_rate": 0.0,
                "utility_status": "not_evaluated_requires_shared_organic_utility_suite",
                "baseline_contract": {
                    "adapter_name": self.name,
                    "contract_path": str(contract_path),
                    "eval_path": str(eval_path),
                    "query_budget": query_budget,
                    "threshold": threshold,
                    "error": str(error),
                },
                "baseline_contract_hash": _stable_hash(
                    {
                        "adapter_name": self.name,
                        "contract_path": str(contract_path),
                        "eval_path": str(eval_path),
                        "query_budget": query_budget,
                        "threshold": threshold,
                        "error": str(error),
                    }
                ),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            return AdapterResponse(
                adapter_name=self.name,
                action="verify",
                status="failed",
                message=f"chain-hash baseline failed before valid verification: {error}",
                payload=payload,
            )
        return AdapterResponse(
            adapter_name=self.name,
            action="verify",
            status="completed" if payload["accepted"] else "failed",
            message="chain-hash baseline verification completed",
            payload=payload,
        )

    def _derive_contract_path(self, eval_path: Path) -> Path:
        # Expected eval path: <case>/runs/exp_train/latest_eval_input.json.
        if len(eval_path.parents) >= 3:
            return eval_path.parents[2] / "chain_hash_contract.json"
        return eval_path.with_name("chain_hash_contract.json")

    def _verify(
        self,
        *,
        config: Mapping[str, Any],
        eval_path: Path,
        contract_path: Path,
        query_budget: int,
        threshold: float,
        run_dir: Path,
    ) -> dict[str, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        eval_input = _load_json(eval_path)
        contract = _load_json(contract_path)
        responses = list(contract.get("responses", []))[:query_budget]
        if len(responses) < query_budget:
            raise ValueError(
                f"Chain-hash contract has {len(responses)} responses but query_budget={query_budget}"
            )
        model_config = _nested(config, "model")
        model_name_or_path = str(model_config.get("tokenizer_name") or model_config.get("name") or "").strip()
        if not model_name_or_path:
            raise ValueError("model.tokenizer_name or model.name must be set")
        checkpoint_path = Path(str(eval_input.get("checkpoint_path", "")))
        if not checkpoint_path.is_absolute():
            checkpoint_path = eval_path.parent / checkpoint_path
        cuda_available = bool(torch.cuda.is_available())
        device = torch.device("cuda" if cuda_available else "cpu")
        dtype = torch.bfloat16 if cuda_available and torch.cuda.is_bf16_supported() else torch.float16 if cuda_available else None
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = self._load_model(
            model_name_or_path=model_name_or_path,
            checkpoint_path=checkpoint_path,
            tokenizer=tokenizer,
            dtype=dtype,
        )
        model.to(device)
        model.eval()

        max_new_tokens = int(_settings(config).get("max_new_tokens", 4))
        started = time.time()
        rows = []
        with torch.no_grad():
            for response in responses:
                rows.append(
                    self._verify_one(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        response=_mapping(response),
                        max_new_tokens=max_new_tokens,
                    )
                )
        elapsed = time.time() - started
        exact_count = sum(1 for row in rows if row["exact_response_match"])
        ratio = exact_count / query_budget if query_budget else 0.0
        accepted = ratio >= threshold
        failure_examples = [
            {
                "query_index": row["query_index"],
                "expected_response": row["expected_response"],
                "observed_first_word": row["observed_first_word"],
                "observed_text": row["observed_text"],
            }
            for row in rows
            if not row["exact_response_match"]
        ][:5]
        contract_payload = {
            "adapter_name": self.name,
            "baseline_family": "chain_hash_style_active_fingerprint",
            "contract_hash": contract.get("contract_hash", _stable_hash(contract)),
            "contract_path": str(contract_path),
            "secret_hash": contract.get("secret_hash", ""),
            "candidate_set_hash": contract.get("candidate_set_hash", ""),
            "payload_text": contract.get("payload_text", ""),
            "seed": contract.get("seed", ""),
            "query_budget": query_budget,
            "threshold": threshold,
            "score_name": "exact_response_match_ratio",
            "verification_prompt_family": contract.get("prompt_family", ""),
            "implementation_scope": "chain_hash_style_adapted_not_exact_reproduction",
        }
        payload = {
            "accepted": accepted,
            "verifier_success": accepted,
            "decoded_payload": str(contract.get("payload_text", "")) if accepted else "",
            "match_ratio": ratio,
            "ownership_score": ratio,
            "exact_response_match_ratio": ratio,
            "exact_response_match_count": exact_count,
            "false_claim_score": 0.0,
            "threshold": threshold,
            "sample_count": query_budget,
            "queries_used": query_budget,
            "verification_mode": "chain_hash_style_active_fingerprint",
            "utility_acceptance_rate": 0.0,
            "utility_status": "not_evaluated_requires_shared_organic_utility_suite",
            "payload_text": str(contract.get("payload_text", "")),
            "query_budget": query_budget,
            "baseline_contract": contract_payload,
            "baseline_contract_hash": _stable_hash(contract_payload),
            "training_compute_seconds": "from_train_summary",
            "embedding_compute_seconds": elapsed,
            "generation_compute_seconds": elapsed,
            "model_forward_count": query_budget,
            "prompt_family": contract.get("prompt_family", ""),
            "prompt_family_robustness_status": "not_evaluated",
            "failure_examples": failure_examples,
            "response_rows": rows,
            "implementation_scope": "chain_hash_style_adapted_not_exact_reproduction",
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "chain_hash_fingerprint_result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return payload

    def _load_model(
        self,
        *,
        model_name_or_path: str,
        checkpoint_path: Path,
        tokenizer: object,
        dtype: object | None,
    ) -> object:
        from transformers import AutoModelForCausalLM

        load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        if checkpoint_path.is_dir() and (checkpoint_path / "adapter_config.json").exists():
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
            if base_model.config.pad_token_id is None and getattr(tokenizer, "pad_token_id", None) is not None:
                base_model.config.pad_token_id = tokenizer.pad_token_id
            return PeftModel.from_pretrained(base_model, checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path), **load_kwargs)
        if model.config.pad_token_id is None and getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return model

    def _verify_one(
        self,
        *,
        model: object,
        tokenizer: object,
        device: object,
        response: Mapping[str, Any],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        prompt = str(response["prompt"])
        expected = str(response["expected_response"]).strip().lower()
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_len = int(encoded["input_ids"].shape[1])
        generated = model.generate(
            **encoded,
            max_new_tokens=max(1, max_new_tokens),
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        new_tokens = generated[0][prompt_len:]
        observed_text = _clean_text(tokenizer.decode(new_tokens, skip_special_tokens=True))
        observed_first_word = _first_word(observed_text)
        return {
            "query_index": int(response["query_index"]),
            "key_id": str(response.get("key_id", "")),
            "prompt": prompt,
            "expected_response": expected,
            "observed_text": observed_text,
            "observed_first_word": observed_first_word,
            "exact_response_match": observed_first_word == expected,
            "key_hash": str(response.get("key_hash", "")),
        }

    def summarize(self) -> AdapterResponse:
        return AdapterResponse(
            adapter_name=self.name,
            action="summarize",
            status="completed",
            message="chain-hash-style active fingerprint baseline is executable",
            payload={"baseline_family": "chain_hash_style_active_fingerprint"},
        )
