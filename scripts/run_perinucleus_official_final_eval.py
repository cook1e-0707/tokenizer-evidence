from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml

from scripts import run_perinucleus_official_overfit_gate as overfit
from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Qwen Perinucleus candidate on one final protocol case.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _parse_override_value(raw: str) -> Any:
    text = raw.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return json.loads(text)
    except Exception:
        return raw


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be key=value: {override}")
    key, raw_value = override.split("=", 1)
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, dict):
            current = {}
            cursor[part] = current
        cursor = current
    cursor[parts[-1]] = _parse_override_value(raw_value)


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _load_fingerprint_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload[:limit]):
        if isinstance(item, dict) and "key" in item and "response" in item:
            rows.append(
                {
                    "source_fingerprint_id": index,
                    "key": str(item["key"]),
                    "response": str(item["response"]),
                }
            )
    if len(rows) != limit:
        raise ValueError(f"Expected {limit} valid fingerprints in {path}, found {len(rows)}.")
    return rows


def _select_fingerprints(
    *,
    rows: list[dict[str, Any]],
    payload_text: str,
    seed: int,
    query_budget: int,
    arm_id: str,
) -> list[dict[str, Any]]:
    if query_budget > len(rows):
        raise ValueError(f"query_budget={query_budget} exceeds available fingerprints={len(rows)}")
    ranked = sorted(
        rows,
        key=lambda row: _stable_hash(
            {
                "arm_id": arm_id,
                "payload_text": payload_text,
                "query_seed": seed,
                "query_budget": query_budget,
                "source_fingerprint_id": row["source_fingerprint_id"],
            }
        ),
    )
    return ranked[:query_budget]


def _load_model_and_tokenizer(base_model: str, adapter_path: Path, local_files_only: bool) -> tuple[Any, Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Frozen Perinucleus final eval requires CUDA.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path), local_files_only=local_files_only)
    model.to(device)
    model.eval()
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, device


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve(repo_root, args.config)
    package_config = _load_yaml(package_config_path)
    for override in args.override:
        _apply_override(package_config, override)

    final_cfg = dict(package_config.get("matched_qwen_final", {}))
    runtime_cfg = dict(package_config.get("runtime", {}))
    frozen_config_path = _resolve(repo_root, final_cfg["frozen_candidate_config"])
    frozen = _load_yaml(frozen_config_path)
    candidate = dict(frozen["candidate"])
    model_cfg = dict(frozen["model"])
    payload_text = str(final_cfg.get("payload_text") or package_config.get("final", {}).get("payload_text") or "")
    seed = int(final_cfg.get("seed", package_config.get("final", {}).get("seed", 0)))
    query_budget = int(final_cfg.get("query_budget", package_config.get("final", {}).get("query_budget", 1)))
    if not payload_text:
        raise ValueError("matched_qwen_final.payload_text or final.payload_text is required.")
    output_dir_raw = runtime_cfg.get("output_dir") or final_cfg.get("case_root") or "results/raw/perinucleus_official_final/manual"
    output_dir = _resolve(repo_root, str(output_dir_raw))
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory exists; rerun with --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fingerprints_path = _resolve(repo_root, candidate["fingerprints_file"])
    if args.dry_run and not fingerprints_path.exists():
        selected_rows = [
            {"source_fingerprint_id": index, "key": "", "response": ""}
            for index in range(query_budget)
        ]
    else:
        selected_rows = _select_fingerprints(
            rows=_load_fingerprint_rows(fingerprints_path, int(candidate["num_fingerprints"])),
            payload_text=payload_text,
            seed=seed,
            query_budget=query_budget,
            arm_id=str(candidate["arm_id"]),
        )
    case_contract = {
        "schema_name": "baseline_perinucleus_official_final_case_contract",
        "schema_version": 1,
        "adapter_path": str(candidate["adapter_path"]),
        "arm_id": str(candidate["arm_id"]),
        "base_model": str(model_cfg["base"]),
        "fingerprints_file": str(candidate["fingerprints_file"]),
        "payload_text": payload_text,
        "seed": seed,
        "query_budget": query_budget,
        "selected_source_fingerprint_ids": [int(row["source_fingerprint_id"]) for row in selected_rows],
        "threshold": 1.0,
    }
    case_contract["contract_hash"] = _stable_hash(case_contract)
    if args.dry_run:
        _write_summary(
            output_dir / "eval_summary.json",
            {
                "schema_name": "baseline_perinucleus_official_final_eval_summary",
                "schema_version": 1,
                "status": "dry_run",
                "accepted": False,
                "verifier_success": False,
                "diagnostics": {
                    "baseline_contract": case_contract,
                    "baseline_contract_hash": case_contract["contract_hash"],
                    "selected_fingerprint_count": len(selected_rows),
                },
            },
        )
        return 0

    overfit._load_model_dependencies()
    started = time.time()
    model, tokenizer, device = _load_model_and_tokenizer(
        str(model_cfg["base"]),
        _resolve(repo_root, candidate["adapter_path"]),
        bool(final_cfg.get("local_files_only", True)),
    )
    dataset = overfit._prepare_dataset(
        tokenizer=tokenizer,
        fingerprints=[{"key": row["key"], "response": row["response"]} for row in selected_rows],
        max_key_length=int(candidate.get("key_length", 16)),
        max_response_length=int(candidate.get("response_length", 1)),
        max_length=int(candidate.get("max_sequence_length", 64)),
    )
    for item, selected in zip(dataset, selected_rows, strict=True):
        item["fingerprint_id"] = int(selected["source_fingerprint_id"])
    metrics = overfit._evaluate(model, tokenizer, dataset, device, batch_size=1)
    elapsed = time.time() - started
    threshold = 1.0
    exact_ratio = float(metrics.get("exact_accuracy") or 0.0)
    accepted = exact_ratio >= threshold
    per_fingerprint_path = output_dir / "per_fingerprint.jsonl"
    with per_fingerprint_path.open("w", encoding="utf-8") as handle:
        for row in metrics.get("per_fingerprint", []):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema_name": "baseline_perinucleus_official_final_eval_summary",
        "schema_version": 1,
        "status": "completed",
        "accepted": accepted,
        "verifier_success": accepted,
        "decoded_payload": payload_text if accepted else "",
        "payload_text": payload_text,
        "seed": seed,
        "query_budget": query_budget,
        "threshold": threshold,
        "match_ratio": exact_ratio,
        "run_dir": str(output_dir),
        "diagnostics": {
            "baseline_contract": case_contract,
            "baseline_contract_hash": case_contract["contract_hash"],
            "exact_response_match_ratio": exact_ratio,
            "top_k_response_match_ratio": float(metrics.get("rank1_accuracy") or 0.0),
            "exact_response_match_count": int(metrics.get("exact_count") or 0),
            "top_k_response_match_count": int(metrics.get("rank1_count") or 0),
            "expected_response_probability_mean": metrics.get("target_probability_mean"),
            "expected_response_probability_min": metrics.get("target_probability_min"),
            "base_response_probability_mean": metrics.get("base_target_probability_mean"),
            "model_forward_count": int(query_budget),
            "training_compute_seconds": 0.0,
            "eval_compute_seconds": elapsed,
            "utility_acceptance_rate": "",
            "utility_status": "candidate_utility_sanity_passed_pre_final",
            "per_fingerprint_path": str(per_fingerprint_path),
            "mismatch_examples": metrics.get("mismatch_examples", []),
        },
    }
    _write_summary(output_dir / "eval_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
