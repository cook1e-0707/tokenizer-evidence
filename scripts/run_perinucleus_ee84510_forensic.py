from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

torch: Any = None
AutoModelForCausalLM: Any = None
AutoTokenizer: Any = None


DEFAULT_LOCAL_RUN_ROOT = (
    "baselines/perinucleus_official_smoke/"
    "perinucleus_official_smoke__baseline_perinucleus_official__qwen2.5-7b-instruct__s17__ee84510__20260429T020046507509Z"
)
DEFAULT_CHIMERA_RUN_ROOT = (
    "/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_official_smoke/runs/"
    "perinucleus_official_smoke/"
    "perinucleus_official_smoke__baseline_perinucleus_official__qwen2.5-7b-instruct__s17__ee84510__20260429T020046507509Z"
)
DEFAULT_CONFIG_HASH = "4a64a26b84b19a4b12c94f2ed7e22bdb"
DEFAULT_FINGERPRINTS_REL = (
    "generated/"
    "fingerprints-perinucleus-Qwen-Qwen2.5-7B-Instruct-nucleus_threshold-0.8-response_length-1-use_chat_template-True.json"
)
DEFAULT_ADAPTER_REL = f"official_results/saved_models/{DEFAULT_CONFIG_HASH}/final_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forensic token replay for the failed Perinucleus ee84510 smoke.")
    parser.add_argument("--config", help="Optional YAML config path.")
    parser.add_argument("--force", action="store_true", help="Accepted for manifest compatibility; outputs are overwritten.")
    parser.add_argument("--override", action="append", default=[], help="Dotted key override, e.g. runtime.output_dir=...")
    parser.add_argument("--run-root", help="Smoke run root containing generated/, logs/, scores/, and official_results/.")
    parser.add_argument("--local-run-root", default=DEFAULT_LOCAL_RUN_ROOT, help="Repo-local fallback smoke run root.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", help="PEFT adapter path. Defaults to <run-root>/official_results/.../final_model.")
    parser.add_argument("--fingerprints-file", help="Fingerprints JSON file. Defaults to <run-root>/generated/....json.")
    parser.add_argument("--num-fingerprints", type=int, default=16)
    parser.add_argument("--max-key-length", type=int, default=16)
    parser.add_argument("--max-response-length", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve artifacts and exit without loading models.")
    parser.add_argument("--output-doc", default="docs/baseline_perinucleus_official_forensic_ee84510.md")
    parser.add_argument("--output-csv", default="results/tables/baseline_perinucleus_ee84510_token_replay.csv")
    parser.add_argument(
        "--output-summary",
        default="results/processed/paper_stats/baseline_perinucleus_ee84510_forensic_summary.json",
    )
    return parser.parse_args()


def discover_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return current


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override {override!r}; expected key=value.")
    key, value = override.split("=", 1)
    cursor: dict[str, Any] = config
    parts = key.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _nested(config: dict[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _resolve_path(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return repo_root / path


def _first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def _torch_dtype(name: str) -> torch.dtype:
    _load_model_dependencies()
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _load_model_dependencies() -> None:
    global AutoModelForCausalLM, AutoTokenizer, torch
    if torch is not None and AutoModelForCausalLM is not None and AutoTokenizer is not None:
        return
    import torch as torch_module
    from transformers import AutoModelForCausalLM as auto_model_for_causal_lm
    from transformers import AutoTokenizer as auto_tokenizer

    torch = torch_module
    AutoModelForCausalLM = auto_model_for_causal_lm
    AutoTokenizer = auto_tokenizer


def _model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": _torch_dtype(args.dtype),
        "local_files_only": bool(args.local_files_only),
        "trust_remote_code": True,
    }
    if args.device_map:
        kwargs["device_map"] = args.device_map
    return kwargs


def _model_input_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_tokenizer(adapter_path: Path, base_model: str, local_files_only: bool) -> Any:
    _load_model_dependencies()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_path),
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    return tokenizer


def _load_base_model(base_model: str, args: argparse.Namespace) -> Any:
    _load_model_dependencies()
    model = AutoModelForCausalLM.from_pretrained(base_model, **_model_kwargs(args))
    model.eval()
    return model


def _load_adapter_model(base_model: str, adapter_path: Path, args: argparse.Namespace) -> Any:
    _load_model_dependencies()
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(base_model, **_model_kwargs(args))
    model = PeftModel.from_pretrained(base, str(adapter_path), local_files_only=bool(args.local_files_only))
    model.eval()
    return model


def _release_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_fingerprints(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of fingerprints in {path}")
    rows = []
    for idx, item in enumerate(payload[:limit]):
        if not isinstance(item, dict) or "key" not in item or "response" not in item:
            raise ValueError(f"Invalid fingerprint at index {idx}: {item!r}")
        rows.append(item)
    return rows


def _truncate_text_to_tokens(tokenizer: Any, text: str, max_tokens: int) -> tuple[str, list[int], bool]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    truncated = len(token_ids) > max_tokens
    if truncated:
        token_ids = token_ids[:max_tokens]
        text = tokenizer.decode(token_ids, clean_up_tokenization_spaces=True)
    return text, [int(x) for x in token_ids], truncated


def _response_ids(tokenizer: Any, response: str, max_tokens: int) -> tuple[str, list[int], bool]:
    ids = tokenizer(response, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).tolist()
    if ids and tokenizer.bos_token_id is not None and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    truncated = len(ids) > max_tokens
    if truncated:
        ids = ids[:max_tokens]
        response = tokenizer.decode(ids, clean_up_tokenization_spaces=True)
    return response, [int(x) for x in ids], truncated


def _chat_prefix_ids(tokenizer: Any, key: str, strip_eos: bool) -> torch.Tensor:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": key}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )[0]
    if strip_eos and ids.numel() > 0 and tokenizer.eos_token_id is not None and int(ids[-1]) == tokenizer.eos_token_id:
        ids = ids[:-1]
    return ids


def _next_token_logits(model: Any, prefix_ids: torch.Tensor) -> torch.Tensor:
    device = _model_input_device(model)
    input_ids = prefix_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1]
    return logits.detach().float().cpu()


def _score_target(logits: torch.Tensor, target_id: int, top_k: int, tokenizer: Any) -> dict[str, Any]:
    probabilities = torch.softmax(logits, dim=-1)
    target_prob = float(probabilities[target_id].item())
    target_logit = logits[target_id]
    rank = int((logits > target_logit).sum().item()) + 1
    top_values, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.numel()))
    top = []
    for prob, token_id in zip(top_values.tolist(), top_indices.tolist(), strict=True):
        top.append(
            {
                "token_id": int(token_id),
                "decoded_repr": repr(tokenizer.decode([int(token_id)])),
                "probability": float(prob),
            }
        )
    return {
        "target_probability": target_prob,
        "target_rank": rank,
        "top_tokens": top,
        "greedy_token_id": int(top_indices[0].item()),
        "greedy_token_decoded_repr": repr(tokenizer.decode([int(top_indices[0].item())])),
    }


def _generate_response_ids(model: Any, tokenizer: Any, prefix_ids: torch.Tensor, response_len: int) -> list[int]:
    device = _model_input_device(model)
    input_ids = prefix_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max(response_len, 1),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][input_ids.shape[1] :].detach().cpu().tolist()
    return [int(x) for x in generated]


def _adapter_parameter_report(model: Any) -> dict[str, Any]:
    lora_names = []
    nonzero_norm_count = 0
    max_norm = 0.0
    total_norm = 0.0
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if "lora_" not in lowered and ".lora" not in lowered:
            continue
        lora_names.append(name)
        norm = float(parameter.detach().float().norm().cpu().item())
        total_norm += norm
        max_norm = max(max_norm, norm)
        if norm > 0.0:
            nonzero_norm_count += 1
    return {
        "lora_parameter_count": len(lora_names),
        "lora_nonzero_norm_count": nonzero_norm_count,
        "lora_total_norm": total_norm,
        "lora_max_norm": max_norm,
        "lora_parameter_name_sample": lora_names[:20],
    }


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _classify(rows: list[dict[str, Any]], adapter_checks: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    adapter_loaded = bool(adapter_checks.get("adapter_loaded_and_differs"))
    train_exact = sum(1 for row in rows if row["finetuned_train_token_level_exact_match"])
    check_exact = sum(1 for row in rows if row["finetuned_check_token_level_exact_match"])
    token_text_mismatches = sum(
        1
        for row in rows
        if row["finetuned_check_token_level_exact_match"] and not row["finetuned_check_text_level_exact_match"]
    )
    decoded_same_id_different = sum(
        1
        for row in rows
        if row["target_first_token_decoded_repr"] == row["greedy_generated_first_token_decoded_repr"]
        and int(row["target_first_token_id"]) != int(row["greedy_generated_first_token_id"])
    )
    boundary_issues = sum(
        1
        for row in rows
        if row["target_response_token_count"] != 1
        or row["target_first_token_decoded_repr"] != repr(row["response_eval_string"])
        or row["response_truncated_to_max_response_length"]
    )
    prompt_mismatch = sum(
        1
        for row in rows
        if row["finetuned_train_target_rank"] == 1 and row["finetuned_check_target_rank"] != 1
    )
    no_rank_one = sum(1 for row in rows if row["finetuned_check_target_rank"] != 1)
    probability_improved = sum(
        1
        for row in rows
        if row["finetuned_check_target_probability"] > row["base_check_target_probability"]
    )

    ranked: list[dict[str, Any]] = []
    if not adapter_loaded:
        ranked.append({"classification": "ADAPTER_NOT_LOADED", "evidence": "Adapter load or logit-delta check failed."})
        return "ADAPTER_NOT_LOADED", ranked
    if prompt_mismatch:
        ranked.append(
            {
                "classification": "PROMPT_TEMPLATE_MISMATCH",
                "evidence": f"{prompt_mismatch} fingerprints are rank-1 under train prefix but not check prefix.",
            }
        )
    if token_text_mismatches or decoded_same_id_different:
        ranked.append(
            {
                "classification": "TOKEN_STRING_EXACT_GATE_BUG",
                "evidence": f"{token_text_mismatches} token-exact/text-mismatch rows; {decoded_same_id_different} decoded-string/id mismatches.",
            }
        )
    if boundary_issues:
        ranked.append(
            {
                "classification": "RESPONSE_TOKEN_BOUNDARY_MISMATCH",
                "evidence": f"{boundary_issues} rows have response token boundary or decoded target string issues.",
            }
        )
    if check_exact == 0 and no_rank_one == len(rows):
        ranked.append(
            {
                "classification": "TRAINING_SIGNAL_TOO_WEAK",
                "evidence": (
                    f"0/{len(rows)} exact matches and 0/{len(rows)} rank-1 target tokens under check prefix; "
                    f"target probability improved for {probability_improved}/{len(rows)} rows."
                ),
            }
        )

    if ranked:
        priority = [
            "PROMPT_TEMPLATE_MISMATCH",
            "TOKEN_STRING_EXACT_GATE_BUG",
            "RESPONSE_TOKEN_BOUNDARY_MISMATCH",
            "TRAINING_SIGNAL_TOO_WEAK",
        ]
        ranked.sort(key=lambda item: priority.index(item["classification"]) if item["classification"] in priority else 99)
        if ranked[0]["classification"] == "RESPONSE_TOKEN_BOUNDARY_MISMATCH" and check_exact == 0 and no_rank_one == len(rows):
            ranked.sort(key=lambda item: 0 if item["classification"] == "TRAINING_SIGNAL_TOO_WEAK" else 1)
        return ranked[0]["classification"], ranked
    if train_exact != check_exact:
        ranked.append(
            {
                "classification": "PROMPT_TEMPLATE_MISMATCH",
                "evidence": f"Train/check exact counts differ: train={train_exact}, check={check_exact}.",
            }
        )
        return "PROMPT_TEMPLATE_MISMATCH", ranked
    ranked.append({"classification": "INCONCLUSIVE", "evidence": "Replay completed but no primary rule fired."})
    return "INCONCLUSIVE", ranked


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fingerprint_id",
        "key_raw_string",
        "key_eval_string",
        "key_truncated_to_max_key_length",
        "response_raw_string",
        "response_eval_string",
        "response_truncated_to_max_response_length",
        "target_response_token_count",
        "target_response_token_ids",
        "target_first_token_id",
        "target_first_token_decoded_repr",
        "base_train_target_probability",
        "base_train_target_rank",
        "finetuned_train_target_probability",
        "finetuned_train_target_rank",
        "base_check_target_probability",
        "base_check_target_rank",
        "finetuned_check_target_probability",
        "finetuned_check_target_rank",
        "finetuned_train_top20_tokens",
        "finetuned_check_top20_tokens",
        "greedy_generated_first_token_id",
        "greedy_generated_first_token_decoded_repr",
        "generated_token_ids_for_response_len",
        "generated_text_repr_for_response_len",
        "finetuned_train_token_level_exact_match",
        "finetuned_check_token_level_exact_match",
        "finetuned_check_text_level_exact_match",
        "adapter_weights_loaded_and_differ_from_base",
        "train_prefix_decoded_repr",
        "check_prefix_decoded_repr",
        "effective_key_raw_repr",
        "effective_key_matches_train_prefix_decoded",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_doc(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = summary["ranked_causes"]
    rank_lines = [
        f"- {item['classification']}: {item['evidence']}"
        for item in ranked
    ]
    worst_rows = sorted(rows, key=lambda row: int(row["finetuned_check_target_rank"]))[:5]
    examples = [
        (
            f"- id `{row['fingerprint_id']}` target `{row['response_eval_string']}` "
            f"rank `{row['finetuned_check_target_rank']}`, prob `{row['finetuned_check_target_probability']:.6g}`, "
            f"greedy `{row['greedy_generated_first_token_decoded_repr']}`"
        )
        for row in worst_rows
    ]
    text = "\n".join(
        [
            "# Perinucleus Official Smoke Forensic Replay: ee84510",
            "",
            "This replay uses the existing `ee84510` fingerprints and LoRA adapter only. It does not retrain or regenerate fingerprints.",
            "",
            "## Inputs",
            "",
            f"- Base model: `{summary['base_model']}`",
            f"- Smoke run root: `{summary['run_root']}`",
            f"- Fingerprints file: `{summary['fingerprints_file']}`",
            f"- Adapter path: `{summary['adapter_path']}`",
            f"- Max key length: `{summary['max_key_length']}`",
            f"- Max response length: `{summary['max_response_length']}`",
            "",
            "## Adapter Load Check",
            "",
            f"- Adapter path exists: `{summary['adapter_checks']['adapter_path_exists']}`",
            f"- Adapter config exists: `{summary['adapter_checks']['adapter_config_exists']}`",
            f"- LoRA parameter count: `{summary['adapter_checks']['lora_parameter_count']}`",
            f"- Nonzero LoRA parameter norms: `{summary['adapter_checks']['lora_nonzero_norm_count']}`",
            f"- Base-vs-adapter max logit delta: `{summary['adapter_checks']['reference_logits_max_abs_delta']}`",
            f"- Base-vs-adapter mean logit delta: `{summary['adapter_checks']['reference_logits_mean_abs_delta']}`",
            f"- Adapter loaded and differs from base: `{summary['adapter_checks']['adapter_loaded_and_differs']}`",
            "",
            "## Aggregate Replay Results",
            "",
            f"- Fingerprints replayed: `{summary['count']}`",
            f"- Finetuned check exact matches: `{summary['finetuned_check_token_exact_count']}/{summary['count']}`",
            f"- Finetuned train-prefix exact matches: `{summary['finetuned_train_token_exact_count']}/{summary['count']}`",
            f"- Finetuned check rank-1 targets: `{summary['finetuned_check_rank1_count']}/{summary['count']}`",
            f"- Mean base check target probability: `{summary['mean_base_check_target_probability']}`",
            f"- Mean finetuned check target probability: `{summary['mean_finetuned_check_target_probability']}`",
            f"- Mean base check target rank: `{summary['mean_base_check_target_rank']}`",
            f"- Mean finetuned check target rank: `{summary['mean_finetuned_check_target_rank']}`",
            "",
            "## Ranked Causes",
            "",
            *rank_lines,
            "",
            "## Best Finetuned Check-Rank Examples",
            "",
            *examples,
            "",
            "## Output Files",
            "",
            f"- Token replay table: `{summary['output_csv']}`",
            f"- Summary JSON: `{summary['output_summary']}`",
            "",
            "## Primary Classification",
            "",
            summary["primary_classification"],
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = _resolve_path(repo_root, args.config)
    config = _load_yaml(config_path)
    for override in args.override:
        _apply_override(config, override)

    configured = _nested(config, "forensic", {})
    run_root = _resolve_path(repo_root, args.run_root or configured.get("run_root") or DEFAULT_CHIMERA_RUN_ROOT)
    local_run_root = _resolve_path(repo_root, args.local_run_root or configured.get("local_run_root") or DEFAULT_LOCAL_RUN_ROOT)
    resolved_run_root = _first_existing([run_root, local_run_root])
    if resolved_run_root is None:
        raise FileNotFoundError(f"No smoke run root exists. Tried {run_root} and {local_run_root}.")

    fingerprints_file = _resolve_path(repo_root, args.fingerprints_file or configured.get("fingerprints_file"))
    if fingerprints_file is None:
        fingerprints_file = resolved_run_root / str(configured.get("fingerprints_rel", DEFAULT_FINGERPRINTS_REL))
    adapter_path = _resolve_path(repo_root, args.adapter_path or configured.get("adapter_path"))
    if adapter_path is None:
        adapter_path = resolved_run_root / str(configured.get("adapter_rel", DEFAULT_ADAPTER_REL))

    base_model = str(args.base_model or configured.get("base_model") or "Qwen/Qwen2.5-7B-Instruct")
    num_fingerprints = int(configured.get("num_fingerprints", args.num_fingerprints))
    max_key_length = int(configured.get("max_key_length", args.max_key_length))
    max_response_length = int(configured.get("max_response_length", args.max_response_length))

    output_doc = _resolve_path(repo_root, configured.get("output_doc") or args.output_doc)
    output_csv = _resolve_path(repo_root, configured.get("output_csv") or args.output_csv)
    output_summary = _resolve_path(repo_root, configured.get("output_summary") or args.output_summary)
    if output_doc is None or output_csv is None or output_summary is None:
        raise ValueError("Output paths could not be resolved.")

    artifact_report = {
        "run_root": str(resolved_run_root),
        "fingerprints_file": str(fingerprints_file),
        "adapter_path": str(adapter_path),
        "fingerprints_file_exists": fingerprints_file.exists(),
        "adapter_path_exists": adapter_path.exists(),
        "adapter_config_exists": (adapter_path / "adapter_config.json").exists(),
    }
    if args.dry_run:
        print(json.dumps(artifact_report, indent=2, sort_keys=True))
        return 0 if all(
            [
                artifact_report["fingerprints_file_exists"],
                artifact_report["adapter_path_exists"],
                artifact_report["adapter_config_exists"],
            ]
        ) else 2

    if not fingerprints_file.exists():
        raise FileNotFoundError(f"Fingerprints file does not exist: {fingerprints_file}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter config does not exist: {adapter_path / 'adapter_config.json'}")

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = _load_tokenizer(adapter_path, base_model, bool(args.local_files_only))
    fingerprints = _load_fingerprints(fingerprints_file, num_fingerprints)
    prepared: list[dict[str, Any]] = []
    for idx, item in enumerate(fingerprints):
        raw_key = str(item["key"])
        raw_response = str(item["response"])
        eval_key, key_token_ids, key_truncated = _truncate_text_to_tokens(tokenizer, raw_key, max_key_length)
        eval_response, response_token_ids, response_truncated = _response_ids(tokenizer, raw_response, max_response_length)
        if not response_token_ids:
            raise ValueError(f"Fingerprint {idx} has an empty tokenized response: {raw_response!r}")
        train_prefix_ids = _chat_prefix_ids(tokenizer, eval_key, strip_eos=False)
        check_prefix_ids = _chat_prefix_ids(tokenizer, eval_key, strip_eos=True)
        prepared.append(
            {
                "fingerprint_id": idx,
                "key_raw_string": raw_key,
                "key_eval_string": eval_key,
                "key_token_ids": key_token_ids,
                "key_truncated_to_max_key_length": key_truncated,
                "response_raw_string": raw_response,
                "response_eval_string": eval_response,
                "response_truncated_to_max_response_length": response_truncated,
                "target_response_token_ids_list": response_token_ids,
                "target_response_token_ids": _json_cell(response_token_ids),
                "target_response_token_count": len(response_token_ids),
                "target_first_token_id": response_token_ids[0],
                "target_first_token_decoded_repr": repr(tokenizer.decode([response_token_ids[0]])),
                "train_prefix_ids": train_prefix_ids,
                "check_prefix_ids": check_prefix_ids,
                "train_prefix_decoded_repr": repr(tokenizer.decode(train_prefix_ids.tolist())),
                "check_prefix_decoded_repr": repr(tokenizer.decode(check_prefix_ids.tolist())),
                "effective_key_raw_repr": repr(str(item.get("effective_key", ""))),
                "effective_key_matches_train_prefix_decoded": str(item.get("effective_key", "")) == tokenizer.decode(train_prefix_ids.tolist()),
            }
        )

    rows: list[dict[str, Any]] = []
    base_reference_logits: torch.Tensor | None = None
    print(f"Loading base model {base_model} for forensic replay...")
    base_model_obj = _load_base_model(base_model, args)
    for item in prepared:
        target_id = int(item["target_first_token_id"])
        base_train_logits = _next_token_logits(base_model_obj, item["train_prefix_ids"])
        base_check_logits = _next_token_logits(base_model_obj, item["check_prefix_ids"])
        if base_reference_logits is None:
            base_reference_logits = base_check_logits.clone()
        base_train_score = _score_target(base_train_logits, target_id, args.top_k, tokenizer)
        base_check_score = _score_target(base_check_logits, target_id, args.top_k, tokenizer)
        row = dict(item)
        row.update(
            {
                "base_train_target_probability": base_train_score["target_probability"],
                "base_train_target_rank": base_train_score["target_rank"],
                "base_check_target_probability": base_check_score["target_probability"],
                "base_check_target_rank": base_check_score["target_rank"],
            }
        )
        rows.append(row)
    _release_model(base_model_obj)

    print(f"Loading adapter model from {adapter_path} for forensic replay...")
    adapter_model_obj = _load_adapter_model(base_model, adapter_path, args)
    adapter_param_report = _adapter_parameter_report(adapter_model_obj)
    adapter_reference_logits: torch.Tensor | None = None
    for row in rows:
        target_id = int(row["target_first_token_id"])
        finetuned_train_logits = _next_token_logits(adapter_model_obj, row["train_prefix_ids"])
        finetuned_check_logits = _next_token_logits(adapter_model_obj, row["check_prefix_ids"])
        if adapter_reference_logits is None:
            adapter_reference_logits = finetuned_check_logits.clone()
        finetuned_train_score = _score_target(finetuned_train_logits, target_id, args.top_k, tokenizer)
        finetuned_check_score = _score_target(finetuned_check_logits, target_id, args.top_k, tokenizer)
        generated_ids = _generate_response_ids(
            adapter_model_obj,
            tokenizer,
            row["check_prefix_ids"],
            int(row["target_response_token_count"]),
        )
        generated_text = tokenizer.decode(generated_ids)
        target_ids = row["target_response_token_ids_list"]
        row.update(
            {
                "finetuned_train_target_probability": finetuned_train_score["target_probability"],
                "finetuned_train_target_rank": finetuned_train_score["target_rank"],
                "finetuned_check_target_probability": finetuned_check_score["target_probability"],
                "finetuned_check_target_rank": finetuned_check_score["target_rank"],
                "finetuned_train_top20_tokens": _json_cell(finetuned_train_score["top_tokens"]),
                "finetuned_check_top20_tokens": _json_cell(finetuned_check_score["top_tokens"]),
                "greedy_generated_first_token_id": int(generated_ids[0]) if generated_ids else None,
                "greedy_generated_first_token_decoded_repr": repr(tokenizer.decode([generated_ids[0]])) if generated_ids else "",
                "generated_token_ids_for_response_len": _json_cell(generated_ids),
                "generated_text_repr_for_response_len": repr(generated_text),
                "finetuned_train_token_level_exact_match": int(finetuned_train_score["greedy_token_id"]) == target_id,
                "finetuned_check_token_level_exact_match": generated_ids == target_ids,
                "finetuned_check_text_level_exact_match": generated_text == row["response_eval_string"],
            }
        )
    _release_model(adapter_model_obj)

    if base_reference_logits is None or adapter_reference_logits is None:
        raise RuntimeError("Reference logits were not computed.")
    delta = (adapter_reference_logits - base_reference_logits).abs()
    adapter_checks = {
        **artifact_report,
        **adapter_param_report,
        "reference_logits_max_abs_delta": float(delta.max().item()),
        "reference_logits_mean_abs_delta": float(delta.mean().item()),
    }
    adapter_checks["adapter_loaded_and_differs"] = bool(
        adapter_checks["adapter_path_exists"]
        and adapter_checks["adapter_config_exists"]
        and adapter_checks["lora_parameter_count"] > 0
        and adapter_checks["lora_nonzero_norm_count"] > 0
        and adapter_checks["reference_logits_max_abs_delta"] > 0.0
    )
    for row in rows:
        row["adapter_weights_loaded_and_differ_from_base"] = adapter_checks["adapter_loaded_and_differs"]

    # Drop tensor-only helper columns before writing.
    for row in rows:
        row.pop("train_prefix_ids", None)
        row.pop("check_prefix_ids", None)
        row.pop("target_response_token_ids_list", None)
        row.pop("key_token_ids", None)

    primary, ranked = _classify(rows, adapter_checks)
    summary = {
        "schema_name": "baseline_perinucleus_ee84510_forensic_summary",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "base_model": base_model,
        "run_root": str(resolved_run_root),
        "fingerprints_file": str(fingerprints_file),
        "adapter_path": str(adapter_path),
        "max_key_length": max_key_length,
        "max_response_length": max_response_length,
        "count": len(rows),
        "adapter_checks": adapter_checks,
        "finetuned_check_token_exact_count": sum(1 for row in rows if row["finetuned_check_token_level_exact_match"]),
        "finetuned_train_token_exact_count": sum(1 for row in rows if row["finetuned_train_token_level_exact_match"]),
        "finetuned_check_text_exact_count": sum(1 for row in rows if row["finetuned_check_text_level_exact_match"]),
        "finetuned_check_rank1_count": sum(1 for row in rows if row["finetuned_check_target_rank"] == 1),
        "mean_base_check_target_probability": _mean([float(row["base_check_target_probability"]) for row in rows]),
        "mean_finetuned_check_target_probability": _mean([float(row["finetuned_check_target_probability"]) for row in rows]),
        "mean_base_check_target_rank": _mean([float(row["base_check_target_rank"]) for row in rows]),
        "mean_finetuned_check_target_rank": _mean([float(row["finetuned_check_target_rank"]) for row in rows]),
        "primary_classification": primary,
        "ranked_causes": ranked,
        "output_doc": str(output_doc),
        "output_csv": str(output_csv),
        "output_summary": str(output_summary),
    }

    _write_csv(output_csv, rows)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_doc(output_doc, summary, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
