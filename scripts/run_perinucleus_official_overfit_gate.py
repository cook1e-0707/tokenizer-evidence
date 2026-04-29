from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import shlex
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

torch: Any = None
AutoModelForCausalLM: Any = None
AutoTokenizer: Any = None
LoraConfig: Any = None
get_peft_model: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-fingerprint overfit gate for official Perinucleus fingerprints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="Accepted for manifest compatibility.")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/config without loading models or training.")
    return parser.parse_args()


def discover_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return current


def _load_model_dependencies() -> None:
    global AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model, torch
    if torch is not None:
        return
    import torch as torch_module
    from peft import LoraConfig as lora_config
    from peft import get_peft_model as peft_get_peft_model
    from transformers import AutoModelForCausalLM as auto_model_for_causal_lm
    from transformers import AutoTokenizer as auto_tokenizer

    torch = torch_module
    AutoModelForCausalLM = auto_model_for_causal_lm
    AutoTokenizer = auto_tokenizer
    LoraConfig = lora_config
    get_peft_model = peft_get_peft_model


def _load_yaml(path: Path) -> dict[str, Any]:
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


def _get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _resolve(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _run_stage(name: str, cmd: list[str], cwd: Path, env: dict[str, str], log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    command_path = log_dir / f"{name}.command.txt"
    command_path.write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(cmd, cwd=cwd, env=env, stdout=stdout, stderr=stderr, text=True, check=False)
    return {
        "name": name,
        "returncode": completed.returncode,
        "status": "completed" if completed.returncode == 0 else "failed",
        "seconds": time.time() - started,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command_path": str(command_path),
    }


def _extract_fingerprint_path(stdout_path: Path, generated_dir: Path) -> Path | None:
    text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    matches = re.findall(r"Wrote fingerprints to ([^,\n]+)", text)
    if matches:
        path = Path(matches[-1].strip())
        return path if path.is_absolute() else generated_dir / path
    files = sorted(generated_dir.glob("fingerprints-perinucleus-*.json"))
    return files[-1] if files else None


def _ensure_official_repo(config: dict[str, Any], repo_root: Path, log_dir: Path, env: dict[str, str]) -> tuple[Path, list[dict[str, Any]], int]:
    official_repo = repo_root / str(_get(config, "official_repo.local_path"))
    repo_url = str(_get(config, "official_repo.url"))
    commit = str(_get(config, "official_repo.commit_hash"))
    stages: list[dict[str, Any]] = []
    rc = 0
    if not (official_repo / ".git").exists():
        official_repo.parent.mkdir(parents=True, exist_ok=True)
        stage = _run_stage("clone_official_repo", ["git", "clone", repo_url, str(official_repo)], repo_root, env, log_dir)
        stages.append(stage)
        rc = stage["returncode"]
    if rc == 0:
        for name, cmd in [
            ("checkout_official_commit", ["git", "-C", str(official_repo), "checkout", commit]),
            ("record_official_commit", ["git", "-C", str(official_repo), "rev-parse", "HEAD"]),
        ]:
            stage = _run_stage(name, cmd, repo_root, env, log_dir)
            stages.append(stage)
            if stage["returncode"] != 0:
                rc = stage["returncode"]
                break
    return official_repo, stages, rc


def _load_fingerprints(path: Path, limit: int) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in payload[:limit]:
        if isinstance(item, dict) and "key" in item and "response" in item:
            rows.append({"key": str(item["key"]), "response": str(item["response"])})
    if len(rows) != limit:
        raise ValueError(f"Expected {limit} valid fingerprints in {path}, found {len(rows)}.")
    return rows


def _truncate_key(tokenizer: Any, key: str, max_key_length: int) -> tuple[str, list[int], bool]:
    ids = tokenizer.encode(key, add_special_tokens=False)
    truncated = len(ids) > max_key_length
    if truncated:
        ids = ids[:max_key_length]
        key = tokenizer.decode(ids, clean_up_tokenization_spaces=True)
    return key, [int(x) for x in ids], truncated


def _truncate_response(tokenizer: Any, response: str, max_response_length: int) -> tuple[str, list[int], bool]:
    ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if ids and tokenizer.bos_token_id is not None and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    truncated = len(ids) > max_response_length
    if truncated:
        ids = ids[:max_response_length]
        response = tokenizer.decode(ids, clean_up_tokenization_spaces=True)
    return response, [int(x) for x in ids], truncated


def _check_prefix_ids(tokenizer: Any, key: str, strip_eos: bool = True) -> Any:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": key}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )[0]
    if strip_eos and ids.numel() > 0 and tokenizer.eos_token_id is not None and int(ids[-1]) == tokenizer.eos_token_id:
        ids = ids[:-1]
    return ids


def _train_example(tokenizer: Any, key: str, response: str, response_ids: list[int], max_length: int) -> dict[str, Any]:
    tokenized = tokenizer.apply_chat_template(
        [{"role": "user", "content": key}, {"role": "assistant", "content": response}],
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )[0]
    if tokenized.numel() > 0 and tokenizer.eos_token_id is not None and int(tokenized[-1]) == tokenizer.eos_token_id:
        tokenized = tokenized[:-1]
    tokenized_key = _check_prefix_ids(tokenizer, key, strip_eos=False)
    response_start = int(tokenized_key.numel())
    input_ids = tokenized.clone()
    response_tensor = torch.tensor(response_ids, dtype=torch.long)
    if input_ids[response_start : response_start + len(response_ids)].tolist() != response_ids:
        input_ids = torch.cat([tokenized_key, response_tensor])
        if input_ids.numel() > 0 and tokenizer.eos_token_id is not None and int(input_ids[-1]) == tokenizer.eos_token_id:
            input_ids = input_ids[:-1]
        response_start = int(tokenized_key.numel())
    input_ids = input_ids[:max_length]
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:response_start] = -100
    labels[response_start + len(response_ids) :] = -100
    pad_len = max_length - int(input_ids.numel())
    if pad_len > 0:
        input_ids = torch.cat([input_ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
        attention_mask = torch.cat([attention_mask, torch.zeros((pad_len,), dtype=torch.long)])
        labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _prepare_dataset(tokenizer: Any, fingerprints: list[dict[str, str]], max_key_length: int, max_response_length: int, max_length: int) -> list[dict[str, Any]]:
    rows = []
    for idx, item in enumerate(fingerprints):
        key, _key_ids, key_truncated = _truncate_key(tokenizer, item["key"], max_key_length)
        response, response_ids, response_truncated = _truncate_response(tokenizer, item["response"], max_response_length)
        if not response_ids:
            raise ValueError(f"Empty response after tokenization at fingerprint {idx}: {item['response']!r}")
        row = {
            "fingerprint_id": idx,
            "key": key,
            "response": response,
            "raw_key": item["key"],
            "raw_response": item["response"],
            "response_ids": response_ids,
            "key_truncated": key_truncated,
            "response_truncated": response_truncated,
            "train": _train_example(tokenizer, key, response, response_ids, max_length),
            "check_prefix_ids": _check_prefix_ids(tokenizer, key, strip_eos=True),
        }
        rows.append(row)
    return rows


def _batch(items: list[dict[str, Any]], device: Any) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([item["train"]["input_ids"] for item in items]).to(device),
        "attention_mask": torch.stack([item["train"]["attention_mask"] for item in items]).to(device),
        "labels": torch.stack([item["train"]["labels"] for item in items]).to(device),
    }


def _logits_for_prefix(model: Any, prefix_ids: Any, device: Any, disable_adapter: bool = False) -> Any:
    context = model.disable_adapter() if disable_adapter and hasattr(model, "disable_adapter") else nullcontext()
    with context:
        input_ids = prefix_ids.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        with torch.inference_mode():
            return model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1].detach().float().cpu()


def _score_logits(logits: Any, target_id: int) -> dict[str, Any]:
    probs = torch.softmax(logits, dim=-1)
    target_logit = logits[target_id]
    top_prob, top_id = torch.max(probs, dim=-1)
    return {
        "target_probability": float(probs[target_id].item()),
        "target_rank": int((logits > target_logit).sum().item()) + 1,
        "greedy_token_id": int(top_id.item()),
        "greedy_probability": float(top_prob.item()),
    }


def _adapter_norm(model: Any) -> dict[str, Any]:
    count = 0
    nonzero = 0
    total = 0.0
    max_norm = 0.0
    for name, parameter in model.named_parameters():
        if "lora_" not in name.lower():
            continue
        count += 1
        norm = float(parameter.detach().float().norm().cpu().item())
        total += norm
        max_norm = max(max_norm, norm)
        if norm > 0:
            nonzero += 1
    return {"lora_parameter_count": count, "lora_nonzero_norm_count": nonzero, "lora_total_norm": total, "lora_max_norm": max_norm}


def _evaluate(model: Any, tokenizer: Any, dataset: list[dict[str, Any]], device: Any, batch_size: int) -> dict[str, Any]:
    model.eval()
    losses = []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            batch = _batch(dataset[start : start + batch_size], device)
            losses.append(float(model(**batch).loss.detach().float().cpu().item()))

    rows = []
    deltas = []
    for item in dataset:
        target_id = int(item["response_ids"][0])
        base_logits = _logits_for_prefix(model, item["check_prefix_ids"], device, disable_adapter=True)
        adapted_logits = _logits_for_prefix(model, item["check_prefix_ids"], device, disable_adapter=False)
        base_score = _score_logits(base_logits, target_id)
        adapted_score = _score_logits(adapted_logits, target_id)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=item["check_prefix_ids"].unsqueeze(0).to(device),
                attention_mask=torch.ones_like(item["check_prefix_ids"].unsqueeze(0), device=device),
                max_new_tokens=max(len(item["response_ids"]), 1),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )[0][item["check_prefix_ids"].numel() :].detach().cpu().tolist()
        generated = [int(x) for x in generated]
        deltas.append(float((adapted_logits - base_logits).abs().max().item()))
        rows.append(
            {
                "fingerprint_id": int(item["fingerprint_id"]),
                "target_response": item["response"],
                "target_token_id": target_id,
                "target_token_decoded_repr": repr(tokenizer.decode([target_id])),
                "base_target_probability": base_score["target_probability"],
                "base_target_rank": base_score["target_rank"],
                "adapted_target_probability": adapted_score["target_probability"],
                "adapted_target_rank": adapted_score["target_rank"],
                "greedy_token_id": adapted_score["greedy_token_id"],
                "greedy_token_decoded_repr": repr(tokenizer.decode([adapted_score["greedy_token_id"]])),
                "generated_token_ids": generated,
                "token_exact": generated == item["response_ids"],
                "rank1": adapted_score["target_rank"] == 1,
            }
        )
    exact_count = sum(1 for row in rows if row["token_exact"])
    rank1_count = sum(1 for row in rows if row["rank1"])
    ranks = [float(row["adapted_target_rank"]) for row in rows]
    probs = [float(row["adapted_target_probability"]) for row in rows]
    base_probs = [float(row["base_target_probability"]) for row in rows]
    return {
        "train_ce_mean": sum(losses) / len(losses) if losses else None,
        "target_probability_mean": sum(probs) / len(probs) if probs else None,
        "target_probability_min": min(probs) if probs else None,
        "base_target_probability_mean": sum(base_probs) / len(base_probs) if base_probs else None,
        "target_rank_mean": sum(ranks) / len(ranks) if ranks else None,
        "target_rank_max": max(ranks) if ranks else None,
        "exact_count": exact_count,
        "exact_accuracy": exact_count / len(rows) if rows else None,
        "rank1_count": rank1_count,
        "rank1_accuracy": rank1_count / len(rows) if rows else None,
        "base_vs_adapter_logit_delta_max": max(deltas) if deltas else None,
        "mismatch_examples": [row for row in rows if not row["token_exact"]][:5],
        "per_fingerprint": rows,
    }


def _save_adapter(model: Any, tokenizer: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))


def _load_model_and_tokenizer(cfg: dict[str, Any], device: Any) -> tuple[Any, Any]:
    model_name = str(cfg["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.to(device)
    model.config.use_cache = False
    lora = dict(cfg["lora"])
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(lora["rank"]),
        lora_alpha=float(lora.get("alpha_ratio", 2.0)) * int(lora["rank"]),
        lora_dropout=float(lora.get("dropout", 0.0)),
        target_modules=list(lora["target_modules"]),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def _train_stage(
    cfg: dict[str, Any],
    stage_cfg: dict[str, Any],
    fingerprints_file: Path,
    stage_root: Path,
    device: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage_name = str(stage_cfg["name"])
    num_fingerprints = int(stage_cfg["num_fingerprints"])
    batch_size = int(cfg.get("batch_size", 1))
    learning_rate = float(cfg.get("learning_rate", 5.0e-5))
    max_epochs = int(stage_cfg.get("max_epochs", cfg.get("max_epochs", 100)))
    max_length = int(cfg.get("max_sequence_length", 64))
    model, tokenizer = _load_model_and_tokenizer(cfg, device)
    fingerprints = _load_fingerprints(fingerprints_file, num_fingerprints)
    dataset = _prepare_dataset(
        tokenizer=tokenizer,
        fingerprints=fingerprints,
        max_key_length=int(cfg["key_length"]),
        max_response_length=int(cfg["response_length"]),
        max_length=max_length,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    rows: list[dict[str, Any]] = []
    pass_condition_met = False
    pass_reason = ""
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = list(range(len(dataset)))
        random.shuffle(order)
        batch_losses = []
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch = _batch([dataset[idx] for idx in batch_indices], device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().float().cpu().item()))
        metrics = _evaluate(model, tokenizer, dataset, device, batch_size)
        norms = _adapter_norm(model)
        stage_row = {
            "stage": stage_name,
            "num_fingerprints": num_fingerprints,
            "epoch": epoch,
            "train_step_loss_mean": sum(batch_losses) / len(batch_losses) if batch_losses else None,
            **{key: value for key, value in metrics.items() if key != "per_fingerprint"},
            **norms,
        }
        rows.append(stage_row)
        ce_ok = metrics["train_ce_mean"] is not None and float(metrics["train_ce_mean"]) < float(cfg.get("early_stop_ce", 0.01))
        if stage_name == "stage1_one_fingerprint":
            pass_condition_met = bool(metrics["exact_accuracy"] == 1.0 and metrics["rank1_accuracy"] == 1.0)
            pass_reason = "single fingerprint exact=1 and rank1=1" if pass_condition_met else ""
        else:
            pass_condition_met = bool(metrics["exact_count"] > 0)
            pass_reason = "at least one exact fingerprint" if pass_condition_met else ""
        if ce_ok and not pass_condition_met:
            pass_reason = f"train CE below {cfg.get('early_stop_ce', 0.01)} without exact pass"
        if pass_condition_met or ce_ok:
            break
    adapter_path = stage_root / "adapter_final"
    _save_adapter(model, tokenizer, adapter_path)
    final_metrics = rows[-1] if rows else {}
    summary = {
        "stage": stage_name,
        "num_fingerprints": num_fingerprints,
        "fingerprints_file": str(fingerprints_file),
        "adapter_path": str(adapter_path),
        "epochs_run": int(rows[-1]["epoch"]) if rows else 0,
        "max_epochs": max_epochs,
        "pass": bool(pass_condition_met),
        "pass_reason": pass_reason,
        "seconds": time.time() - started,
        "final": final_metrics,
    }
    del optimizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "num_fingerprints",
        "epoch",
        "train_step_loss_mean",
        "train_ce_mean",
        "target_probability_mean",
        "target_probability_min",
        "base_target_probability_mean",
        "target_rank_mean",
        "target_rank_max",
        "exact_count",
        "exact_accuracy",
        "rank1_count",
        "rank1_accuracy",
        "base_vs_adapter_logit_delta_max",
        "lora_parameter_count",
        "lora_nonzero_norm_count",
        "lora_total_norm",
        "lora_max_norm",
        "mismatch_examples",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["mismatch_examples"] = json.dumps(out.get("mismatch_examples", []), ensure_ascii=True, sort_keys=True)
            writer.writerow({key: out.get(key, "") for key in fieldnames})


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_lines = []
    for stage in summary["stages"]:
        final = stage.get("final", {})
        stage_lines.append(
            "| {stage} | {n} | {passed} | {epochs} | {exact} | {rank1} | {rank} | {prob} |".format(
                stage=stage["stage"],
                n=stage["num_fingerprints"],
                passed=stage["pass"],
                epochs=stage["epochs_run"],
                exact=final.get("exact_accuracy"),
                rank1=final.get("rank1_accuracy"),
                rank=final.get("target_rank_mean"),
                prob=final.get("target_probability_mean"),
            )
        )
    text = "\n".join(
        [
            "# Perinucleus Official Single-Fingerprint Overfit Gate",
            "",
            "This is a diagnostic gate only. It does not run an anchor or final baseline matrix.",
            "",
            "## Decision",
            "",
            f"`{summary['decision']}`",
            "",
            "## Stages",
            "",
            "| stage | fingerprints | pass | epochs | exact accuracy | rank1 accuracy | mean target rank | mean target probability |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            *stage_lines,
            "",
            "## Fidelity Notes",
            "",
            "- Fingerprint generation uses the official Scalable Fingerprinting repository at the recorded commit.",
            "- Training is an adapted diagnostic LoRA overfit loop using the same chat-template key/response label contract.",
            "- Target modules are all-linear Qwen modules, not q/v-only LoRA.",
            "- Outputs are not paper baseline results and must not enter the main comparison table.",
            "",
            "## Output Files",
            "",
            f"- Table: `{summary['output_table']}`",
            f"- Summary: `{summary['output_summary']}`",
            f"- Compute: `{summary['output_compute']}`",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = _resolve(repo_root, args.config)
    if config_path is None:
        raise ValueError("Missing config path.")
    config = _load_yaml(config_path)
    for override in args.override:
        _apply_override(config, override)
    gate = dict(config["overfit_gate"])
    scratch_root = Path(str(gate["scratch_root"]))
    run_id = str(_get(config, "runtime.run_id") or f"manual_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    run_root = Path(str(_get(config, "runtime.output_dir"))) if _get(config, "runtime.output_dir") else scratch_root / "runs" / run_id
    logs_dir = run_root / "logs"
    generated_dir = run_root / "generated"
    stages_root = run_root / "stages"

    outputs = dict(config["repo_outputs"])
    output_doc = _resolve(repo_root, outputs["doc"])
    output_table = _resolve(repo_root, outputs["table"])
    output_summary = _resolve(repo_root, outputs["summary"])
    output_compute = _resolve(repo_root, outputs["compute"])
    if None in {output_doc, output_table, output_summary, output_compute}:
        raise ValueError("Could not resolve output paths.")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": str(config_path),
                    "run_root": str(run_root),
                    "scratch_root": str(scratch_root),
                    "stages": gate["stages"],
                    "outputs": outputs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    for path in [logs_dir, generated_dir, stages_root]:
        path.mkdir(parents=True, exist_ok=True)

    _load_model_dependencies()
    seed = int(gate.get("seed", 17))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Overfit gate requires a CUDA GPU.")
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HOME", str(scratch_root / "hf_home"))
    env.setdefault("TRANSFORMERS_CACHE", str(scratch_root / "hf_home"))

    all_setup_stages: list[dict[str, Any]] = []
    official_repo, setup_stages, setup_rc = _ensure_official_repo(config, repo_root, logs_dir, env)
    all_setup_stages.extend(setup_stages)
    if setup_rc != 0:
        raise RuntimeError(f"Official repo setup failed with rc={setup_rc}")

    rows: list[dict[str, Any]] = []
    stage_summaries: list[dict[str, Any]] = []
    decision = "BLOCKED"
    started = time.time()
    for stage_cfg in gate["stages"]:
        stage_name = str(stage_cfg["name"])
        num_fingerprints = int(stage_cfg["num_fingerprints"])
        stage_root = stages_root / stage_name
        stage_generated_dir = generated_dir / stage_name
        stage_generated_dir.mkdir(parents=True, exist_ok=True)
        generate_output = stage_generated_dir / "fingerprints.json"
        cmd = [
            "python3",
            "generate_finetuning_data.py",
            "--num_fingerprints",
            str(num_fingerprints),
            "--response_length",
            str(gate["response_length"]),
            "--key_length",
            str(gate["key_length"]),
            "--batch_size",
            str(gate.get("generation_batch_size", 1)),
            "--seed",
            str(seed),
            "--key_response_strategy",
            "perinucleus",
            "--model_used_for_key_generation",
            str(gate["model"]),
            "--perinucleus_model",
            str(gate["model"]),
            "--nucleus_t",
            str(gate.get("nucleus_t", 0.8)),
            "--nucleus_k",
            str(gate.get("nucleus_k", 3)),
            "--output_file_path",
            str(generate_output),
            "--use_chat_template",
        ]
        gen_stage = _run_stage(f"generate_{stage_name}", cmd, official_repo, env, logs_dir)
        all_setup_stages.append(gen_stage)
        if gen_stage["returncode"] != 0:
            stage_summaries.append({"stage": stage_name, "num_fingerprints": num_fingerprints, "pass": False, "failure": "fingerprint_generation_failed", "generate_stage": gen_stage})
            decision = "BLOCKED: fingerprint generation failed."
            break
        fingerprints_file = _extract_fingerprint_path(Path(gen_stage["stdout_path"]), stage_generated_dir)
        if fingerprints_file is None or not fingerprints_file.exists():
            stage_summaries.append({"stage": stage_name, "num_fingerprints": num_fingerprints, "pass": False, "failure": "fingerprints_file_missing"})
            decision = "BLOCKED: fingerprints file missing."
            break
        stage_rows, stage_summary = _train_stage(gate, stage_cfg, fingerprints_file, stage_root, device)
        rows.extend(stage_rows)
        stage_summaries.append(stage_summary)
        if not stage_summary["pass"]:
            if stage_name == "stage1_one_fingerprint":
                decision = "BLOCKED_STAGE1: single fingerprint overfit failed; do not run anchor."
            else:
                decision = f"BLOCKED_{stage_name}: ladder stopped before final stage; proceed to capacity/debug sweep, not anchor."
            break
        decision = "STAGE1_PASS: continue ladder." if stage_name == "stage1_one_fingerprint" else "LADDER_PASS_SO_FAR"
    else:
        decision = "OVERFIT_GATE_PASS: all configured ladder stages passed; anchor may be considered only after review."

    summary = {
        "schema_name": "baseline_perinucleus_official_overfit_summary",
        "schema_version": 1,
        "generated_at": _utc_now(),
        "decision": decision,
        "run_root": str(run_root),
        "official_repo": str(official_repo),
        "official_commit": str(_get(config, "official_repo.commit_hash")),
        "model": str(gate["model"]),
        "seed": seed,
        "lora": gate["lora"],
        "stages": stage_summaries,
        "setup_stages": all_setup_stages,
        "output_doc": str(output_doc),
        "output_table": str(output_table),
        "output_summary": str(output_summary),
        "output_compute": str(output_compute),
    }
    compute = {
        "schema_name": "baseline_perinucleus_official_overfit_compute",
        "schema_version": 1,
        "generated_at": summary["generated_at"],
        "run_root": str(run_root),
        "seconds": time.time() - started,
        "device": str(device),
        "stage_seconds": [{stage["stage"]: stage.get("seconds")} for stage in stage_summaries],
    }
    _write_csv(output_table, rows)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    output_compute.parent.mkdir(parents=True, exist_ok=True)
    output_compute.write_text(json.dumps(compute, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_doc, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if decision.startswith("OVERFIT_GATE_PASS") or decision.startswith("LADDER_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
