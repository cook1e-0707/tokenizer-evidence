from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.natural_evidence_v1.analyze_qwen_on_policy_survival import _entry_source
from scripts.natural_evidence_v1.common import read_jsonl, write_csv, write_json, write_jsonl
from scripts.natural_evidence_v1.replay_qwen_frame_completion import _as_int, _hash_file, _rate


SCHEMA_NAME = "natural_evidence_qwen_846699_teacher_forced_bucket_mass_probe_v1"
POSITION_SCHEMA = "natural_evidence_qwen_846699_teacher_forced_bucket_mass_position_v1"
MODEL_CONDITIONS = ("base", "protected_trained", "task_only_lora")
SUMMARY_FIELDS = [
    "model_condition",
    "payload_id",
    "seed",
    "position_rows",
    "mean_target_candidate_mass",
    "mean_best_other_candidate_mass",
    "mean_target_margin",
    "mean_full_vocab_target_mass",
    "mean_target_rank",
    "target_rank1_rate",
    "target_rank_le2_rate",
    "positive_margin_rate",
    "reference_target_bucket_hit_rate",
]
SLICE_FIELDS = [
    "slice_kind",
    "slice_value",
    *SUMMARY_FIELDS,
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-only teacher-forced bucket-mass probe for Qwen 846699. "
            "Scores base/protected/task-only models at committed variable-radix "
            "prefixes. This never trains, generates, or claims payload recovery."
        )
    )
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--survival-summary-json", required=True)
    parser.add_argument("--lift-summary-json", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--training-job-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--payload-ids", default="P0421,P1729")
    parser.add_argument("--seeds", default="17,23")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-rows-per-payload", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _format_float(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return f"{float(value):.17g}"


def _token_ids(values: object) -> list[int]:
    if not isinstance(values, list):
        return []
    output: list[int] = []
    for value in values:
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            continue
    return output


def _unique_ints(values: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def bucket_probe_from_token_logits(
    *,
    token_logits: Mapping[int, float],
    bucket_to_token_ids: Mapping[str, Sequence[int]],
    target_bucket: str,
) -> dict[str, Any]:
    candidate_ids = _unique_ints(
        [
            int(token_id)
            for token_ids in bucket_to_token_ids.values()
            for token_id in token_ids
            if int(token_id) in token_logits
        ]
    )
    if not candidate_ids:
        return {
            "target_candidate_mass": 0.0,
            "best_other_candidate_mass": 0.0,
            "target_margin": 0.0,
            "target_rank": 0,
            "bucket_masses": {},
        }
    max_logit = max(float(token_logits[token_id]) for token_id in candidate_ids)
    denom = sum(math.exp(float(token_logits[token_id]) - max_logit) for token_id in candidate_ids)
    token_probs = {
        token_id: math.exp(float(token_logits[token_id]) - max_logit) / denom
        for token_id in candidate_ids
    }
    bucket_masses: dict[str, float] = {}
    for bucket_id, token_ids in sorted(bucket_to_token_ids.items(), key=lambda item: str(item[0])):
        bucket_masses[str(bucket_id)] = sum(token_probs.get(int(token_id), 0.0) for token_id in token_ids)
    target_mass = float(bucket_masses.get(str(target_bucket), 0.0))
    other_masses = [
        mass
        for bucket_id, mass in bucket_masses.items()
        if str(bucket_id) != str(target_bucket)
    ]
    best_other = max(other_masses) if other_masses else 0.0
    target_rank = 1 + sum(1 for mass in other_masses if mass > target_mass)
    return {
        "target_candidate_mass": target_mass,
        "best_other_candidate_mass": best_other,
        "target_margin": target_mass - best_other,
        "target_rank": target_rank,
        "bucket_masses": bucket_masses,
    }


def _encode_no_special(tokenizer: Any, text: str) -> list[int]:
    try:
        return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
    except TypeError:
        return [int(token_id) for token_id in tokenizer.encode(text)]


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return ""


def _token_surface_class(token_text: str) -> str:
    from scripts.natural_evidence_v1.common import token_surface_class

    return token_surface_class(token_text) if token_text else ""


def _classify_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    classes = [
        _token_surface_class(_decode_token(tokenizer, token_id))
        for token_id in list(token_ids)[:4]
    ]
    counts = Counter(item for item in classes if item)
    return counts.most_common(1)[0][0] if counts else ""


def _load_train_rows(train_data_dir: Path, payload_ids: Sequence[str], max_rows_per_payload: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows_by_payload: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {"train_data_dir": str(train_data_dir), "payloads": {}}
    for payload_id in payload_ids:
        train_path = train_data_dir / payload_id / "variable_radix_train.jsonl"
        contract_path = train_data_dir / payload_id / "variable_radix_train_contract.json"
        if not train_path.is_file() or train_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing train JSONL for {payload_id}: {train_path}")
        if not contract_path.is_file() or contract_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing train contract for {payload_id}: {contract_path}")
        rows = [dict(row) for row in read_jsonl(train_path)]
        if max_rows_per_payload > 0:
            rows = rows[:max_rows_per_payload]
        rows_by_payload[payload_id] = rows
        position_count = sum(len(row.get("eligible_positions", [])) for row in rows if isinstance(row.get("eligible_positions", []), list))
        contract = _read_json(contract_path)
        summary["payloads"][payload_id] = {
            "train_jsonl": _hash_file(train_path),
            "contract_json": _hash_file(contract_path),
            "row_count": len(rows),
            "position_count": position_count,
            "contract_schema": contract.get("schema_name", ""),
            "encoding_mode": contract.get("encoding_mode", ""),
            "variable_radix_frame_policy": contract.get("variable_radix_frame_policy", ""),
        }
    return rows_by_payload, summary


def _checkpoint_pattern(checkpoint_root: Path, arm: str, payload_id: str, seed: int, training_job_id: str) -> Path:
    return checkpoint_root / f"{arm}_{payload_id}_seed{seed}_{training_job_id}" / "checkpoints" / "natural_bucket_lora_last"


def _resolve_checkpoint(checkpoint_root: Path, arm: str, payload_id: str, seed: int, training_job_id: str) -> Path:
    checkpoint = _checkpoint_pattern(checkpoint_root, arm, payload_id, seed, training_job_id)
    if not (checkpoint / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing adapter checkpoint: {checkpoint}")
    return checkpoint


def _load_model(
    *,
    model_name: str,
    tokenizer_name: str,
    adapter_dir: Path | None,
    require_cuda: bool,
) -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("teacher-forced bucket-mass probe requires torch and transformers") from error

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True, "trust_remote_code": True}
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("teacher-forced adapter probe requires peft") from error
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _release_model(torch_module: Any, model: Any) -> None:
    del model
    if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _batch_inputs(
    *,
    torch_module: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    max_length: int,
    device: Any,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    pad_token_id = int(tokenizer.pad_token_id)
    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    specs: list[dict[str, Any]] = []
    max_width = 0
    for batch_row_index, row in enumerate(rows):
        prompt_ids = _encode_no_special(tokenizer, str(row.get("prompt", "")))
        response_ids = _encode_no_special(tokenizer, str(row.get("response_text", "")))
        input_ids = [*prompt_ids, *response_ids]
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        for prompt_slot, position in enumerate(row.get("eligible_positions", [])):
            if not isinstance(position, dict):
                continue
            token_index = _as_int(position.get("token_index", -1), -1)
            prediction_index = len(prompt_ids) + token_index - 1
            target_index = len(prompt_ids) + token_index
            if prediction_index < 0 or target_index >= len(input_ids) or token_index >= len(response_ids):
                continue
            bucket_to_token_ids = {
                str(bucket_id): _token_ids(token_ids)
                for bucket_id, token_ids in dict(position.get("bucket_to_token_ids", {})).items()
            }
            target_bucket = str(position.get("target_bucket", ""))
            target_bucket_token_ids = _token_ids(position.get("target_bucket_token_ids", []))
            candidate_ids = _unique_ints([token_id for token_ids in bucket_to_token_ids.values() for token_id in token_ids])
            if not bucket_to_token_ids or target_bucket not in bucket_to_token_ids or not candidate_ids:
                continue
            source = _entry_source(str(position.get("entry_key", "")))
            specs.append(
                {
                    "batch_row_index": batch_row_index,
                    "prediction_index": prediction_index,
                    "payload_id": str(row.get("payload_id", "")),
                    "prompt_id": str(row.get("prompt_id", "")),
                    "prompt_slot": prompt_slot,
                    "token_index": token_index,
                    "reference_token_id": int(response_ids[token_index]),
                    "bucket_to_token_ids": bucket_to_token_ids,
                    "candidate_token_ids": candidate_ids,
                    "target_bucket": target_bucket,
                    "target_bucket_token_ids": target_bucket_token_ids,
                    "target_digit": _as_int(position.get("target_digit", -1), -1),
                    "target_radix": _as_int(position.get("target_radix", 0)),
                    "frame_index": _as_int(position.get("frame_index", 0)),
                    "frame_digit_index": _as_int(position.get("frame_digit_index", 0)),
                    "frame_digit_count": _as_int(position.get("frame_digit_count", 0)),
                    "payload_digit_index": _as_int(position.get("payload_digit_index", 0)),
                    "entry_key": str(position.get("entry_key", "")),
                    "source_prompt_split": str(row.get("prompt_split", "")) or source.get("source_prompt_split", ""),
                    "source_model_condition": source.get("source_model_condition", ""),
                    "source_example_role": str(row.get("example_role", "")),
                }
            )
        input_rows.append(input_ids)
        attention_rows.append([1] * len(input_ids))
        max_width = max(max_width, len(input_ids))
    for input_ids, attention in zip(input_rows, attention_rows, strict=True):
        pad_width = max_width - len(input_ids)
        if pad_width > 0:
            input_ids.extend([pad_token_id] * pad_width)
            attention.extend([0] * pad_width)
    return (
        torch_module.tensor(input_rows, dtype=torch_module.long, device=device),
        torch_module.tensor(attention_rows, dtype=torch_module.long, device=device),
        specs,
    )


def _empty_stats() -> dict[str, float | int]:
    return {
        "position_rows": 0,
        "target_candidate_mass_sum": 0.0,
        "best_other_candidate_mass_sum": 0.0,
        "target_margin_sum": 0.0,
        "full_vocab_target_mass_sum": 0.0,
        "target_rank_sum": 0.0,
        "target_rank1_rows": 0,
        "target_rank_le2_rows": 0,
        "positive_margin_rows": 0,
        "reference_target_bucket_hit_rows": 0,
    }


def _update_stats(stats: dict[str, float | int], row: Mapping[str, Any]) -> None:
    stats["position_rows"] = int(stats["position_rows"]) + 1
    stats["target_candidate_mass_sum"] = float(stats["target_candidate_mass_sum"]) + float(row["target_candidate_mass"])
    stats["best_other_candidate_mass_sum"] = float(stats["best_other_candidate_mass_sum"]) + float(row["best_other_candidate_mass"])
    stats["target_margin_sum"] = float(stats["target_margin_sum"]) + float(row["target_margin"])
    stats["full_vocab_target_mass_sum"] = float(stats["full_vocab_target_mass_sum"]) + float(row["full_vocab_target_mass"])
    stats["target_rank_sum"] = float(stats["target_rank_sum"]) + float(row["target_rank"])
    stats["target_rank1_rows"] = int(stats["target_rank1_rows"]) + int(int(row["target_rank"]) == 1)
    stats["target_rank_le2_rows"] = int(stats["target_rank_le2_rows"]) + int(0 < int(row["target_rank"]) <= 2)
    stats["positive_margin_rows"] = int(stats["positive_margin_rows"]) + int(float(row["target_margin"]) > 0.0)
    stats["reference_target_bucket_hit_rows"] = int(stats["reference_target_bucket_hit_rows"]) + int(bool(row["reference_target_bucket_hit"]))


def _summary_row(key: tuple[str, str, str], stats: Mapping[str, float | int]) -> dict[str, Any]:
    model_condition, payload_id, seed = key
    rows = int(stats["position_rows"])
    return {
        "model_condition": model_condition,
        "payload_id": payload_id,
        "seed": seed,
        "position_rows": rows,
        "mean_target_candidate_mass": _format_float(_rate(float(stats["target_candidate_mass_sum"]), rows)),
        "mean_best_other_candidate_mass": _format_float(_rate(float(stats["best_other_candidate_mass_sum"]), rows)),
        "mean_target_margin": _format_float(_rate(float(stats["target_margin_sum"]), rows)),
        "mean_full_vocab_target_mass": _format_float(_rate(float(stats["full_vocab_target_mass_sum"]), rows)),
        "mean_target_rank": _format_float(_rate(float(stats["target_rank_sum"]), rows)),
        "target_rank1_rate": _format_float(_rate(int(stats["target_rank1_rows"]), rows)),
        "target_rank_le2_rate": _format_float(_rate(int(stats["target_rank_le2_rows"]), rows)),
        "positive_margin_rate": _format_float(_rate(int(stats["positive_margin_rows"]), rows)),
        "reference_target_bucket_hit_rate": _format_float(_rate(int(stats["reference_target_bucket_hit_rows"]), rows)),
    }


def _slice_values(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        ("all", "all"),
        ("payload_id", str(row.get("payload_id", ""))),
        ("seed", str(row.get("seed", ""))),
        ("payload_seed", f"{row.get('payload_id', '')}|{row.get('seed', '')}"),
        ("prompt_slot", str(row.get("prompt_slot", ""))),
        ("frame_digit_index", str(row.get("frame_digit_index", ""))),
        ("target_radix", str(row.get("target_radix", ""))),
        ("target_bucket_token_class", str(row.get("target_bucket_token_class", ""))),
        ("source_model_condition", str(row.get("source_model_condition", ""))),
        ("source_prompt_split", str(row.get("source_prompt_split", ""))),
        ("source_example_role", str(row.get("source_example_role", ""))),
        ("target_bucket", str(row.get("target_bucket", ""))),
        ("frame_digit_count", str(row.get("frame_digit_count", ""))),
        ("payload_seed_target_radix", f"{row.get('payload_id', '')}|{row.get('seed', '')}|radix={row.get('target_radix', '')}"),
        ("payload_seed_target_bucket_token_class", f"{row.get('payload_id', '')}|{row.get('seed', '')}|target_class={row.get('target_bucket_token_class', '')}"),
        ("payload_seed_source_model_condition", f"{row.get('payload_id', '')}|{row.get('seed', '')}|source={row.get('source_model_condition', '')}"),
    ]


def _score_rows(
    *,
    torch_module: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    rows_by_payload: Mapping[str, Sequence[Mapping[str, Any]]],
    payload_ids: Sequence[str],
    model_condition: str,
    seed: str,
    batch_size: int,
    max_length: int,
    adapter_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_condition: dict[tuple[str, str, str], dict[str, float | int]] = defaultdict(_empty_stats)
    by_slice: dict[tuple[str, str, str, str, str], dict[str, float | int]] = defaultdict(_empty_stats)
    position_rows: list[dict[str, Any]] = []
    with torch_module.no_grad():
        for payload_id in payload_ids:
            rows = list(rows_by_payload[payload_id])
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                input_ids, attention_mask, specs = _batch_inputs(
                    torch_module=torch_module,
                    tokenizer=tokenizer,
                    rows=batch,
                    max_length=max_length,
                    device=device,
                )
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                for spec in specs:
                    row_logits = outputs.logits[int(spec["batch_row_index"]), int(spec["prediction_index"]), :].float()
                    token_logits = {
                        int(token_id): float(row_logits[int(token_id)].detach().cpu().item())
                        for token_id in spec["candidate_token_ids"]
                    }
                    probe = bucket_probe_from_token_logits(
                        token_logits=token_logits,
                        bucket_to_token_ids=spec["bucket_to_token_ids"],
                        target_bucket=str(spec["target_bucket"]),
                    )
                    target_token_ids = _unique_ints(spec["target_bucket_token_ids"])
                    if target_token_ids:
                        target_logits = row_logits[target_token_ids]
                        full_vocab_log_denom = torch_module.logsumexp(row_logits, dim=0)
                        full_vocab_target_mass = float(
                            torch_module.exp(torch_module.logsumexp(target_logits, dim=0) - full_vocab_log_denom)
                            .detach()
                            .cpu()
                            .item()
                        )
                    else:
                        full_vocab_target_mass = 0.0
                    reference_token_id = int(spec["reference_token_id"])
                    reference_target_bucket_hit = reference_token_id in set(target_token_ids)
                    row = {
                        "schema_name": POSITION_SCHEMA,
                        "model_condition": model_condition,
                        "payload_id": payload_id,
                        "seed": seed,
                        "adapter_dir": str(adapter_dir) if adapter_dir is not None else "",
                        "prompt_id": spec["prompt_id"],
                        "prompt_slot": int(spec["prompt_slot"]),
                        "token_index": int(spec["token_index"]),
                        "frame_index": int(spec["frame_index"]),
                        "frame_digit_index": int(spec["frame_digit_index"]),
                        "frame_digit_count": int(spec["frame_digit_count"]),
                        "payload_digit_index": int(spec["payload_digit_index"]),
                        "target_digit": int(spec["target_digit"]),
                        "target_radix": int(spec["target_radix"]),
                        "target_bucket": str(spec["target_bucket"]),
                        "target_bucket_token_class": _classify_token_ids(tokenizer, target_token_ids),
                        "source_prompt_split": spec["source_prompt_split"],
                        "source_model_condition": spec["source_model_condition"],
                        "source_example_role": spec["source_example_role"],
                        "reference_token_id": reference_token_id,
                        "reference_token_text": _decode_token(tokenizer, reference_token_id),
                        "reference_target_bucket_hit": bool(reference_target_bucket_hit),
                        "target_candidate_mass": float(probe["target_candidate_mass"]),
                        "best_other_candidate_mass": float(probe["best_other_candidate_mass"]),
                        "target_margin": float(probe["target_margin"]),
                        "target_rank": int(probe["target_rank"]),
                        "full_vocab_target_mass": full_vocab_target_mass,
                    }
                    position_rows.append(row)
                    condition_key = (model_condition, payload_id, seed)
                    _update_stats(by_condition[condition_key], row)
                    for slice_kind, slice_value in _slice_values(row):
                        _update_stats(by_slice[(slice_kind, slice_value, model_condition, payload_id, seed)], row)
    condition_rows = [
        _summary_row(key, stats)
        for key, stats in sorted(by_condition.items())
    ]
    slice_rows = [
        {"slice_kind": slice_kind, "slice_value": slice_value, **_summary_row((model_condition, payload_id, seed), stats)}
        for (slice_kind, slice_value, model_condition, payload_id, seed), stats in sorted(by_slice.items())
    ]
    return condition_rows, slice_rows, position_rows


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary": output_dir / "qwen_846699_teacher_forced_bucket_mass_probe_summary.json",
        "by_condition": output_dir / "qwen_846699_teacher_forced_bucket_mass_by_condition.csv",
        "by_slice": output_dir / "qwen_846699_teacher_forced_bucket_mass_by_slice.csv",
        "positions": output_dir / "qwen_846699_teacher_forced_bucket_mass_positions.jsonl",
    }


def _aggregate_by_condition(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (str(row["model_condition"]), str(row["payload_id"]), str(row["seed"])): row
        for row in rows
    }


def run_probe(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    payload_ids = _parse_csv_list(args.payload_ids)
    seeds = _parse_int_list(args.seeds)
    train_data_dir = _resolve(args.train_data_dir)
    checkpoint_root = _resolve(args.checkpoint_root)
    rows_by_payload, train_summary = _load_train_rows(train_data_dir, payload_ids, int(args.max_rows_per_payload))
    survival_summary = _read_json(_resolve(args.survival_summary_json))
    lift_summary = _read_json(_resolve(args.lift_summary_json))
    all_condition_rows: list[dict[str, Any]] = []
    all_slice_rows: list[dict[str, Any]] = []
    all_position_rows: list[dict[str, Any]] = []
    score_units: list[dict[str, Any]] = [
        {
            "model_condition": "base",
            "seed": "",
            "payload_ids": payload_ids,
            "adapter_dir": None,
        }
    ]
    for payload_id in payload_ids:
        for seed in seeds:
            score_units.append(
                {
                    "model_condition": "protected_trained",
                    "seed": str(seed),
                    "payload_ids": [payload_id],
                    "adapter_dir": _resolve_checkpoint(checkpoint_root, "qwen_protected", payload_id, seed, args.training_job_id),
                }
            )
            score_units.append(
                {
                    "model_condition": "task_only_lora",
                    "seed": str(seed),
                    "payload_ids": [payload_id],
                    "adapter_dir": _resolve_checkpoint(checkpoint_root, "qwen_task_only_lora", payload_id, seed, args.training_job_id),
                }
            )
    for unit in score_units:
        torch_module, tokenizer, model, device = _load_model(
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            adapter_dir=unit["adapter_dir"],
            require_cuda=bool(args.require_cuda),
        )
        condition_rows, slice_rows, position_rows = _score_rows(
            torch_module=torch_module,
            tokenizer=tokenizer,
            model=model,
            device=device,
            rows_by_payload=rows_by_payload,
            payload_ids=unit["payload_ids"],
            model_condition=str(unit["model_condition"]),
            seed=str(unit["seed"]),
            batch_size=max(1, int(args.batch_size)),
            max_length=max(1, int(args.max_length)),
            adapter_dir=unit["adapter_dir"],
        )
        all_condition_rows.extend(condition_rows)
        all_slice_rows.extend(slice_rows)
        all_position_rows.extend(position_rows)
        _release_model(torch_module, model)
    condition_index = _aggregate_by_condition(all_condition_rows)
    protected_rows = [
        row for row in all_condition_rows if row["model_condition"] == "protected_trained"
    ]
    task_rows = [
        row for row in all_condition_rows if row["model_condition"] == "task_only_lora"
    ]
    base_rows = [row for row in all_condition_rows if row["model_condition"] == "base"]
    def _mean_metric(rows: Sequence[Mapping[str, Any]], field: str) -> float:
        weights = [int(row["position_rows"]) for row in rows]
        total = sum(weights)
        return sum(float(row[field]) * weight for row, weight in zip(rows, weights, strict=True)) / total if total else 0.0
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_TEACHER_FORCED_BUCKET_MASS_PROBE",
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "generation_started": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "result_claim": "teacher_forced_bucket_mass_probe_not_payload_recovery_not_far",
        "inputs": {
            "train_data_dir": train_summary,
            "checkpoint_root": str(checkpoint_root),
            "training_job_id": str(args.training_job_id),
            "survival_summary_json": {
                "path": str(_resolve(args.survival_summary_json)),
                "status": survival_summary.get("status", ""),
            },
            "lift_summary_json": {
                "path": str(_resolve(args.lift_summary_json)),
                "status": lift_summary.get("status", ""),
            },
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "payload_ids": payload_ids,
            "seeds": seeds,
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
            "max_rows_per_payload": int(args.max_rows_per_payload),
        },
        "aggregate": {
            "base_mean_target_candidate_mass": _mean_metric(base_rows, "mean_target_candidate_mass"),
            "protected_mean_target_candidate_mass": _mean_metric(protected_rows, "mean_target_candidate_mass"),
            "task_only_mean_target_candidate_mass": _mean_metric(task_rows, "mean_target_candidate_mass"),
            "protected_minus_task_only_target_candidate_mass": (
                _mean_metric(protected_rows, "mean_target_candidate_mass")
                - _mean_metric(task_rows, "mean_target_candidate_mass")
            ),
            "protected_minus_base_target_candidate_mass": (
                _mean_metric(protected_rows, "mean_target_candidate_mass")
                - _mean_metric(base_rows, "mean_target_candidate_mass")
            ),
            "protected_target_rank1_rate": _mean_metric(protected_rows, "target_rank1_rate"),
            "task_only_target_rank1_rate": _mean_metric(task_rows, "target_rank1_rate"),
            "base_target_rank1_rate": _mean_metric(base_rows, "target_rank1_rate"),
            "condition_count": len(condition_index),
            "position_row_count": len(all_position_rows),
        },
        "interpretation": {
            "protected_target_mass_higher_than_task_only": (
                _mean_metric(protected_rows, "mean_target_candidate_mass")
                > _mean_metric(task_rows, "mean_target_candidate_mass")
            ),
            "protected_target_mass_higher_than_base": (
                _mean_metric(protected_rows, "mean_target_candidate_mass")
                > _mean_metric(base_rows, "mean_target_candidate_mass")
            ),
            "next_required_diagnostic": "decoder_oracle_substitution",
        },
        "outputs": {
            "by_condition_csv": "qwen_846699_teacher_forced_bucket_mass_by_condition.csv",
            "by_slice_csv": "qwen_846699_teacher_forced_bucket_mass_by_slice.csv",
            "positions_jsonl": "qwen_846699_teacher_forced_bucket_mass_positions.jsonl",
        },
    }
    return summary, all_condition_rows, all_slice_rows, all_position_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = _resolve(args.output_dir)
    paths = _output_paths(output_dir)
    for path in paths.values():
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing teacher-forced probe artifact: {path}")
    summary, condition_rows, slice_rows, position_rows = run_probe(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(paths["summary"], summary)
    write_csv(paths["by_condition"], condition_rows, SUMMARY_FIELDS)
    write_csv(paths["by_slice"], slice_rows, SLICE_FIELDS)
    write_jsonl(paths["positions"], position_rows)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
