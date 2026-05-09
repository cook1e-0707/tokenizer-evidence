from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.natural_evidence_v1.build_bucket_bank import _bucketize
from scripts.natural_evidence_v1.common import (
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    stable_hash_hex,
    write_csv,
    write_json,
    write_jsonl,
)
from src.core.payload_codec import BucketPayloadCodec, PayloadCodecError, decode_bytes_variable_radices
from src.core.rs_codec import ReedSolomonCodec


SCHEMA_NAME = "natural_evidence_qwen_diagnostic_e2e_eval_v1"
REQUIRED_CONDITION = "diagnostic_high_risk"
REQUIRED_CLAIM_STATUS = "NO_PAPER_CLAIM"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen diagnostic high-risk E2E evaluation from completed natural "
            "bucket LoRA checkpoints. This is not a paper-facing success claim."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--heldout-reference-outputs", required=True)
    parser.add_argument("--bucket-bank-entries", required=True)
    parser.add_argument("--compatibility-jsonl", required=True)
    parser.add_argument("--compatibility-by-entry-csv", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--tokenizer-name", default="")
    parser.add_argument("--payload-ids", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--query-budgets", default="")
    parser.add_argument("--eval-owner-probes", type=int, default=2048)
    parser.add_argument("--max-prompts", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--wrong-key-count", type=int, default=0)
    parser.add_argument("--condition", default=REQUIRED_CONDITION)
    parser.add_argument("--paper-claim-status", default=REQUIRED_CLAIM_STATUS)
    parser.add_argument("--start-eval", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _parse_int_list(value: str, fallback: Sequence[int]) -> list[int]:
    if not value.strip():
        return [int(item) for item in fallback]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str, fallback: Sequence[str]) -> list[str]:
    if not value.strip():
        return [str(item) for item in fallback]
    return [item.strip() for item in value.split(",") if item.strip()]


def _payload_text_by_id(config: Mapping[str, Any]) -> dict[str, str]:
    payloads = config.get("payloads", [])
    if not isinstance(payloads, list):
        return {}
    output: dict[str, str] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        payload_id = str(payload.get("payload_id", ""))
        if payload_id:
            output[payload_id] = str(payload.get("payload_text", payload_id))
    return output


def _model_config(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, dict) or key not in models or not isinstance(models[key], dict):
        raise ValueError(f"missing model config for {key!r}")
    return dict(models[key])


def _diagnostic_scale(config: Mapping[str, Any]) -> dict[str, Any]:
    scale = config.get("diagnostic_high_risk_pilot_scale", {})
    return dict(scale) if isinstance(scale, dict) else {}


def _min1_entry_ids(rows: Sequence[Mapping[str, str]]) -> set[str]:
    return {
        str(row.get("bank_entry_id", ""))
        for row in rows
        if row.get("bank_entry_id") and _as_bool(row.get("would_accept_min1", ""))
    }


def _compatible_rows_by_entry(
    rows: Sequence[Mapping[str, Any]],
    min1_entry_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        entry_id = str(row.get("bank_entry_id", ""))
        if entry_id not in min1_entry_ids or not _as_bool(row.get("compatibility_pass", False)):
            continue
        grouped[entry_id].append(dict(row))
    return dict(grouped)


def _filter_entries(
    *,
    bank_entries: Sequence[Mapping[str, Any]],
    compatible_by_entry: Mapping[str, Sequence[Mapping[str, Any]]],
    bucket_count: int,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    compatible_token_ids: dict[str, dict[str, set[int]]] = {}
    for entry_id, rows in compatible_by_entry.items():
        by_bucket: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            bucket_id = str(row.get("bucket_id", ""))
            if bucket_id == "":
                continue
            by_bucket[bucket_id].add(int(row["token_id"]))
        compatible_token_ids[entry_id] = dict(by_bucket)

    for entry in bank_entries:
        entry_id = str(entry.get("bank_entry_id", ""))
        allowed = compatible_token_ids.get(entry_id)
        if not allowed:
            continue
        buckets = dict(entry.get("buckets", {}))
        filtered_buckets: dict[str, list[int]] = {}
        for bucket_id in range(bucket_count):
            bucket_key = str(bucket_id)
            token_ids = [
                int(token_id)
                for token_id in buckets.get(bucket_key, [])
                if int(token_id) in allowed.get(bucket_key, set())
            ]
            if not token_ids:
                break
            filtered_buckets[bucket_key] = token_ids
        else:
            updated = dict(entry)
            updated["buckets"] = filtered_buckets
            updated["bucket_count"] = bucket_count
            filtered.append(updated)
    return filtered


def _entry_token_index(entry: Mapping[str, Any]) -> int:
    for key in ("prefix_response_token_count", "token_index", "token_position", "prefix_token_count"):
        if key in entry and str(entry.get(key, "")) != "":
            return int(entry[key])
    return 0


def _spaced_entries(entries: Sequence[Mapping[str, Any]], min_spacing_tokens: int, max_positions: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    last_token_index: int | None = None
    for entry in sorted(entries, key=_entry_token_index):
        token_index = _entry_token_index(entry)
        if last_token_index is not None and token_index - last_token_index < min_spacing_tokens:
            continue
        selected.append(dict(entry))
        last_token_index = token_index
        if len(selected) >= max_positions:
            break
    return selected


def _entries_by_prompt(
    entries: Sequence[Mapping[str, Any]],
    *,
    min_spacing_tokens: int,
    max_positions: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        prompt_id = str(entry.get("prompt_id", ""))
        if prompt_id:
            grouped[prompt_id].append(dict(entry))
    return {
        prompt_id: _spaced_entries(prompt_entries, min_spacing_tokens, max_positions)
        for prompt_id, prompt_entries in grouped.items()
    }


def _token_to_bucket(entry: Mapping[str, Any]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for bucket_id, token_ids in dict(entry.get("buckets", {})).items():
        for token_id in token_ids:
            mapping[int(token_id)] = int(bucket_id)
    return mapping


def _wrong_key_maps(
    *,
    compatible_by_entry: Mapping[str, Sequence[Mapping[str, Any]]],
    wrong_key_id: str,
    bucket_count: int,
    protocol_id: str,
    bank_id: str,
    bucket_assignment: str,
) -> dict[str, dict[int, int]]:
    output: dict[str, dict[int, int]] = {}
    for entry_id, rows in compatible_by_entry.items():
        candidates_by_token: dict[int, dict[str, Any]] = {}
        context_signature = ""
        for row in rows:
            token_id = int(row["token_id"])
            context_signature = context_signature or str(row.get("context_signature", ""))
            candidates_by_token[token_id] = {
                "token_id": token_id,
                "token_text": str(row.get("token_text", row.get("source_token_text", ""))),
                "probability": float(row.get("probability", row.get("reference_probability", 1.0))),
            }
        if len(candidates_by_token) < bucket_count:
            continue
        buckets = _bucketize(
            candidates=list(candidates_by_token.values()),
            bucket_count=bucket_count,
            min_members_per_bucket=1,
            key=wrong_key_id,
            protocol_id=protocol_id,
            bank_id=bank_id,
            prefix_signature=context_signature or entry_id,
            assignment_mode=bucket_assignment,
        )
        token_map: dict[int, int] = {}
        for bucket_id, members in buckets.items():
            for member in members:
                token_map[int(member["token_id"])] = int(bucket_id)
        output[entry_id] = token_map
    return output


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded.get("input_ids", [])
    if not isinstance(token_ids, list):
        return []
    return [int(token_id) for token_id in token_ids]


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))


def _generated_prompt(row: Mapping[str, Any]) -> str:
    prompt = str(row.get("prompt", ""))
    if prompt:
        return prompt
    user_probe = str(row.get("user_probe", ""))
    return f"User: {user_probe}\nAssistant:"


def _decode_completion(tokenizer: Any, generated_ids: Any, completion_start: int) -> str:
    completion_ids = generated_ids[completion_start:]
    return str(tokenizer.decode(completion_ids, skip_special_tokens=True)).strip()


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
        raise RuntimeError("diagnostic E2E eval requires torch and transformers") from error

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("diagnostic E2E adapter eval requires peft") from error
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _generate_outputs(
    *,
    torch_module: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    do_sample = temperature > 0.0
    with torch_module.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = list(rows[start : start + batch_size])
            prompts = [_generated_prompt(row) for row in batch]
            tokenized = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            completion_start = int(tokenized["input_ids"].shape[1])
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max(1, max_new_tokens),
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = float(temperature)
            generated = model.generate(**tokenized, **generation_kwargs)
            for row, rendered_prompt, generated_row in zip(batch, prompts, generated, strict=True):
                outputs.append(
                    {
                        "schema_name": "natural_evidence_qwen_diagnostic_generated_output_v1",
                        **metadata,
                        "prompt_id": str(row.get("prompt_id", "")),
                        "prompt_split": str(row.get("prompt_split", row.get("split", "heldout"))),
                        "query_index": len(outputs),
                        "user_probe": str(row.get("user_probe", "")),
                        "prompt": rendered_prompt,
                        "response_text": _decode_completion(tokenizer, generated_row, completion_start),
                    }
                )
    return outputs


def _observe_outputs(
    *,
    tokenizer: Any,
    generated_rows: Sequence[Mapping[str, Any]],
    entries_by_prompt: Mapping[str, Sequence[Mapping[str, Any]]],
    token_maps_by_entry: Mapping[str, Mapping[int, int]],
    observation_condition: str,
    extra_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in generated_rows:
        prompt = str(row.get("prompt", ""))
        response = str(row.get("response_text", ""))
        prompt_ids = _token_ids(tokenizer, prompt)
        response_ids = _token_ids(tokenizer, response)
        prompt_id = str(row.get("prompt_id", ""))
        for position_index, entry in enumerate(entries_by_prompt.get(prompt_id, [])):
            entry_id = str(entry.get("bank_entry_id", ""))
            offset = _entry_token_index(entry)
            prefix_token_ids = [int(token_id) for token_id in entry.get("prefix_token_ids", [])]
            actual_prefix = [*prompt_ids, *response_ids[:offset]]
            strict_prefix_match = actual_prefix == prefix_token_ids
            observed_token_id: int | None = None
            erasure_reason = ""
            if offset >= len(response_ids):
                erasure_reason = "offset_out_of_response"
            else:
                observed_token_id = int(response_ids[offset])
            token_to_bucket = token_maps_by_entry.get(entry_id, {})
            bucket_id = token_to_bucket.get(observed_token_id) if observed_token_id is not None else None
            compatible_bucket_ids = [str(value) for value in entry.get("compatible_bucket_ids", [])]
            digit = ""
            radix = ""
            if bucket_id is not None and compatible_bucket_ids:
                bucket_key = str(bucket_id)
                if bucket_key in compatible_bucket_ids:
                    digit = compatible_bucket_ids.index(bucket_key)
                    radix = len(compatible_bucket_ids)
                else:
                    erasure_reason = "observed_bucket_not_variable_radix_compatible"
                    bucket_id = None
            if bucket_id is None and not erasure_reason:
                erasure_reason = "observed_token_not_in_bucket_set"
            if not strict_prefix_match:
                erasure_reason = "strict_prefix_mismatch"
                bucket_id = None
                digit = ""
                radix = ""
            observations.append(
                {
                    "schema_name": "natural_evidence_qwen_diagnostic_bucket_observation_v1",
                    **extra_metadata,
                    "observation_condition": observation_condition,
                    "prompt_id": prompt_id,
                    "query_index": int(row.get("query_index", 0)),
                    "position_index": position_index,
                    "bank_entry_id": entry_id,
                    "prefix_response_token_count": offset,
                    "strict_prefix_match": strict_prefix_match,
                    "observed_token_id": "" if observed_token_id is None else observed_token_id,
                    "observed_token_text": "" if observed_token_id is None else _decode_token(tokenizer, observed_token_id),
                    "bucket_id": "" if bucket_id is None else int(bucket_id),
                    "digit": digit,
                    "radix": radix,
                    "compatible_bucket_ids": compatible_bucket_ids,
                    "erasure": bucket_id is None,
                    "erasure_reason": erasure_reason,
                }
            )
    return observations


def _decode_bucket_ids(
    bucket_ids: Sequence[int],
    *,
    bucket_tuple_width: int,
    bucket_radix: int,
    rs_parity_symbols: int,
) -> tuple[str, str]:
    usable_count = (len(bucket_ids) // bucket_tuple_width) * bucket_tuple_width
    if usable_count == 0:
        return "", "insufficient_symbols"
    tuples = [
        list(bucket_ids[index : index + bucket_tuple_width])
        for index in range(0, usable_count, bucket_tuple_width)
    ]
    codec = BucketPayloadCodec(
        bucket_radices=tuple(bucket_radix for _ in range(bucket_tuple_width)),
        rs_codec=ReedSolomonCodec(parity_symbols=rs_parity_symbols),
    )
    try:
        decoded = codec.decode_bytes(tuples, apply_rs=rs_parity_symbols > 0)
    except (PayloadCodecError, ValueError) as error:
        return "", f"decode_error:{error}"
    try:
        return decoded.decode("utf-8"), "decoded"
    except UnicodeDecodeError:
        return decoded.hex(), "decoded_non_utf8"


def _decode_variable_radix_digits(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[str, str, int, int]:
    digits: list[int] = []
    radices: list[int] = []
    for row in observations:
        digit_raw = row.get("digit", "")
        radix_raw = row.get("radix", "")
        if str(digit_raw) == "" or str(radix_raw) == "":
            continue
        digits.append(int(digit_raw))
        radices.append(int(radix_raw))
    if not digits:
        return "", "insufficient_symbols", 0, 0
    try:
        decoded, byte_groups = decode_bytes_variable_radices(digits, radices)
    except (PayloadCodecError, ValueError) as error:
        if "Insufficient variable-radix capacity" in str(error):
            return "", "insufficient_symbols", len(digits), 0
        return "", f"decode_error:{error}", len(digits), 0
    if not decoded:
        return "", "insufficient_symbols", len(digits), 0
    try:
        return decoded.decode("utf-8"), "decoded", len(digits), len(byte_groups)
    except UnicodeDecodeError:
        return decoded.hex(), "decoded_non_utf8", len(digits), len(byte_groups)


def _decode_variable_radix_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    expected_payload: str,
) -> dict[str, Any]:
    if any(str(row.get("frame_index", "")) != "" for row in observations):
        grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in observations:
            if str(row.get("frame_index", "")) == "":
                continue
            grouped[int(row["frame_index"])].append(row)
        decoded_frames: list[dict[str, Any]] = []
        usable_symbols = 0
        for frame_index, frame_rows in sorted(grouped.items()):
            ordered = sorted(frame_rows, key=lambda row: int(row.get("frame_digit_index", 0) or 0))
            expected_count = max(int(row.get("frame_digit_count", 0) or 0) for row in ordered)
            observed_by_digit_index: dict[int, Mapping[str, Any]] = {}
            for fallback_index, row in enumerate(ordered):
                if str(row.get("digit", "")) == "" or str(row.get("radix", "")) == "":
                    continue
                digit_index = int(row.get("frame_digit_index", fallback_index) or 0)
                observed_by_digit_index.setdefault(digit_index, row)
            expected_slots = set(range(expected_count)) if expected_count else set(observed_by_digit_index)
            missing_slots = expected_slots - set(observed_by_digit_index)
            complete_rows = [
                observed_by_digit_index[index]
                for index in sorted(expected_slots)
                if index in observed_by_digit_index
            ]
            if expected_count and missing_slots:
                decoded_frames.append(
                    {
                        "frame_index": frame_index,
                        "status": "incomplete_frame",
                        "observed_symbols": len(observed_by_digit_index),
                        "expected_symbols": expected_count,
                        "recovered_payload": "",
                        "accepted": False,
                    }
                )
                continue
            recovered_payload, decode_status, frame_symbols, _decoded_bytes = (
                _decode_variable_radix_digits(complete_rows)
            )
            usable_symbols += frame_symbols
            decoded_frames.append(
                {
                    "frame_index": frame_index,
                    "status": decode_status,
                    "observed_symbols": len(observed_by_digit_index),
                    "expected_symbols": expected_count,
                    "recovered_payload": recovered_payload,
                    "accepted": recovered_payload == expected_payload,
                }
            )
        accepted_frames = [frame for frame in decoded_frames if bool(frame["accepted"])]
        decoded_complete_frames = [
            frame for frame in decoded_frames if str(frame["status"]).startswith("decoded")
        ]
        complete_frame_count = sum(
            1 for frame in decoded_frames if str(frame["status"]) != "incomplete_frame"
        )
        incomplete_frames = [
            frame
            for frame in decoded_frames
            if str(frame["status"]) == "incomplete_frame" and int(frame["observed_symbols"]) > 0
        ]
        partial_frame_symbol_count = sum(
            int(frame["observed_symbols"]) for frame in incomplete_frames
        )
        max_partial_frame_symbols = max(
            (int(frame["observed_symbols"]) for frame in incomplete_frames),
            default=0,
        )
        frame_diagnostics = {
            "frame_count": len(decoded_frames),
            "complete_frame_count": complete_frame_count,
            "incomplete_frame_count": len(incomplete_frames),
            "partial_frame_symbol_count": partial_frame_symbol_count,
            "max_partial_frame_symbols": max_partial_frame_symbols,
        }
        if accepted_frames:
            return {
                "accepted": True,
                "recovered_payload": expected_payload,
                "decode_status": "decoded_frame_accept",
                "usable_symbols": usable_symbols,
                "decoded_frame_count": len(decoded_complete_frames),
                "accepted_frame_count": len(accepted_frames),
                **frame_diagnostics,
            }
        if decoded_complete_frames:
            return {
                "accepted": False,
                "recovered_payload": str(decoded_complete_frames[0]["recovered_payload"]),
                "decode_status": "decoded_frames_no_accept",
                "usable_symbols": usable_symbols,
                "decoded_frame_count": len(decoded_complete_frames),
                "accepted_frame_count": 0,
                **frame_diagnostics,
            }
        return {
            "accepted": False,
            "recovered_payload": "",
            "decode_status": "insufficient_symbols",
            "usable_symbols": usable_symbols,
            "decoded_frame_count": 0,
            "accepted_frame_count": 0,
            **frame_diagnostics,
        }

    recovered_payload, decode_status, usable_symbols, _decoded_bytes = _decode_variable_radix_digits(
        observations
    )
    return {
        "accepted": recovered_payload == expected_payload,
        "recovered_payload": recovered_payload,
        "decode_status": decode_status,
        "usable_symbols": usable_symbols,
        "decoded_frame_count": 0,
        "accepted_frame_count": int(recovered_payload == expected_payload),
        "frame_count": 0,
        "complete_frame_count": 0,
        "incomplete_frame_count": 0,
        "partial_frame_symbol_count": 0,
        "max_partial_frame_symbols": 0,
    }


def _decode_observation_group(
    *,
    observations: Sequence[Mapping[str, Any]],
    query_budgets: Sequence[int],
    bucket_tuple_width: int,
    bucket_radix: int,
    rs_parity_symbols: int,
    expected_payload: str,
    base: Mapping[str, Any],
    decoder_mode: str = "fixed_radix",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sorted_observations = sorted(
        observations,
        key=lambda row: (
            int(row.get("query_index", 0)),
            int(row.get("position_index", 0)),
            str(row.get("bank_entry_id", "")),
        ),
    )
    for budget in query_budgets:
        budget_rows = [row for row in sorted_observations if int(row.get("query_index", 0)) < int(budget)]
        bucket_ids = [
            int(row["bucket_id"])
            for row in budget_rows
            if str(row.get("bucket_id", "")) != ""
        ]
        if decoder_mode == "variable_radix":
            variable_decode = _decode_variable_radix_observations(
                budget_rows,
                expected_payload=expected_payload,
            )
            recovered_payload = str(variable_decode["recovered_payload"])
            decode_status = str(variable_decode["decode_status"])
            usable_symbols = int(variable_decode["usable_symbols"])
            accepted = bool(variable_decode["accepted"])
            decoded_frame_count = int(variable_decode["decoded_frame_count"])
            accepted_frame_count = int(variable_decode["accepted_frame_count"])
            frame_count = int(variable_decode["frame_count"])
            complete_frame_count = int(variable_decode["complete_frame_count"])
            incomplete_frame_count = int(variable_decode["incomplete_frame_count"])
            partial_frame_symbol_count = int(variable_decode["partial_frame_symbol_count"])
            max_partial_frame_symbols = int(variable_decode["max_partial_frame_symbols"])
        else:
            recovered_payload, decode_status = _decode_bucket_ids(
                bucket_ids,
                bucket_tuple_width=bucket_tuple_width,
                bucket_radix=bucket_radix,
                rs_parity_symbols=rs_parity_symbols,
            )
            usable_symbols = (len(bucket_ids) // bucket_tuple_width) * bucket_tuple_width
            accepted = recovered_payload == expected_payload
            decoded_frame_count = 0
            accepted_frame_count = 0
            frame_count = 0
            complete_frame_count = 0
            incomplete_frame_count = 0
            partial_frame_symbol_count = 0
            max_partial_frame_symbols = 0
        rows.append(
            {
                **base,
                "query_budget": int(budget),
                "accepted": accepted,
                "recovered_payload": recovered_payload,
                "expected_payload": expected_payload,
                "eligible_positions": len(budget_rows),
                "observed_symbols": len(bucket_ids),
                "usable_symbols": usable_symbols,
                "erasures": len(budget_rows) - len(bucket_ids),
                "decode_status": decode_status,
                "decoder_mode": decoder_mode,
                "decoded_frame_count": decoded_frame_count,
                "accepted_frame_count": accepted_frame_count,
                "frame_count": frame_count,
                "complete_frame_count": complete_frame_count,
                "incomplete_frame_count": incomplete_frame_count,
                "partial_frame_symbol_count": partial_frame_symbol_count,
                "max_partial_frame_symbols": max_partial_frame_symbols,
                "strict_observations_only": True,
                "result_claim": "diagnostic_eval_not_paper_claim",
            }
        )
    return rows


def _checkpoint(run_root: Path, arm: str, payload_id: str, seed: int) -> Path:
    return run_root / f"{arm}_{payload_id}_seed{seed}" / "checkpoints" / "natural_bucket_lora_last"


def _release_model(torch_module: Any, model: Any) -> None:
    del model
    gc.collect()
    if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _write_preflight(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "qwen_diagnostic_e2e_eval_preflight.json", payload)


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _append_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _prepare_decode_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated.setdefault("utility_metric", "NEEDS_RESULTS")
        updated.setdefault("naturalness_metric", "NEEDS_RESULTS")
        prepared.append(updated)
    return prepared


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    protocol_id = str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1"))
    model_cfg = _model_config(config, "qwen")
    scale = _diagnostic_scale(config)
    payload_texts = _payload_text_by_id(config)
    payload_ids = _parse_str_list(args.payload_ids, list(payload_texts.keys()))
    seeds = _parse_int_list(args.seeds, [int(seed) for seed in scale.get("seeds", [17, 23])])
    query_budgets = _parse_int_list(args.query_budgets, [int(value) for value in scale.get("query_budgets", [64, 128, 256, 512])])
    model_name = args.model_name or str(model_cfg.get("model_name", ""))
    tokenizer_name = args.tokenizer_name or str(model_cfg.get("tokenizer_name", model_name))
    run_root = resolve_repo_path(args.run_root, root)
    output_dir = resolve_repo_path(args.output_dir, root)

    errors: list[str] = []
    if args.condition != REQUIRED_CONDITION:
        errors.append("condition must be diagnostic_high_risk")
    if args.paper_claim_status != REQUIRED_CLAIM_STATUS:
        errors.append("paper-claim-status must be NO_PAPER_CLAIM")
    if int(args.eval_owner_probes) < int(scale.get("eval_owner_probes", 2048)):
        errors.append("eval-owner-probes below diagnostic minimum")
    if query_budgets != [64, 128, 256, 512]:
        errors.append("query budgets must be [64, 128, 256, 512]")
    if len(payload_ids) < 2:
        errors.append("diagnostic eval requires at least two payloads")
    if len(seeds) < 2:
        errors.append("diagnostic eval requires at least two seeds")
    if not model_name or not tokenizer_name:
        errors.append("missing model/tokenizer name")
    for payload_id in payload_ids:
        for seed in seeds:
            for arm in ("qwen_protected", "qwen_task_only_lora"):
                checkpoint = _checkpoint(run_root, arm, payload_id, seed)
                if not checkpoint.exists():
                    errors.append(f"missing checkpoint: {checkpoint}")

    preflight = {
        "schema_name": SCHEMA_NAME,
        "status": "PASS_PREFLIGHT_READY_TO_EVAL" if not errors else "FAIL_PREFLIGHT",
        "errors": errors,
        "paper_claim_allowed": False,
        "gpu_required_for_generation": True,
        "eval_started": False,
        "model": model_name,
        "tokenizer": tokenizer_name,
        "payload_ids": payload_ids,
        "seeds": seeds,
        "query_budgets": query_budgets,
        "eval_owner_probes": int(args.eval_owner_probes),
        "result_claim": "diagnostic_eval_preflight_not_payload_recovery",
    }
    _write_preflight(output_dir, preflight)
    if errors:
        print(json.dumps(preflight, sort_keys=True))
        return 1
    if not args.start_eval:
        print(json.dumps(preflight, sort_keys=True))
        return 0

    if output_dir.exists() and any(path.name != "qwen_diagnostic_e2e_eval_preflight.json" for path in output_dir.iterdir()):
        raise RuntimeError(f"output-dir must be empty or preflight-only before eval: {output_dir}")

    heldout_rows = read_jsonl(resolve_repo_path(args.heldout_reference_outputs, root))[: args.max_prompts]
    if len(heldout_rows) < int(args.eval_owner_probes):
        raise RuntimeError(f"heldout rows below eval owner probe minimum: {len(heldout_rows)}")

    bucket_count = 4
    min1_ids = _min1_entry_ids(_read_csv(resolve_repo_path(args.compatibility_by_entry_csv, root)))
    compatible_by_entry = _compatible_rows_by_entry(
        read_jsonl(resolve_repo_path(args.compatibility_jsonl, root)),
        min1_ids,
    )
    filtered_entries = _filter_entries(
        bank_entries=read_jsonl(resolve_repo_path(args.bucket_bank_entries, root)),
        compatible_by_entry=compatible_by_entry,
        bucket_count=bucket_count,
    )
    selector_cfg = dict(config.get("selector", {}))
    bucket_cfg = dict(config.get("bucket_bank", {}))
    entries_by_prompt = _entries_by_prompt(
        filtered_entries,
        min_spacing_tokens=int(selector_cfg.get("min_spacing_tokens", bucket_cfg.get("min_spacing_tokens", 12))),
        max_positions=int(selector_cfg.get("max_evidence_positions_per_response", bucket_cfg.get("max_evidence_positions_per_response", 4))),
    )
    correct_maps = {str(entry.get("bank_entry_id", "")): _token_to_bucket(entry) for entry in filtered_entries}
    wrong_key_count = args.wrong_key_count or int(dict(config.get("null_evaluations", {})).get("wrong_key_count", 4))
    wrong_maps_by_index = [
        _wrong_key_maps(
            compatible_by_entry=compatible_by_entry,
            wrong_key_id=f"{selector_cfg.get('audit_key_id', 'K001')}_WRONG_{index}",
            bucket_count=bucket_count,
            protocol_id=protocol_id,
            bank_id="qwen_natural_bucket_bank_v1",
            bucket_assignment=str(bucket_cfg.get("bucket_assignment", "keyed_mass_balance")),
        )
        for index in range(wrong_key_count)
    ]
    decoder_cfg = dict(config.get("decoder", {}))
    bucket_tuple_width = int(decoder_cfg.get("bucket_tuple_width", 3))
    rs_parity_symbols = int(decoder_cfg.get("rs_parity_symbols", 0))

    outputs_dir = output_dir
    generated_path = outputs_dir / "qwen_diagnostic_generated_outputs.jsonl"
    observations_path = outputs_dir / "qwen_diagnostic_bucket_observations.jsonl"
    decode_path = outputs_dir / "qwen_diagnostic_decode_trace.csv"
    progress_path = outputs_dir / "qwen_diagnostic_e2e_eval_progress.json"
    summary_path = outputs_dir / "qwen_diagnostic_e2e_eval_summary.json"
    decode_fieldnames = [
        "model_family",
        "model_condition",
        "tokenizer",
        "bucket_bank_id",
        "payload_id",
        "expected_payload_id",
        "seed",
        "query_budget",
        "accepted",
        "recovered_payload",
        "expected_payload",
        "far_family",
        "utility_metric",
        "naturalness_metric",
        "protocol_id",
        "eligible_positions",
        "observed_symbols",
        "usable_symbols",
        "erasures",
        "decode_status",
        "decoder_mode",
        "decoded_frame_count",
        "accepted_frame_count",
        "frame_count",
        "complete_frame_count",
        "incomplete_frame_count",
        "partial_frame_symbol_count",
        "max_partial_frame_symbols",
        "strict_observations_only",
        "result_claim",
    ]
    for partial_path in (generated_path, observations_path, decode_path):
        if partial_path.exists():
            partial_path.unlink()

    generated_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    completed_units: list[str] = []
    _write_progress(
        progress_path,
        {
            "schema_name": "natural_evidence_qwen_diagnostic_e2e_eval_progress_v1",
            "status": "RUNNING",
            "stage": "start_eval",
            "completed_units": completed_units,
            "generated_output_count": 0,
            "observation_count": 0,
            "decode_row_count": 0,
            "result_claim": "diagnostic_eval_progress_not_payload_recovery",
        },
    )

    torch_module, tokenizer, model, device = _load_model(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        adapter_dir=None,
        require_cuda=args.require_cuda,
    )
    raw_generated = _generate_outputs(
        torch_module=torch_module,
        tokenizer=tokenizer,
        model=model,
        device=device,
        rows=heldout_rows,
        batch_size=max(1, args.batch_size),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        metadata={
            "protocol_id": protocol_id,
            "model_family": "qwen",
            "model_condition": "raw",
            "payload_id": "",
            "seed": "",
        },
    )
    generated_rows.extend(raw_generated)
    raw_obs = _observe_outputs(
        tokenizer=tokenizer,
        generated_rows=raw_generated,
        entries_by_prompt=entries_by_prompt,
        token_maps_by_entry=correct_maps,
        observation_condition="correct_key",
        extra_metadata={"model_family": "qwen", "model_condition": "raw", "payload_id": "", "seed": "", "protocol_id": protocol_id},
    )
    observations.extend(raw_obs)
    raw_decode_rows: list[dict[str, Any]] = []
    for payload_id in payload_ids:
        for seed in seeds:
            raw_decode_rows.extend(
                _decode_observation_group(
                    observations=raw_obs,
                    query_budgets=query_budgets,
                    bucket_tuple_width=bucket_tuple_width,
                    bucket_radix=bucket_count,
                    rs_parity_symbols=rs_parity_symbols,
                    expected_payload=payload_texts[payload_id],
                    base={
                        "model_family": "qwen",
                        "model_condition": "raw",
                        "tokenizer": tokenizer_name,
                        "bucket_bank_id": "qwen_4way_min1_compatible",
                        "payload_id": payload_id,
                        "expected_payload_id": payload_id,
                        "seed": seed,
                        "far_family": "raw_exact_model_pre_null",
                        "protocol_id": protocol_id,
                    },
                )
            )
    raw_decode_rows = _prepare_decode_rows(raw_decode_rows)
    decode_rows.extend(raw_decode_rows)
    completed_units.append("raw")
    _append_jsonl(generated_path, raw_generated)
    _append_jsonl(observations_path, raw_obs)
    _append_csv(decode_path, raw_decode_rows, decode_fieldnames)
    _write_progress(
        progress_path,
        {
            "schema_name": "natural_evidence_qwen_diagnostic_e2e_eval_progress_v1",
            "status": "RUNNING",
            "stage": "completed_raw",
            "completed_units": completed_units,
            "generated_output_count": len(generated_rows),
            "observation_count": len(observations),
            "decode_row_count": len(decode_rows),
            "result_claim": "diagnostic_eval_progress_not_payload_recovery",
        },
    )
    _release_model(torch_module, model)

    for payload_id in payload_ids:
        for seed in seeds:
            for arm, model_condition in (
                ("qwen_protected", "protected_trained"),
                ("qwen_task_only_lora", "task_only_lora"),
            ):
                adapter_dir = _checkpoint(run_root, arm, payload_id, seed)
                torch_module, tokenizer, model, device = _load_model(
                    model_name=model_name,
                    tokenizer_name=tokenizer_name,
                    adapter_dir=adapter_dir,
                    require_cuda=args.require_cuda,
                )
                run_generated = _generate_outputs(
                    torch_module=torch_module,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    rows=heldout_rows,
                    batch_size=max(1, args.batch_size),
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    metadata={
                        "protocol_id": protocol_id,
                        "model_family": "qwen",
                        "model_condition": model_condition,
                        "payload_id": payload_id,
                        "seed": seed,
                    },
                )
                generated_rows.extend(run_generated)
                run_obs = _observe_outputs(
                    tokenizer=tokenizer,
                    generated_rows=run_generated,
                    entries_by_prompt=entries_by_prompt,
                    token_maps_by_entry=correct_maps,
                    observation_condition="correct_key",
                    extra_metadata={
                        "model_family": "qwen",
                        "model_condition": model_condition,
                        "payload_id": payload_id,
                        "seed": seed,
                        "protocol_id": protocol_id,
                    },
                )
                observations.extend(run_obs)
                rows_to_persist = list(run_obs)
                run_decode_rows: list[dict[str, Any]] = []
                run_decode_rows.extend(
                    _decode_observation_group(
                        observations=run_obs,
                        query_budgets=query_budgets,
                        bucket_tuple_width=bucket_tuple_width,
                        bucket_radix=bucket_count,
                        rs_parity_symbols=rs_parity_symbols,
                        expected_payload=payload_texts[payload_id],
                        base={
                            "model_family": "qwen",
                            "model_condition": model_condition,
                            "tokenizer": tokenizer_name,
                            "bucket_bank_id": "qwen_4way_min1_compatible",
                            "payload_id": payload_id,
                            "expected_payload_id": payload_id,
                            "seed": seed,
                            "far_family": "protected" if model_condition == "protected_trained" else "task_only_lora_null",
                            "protocol_id": protocol_id,
                        },
                    )
                )
                if model_condition == "protected_trained":
                    for wrong_index, wrong_maps in enumerate(wrong_maps_by_index):
                        wrong_obs = _observe_outputs(
                            tokenizer=tokenizer,
                            generated_rows=run_generated,
                            entries_by_prompt=entries_by_prompt,
                            token_maps_by_entry=wrong_maps,
                            observation_condition=f"wrong_key_{wrong_index}",
                            extra_metadata={
                                "model_family": "qwen",
                                "model_condition": "wrong_key",
                                "payload_id": payload_id,
                                "seed": seed,
                                "wrong_key_index": wrong_index,
                                "protocol_id": protocol_id,
                            },
                        )
                        observations.extend(wrong_obs)
                        rows_to_persist.extend(wrong_obs)
                        run_decode_rows.extend(
                            _decode_observation_group(
                                observations=wrong_obs,
                                query_budgets=query_budgets,
                                bucket_tuple_width=bucket_tuple_width,
                                bucket_radix=bucket_count,
                                rs_parity_symbols=rs_parity_symbols,
                                expected_payload=payload_texts[payload_id],
                                base={
                                    "model_family": "qwen",
                                    "model_condition": "wrong_key",
                                    "tokenizer": tokenizer_name,
                                    "bucket_bank_id": "qwen_4way_min1_compatible",
                                    "payload_id": payload_id,
                                    "expected_payload_id": payload_id,
                                    "seed": seed,
                                    "far_family": f"wrong_key_{wrong_index}",
                                    "protocol_id": protocol_id,
                                },
                            )
                        )
                    wrong_payload_id = next(candidate for candidate in payload_ids if candidate != payload_id)
                    run_decode_rows.extend(
                        _decode_observation_group(
                            observations=run_obs,
                            query_budgets=query_budgets,
                            bucket_tuple_width=bucket_tuple_width,
                            bucket_radix=bucket_count,
                            rs_parity_symbols=rs_parity_symbols,
                            expected_payload=payload_texts[wrong_payload_id],
                            base={
                                "model_family": "qwen",
                                "model_condition": "wrong_payload",
                                "tokenizer": tokenizer_name,
                                "bucket_bank_id": "qwen_4way_min1_compatible",
                                "payload_id": payload_id,
                                "expected_payload_id": wrong_payload_id,
                                "seed": seed,
                                "far_family": "wrong_payload",
                                "protocol_id": protocol_id,
                            },
                        )
                    )
                run_decode_rows = _prepare_decode_rows(run_decode_rows)
                decode_rows.extend(run_decode_rows)
                completed_units.append(f"{model_condition}_{payload_id}_seed{seed}")
                _append_jsonl(generated_path, run_generated)
                _append_jsonl(observations_path, rows_to_persist)
                _append_csv(decode_path, run_decode_rows, decode_fieldnames)
                _write_progress(
                    progress_path,
                    {
                        "schema_name": "natural_evidence_qwen_diagnostic_e2e_eval_progress_v1",
                        "status": "RUNNING",
                        "stage": f"completed_{model_condition}_{payload_id}_seed{seed}",
                        "completed_units": completed_units,
                        "generated_output_count": len(generated_rows),
                        "observation_count": len(observations),
                        "decode_row_count": len(decode_rows),
                        "result_claim": "diagnostic_eval_progress_not_payload_recovery",
                    },
                )
                _release_model(torch_module, model)

    protected_accepts = [row for row in decode_rows if row["model_condition"] == "protected_trained" and row["accepted"]]
    null_accepts = [row for row in decode_rows if row["model_condition"] != "protected_trained" and row["accepted"]]
    strict_matches = sum(1 for row in observations if bool(row.get("strict_prefix_match", False)))
    observed_symbols = sum(1 for row in observations if str(row.get("bucket_id", "")) != "")
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "EVAL_COMPLETE_DIAGNOSTIC_HIGH_RISK_NOT_PAPER_CLAIM",
        "paper_claim_allowed": False,
        "protocol_id": protocol_id,
        "model": model_name,
        "tokenizer": tokenizer_name,
        "payload_ids": payload_ids,
        "seeds": seeds,
        "query_budgets": query_budgets,
        "heldout_prompt_count": len(heldout_rows),
        "filtered_min1_entries": len(filtered_entries),
        "generated_output_count": len(generated_rows),
        "observation_count": len(observations),
        "strict_prefix_match_count": strict_matches,
        "observed_symbol_count": observed_symbols,
        "decode_row_count": len(decode_rows),
        "protected_accept_count": len(protected_accepts),
        "null_accept_count": len(null_accepts),
        "diagnostic_recovery_observed": bool(protected_accepts),
        "null_accept_observed": bool(null_accepts),
        "not_full_far": True,
        "result_claim": "qwen_diagnostic_e2e_eval_not_paper_claim",
        "outputs": {
            "generated_outputs_jsonl": str(generated_path),
            "bucket_observations_jsonl": str(observations_path),
            "decode_trace_csv": str(decode_path),
            "summary_json": str(summary_path),
        },
    }
    write_json(summary_path, summary)
    _write_progress(
        progress_path,
        {
            "schema_name": "natural_evidence_qwen_diagnostic_e2e_eval_progress_v1",
            "status": "COMPLETE",
            "stage": "complete",
            "completed_units": completed_units,
            "generated_output_count": len(generated_rows),
            "observation_count": len(observations),
            "decode_row_count": len(decode_rows),
            "summary_json": str(summary_path),
            "result_claim": "diagnostic_eval_progress_not_payload_recovery",
        },
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
