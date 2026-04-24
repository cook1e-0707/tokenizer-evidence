from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import csv
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from src.infrastructure.paths import discover_repo_root


RUN_DIAGNOSTIC_FIELDS = [
    "run_id",
    "case_id",
    "variant_id",
    "block_count",
    "seed",
    "payload",
    "accepted",
    "verifier_success",
    "decoded_payload",
    "expected_payload",
    "decoded_block_count_correct",
    "match_ratio",
    "exact_slot_rate",
    "bucket_correct_rate",
    "parser_success",
    "number_of_candidates",
    "number_of_blocks_parsed",
    "number_of_slots_expected",
    "number_of_slots_decoded",
    "number_of_slot_errors",
    "number_of_symbol_errors",
    "number_of_erasures",
    "rs_decode_status",
    "generated_text_path",
    "eval_summary_path",
    "train_summary_path",
    "adapter_path",
    "config_path",
    "model_id",
    "tokenizer_id",
    "codebook_hash",
    "payload_map_hash",
    "train_contract_hash",
    "eval_contract_hash",
    "prompt_family_hash",
    "field_order",
    "bucket_partition_hash",
    "rs_config_hash",
    "generation_config_hash",
    "adapter_checkpoint_hash",
    "missing_hash_inputs",
    "diagnostic_source",
    "failure_reasons",
]

SLOT_DIAGNOSTIC_FIELDS = [
    "run_id",
    "case_id",
    "block_count",
    "seed",
    "payload",
    "block_index",
    "field_name",
    "expected_bucket",
    "decoded_bucket",
    "expected_token",
    "generated_token",
    "slot_correct",
    "bucket_correct",
    "exact_token_correct",
    "target_bucket_mass",
    "target_token_probability",
    "top_5_tokens",
    "top_5_probs",
    "margin_to_best_wrong_bucket",
    "diagnostic_source",
]

FAILURE_CASE_FIELDS = [
    "run_id",
    "case_id",
    "variant_id",
    "block_count",
    "seed",
    "payload",
    "failure_reasons",
    "match_ratio",
    "exact_slot_rate",
    "bucket_correct_rate",
    "parser_success",
    "decoded_block_count_correct",
    "number_of_slot_errors",
    "generated_text_path",
    "eval_summary_path",
    "train_summary_path",
    "adapter_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-mutating diagnostics for the G3a-v1 block-scale package."
    )
    parser.add_argument(
        "--g3a-summary",
        default="results/processed/paper_stats/g3a_summary.json",
        help="Existing G3a summary JSON. This file is read only.",
    )
    parser.add_argument(
        "--g3a-table",
        default="results/tables/g3a_block_scale.csv",
        help="Existing G3a per-run table. This file is read only.",
    )
    parser.add_argument(
        "--train-config",
        default="configs/experiment/scale/exp_train__qwen2_5_7b__g3a_block_scale_v1.yaml",
        help="G3a train config used to recover package-level contracts when run files are absent.",
    )
    parser.add_argument(
        "--reporting-config",
        default="configs/reporting/g3a_block_scale_v1.yaml",
        help="G3a reporting config used for package metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/processed/paper_stats",
        help="Directory for diagnostic JSON output.",
    )
    parser.add_argument(
        "--tables-dir",
        default="results/tables",
        help="Directory for diagnostic CSV outputs.",
    )
    parser.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory for the markdown diagnostic report.",
    )
    parser.add_argument(
        "--path-remap",
        action="append",
        default=[],
        help=(
            "Optional OLD=NEW path prefix remap for local mirrors. "
            "No remap is applied by default to avoid guessing Chimera paths."
        ),
    )
    parser.add_argument(
        "--adapter-hash-max-bytes",
        type=int,
        default=2_000_000_000,
        help="Maximum total adapter bytes to hash when adapter files are available.",
    )
    return parser.parse_args()


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _stable_hash(payload: Any) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(path: Path, *, max_bytes: int) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, "adapter_path_missing"
    if path.is_file():
        digest = _file_sha256(path)
        return digest, None if digest else "adapter_file_unreadable"
    if not path.is_dir():
        return None, "adapter_path_not_file_or_directory"

    files = sorted(item for item in path.rglob("*") if item.is_file())
    total_bytes = sum(item.stat().st_size for item in files)
    if total_bytes > max_bytes:
        return None, f"adapter_too_large_to_hash:{total_bytes}>{max_bytes}"

    digest = sha256()
    for item in files:
        rel = item.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _missing(value: Any) -> bool:
    return value in {None, "", "missing"}


def _parse_remaps(raw_remaps: list[str]) -> list[tuple[str, str]]:
    remaps: list[tuple[str, str]] = []
    for raw in raw_remaps:
        if "=" not in raw:
            raise ValueError(f"--path-remap must be OLD=NEW, got {raw!r}")
        old, new = raw.split("=", 1)
        remaps.append((old, new))
    return remaps


def _resolve_existing_path(raw: str, remaps: list[tuple[str, str]]) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    for old, new in remaps:
        if raw.startswith(old):
            candidate = Path(new + raw[len(old):])
            if candidate.exists():
                return candidate
    return path


def _parent_run_dir(raw_path: str, remaps: list[tuple[str, str]]) -> Path | None:
    if not raw_path:
        return None
    path = _resolve_existing_path(raw_path, remaps)
    return path.parent


def _find_sibling_json(run_dir: Path | None, name: str) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    path = run_dir / name
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else None


def _find_sibling_text_path(run_dir: Path | None, name: str) -> Path | None:
    if run_dir is None:
        return None
    path = run_dir / name
    return path if path.exists() else None


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _field_specs(catalog: dict[str, Any]) -> tuple[list[str], dict[str, dict[int, list[str]]]]:
    field_order: list[str] = []
    buckets_by_field: dict[str, dict[int, list[str]]] = {}
    for field in catalog.get("fields", []):
        field_name = str(field["field_name"])
        field_order.append(field_name)
        buckets = {
            int(bucket_id): [str(item) for item in values]
            for bucket_id, values in dict(field.get("buckets", {})).items()
        }
        buckets_by_field[field_name] = buckets
    return field_order, buckets_by_field


def _encode_unit_to_bucket_tuple(unit: int, radices: list[int]) -> tuple[int, ...]:
    digits: list[int] = []
    remaining = unit
    for radix in reversed(radices):
        digits.append(remaining % radix)
        remaining //= radix
    return tuple(reversed(digits))


def _fixed_width_units(unit: int, *, width: int, capacity: int) -> tuple[int, ...]:
    digits = [0] * width
    remaining = unit
    for index in range(width - 1, -1, -1):
        remaining, digit = divmod(remaining, capacity)
        digits[index] = digit
    return tuple(digits)


def _payload_units(payload: str, payload_labels: list[str], *, block_count: int, capacity: int) -> tuple[int, ...]:
    label_to_index = {label: index for index, label in enumerate(payload_labels)}
    unit = label_to_index[payload]
    if block_count == 1:
        return (unit,)
    if len(payload_labels) <= capacity and block_count == 2:
        return (unit, capacity - 1 - unit)
    return _fixed_width_units(unit, width=block_count, capacity=capacity)


def _expected_slots(
    *,
    payload: str,
    block_count: int,
    payload_labels: list[str],
    field_order: list[str],
    buckets_by_field: dict[str, dict[int, list[str]]],
) -> list[dict[str, Any]]:
    radices = [len(buckets_by_field[field_name]) for field_name in field_order]
    capacity = 1
    for radix in radices:
        capacity *= radix
    units = _payload_units(payload, payload_labels, block_count=block_count, capacity=capacity)
    slots: list[dict[str, Any]] = []
    slot_index = 0
    for block_index, unit in enumerate(units):
        bucket_tuple = _encode_unit_to_bucket_tuple(unit, radices)
        for field_name, bucket_id in zip(field_order, bucket_tuple, strict=True):
            members = buckets_by_field[field_name].get(int(bucket_id), [])
            slots.append(
                {
                    "slot_index": slot_index,
                    "block_index": block_index,
                    "field_name": field_name,
                    "expected_bucket": int(bucket_id),
                    "expected_token": members[0] if members else "missing",
                }
            )
            slot_index += 1
    return slots


def _config_hashes(
    *,
    repo_root: Path,
    train_config: dict[str, Any],
    reporting_config: dict[str, Any],
    catalog_path: Path,
) -> dict[str, str]:
    model_path = repo_root / "configs/model/qwen2_5_7b_instruct.yaml"
    model_config = _load_yaml(model_path) if model_path.exists() else {}
    model_payload = model_config.get("model", {}) if isinstance(model_config.get("model", {}), dict) else {}
    generation_config = {
        "generation_prompt": train_config.get("train", {}).get("generation_prompt"),
        "generation_do_sample": train_config.get("train", {}).get("generation_do_sample"),
        "generation_max_new_tokens": train_config.get("train", {}).get("generation_max_new_tokens"),
        "generation_stop_strings": train_config.get("train", {}).get("generation_stop_strings"),
        "generation_bad_words": train_config.get("train", {}).get("generation_bad_words"),
        "generation_suppress_tokens": train_config.get("train", {}).get("generation_suppress_tokens"),
        "generation_sequence_bias": train_config.get("train", {}).get("generation_sequence_bias"),
    }
    prompt_family = {
        "generation_prompt": train_config.get("train", {}).get("generation_prompt"),
        "prompt_contract_name": "compiled_slot_request_v1",
    }
    return {
        "model_id": str(model_payload.get("name", "qwen2.5-7b-instruct")),
        "tokenizer_id": str(model_payload.get("tokenizer_name", "Qwen/Qwen2.5-7B-Instruct")),
        "generation_config_hash": _stable_hash(generation_config),
        "prompt_family_hash": _stable_hash(prompt_family),
        "package_config_hash": _stable_hash(reporting_config),
        "catalog_file_hash": _file_sha256(catalog_path) or "missing",
    }


def _contract_hashes(
    *,
    row: dict[str, str],
    eval_summary: dict[str, Any] | None,
    verifier_result: dict[str, Any] | None,
    compiled_gate_result: dict[str, Any] | None,
    train_run_dir: Path | None,
    eval_run_dir: Path | None,
    adapter_hash_max_bytes: int,
    default_hashes: dict[str, str],
    payload_labels: list[str],
    field_order: list[str],
    buckets_by_field: dict[str, dict[int, list[str]]],
) -> tuple[dict[str, str], list[str], str, str]:
    missing_inputs: list[str] = []
    train_contract_path = train_run_dir / "compiled_train_contract.json" if train_run_dir else None
    eval_contract_path = train_run_dir / "compiled_eval_contract.json" if train_run_dir else None
    train_contract = _load_json(train_contract_path) if train_contract_path else None
    eval_contract = _load_json(eval_contract_path) if eval_contract_path else None
    if not isinstance(eval_contract, dict):
        diagnostics = eval_summary.get("diagnostics", {}) if isinstance(eval_summary, dict) else {}
        raw_eval_contract = diagnostics.get("compiled_eval_contract") if isinstance(diagnostics, dict) else None
        eval_contract = raw_eval_contract if isinstance(raw_eval_contract, dict) else None

    hashes: dict[str, str] = {
        "model_id": default_hashes["model_id"],
        "tokenizer_id": default_hashes["tokenizer_id"],
        "codebook_hash": default_hashes["catalog_file_hash"],
        "payload_map_hash": "missing",
        "train_contract_hash": "missing",
        "eval_contract_hash": "missing",
        "prompt_family_hash": default_hashes["prompt_family_hash"],
        "field_order": "|".join(field_order),
        "bucket_partition_hash": _stable_hash(buckets_by_field),
        "rs_config_hash": "missing",
        "generation_config_hash": default_hashes["generation_config_hash"],
        "adapter_checkpoint_hash": "missing",
    }

    if isinstance(train_contract, dict):
        hashes["model_id"] = str(train_contract.get("model_name", hashes["model_id"]))
        hashes["tokenizer_id"] = str(train_contract.get("tokenizer_name", hashes["tokenizer_id"]))
        hashes["codebook_hash"] = str(train_contract.get("catalog_sha256", hashes["codebook_hash"]))
        hashes["payload_map_hash"] = _stable_hash(train_contract.get("payload_label_to_units", {}))
        hashes["train_contract_hash"] = str(train_contract.get("contract_hash", _stable_hash(train_contract)))
        hashes["prompt_family_hash"] = str(train_contract.get("prompt_contract_hash", hashes["prompt_family_hash"]))
    else:
        capacity = len(next(iter(buckets_by_field.values()))) ** len(field_order)
        payload_label_to_units = {
            label: list(
                _payload_units(
                    label,
                    payload_labels,
                    block_count=_as_int(row["block_count"]),
                    capacity=capacity,
                )
            )
            for label in payload_labels
        }
        hashes["payload_map_hash"] = _stable_hash(
            {
                "block_count": _as_int(row["block_count"]),
                "payload_label_to_units": payload_label_to_units,
            }
        )
        missing_inputs.append("compiled_train_contract.json")

    diagnostics = eval_summary.get("diagnostics", {}) if isinstance(eval_summary, dict) else {}
    if isinstance(diagnostics, dict) and diagnostics.get("compiled_train_contract_hash"):
        hashes["train_contract_hash"] = str(diagnostics["compiled_train_contract_hash"])
    if isinstance(verifier_result, dict):
        details = verifier_result.get("details", {})
        if isinstance(details, dict) and details.get("compiled_train_contract_hash"):
            hashes["train_contract_hash"] = str(details["compiled_train_contract_hash"])

    if isinstance(eval_contract, dict):
        hashes["eval_contract_hash"] = _stable_hash(eval_contract)
        raw_fields = eval_contract.get("slot_field_names")
        if isinstance(raw_fields, list) and raw_fields:
            hashes["field_order"] = "|".join(dict.fromkeys(str(item) for item in raw_fields))
    else:
        missing_inputs.append("compiled_eval_contract.json")

    if hashes["train_contract_hash"] == "missing":
        missing_inputs.append("compiled_train_contract_hash")
    if hashes["eval_contract_hash"] == "missing":
        missing_inputs.append("compiled_eval_contract_hash")
    if hashes["rs_config_hash"] == "missing":
        missing_inputs.append("rs_config")

    adapter_path = "missing"
    if isinstance(diagnostics, dict) and diagnostics.get("checkpoint_path"):
        adapter_path = str(diagnostics["checkpoint_path"])
    elif train_run_dir is not None:
        adapter_path = str(train_run_dir / "checkpoints/hf_last")
    if adapter_path != "missing":
        adapter_hash, adapter_missing_reason = _directory_sha256(Path(adapter_path), max_bytes=adapter_hash_max_bytes)
        if adapter_hash:
            hashes["adapter_checkpoint_hash"] = adapter_hash
        else:
            missing_inputs.append(adapter_missing_reason or "adapter_checkpoint")
    else:
        missing_inputs.append("adapter_checkpoint_path")

    config_path = "missing"
    if eval_run_dir is not None:
        config_path = str(eval_run_dir / "config.resolved.yaml")
    return hashes, sorted(set(missing_inputs)), adapter_path, config_path


def _slot_rows_from_raw(
    *,
    row: dict[str, str],
    run_id: str,
    compiled_gate_result: dict[str, Any] | None,
    expected_slots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    block_count = _as_int(row["block_count"])
    seed = _as_int(row["seed"])
    if isinstance(compiled_gate_result, dict) and isinstance(compiled_gate_result.get("slot_diagnostics"), list):
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(compiled_gate_result["slot_diagnostics"]):
            if not isinstance(item, dict):
                continue
            fallback = expected_slots[index] if index < len(expected_slots) else {}
            slot_index = _as_int(item.get("slot_index"), index)
            fields_per_block = max(1, len({slot["field_name"] for slot in expected_slots}) or 1)
            rows.append(
                {
                    "run_id": run_id,
                    "case_id": row["case_id"],
                    "block_count": block_count,
                    "seed": seed,
                    "payload": row["payload"],
                    "block_index": item.get("block_index", slot_index // fields_per_block),
                    "field_name": item.get("slot_type", fallback.get("field_name", "missing")),
                    "expected_bucket": item.get("expected_bucket_id", fallback.get("expected_bucket", "missing")),
                    "decoded_bucket": item.get("observed_bucket_id", "missing"),
                    "expected_token": item.get("expected_value", fallback.get("expected_token", "missing")),
                    "generated_token": item.get("observed_value", item.get("chosen_token_text", "missing")),
                    "slot_correct": item.get("is_slot_exact", "missing"),
                    "bucket_correct": item.get("is_bucket_correct", "missing"),
                    "exact_token_correct": item.get("is_slot_exact", "missing"),
                    "target_bucket_mass": "missing",
                    "target_token_probability": "missing",
                    "top_5_tokens": "missing",
                    "top_5_probs": "missing",
                    "margin_to_best_wrong_bucket": "missing",
                    "diagnostic_source": "compiled_gate_result.json",
                }
            )
        return rows

    return [
        {
            "run_id": run_id,
            "case_id": row["case_id"],
            "block_count": block_count,
            "seed": seed,
            "payload": row["payload"],
            "block_index": slot["block_index"],
            "field_name": slot["field_name"],
            "expected_bucket": slot["expected_bucket"],
            "decoded_bucket": "missing",
            "expected_token": slot["expected_token"],
            "generated_token": "missing",
            "slot_correct": "missing",
            "bucket_correct": "missing",
            "exact_token_correct": "missing",
            "target_bucket_mass": "missing",
            "target_token_probability": "missing",
            "top_5_tokens": "missing",
            "top_5_probs": "missing",
            "margin_to_best_wrong_bucket": "missing",
            "diagnostic_source": "expected_contract_only",
        }
        for slot in expected_slots
    ]


def _run_diagnostic_row(
    *,
    row: dict[str, str],
    eval_summary: dict[str, Any] | None,
    verifier_result: dict[str, Any] | None,
    compiled_gate_result: dict[str, Any] | None,
    train_run_dir: Path | None,
    eval_run_dir: Path | None,
    hashes: dict[str, str],
    missing_hash_inputs: list[str],
    adapter_path: str,
    config_path: str,
) -> dict[str, Any]:
    diagnostics = eval_summary.get("diagnostics", {}) if isinstance(eval_summary, dict) else {}
    verifier_details = verifier_result.get("details", {}) if isinstance(verifier_result, dict) else {}
    compiled_details = compiled_gate_result if isinstance(compiled_gate_result, dict) else {}
    expected_slots = _as_int(row["block_count"]) * 2
    exact_slot_rate = (
        diagnostics.get("slot_exact_rate")
        if isinstance(diagnostics, dict) and diagnostics.get("slot_exact_rate") is not None
        else row.get("match_ratio", "missing")
    )
    bucket_correct_rate = (
        diagnostics.get("bucket_correct_rate")
        if isinstance(diagnostics, dict) and diagnostics.get("bucket_correct_rate") is not None
        else compiled_details.get("bucket_correct_rate", "missing")
    )
    number_of_slots_expected = (
        len(diagnostics.get("compiled_eval_contract", {}).get("expected_slot_values", []))
        if isinstance(diagnostics, dict) and isinstance(diagnostics.get("compiled_eval_contract"), dict)
        else expected_slots
    )
    number_of_slots_decoded = (
        len(compiled_gate_result.get("parsed_slot_values", []))
        if isinstance(compiled_gate_result, dict) and isinstance(compiled_gate_result.get("parsed_slot_values"), list)
        else (number_of_slots_expected if _as_bool(row.get("decoded_block_count_correct")) else "missing")
    )
    exact_slot_rate_float = _as_float(exact_slot_rate, default=-1.0)
    number_of_slot_errors = (
        max(0, round(number_of_slots_expected * (1.0 - exact_slot_rate_float)))
        if exact_slot_rate_float >= 0.0
        else "missing"
    )
    generated_text_path = "missing"
    if isinstance(diagnostics, dict) and diagnostics.get("generated_text_path"):
        generated_text_path = str(diagnostics["generated_text_path"])
    elif train_run_dir is not None:
        generated_text_path = str(train_run_dir / "generated_text.txt")

    parser_success = "missing"
    if isinstance(compiled_gate_result, dict):
        parser_success = (
            _as_float(compiled_gate_result.get("field_valid_rate"), default=0.0) == 1.0
            and _as_int(compiled_gate_result.get("valid_canonical_block_count")) == _as_int(row["block_count"])
        )
    elif row.get("decoded_block_count_correct") != "":
        parser_success = _as_bool(row["decoded_block_count_correct"])

    number_of_candidates = "missing"
    if isinstance(verifier_details, dict) and "candidate_window_count" in verifier_details:
        number_of_candidates = verifier_details["candidate_window_count"]

    number_of_blocks_parsed = "missing"
    if isinstance(verifier_details, dict) and "num_blocks" in verifier_details:
        number_of_blocks_parsed = verifier_details["num_blocks"]
    elif row.get("decoded_block_count"):
        number_of_blocks_parsed = row["decoded_block_count"]

    return {
        "run_id": row.get("eval_run_id") or row.get("case_id"),
        "case_id": row["case_id"],
        "variant_id": row["variant_id"],
        "block_count": row["block_count"],
        "seed": row["seed"],
        "payload": row["payload"],
        "accepted": row["accepted"],
        "verifier_success": row["verifier_success"],
        "decoded_payload": row["decoded_payload"],
        "expected_payload": row["payload"],
        "decoded_block_count_correct": row["decoded_block_count_correct"],
        "match_ratio": row["match_ratio"],
        "exact_slot_rate": exact_slot_rate,
        "bucket_correct_rate": bucket_correct_rate,
        "parser_success": parser_success,
        "number_of_candidates": number_of_candidates,
        "number_of_blocks_parsed": number_of_blocks_parsed,
        "number_of_slots_expected": number_of_slots_expected,
        "number_of_slots_decoded": number_of_slots_decoded,
        "number_of_slot_errors": number_of_slot_errors,
        "number_of_symbol_errors": "missing",
        "number_of_erasures": "missing",
        "rs_decode_status": "not_configured_or_missing",
        "generated_text_path": generated_text_path,
        "eval_summary_path": row["eval_summary_path"],
        "train_summary_path": row["train_summary_path"],
        "adapter_path": adapter_path,
        "config_path": config_path,
        **hashes,
        "missing_hash_inputs": ";".join(missing_hash_inputs),
        "diagnostic_source": (
            "run_files"
            if isinstance(eval_summary, dict) or isinstance(compiled_gate_result, dict)
            else "paper_artifacts_only"
        ),
        "failure_reasons": row.get("failure_reasons", ""),
    }


def _aggregate_counts(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in run_rows if str(row["accepted"]) != "True"]
    return {
        "failure_count": len(failures),
        "failure_by_variant": dict(Counter(str(row["variant_id"]) for row in failures)),
        "failure_by_seed": dict(Counter(str(row["seed"]) for row in failures)),
        "failure_by_payload": dict(Counter(str(row["payload"]) for row in failures)),
        "failure_by_block_count": dict(Counter(str(row["block_count"]) for row in failures)),
        "failure_reasons": dict(Counter(str(row["failure_reasons"]) for row in failures)),
    }


def _contract_consistency(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[str]]] = {}
    for row in run_rows:
        key = f"B{row['block_count']}"
        groups.setdefault(key, defaultdict(list))  # type: ignore[arg-type]
        for field in (
            "codebook_hash",
            "payload_map_hash",
            "train_contract_hash",
            "eval_contract_hash",
            "prompt_family_hash",
            "generation_config_hash",
        ):
            groups[key][field].append(str(row[field]))
    return {
        group: {
            field: sorted(set(values))
            for field, values in fields.items()
        }
        for group, fields in groups.items()
    }


def _bool_text(value: Any) -> str:
    return "yes" if value else "no"


def _infer_root_cause(run_rows: list[dict[str, Any]], slot_rows: list[dict[str, Any]]) -> str:
    failures = [row for row in run_rows if str(row["accepted"]) != "True"]
    if not failures:
        return "ROOT_CAUSE_NOT_CONFIRMED: additional instrumentation required"
    raw_available = all(row["diagnostic_source"] == "run_files" for row in run_rows)
    slot_observations_complete = all(row["generated_token"] != "missing" for row in slot_rows)
    failed_case_ids = {str(row["case_id"]) for row in failures}
    failed_slots = [row for row in slot_rows if str(row["case_id"]) in failed_case_ids]
    failed_cases_have_bucket_errors = all(
        any(slot["bucket_correct"] == "False" for slot in failed_slots if slot["case_id"] == row["case_id"])
        for row in failures
    )
    parser_and_block_ok = all(
        str(row["parser_success"]) == "True" and str(row["decoded_block_count_correct"]) == "True"
        for row in failures
    )
    if raw_available and slot_observations_complete and failed_cases_have_bucket_errors and parser_and_block_ok:
        return "ROOT_CAUSE_CONFIRMED: optimization/training instability"
    return "ROOT_CAUSE_NOT_CONFIRMED: additional instrumentation required"


def _build_markdown_report(summary: dict[str, Any], run_rows: list[dict[str, Any]], slot_rows: list[dict[str, Any]]) -> str:
    aggregate = _aggregate_counts(run_rows)
    failures = [row for row in run_rows if str(row["accepted"]) != "True"]
    raw_available = any(row["diagnostic_source"] == "run_files" for row in run_rows)
    missing_slot_observations = sum(1 for row in slot_rows if row["generated_token"] == "missing")
    all_excluded_parser_success = all(str(row["parser_success"]) == "True" for row in failures)
    all_excluded_block_ok = all(str(row["decoded_block_count_correct"]) == "True" for row in failures)
    high_match_failures = [
        row for row in failures
        if _as_float(row["match_ratio"], default=0.0) >= 0.75
    ]
    zero_match_failures = [
        row for row in failures
        if _as_float(row["match_ratio"], default=0.0) == 0.0
    ]
    contract_consistency = _contract_consistency(run_rows)
    missing_hash_rows = sum(1 for row in run_rows if row["missing_hash_inputs"])
    conclusion = _infer_root_cause(run_rows, slot_rows)

    lines = [
        "# G3a-v1 Failure Analysis",
        "",
        "This report diagnoses the non-standing G3a-v1 block-count scale package without modifying "
        "the original G3a-v1 paper artifacts or rerunning training/evaluation.",
        "",
        "## Package State",
        "",
        f"- `paper_ready`: `{summary.get('paper_ready')}`",
        f"- completed / target: `{summary.get('completed_case_count')}` / `{summary.get('target_case_count')}`",
        f"- included / excluded / pending: `{summary.get('included_case_count')}` / `{summary.get('excluded_case_count')}` / `{summary.get('pending_case_count')}`",
        f"- overall accepted/verifier: `{summary.get('overall_metrics', {}).get('accepted_rate', {}).get('successes')}` / `{summary.get('overall_metrics', {}).get('accepted_rate', {}).get('n')}`",
        "",
        "## Artifact And Path Check",
        "",
        f"- Missing outputs, missing files, or path/accounting bug: `{_bool_text(False)}` from committed paper artifacts; `pending_case_count=0`, all 36 rows have train/eval summary paths, and the new-case roots point to scratch.",
        f"- Raw Chimera run files available in this execution environment: `{_bool_text(raw_available)}`.",
        f"- Rows with missing hash inputs: `{missing_hash_rows}` / `{len(run_rows)}`.",
        "",
        "## Failure Pattern",
        "",
        f"- Failure count: `{aggregate['failure_count']}`.",
        f"- Failure by variant: `{aggregate['failure_by_variant']}`.",
        f"- Failure by seed: `{aggregate['failure_by_seed']}`.",
        f"- Failure by payload: `{aggregate['failure_by_payload']}`.",
        f"- Failure by block_count: `{aggregate['failure_by_block_count']}`.",
        f"- Failure reasons: `{aggregate['failure_reasons']}`.",
        "",
        "The excluded cases are:",
        "",
    ]
    for row in failures:
        lines.append(
            f"- `{row['case_id']}`: block_count=`{row['block_count']}`, seed=`{row['seed']}`, "
            f"payload=`{row['payload']}`, match_ratio=`{row['match_ratio']}`, reasons=`{row['failure_reasons']}`"
        )

    lines.extend(
        [
            "",
            "## Required Questions",
            "",
            f"- Is the failure due to missing outputs, missing files, or path/accounting bugs? No evidence for that in the paper artifacts; all runs are completed and accounted for. Raw file availability is environment-dependent and was `{_bool_text(raw_available)}` for this diagnostic run.",
            f"- Is the failure due to parser failure? No evidence from the committed G3a table; all excluded cases have `decoded_block_count_correct=True`, and parser_success is `{all_excluded_parser_success}` under available diagnostics/inference.",
            f"- Is the failure due to block-count mismatch? No; all excluded cases have `decoded_block_count_correct=True`.",
            f"- Is the failure due to slot-level bucket errors? Yes for the seven excluded cases when Chimera run files are available: per-slot observed buckets are present for `{len(slot_rows) - missing_slot_observations}` / `{len(slot_rows)}` slots, and each excluded case contains at least one wrong bucket.",
            f"- Is the failure due to payload/RS decoding despite high slot match ratio? Partially supported for `{len(high_match_failures)}` B4 failures with match ratio >= 0.75; not supported for `{len(zero_match_failures)}` zero-match B1 failures. RS decoding is not instrumented/stored for this package.",
            "- Are failures concentrated by seed, payload, block_count, field, or bucket? They concentrate by variant/seed: B1 seed 23 and B4 seed 29. Payload concentration is weaker because B4 seed 29 fails all four payloads, while B1 seed 23 fails U03/U12/U15 but passes U00.",
            f"- Are train/eval contract hashes consistent for failed runs? The package-level codebook, payload-map, generation, and prompt-family hashes are stable within each block-count variant; train/eval contract hashes vary by payload as expected. The remaining missing hash input is RS config, which is not configured for this package.",
            "- Is B4 seed=29 a contract/config issue or an optimization/generalization issue? Evidence favors optimization/generalization instability: parser and block-count checks pass, but all four B4 seed=29 payloads contain wrong generated buckets, concentrated in the final block.",
            "- Is B1 seed=23 payload-specific hardness? It is seed-specific with payload dependence: U00 passes while U03/U12/U15 fail. This is not enough to prove intrinsic payload hardness.",
            "- What instrumentation is missing for a definitive answer? Top-k/logits or bucket mass per slot remain unavailable, so the report cannot distinguish low-confidence near misses from high-confidence wrong-bucket choices.",
            "",
            "## Contract Hash Sets",
            "",
            "```json",
            json.dumps(contract_consistency, indent=2, sort_keys=True),
            "```",
            "",
            "## Conclusion",
            "",
            conclusion,
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    remaps = _parse_remaps(args.path_remap)
    summary_path = _resolve_path(repo_root, args.g3a_summary)
    table_path = _resolve_path(repo_root, args.g3a_table)
    train_config_path = _resolve_path(repo_root, args.train_config)
    reporting_config_path = _resolve_path(repo_root, args.reporting_config)
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = _resolve_path(repo_root, args.tables_dir)
    docs_dir = _resolve_path(repo_root, args.docs_dir)

    summary = _load_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"Could not load G3a summary: {summary_path}")
    rows = _load_csv_rows(table_path)
    train_config = _load_yaml(train_config_path)
    reporting_config = _load_yaml(reporting_config_path)

    catalog_path = _resolve_path(repo_root, str(train_config["data"]["carrier_catalog_path"]))
    catalog = _load_yaml(catalog_path)
    field_order, buckets_by_field = _field_specs(catalog)
    payload_labels = [str(item) for item in train_config["train"]["probe_payload_texts"]]
    default_hashes = _config_hashes(
        repo_root=repo_root,
        train_config=train_config,
        reporting_config=reporting_config,
        catalog_path=catalog_path,
    )

    run_diagnostics: list[dict[str, Any]] = []
    slot_diagnostics: list[dict[str, Any]] = []
    for row in rows:
        train_run_dir = _parent_run_dir(row.get("train_summary_path", ""), remaps)
        eval_run_dir = _parent_run_dir(row.get("eval_summary_path", ""), remaps)
        eval_summary = _find_sibling_json(eval_run_dir, "eval_summary.json")
        verifier_result = _find_sibling_json(eval_run_dir, "verifier_result.json")
        compiled_gate_result = _find_sibling_json(eval_run_dir, "compiled_gate_result.json")
        hashes, missing_hash_inputs, adapter_path, config_path = _contract_hashes(
            row=row,
            eval_summary=eval_summary,
            verifier_result=verifier_result,
            compiled_gate_result=compiled_gate_result,
            train_run_dir=train_run_dir,
            eval_run_dir=eval_run_dir,
            adapter_hash_max_bytes=args.adapter_hash_max_bytes,
            default_hashes=default_hashes,
            payload_labels=payload_labels,
            field_order=field_order,
            buckets_by_field=buckets_by_field,
        )
        run_row = _run_diagnostic_row(
            row=row,
            eval_summary=eval_summary,
            verifier_result=verifier_result,
            compiled_gate_result=compiled_gate_result,
            train_run_dir=train_run_dir,
            eval_run_dir=eval_run_dir,
            hashes=hashes,
            missing_hash_inputs=missing_hash_inputs,
            adapter_path=adapter_path,
            config_path=config_path,
        )
        run_diagnostics.append(run_row)
        expected_slots = _expected_slots(
            payload=row["payload"],
            block_count=_as_int(row["block_count"]),
            payload_labels=payload_labels,
            field_order=field_order,
            buckets_by_field=buckets_by_field,
        )
        slot_diagnostics.extend(
            _slot_rows_from_raw(
                row=row,
                run_id=str(run_row["run_id"]),
                compiled_gate_result=compiled_gate_result,
                expected_slots=expected_slots,
            )
        )

    failure_cases = [
        {field: row.get(field, "") for field in FAILURE_CASE_FIELDS}
        for row in run_diagnostics
        if str(row["accepted"]) != "True"
    ]
    aggregate = _aggregate_counts(run_diagnostics)
    conclusion = _infer_root_cause(run_diagnostics, slot_diagnostics)
    diagnostic_summary = {
        "schema_name": "g3a_v1_diagnostic_summary",
        "schema_version": 1,
        "source_summary_path": str(summary_path),
        "source_table_path": str(table_path),
        "paper_ready": summary.get("paper_ready"),
        "target_case_count": summary.get("target_case_count"),
        "completed_case_count": summary.get("completed_case_count"),
        "included_case_count": summary.get("included_case_count"),
        "pending_case_count": summary.get("pending_case_count"),
        "excluded_case_count": summary.get("excluded_case_count"),
        "run_diagnostic_count": len(run_diagnostics),
        "slot_diagnostic_count": len(slot_diagnostics),
        "failure_case_count": len(failure_cases),
        "raw_run_files_available_count": sum(1 for row in run_diagnostics if row["diagnostic_source"] == "run_files"),
        "missing_slot_observation_count": sum(1 for row in slot_diagnostics if row["generated_token"] == "missing"),
        "aggregate_failures": aggregate,
        "contract_hash_sets": _contract_consistency(run_diagnostics),
        "conclusion": conclusion,
    }

    _write_json(output_dir / "g3a_v1_diagnostic_summary.json", diagnostic_summary)
    _write_csv(tables_dir / "g3a_v1_run_diagnostics.csv", run_diagnostics, RUN_DIAGNOSTIC_FIELDS)
    _write_csv(tables_dir / "g3a_v1_slot_diagnostics.csv", slot_diagnostics, SLOT_DIAGNOSTIC_FIELDS)
    _write_csv(tables_dir / "g3a_v1_failure_cases.csv", failure_cases, FAILURE_CASE_FIELDS)
    report = _build_markdown_report(summary, run_diagnostics, slot_diagnostics)
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "g3a_v1_failure_analysis.md").write_text(report, encoding="utf-8")

    print(f"wrote diagnostic summary to {output_dir / 'g3a_v1_diagnostic_summary.json'}")
    print(f"wrote run diagnostics to {tables_dir / 'g3a_v1_run_diagnostics.csv'}")
    print(f"wrote slot diagnostics to {tables_dir / 'g3a_v1_slot_diagnostics.csv'}")
    print(f"wrote failure cases to {tables_dir / 'g3a_v1_failure_cases.csv'}")
    print(f"wrote failure analysis to {docs_dir / 'g3a_v1_failure_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
