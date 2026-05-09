from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.natural_evidence_v1.common import token_surface_class, write_csv, write_json, write_jsonl
from scripts.natural_evidence_v1.replay_qwen_frame_completion import _as_int, _hash_file, _nonempty, _rate


SCHEMA_NAME = "natural_evidence_qwen_846699_on_policy_survival_v1"
SLICE_FIELDS = [
    "slice_kind",
    "slice_value",
    "observation_rows",
    "compatible_hit_rows",
    "compatible_hit_rate",
    "target_comparable_rows",
    "target_hit_rows",
    "target_hit_rate",
    "erasure_rows",
    "bucket_miss_rows",
    "token_index_out_of_response_rows",
    "observed_token_in_candidate_set_rows",
    "observed_token_in_any_bucket_rows",
    "observed_token_in_target_bucket_rows",
]
COMPACT_FIELDS = [
    "model_condition",
    "payload_id",
    "seed",
    "observation_condition",
    "observation_rows",
    "compatible_hit_rows",
    "compatible_hit_rate",
    "target_comparable_rows",
    "target_hit_rows",
    "target_hit_rate",
    "bucket_miss_rows",
    "token_index_out_of_response_rows",
    "dominant_observed_token_class",
    "dominant_target_bucket_token_class",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze on-policy survival for Qwen 846699 observations by slot/source. "
            "Artifact-only: reads existing JSONL artifacts and never trains, generates, "
            "or claims payload recovery."
        )
    )
    parser.add_argument("--observations-jsonl", required=True)
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--oracle-summary-json", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--payload-ids", default="P0421,P1729")
    parser.add_argument("--tokenizer-name", default="")
    parser.add_argument("--max-examples", type=int, default=240)
    parser.add_argument("--max-token-texts-per-bucket", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _format_float(value: float) -> str:
    return f"{float(value):.17g}"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            yield payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _entry_source(entry_key: str) -> dict[str, str]:
    parts = str(entry_key).split("||")
    return {
        "source_prompt_id": parts[1] if len(parts) > 1 else "",
        "source_prompt_split": parts[2] if len(parts) > 2 else "",
        "source_model_condition": parts[3] if len(parts) > 3 else "",
    }


def _load_train_positions(train_data_dir: Path, payload_ids: Sequence[str]) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    by_prompt_slot: dict[tuple[str, str, int], dict[str, Any]] = {}
    by_entry_key: dict[str, dict[str, Any]] = {}
    payload_counts: dict[str, int] = {}
    for payload_id in payload_ids:
        train_path = train_data_dir / payload_id / "variable_radix_train.jsonl"
        if not train_path.is_file() or train_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing train JSONL for {payload_id}: {train_path}")
        position_count = 0
        for row in _read_jsonl(train_path):
            prompt_id = str(row.get("prompt_id", ""))
            prompt_split = str(row.get("prompt_split", "")) or _entry_source(
                str((row.get("eligible_positions") or [{}])[0].get("entry_key", ""))
                if isinstance(row.get("eligible_positions"), list) and row.get("eligible_positions")
                else ""
            ).get("source_prompt_split", "")
            positions = row.get("eligible_positions", [])
            if not isinstance(positions, list):
                continue
            for prompt_slot, position in enumerate(positions):
                if not isinstance(position, dict):
                    continue
                entry_key = str(position.get("entry_key", ""))
                source = _entry_source(entry_key)
                metadata = {
                    "payload_id": payload_id,
                    "prompt_id": prompt_id,
                    "prompt_slot": prompt_slot,
                    "source_prompt_split": prompt_split or source.get("source_prompt_split", ""),
                    "source_model_condition": source.get("source_model_condition", ""),
                    "source_example_role": str(row.get("example_role", "")),
                    "token_index": _as_int(position.get("token_index", 0)),
                    "frame_index": _as_int(position.get("frame_index", 0)),
                    "frame_digit_index": _as_int(position.get("frame_digit_index", 0)),
                    "frame_digit_count": _as_int(position.get("frame_digit_count", 0)),
                    "payload_digit_index": _as_int(position.get("payload_digit_index", 0)),
                    "target_digit": _as_int(position.get("target_digit", -1), -1),
                    "target_radix": _as_int(position.get("target_radix", 0)),
                    "target_bucket": str(position.get("target_bucket", "")),
                    "compatible_bucket_ids": [str(value) for value in position.get("compatible_bucket_ids", [])],
                    "bucket_to_token_ids": {
                        str(bucket_id): [int(token_id) for token_id in token_ids]
                        for bucket_id, token_ids in dict(position.get("bucket_to_token_ids", {})).items()
                    },
                    "target_bucket_token_ids": [int(value) for value in position.get("target_bucket_token_ids", [])],
                    "candidate_token_ids": [int(value) for value in position.get("candidate_token_ids", [])],
                    "bank_entry_id": str(position.get("bank_entry_id", "")),
                    "entry_key": entry_key,
                }
                by_prompt_slot[(payload_id, prompt_id, prompt_slot)] = metadata
                if entry_key:
                    by_entry_key[entry_key] = metadata
                position_count += 1
        payload_counts[payload_id] = position_count
    return by_prompt_slot, by_entry_key, {
        "train_data_dir": str(train_data_dir),
        "payload_position_counts": payload_counts,
        "metadata_position_count": len(by_prompt_slot),
        "entry_key_count": len(by_entry_key),
    }


def _load_tokenizer(tokenizer_name: str) -> tuple[Any | None, str]:
    if not tokenizer_name:
        return None, "tokenizer_not_requested"
    try:
        from transformers import AutoTokenizer  # type: ignore

        return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True), ""
    except Exception as exc:  # pragma: no cover - exercised on clusters without cache.
        return None, f"{type(exc).__name__}: {exc}"


def _decode_token(tokenizer: Any | None, token_id: int) -> str:
    if tokenizer is None:
        return ""
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return ""


def _bucket_texts(
    *,
    tokenizer: Any | None,
    bucket_to_token_ids: Mapping[str, Sequence[int]],
    max_token_texts_per_bucket: int,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for bucket_id, token_ids in sorted(bucket_to_token_ids.items()):
        rows: list[dict[str, Any]] = []
        for token_id in list(token_ids)[:max_token_texts_per_bucket]:
            token_text = _decode_token(tokenizer, int(token_id))
            rows.append(
                {
                    "token_id": int(token_id),
                    "token_text": token_text,
                    "token_class": token_surface_class(token_text) if token_text else "",
                }
            )
        output[str(bucket_id)] = rows
    return output


def _target_comparable(row: Mapping[str, Any]) -> bool:
    return str(row.get("observation_condition", "")) == "correct_key" and str(row.get("model_condition", "")) in {
        "raw",
        "protected_trained",
        "task_only_lora",
    }


def _metadata_for_observation(
    row: Mapping[str, Any],
    by_prompt_slot: Mapping[tuple[str, str, int], Mapping[str, Any]],
    by_entry_key: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    key = (
        str(row.get("payload_id", "")),
        str(row.get("prompt_id", "")),
        _as_int(row.get("position_index", 0)),
    )
    metadata = by_prompt_slot.get(key)
    if metadata is not None:
        return dict(metadata), "payload_prompt_position"
    entry_key = str(row.get("entry_key", ""))
    metadata = by_entry_key.get(entry_key)
    if metadata is not None:
        return dict(metadata), "entry_key"
    return {}, "missing"


def _empty_stats() -> dict[str, int | Counter[str]]:
    return {
        "observation_rows": 0,
        "compatible_hit_rows": 0,
        "target_comparable_rows": 0,
        "target_hit_rows": 0,
        "erasure_rows": 0,
        "bucket_miss_rows": 0,
        "token_index_out_of_response_rows": 0,
        "observed_token_in_candidate_set_rows": 0,
        "observed_token_in_any_bucket_rows": 0,
        "observed_token_in_target_bucket_rows": 0,
        "observed_token_class_counts": Counter(),
        "target_bucket_token_class_counts": Counter(),
    }


def _update_stats(stats: dict[str, Any], row: Mapping[str, Any]) -> None:
    for field in (
        "observation_rows",
        "compatible_hit_rows",
        "target_comparable_rows",
        "target_hit_rows",
        "erasure_rows",
        "bucket_miss_rows",
        "token_index_out_of_response_rows",
        "observed_token_in_candidate_set_rows",
        "observed_token_in_any_bucket_rows",
        "observed_token_in_target_bucket_rows",
    ):
        stats[field] += int(row.get(field, 0))
    if row.get("observed_token_class"):
        stats["observed_token_class_counts"][str(row["observed_token_class"])] += 1
    if row.get("target_bucket_token_class"):
        stats["target_bucket_token_class_counts"][str(row["target_bucket_token_class"])] += 1


def _stats_row(slice_kind: str, slice_value: str, stats: Mapping[str, Any]) -> dict[str, Any]:
    observations = int(stats["observation_rows"])
    target_comparable = int(stats["target_comparable_rows"])
    return {
        "slice_kind": slice_kind,
        "slice_value": slice_value,
        "observation_rows": observations,
        "compatible_hit_rows": int(stats["compatible_hit_rows"]),
        "compatible_hit_rate": _format_float(_rate(int(stats["compatible_hit_rows"]), observations)),
        "target_comparable_rows": target_comparable,
        "target_hit_rows": int(stats["target_hit_rows"]),
        "target_hit_rate": _format_float(_rate(int(stats["target_hit_rows"]), target_comparable)),
        "erasure_rows": int(stats["erasure_rows"]),
        "bucket_miss_rows": int(stats["bucket_miss_rows"]),
        "token_index_out_of_response_rows": int(stats["token_index_out_of_response_rows"]),
        "observed_token_in_candidate_set_rows": int(stats["observed_token_in_candidate_set_rows"]),
        "observed_token_in_any_bucket_rows": int(stats["observed_token_in_any_bucket_rows"]),
        "observed_token_in_target_bucket_rows": int(stats["observed_token_in_target_bucket_rows"]),
    }


def _dominant(counter: Counter[str]) -> str:
    if not counter:
        return ""
    value, count = counter.most_common(1)[0]
    return f"{value}:{count}"


def _compact_row(key: tuple[str, str, str, str], stats: Mapping[str, Any]) -> dict[str, Any]:
    model_condition, payload_id, seed, observation_condition = key
    base = _stats_row("condition_payload_seed", "|".join(key), stats)
    return {
        "model_condition": model_condition,
        "payload_id": payload_id,
        "seed": seed,
        "observation_condition": observation_condition,
        "observation_rows": base["observation_rows"],
        "compatible_hit_rows": base["compatible_hit_rows"],
        "compatible_hit_rate": base["compatible_hit_rate"],
        "target_comparable_rows": base["target_comparable_rows"],
        "target_hit_rows": base["target_hit_rows"],
        "target_hit_rate": base["target_hit_rate"],
        "bucket_miss_rows": base["bucket_miss_rows"],
        "token_index_out_of_response_rows": base["token_index_out_of_response_rows"],
        "dominant_observed_token_class": _dominant(stats["observed_token_class_counts"]),
        "dominant_target_bucket_token_class": _dominant(stats["target_bucket_token_class_counts"]),
    }


def _slice_values(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        ("model_condition", str(row.get("model_condition", ""))),
        ("payload_id", str(row.get("payload_id", ""))),
        ("seed", str(row.get("seed", ""))),
        ("observation_condition", str(row.get("observation_condition", ""))),
        ("condition_payload_seed", "|".join(str(row.get(key, "")) for key in ("model_condition", "payload_id", "seed", "observation_condition"))),
        ("source_prompt_split", str(row.get("source_prompt_split", ""))),
        ("source_model_condition", str(row.get("source_model_condition", ""))),
        ("source_example_role", str(row.get("source_example_role", ""))),
        ("prompt_slot", str(row.get("prompt_slot", ""))),
        ("token_index", str(row.get("token_index", ""))),
        ("frame_index_mod64", str(_as_int(row.get("frame_index", 0)) % 64)),
        ("frame_digit_index", str(row.get("frame_digit_index", ""))),
        ("frame_digit_count", str(row.get("frame_digit_count", ""))),
        ("payload_digit_index_mod64", str(_as_int(row.get("payload_digit_index", 0)) % 64)),
        ("target_radix", str(row.get("target_radix", ""))),
        ("target_digit", str(row.get("target_digit", ""))),
        ("target_bucket", str(row.get("target_bucket", ""))),
        ("observed_token_class", str(row.get("observed_token_class", ""))),
        ("target_bucket_token_class", str(row.get("target_bucket_token_class", ""))),
        ("erasure_reason", str(row.get("erasure_reason", ""))),
        ("join_source", str(row.get("join_source", ""))),
    ]


def analyze_survival(
    *,
    observations_jsonl: Path,
    train_data_dir: Path,
    oracle_summary_json: Path | None,
    payload_ids: Sequence[str],
    tokenizer_name: str,
    max_examples: int,
    max_token_texts_per_bucket: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    by_prompt_slot, by_entry_key, train_summary = _load_train_positions(train_data_dir, payload_ids)
    tokenizer, tokenizer_error = _load_tokenizer(tokenizer_name)
    total = _empty_stats()
    by_slice: dict[tuple[str, str], dict[str, Any]] = defaultdict(_empty_stats)
    by_compact: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(_empty_stats)
    join_source_counts: Counter[str] = Counter()
    miss_examples: list[dict[str, Any]] = []
    compatible_examples: list[dict[str, Any]] = []
    row_count = 0
    metadata_missing = 0

    for row in _read_jsonl(observations_jsonl):
        row_count += 1
        metadata, join_source = _metadata_for_observation(row, by_prompt_slot, by_entry_key)
        join_source_counts[join_source] += 1
        if not metadata:
            metadata_missing += 1
        observed_token_text = str(row.get("observed_token_text", ""))
        observed_token_id = _as_int(row.get("observed_token_id", -1), -1)
        bucket_to_token_ids = dict(metadata.get("bucket_to_token_ids", {}))
        all_bucket_token_ids = {
            int(token_id)
            for token_ids in bucket_to_token_ids.values()
            for token_id in token_ids
        }
        candidate_token_ids = {int(token_id) for token_id in metadata.get("candidate_token_ids", [])}
        target_bucket = str(metadata.get("target_bucket", ""))
        target_bucket_token_ids = {int(token_id) for token_id in metadata.get("target_bucket_token_ids", [])}
        target_comparable = _target_comparable(row)
        compatible_hit = _nonempty(row.get("digit", "")) and _nonempty(row.get("radix", ""))
        target_hit = target_comparable and compatible_hit and _as_int(row.get("digit", -1), -1) == int(metadata.get("target_digit", -2))
        erasure_reason = str(row.get("erasure_reason", ""))
        target_bucket_token_texts = [
            _decode_token(tokenizer, token_id)
            for token_id in list(target_bucket_token_ids)[:max_token_texts_per_bucket]
        ]
        target_bucket_token_classes = [
            token_surface_class(text)
            for text in target_bucket_token_texts
            if text
        ]
        analysis_row: dict[str, Any] = {
            "observation_rows": 1,
            "compatible_hit_rows": int(compatible_hit),
            "target_comparable_rows": int(target_comparable),
            "target_hit_rows": int(target_hit),
            "erasure_rows": int(bool(erasure_reason)),
            "bucket_miss_rows": int(erasure_reason == "observed_token_not_in_variable_radix_bucket_set"),
            "token_index_out_of_response_rows": int(erasure_reason == "token_index_out_of_response"),
            "observed_token_in_candidate_set_rows": int(observed_token_id in candidate_token_ids),
            "observed_token_in_any_bucket_rows": int(observed_token_id in all_bucket_token_ids),
            "observed_token_in_target_bucket_rows": int(observed_token_id in target_bucket_token_ids),
            "observed_token_class": token_surface_class(observed_token_text) if observed_token_text else "",
            "target_bucket_token_class": target_bucket_token_classes[0] if target_bucket_token_classes else "",
            "model_condition": str(row.get("model_condition", "")),
            "payload_id": str(row.get("payload_id", "")),
            "seed": str(row.get("seed", "")),
            "observation_condition": str(row.get("observation_condition", "")),
            "erasure_reason": erasure_reason,
            "join_source": join_source,
            **metadata,
        }
        _update_stats(total, analysis_row)
        compact_key = (
            str(row.get("model_condition", "")),
            str(row.get("payload_id", "")),
            str(row.get("seed", "")),
            str(row.get("observation_condition", "")),
        )
        _update_stats(by_compact[compact_key], analysis_row)
        for slice_key in _slice_values(analysis_row):
            _update_stats(by_slice[slice_key], analysis_row)

        if erasure_reason == "observed_token_not_in_variable_radix_bucket_set" and len(miss_examples) < max_examples:
            miss_examples.append(
                {
                    "schema_name": "natural_evidence_qwen_846699_bucket_miss_example_v1",
                    "model_condition": row.get("model_condition", ""),
                    "payload_id": row.get("payload_id", ""),
                    "seed": row.get("seed", ""),
                    "observation_condition": row.get("observation_condition", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "prompt_slot": metadata.get("prompt_slot", row.get("position_index", "")),
                    "token_index": metadata.get("token_index", row.get("token_index", "")),
                    "frame_index": metadata.get("frame_index", row.get("frame_index", "")),
                    "frame_digit_index": metadata.get("frame_digit_index", row.get("frame_digit_index", "")),
                    "frame_digit_count": metadata.get("frame_digit_count", row.get("frame_digit_count", "")),
                    "target_digit": metadata.get("target_digit", ""),
                    "target_radix": metadata.get("target_radix", ""),
                    "target_bucket": target_bucket,
                    "compatible_bucket_ids": metadata.get("compatible_bucket_ids", row.get("compatible_bucket_ids", [])),
                    "observed_token_id": row.get("observed_token_id", ""),
                    "observed_token_text": observed_token_text,
                    "observed_token_class": analysis_row["observed_token_class"],
                    "observed_token_in_candidate_set": observed_token_id in candidate_token_ids,
                    "observed_token_in_any_bucket": observed_token_id in all_bucket_token_ids,
                    "observed_token_in_target_bucket": observed_token_id in target_bucket_token_ids,
                    "target_bucket_token_ids": sorted(target_bucket_token_ids),
                    "target_bucket_token_texts": target_bucket_token_texts,
                    "bucket_token_texts": _bucket_texts(
                        tokenizer=tokenizer,
                        bucket_to_token_ids=bucket_to_token_ids,
                        max_token_texts_per_bucket=max_token_texts_per_bucket,
                    ),
                    "source_prompt_split": metadata.get("source_prompt_split", ""),
                    "source_model_condition": metadata.get("source_model_condition", ""),
                    "source_example_role": metadata.get("source_example_role", ""),
                    "join_source": join_source,
                    "entry_key": row.get("entry_key", ""),
                }
            )
        if compatible_hit and len(compatible_examples) < max_examples:
            compatible_examples.append(
                {
                    "schema_name": "natural_evidence_qwen_846699_compatible_hit_example_v1",
                    "model_condition": row.get("model_condition", ""),
                    "payload_id": row.get("payload_id", ""),
                    "seed": row.get("seed", ""),
                    "observation_condition": row.get("observation_condition", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "prompt_slot": metadata.get("prompt_slot", row.get("position_index", "")),
                    "frame_digit_index": metadata.get("frame_digit_index", row.get("frame_digit_index", "")),
                    "target_digit": metadata.get("target_digit", ""),
                    "observed_digit": row.get("digit", ""),
                    "target_hit": bool(target_hit),
                    "observed_token_id": row.get("observed_token_id", ""),
                    "observed_token_text": observed_token_text,
                    "observed_token_class": analysis_row["observed_token_class"],
                    "source_prompt_split": metadata.get("source_prompt_split", ""),
                    "source_model_condition": metadata.get("source_model_condition", ""),
                }
            )

    slice_rows = [
        _stats_row(slice_kind, slice_value, stats)
        for (slice_kind, slice_value), stats in sorted(by_slice.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    compact_rows = [
        _compact_row(key, stats)
        for key, stats in sorted(by_compact.items())
    ]
    observation_hash = _hash_file(observations_jsonl)
    oracle_summary: dict[str, Any] = {}
    oracle_mismatches: list[str] = []
    if oracle_summary_json is not None and str(oracle_summary_json):
        oracle_summary = _read_json(oracle_summary_json)
        oracle_observation_hash = (
            oracle_summary.get("inputs", {})
            .get("observations_jsonl", {})
            .get("sha256", "")
        )
        if oracle_observation_hash and oracle_observation_hash != observation_hash["sha256"]:
            oracle_mismatches.append("observations_jsonl_sha256")
        oracle_observation_rows = (
            oracle_summary.get("inputs", {})
            .get("observations_jsonl", {})
            .get("row_count", None)
        )
        if oracle_observation_rows is not None and int(oracle_observation_rows) != row_count:
            oracle_mismatches.append("observations_jsonl_row_count")

    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_ON_POLICY_SURVIVAL_DIAGNOSTIC",
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "not_full_far": True,
        "result_claim": "on_policy_survival_diagnostic_not_payload_recovery_not_far",
        "inputs": {
            "observations_jsonl": observation_hash,
            "train_data_dir": train_summary,
            "oracle_summary_json": {
                "path": str(oracle_summary_json) if oracle_summary_json is not None else "",
                "status": oracle_summary.get("status", "") if oracle_summary else "",
                "schema_name": oracle_summary.get("schema_name", "") if oracle_summary else "",
                "provenance_mismatches": oracle_mismatches,
            },
            "tokenizer_name": tokenizer_name,
            "tokenizer_load_error": tokenizer_error,
        },
        "aggregate": {
            **_stats_row("all", "all", total),
            "row_count": row_count,
            "metadata_missing_rows": metadata_missing,
            "join_source_counts": _counter_dict(join_source_counts),
            "observed_token_class_counts": _counter_dict(total["observed_token_class_counts"]),
            "target_bucket_token_class_counts": _counter_dict(total["target_bucket_token_class_counts"]),
        },
        "interpretation": {
            "compatible_hit_rate_below_one_percent": _rate(int(total["compatible_hit_rows"]), int(total["observation_rows"])) < 0.01,
            "target_hit_rate_below_one_percent": _rate(int(total["target_hit_rows"]), int(total["target_comparable_rows"])) < 0.01,
            "dominant_erasure_reason": "observed_token_not_in_variable_radix_bucket_set",
            "next_required_diagnostic": "protected_vs_task_only_lift_by_slice",
        },
        "outputs": {
            "survival_by_slice_csv": "qwen_846699_on_policy_survival_by_slice.csv",
            "survival_by_condition_payload_seed_csv": "qwen_846699_on_policy_survival_by_condition_payload_seed.csv",
            "bucket_miss_examples_jsonl": "qwen_846699_on_policy_bucket_miss_examples.jsonl",
            "compatible_hit_examples_jsonl": "qwen_846699_on_policy_compatible_hit_examples.jsonl",
        },
    }
    return summary, slice_rows, compact_rows, miss_examples, compatible_examples


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary": output_dir / "qwen_846699_on_policy_survival_summary.json",
        "by_slice": output_dir / "qwen_846699_on_policy_survival_by_slice.csv",
        "compact": output_dir / "qwen_846699_on_policy_survival_by_condition_payload_seed.csv",
        "miss_examples": output_dir / "qwen_846699_on_policy_bucket_miss_examples.jsonl",
        "compatible_examples": output_dir / "qwen_846699_on_policy_compatible_hit_examples.jsonl",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = _resolve(args.output_dir)
    paths = _output_paths(output_dir)
    for path in paths.values():
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing survival artifact: {path}")
    summary, by_slice, compact, miss_examples, compatible_examples = analyze_survival(
        observations_jsonl=_resolve(args.observations_jsonl),
        train_data_dir=_resolve(args.train_data_dir),
        oracle_summary_json=_resolve(args.oracle_summary_json) if args.oracle_summary_json else None,
        payload_ids=_parse_csv_list(args.payload_ids),
        tokenizer_name=str(args.tokenizer_name),
        max_examples=max(0, int(args.max_examples)),
        max_token_texts_per_bucket=max(1, int(args.max_token_texts_per_bucket)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(paths["summary"], summary)
    write_csv(paths["by_slice"], by_slice, SLICE_FIELDS)
    write_csv(paths["compact"], compact, COMPACT_FIELDS)
    write_jsonl(paths["miss_examples"], miss_examples)
    write_jsonl(paths["compatible_examples"], compatible_examples)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
