from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.build_bucket_bank import _bucketize
from scripts.natural_evidence_v1.common import read_jsonl, read_yaml, resolve_repo_path, stable_hash_hex, write_csv, write_json, write_jsonl
from src.core.payload_codec import BucketPayloadCodec, PayloadCodecError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CPU raw/wrong-key pre-null diagnostic from frozen raw reference "
            "outputs and compatibility-scored natural opportunity candidates."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--reference-outputs", required=True)
    parser.add_argument("--compatibility-jsonl", required=True)
    parser.add_argument("--compatibility-by-entry-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wrong-key-count", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=50)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _payload_texts(config: dict[str, Any]) -> list[tuple[str, str]]:
    payloads = config.get("payloads", [])
    if not isinstance(payloads, list):
        return []
    rows: list[tuple[str, str]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        payload_id = str(payload.get("payload_id", ""))
        if payload_id:
            rows.append((payload_id, str(payload.get("payload_text", payload_id))))
    return rows


def _diagnostic_query_budgets(config: dict[str, Any]) -> list[int]:
    diagnostic = config.get("diagnostic_high_risk_pilot_scale", {})
    if isinstance(diagnostic, dict) and isinstance(diagnostic.get("query_budgets"), list):
        return [int(value) for value in diagnostic["query_budgets"]]
    return [64, 128, 256, 512]


def _diagnostic_seeds(config: dict[str, Any]) -> list[int]:
    diagnostic = config.get("diagnostic_high_risk_pilot_scale", {})
    if isinstance(diagnostic, dict) and isinstance(diagnostic.get("seeds"), list):
        return [int(value) for value in diagnostic["seeds"]]
    return [17, 23]


def _model_config(config: dict[str, Any], tokenizer_key: str) -> dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, dict) or tokenizer_key not in models:
        raise ValueError(f"Missing model config for tokenizer key {tokenizer_key!r}")
    model_cfg = models[tokenizer_key]
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Model config for {tokenizer_key!r} must be a mapping")
    return model_cfg


def _reference_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("prompt_id", "")): row for row in rows if row.get("prompt_id")}


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded.get("input_ids", [])
    if not isinstance(token_ids, list):
        return []
    return [int(token_id) for token_id in token_ids]


def _min1_entry_ids(rows: list[dict[str, str]]) -> set[str]:
    return {str(row.get("bank_entry_id", "")) for row in rows if row.get("bank_entry_id") and _as_bool(row.get("would_accept_min1", ""))}


def _compatible_candidates_by_entry(rows: list[dict[str, Any]], min1_entry_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        entry_id = str(row.get("bank_entry_id", ""))
        if entry_id not in min1_entry_ids or not _as_bool(row.get("compatibility_pass", False)):
            continue
        try:
            token_id = int(row.get("token_id", -1))
        except (TypeError, ValueError):
            continue
        try:
            probability = float(row.get("probability", 1.0))
        except (TypeError, ValueError):
            probability = 1.0
        grouped[entry_id].append(
            {
                "bank_entry_id": entry_id,
                "prompt_id": str(row.get("prompt_id", "")),
                "prefix_response_token_count": int(row.get("prefix_response_token_count", 0)),
                "bucket_id": str(row.get("bucket_id", "")),
                "token_id": token_id,
                "token_text": str(row.get("token_text", row.get("text", ""))),
                "probability": probability,
                "rank": row.get("rank", ""),
                "context_signature": str(row.get("context_signature", "")),
                "protocol_id": str(row.get("protocol_id", "natural_evidence_v1")),
            }
        )
    return grouped


def _entry_metadata(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    first = candidates[0]
    return {
        "bank_entry_id": str(first["bank_entry_id"]),
        "prompt_id": str(first["prompt_id"]),
        "prefix_response_token_count": int(first["prefix_response_token_count"]),
        "context_signature": str(first["context_signature"]),
        "protocol_id": str(first["protocol_id"]),
    }


def _correct_key_token_to_bucket(candidates: list[dict[str, Any]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for candidate in candidates:
        bucket_id = str(candidate.get("bucket_id", ""))
        if bucket_id == "":
            continue
        mapping[int(candidate["token_id"])] = int(bucket_id)
    return mapping


def _wrong_key_token_to_bucket(
    *,
    candidates: list[dict[str, Any]],
    wrong_key_id: str,
    bucket_count: int,
    protocol_id: str,
    bank_id: str,
    prefix_signature: str,
    bucket_assignment: str,
) -> dict[int, int]:
    deduped: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        token_id = int(candidate["token_id"])
        deduped[token_id] = {
            "token_id": token_id,
            "token_text": str(candidate.get("token_text", "")),
            "probability": float(candidate.get("probability", 1.0)),
            "rank": candidate.get("rank", ""),
        }
    buckets = _bucketize(
        candidates=list(deduped.values()),
        bucket_count=bucket_count,
        min_members_per_bucket=1,
        key=wrong_key_id,
        protocol_id=protocol_id,
        bank_id=bank_id,
        prefix_signature=prefix_signature,
        assignment_mode=bucket_assignment,
    )
    mapping: dict[int, int] = {}
    for bucket_id, members in buckets.items():
        for member in members:
            mapping[int(member["token_id"])] = int(bucket_id)
    return mapping


def _observed_token_id(tokenizer: Any, reference: dict[str, Any] | None, offset: int) -> tuple[int | None, str]:
    if reference is None:
        return None, "missing_reference"
    response = str(reference.get("response_text", reference.get("output_text", "")))
    response_ids = _token_ids(tokenizer, response)
    if offset < 0 or offset >= len(response_ids):
        return None, "offset_out_of_response"
    return int(response_ids[offset]), ""


def _ordered_entries(entries: list[dict[str, Any]], seed: int, condition: str) -> list[dict[str, Any]]:
    return sorted(
        entries,
        key=lambda row: stable_hash_hex(
            [
                "natural_evidence_v1_pre_null_order",
                seed,
                condition,
                row["bank_entry_id"],
                row["prompt_id"],
                row["prefix_response_token_count"],
            ]
        ),
    )


def _decode_bucket_ids(bucket_ids: list[int], *, bucket_tuple_width: int, bucket_radix: int, rs_parity_symbols: int) -> tuple[str, str]:
    usable_count = (len(bucket_ids) // bucket_tuple_width) * bucket_tuple_width
    if usable_count == 0:
        return "", "insufficient_symbols"
    bucket_tuples = [
        bucket_ids[index : index + bucket_tuple_width]
        for index in range(0, usable_count, bucket_tuple_width)
    ]
    codec = BucketPayloadCodec(bucket_radices=tuple(bucket_radix for _ in range(bucket_tuple_width)))
    try:
        decoded = codec.decode_bytes(bucket_tuples, apply_rs=rs_parity_symbols > 0)
    except (PayloadCodecError, ValueError) as error:
        return "", f"decode_error:{error}"
    try:
        return decoded.decode("utf-8"), "decoded"
    except UnicodeDecodeError:
        return decoded.hex(), "decoded_non_utf8"


def _condition_entries(
    *,
    tokenizer: Any,
    references: dict[str, dict[str, Any]],
    compatible_by_entry: dict[str, list[dict[str, Any]]],
    condition: str,
    token_to_bucket_by_entry: dict[str, dict[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry_id, candidates in compatible_by_entry.items():
        if not candidates:
            continue
        meta = _entry_metadata(candidates)
        observed_token_id, erasure_reason = _observed_token_id(
            tokenizer,
            references.get(str(meta["prompt_id"])),
            int(meta["prefix_response_token_count"]),
        )
        token_to_bucket = token_to_bucket_by_entry.get(entry_id, {})
        bucket_id = token_to_bucket.get(observed_token_id) if observed_token_id is not None else None
        if bucket_id is None and not erasure_reason:
            erasure_reason = "observed_token_not_in_compatible_bucket_set"
        rows.append(
            {
                **meta,
                "condition": condition,
                "observed_token_id": "" if observed_token_id is None else observed_token_id,
                "bucket_id": "" if bucket_id is None else bucket_id,
                "erasure": bucket_id is None,
                "erasure_reason": erasure_reason,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    model_cfg = _model_config(config, args.tokenizer_key)
    tokenizer_name = str(model_cfg.get("tokenizer_name", model_cfg.get("model_name", "")))
    if not tokenizer_name:
        raise ValueError(f"Missing tokenizer name for {args.tokenizer_key}")

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("raw_wrong_key_pre_null requires transformers tokenizer support") from error

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    references = _reference_index(read_jsonl(resolve_repo_path(args.reference_outputs, root)))
    min1_ids = _min1_entry_ids(_read_csv(resolve_repo_path(args.compatibility_by_entry_csv, root)))
    compatible_rows = read_jsonl(resolve_repo_path(args.compatibility_jsonl, root))
    compatible_by_entry = _compatible_candidates_by_entry(compatible_rows, min1_ids)
    bucket_cfg = dict(config.get("bucket_bank", {}))
    selector_cfg = dict(config.get("selector", {}))
    decoder_cfg = dict(config.get("decoder", {}))
    protocol_id = str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1"))
    bank_id = f"{args.tokenizer_key}_natural_bucket_bank_v1"
    bucket_count = int(dict(bucket_cfg.get("compatibility_adjusted_capacity", {})).get("diagnostic_high_risk_gate", {}).get("bucket_count", 4))
    bucket_assignment = str(bucket_cfg.get("bucket_assignment", "keyed_mass_balance"))
    audit_key_id = str(selector_cfg.get("audit_key_id", "K001"))
    wrong_key_count = args.wrong_key_count or int(dict(config.get("null_evaluations", {})).get("wrong_key_count", 4))
    query_budgets = _diagnostic_query_budgets(config)
    seeds = _diagnostic_seeds(config)
    payloads = _payload_texts(config)
    bucket_tuple_width = int(decoder_cfg.get("bucket_tuple_width", 3))
    rs_parity_symbols = int(decoder_cfg.get("rs_parity_symbols", 0))

    correct_mapping = {
        entry_id: _correct_key_token_to_bucket(candidates)
        for entry_id, candidates in compatible_by_entry.items()
    }
    condition_mappings: dict[str, dict[str, dict[int, int]]] = {"raw_correct_key": correct_mapping}
    for wrong_index in range(wrong_key_count):
        condition = f"wrong_key_{wrong_index}"
        wrong_key_id = f"{audit_key_id}_WRONG_{wrong_index}"
        condition_mappings[condition] = {}
        for entry_id, candidates in compatible_by_entry.items():
            if not candidates:
                continue
            meta = _entry_metadata(candidates)
            condition_mappings[condition][entry_id] = _wrong_key_token_to_bucket(
                candidates=candidates,
                wrong_key_id=wrong_key_id,
                bucket_count=bucket_count,
                protocol_id=protocol_id,
                bank_id=bank_id,
                prefix_signature=str(meta["context_signature"]),
                bucket_assignment=bucket_assignment,
            )

    condition_rows = {
        condition: _condition_entries(
            tokenizer=tokenizer,
            references=references,
            compatible_by_entry=compatible_by_entry,
            condition=condition,
            token_to_bucket_by_entry=token_to_bucket_by_entry,
        )
        for condition, token_to_bucket_by_entry in condition_mappings.items()
    }

    observation_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    for condition, rows in condition_rows.items():
        for seed in seeds:
            ordered = _ordered_entries(rows, seed, condition)
            for payload_id, payload_text in payloads:
                for budget in query_budgets:
                    budget_rows = ordered[:budget]
                    bucket_ids = [
                        int(row["bucket_id"])
                        for row in budget_rows
                        if row.get("bucket_id") not in {None, ""}
                    ]
                    recovered_payload, decode_status = _decode_bucket_ids(
                        bucket_ids,
                        bucket_tuple_width=bucket_tuple_width,
                        bucket_radix=bucket_count,
                        rs_parity_symbols=rs_parity_symbols,
                    )
                    accepted = recovered_payload == payload_text
                    decode_rows.append(
                        {
                            "model_family": str(model_cfg.get("family", args.tokenizer_key)),
                            "model_condition": "raw",
                            "tokenizer": tokenizer_name,
                            "bucket_bank_id": bank_id,
                            "payload_id": payload_id,
                            "seed": seed,
                            "query_budget": budget,
                            "condition": condition,
                            "accepted": accepted,
                            "recovered_payload": recovered_payload,
                            "expected_payload": payload_text,
                            "observed_symbols": len(bucket_ids),
                            "erasures": budget - len(bucket_ids),
                            "usable_symbols": (len(bucket_ids) // bucket_tuple_width) * bucket_tuple_width,
                            "decode_status": decode_status,
                            "protocol_id": protocol_id,
                            "result_claim": "raw_wrong_key_pre_null_diagnostic_not_full_far",
                        }
                    )
            for query_index, row in enumerate(ordered[: max(query_budgets, default=0)]):
                observation_rows.append(
                    {
                        **row,
                        "query_index": query_index,
                        "seed": seed,
                        "model_condition": "raw",
                        "tokenizer": tokenizer_name,
                        "bucket_bank_id": bank_id,
                        "protocol_id": protocol_id,
                    }
                )

    accept_rows = [row for row in decode_rows if row["accepted"]]
    condition_summary = []
    for condition, rows in condition_rows.items():
        erasures = sum(1 for row in rows if row["erasure"])
        condition_summary.append(
            {
                "condition": condition,
                "eligible_entries": len(rows),
                "observed_symbols": len(rows) - erasures,
                "erasures": erasures,
                "observation_rate": (len(rows) - erasures) / len(rows) if rows else 0.0,
            }
        )

    output_dir = resolve_repo_path(args.output_dir, root)
    observations_path = output_dir / "raw_wrong_key_pre_null_observations.jsonl"
    decodes_path = output_dir / "raw_wrong_key_pre_null_decodes.csv"
    condition_summary_path = output_dir / "raw_wrong_key_pre_null_condition_summary.csv"
    examples_path = output_dir / "raw_wrong_key_pre_null_accept_examples.jsonl"
    summary_path = output_dir / "raw_wrong_key_pre_null_summary.json"
    write_jsonl(observations_path, observation_rows)
    decode_fieldnames = [
        "model_family",
        "model_condition",
        "tokenizer",
        "bucket_bank_id",
        "payload_id",
        "seed",
        "query_budget",
        "condition",
        "accepted",
        "recovered_payload",
        "expected_payload",
        "observed_symbols",
        "erasures",
        "usable_symbols",
        "decode_status",
        "protocol_id",
        "result_claim",
    ]
    write_csv(decodes_path, decode_rows, decode_fieldnames)
    write_csv(condition_summary_path, condition_summary, ["condition", "eligible_entries", "observed_symbols", "erasures", "observation_rate"])
    write_jsonl(examples_path, accept_rows[: max(0, args.max_examples)])
    summary = {
        "schema_name": "natural_evidence_raw_wrong_key_pre_null_summary_v1",
        "protocol_id": protocol_id,
        "tokenizer_key": args.tokenizer_key,
        "tokenizer_name": tokenizer_name,
        "reference_outputs": str(resolve_repo_path(args.reference_outputs, root)),
        "compatibility_jsonl": str(resolve_repo_path(args.compatibility_jsonl, root)),
        "compatibility_by_entry_csv": str(resolve_repo_path(args.compatibility_by_entry_csv, root)),
        "min1_compatible_entries": len(min1_ids),
        "entries_with_compatible_candidate_rows": len(compatible_by_entry),
        "conditions": [row["condition"] for row in condition_summary],
        "payload_ids": [payload_id for payload_id, _ in payloads],
        "seeds": seeds,
        "query_budgets": query_budgets,
        "bucket_count": bucket_count,
        "bucket_tuple_width": bucket_tuple_width,
        "decode_rows": len(decode_rows),
        "accept_count": len(accept_rows),
        "accepted": bool(accept_rows),
        "pre_null_status": "HIGH_RISK_ACCEPT" if accept_rows else "PASS_NO_ACCEPTS_DIAGNOSTIC",
        "condition_summary": condition_summary,
        "not_full_far": True,
        "result_claim": "raw_wrong_key_pre_null_diagnostic_not_full_far",
        "outputs": {
            "observations_jsonl": str(observations_path),
            "decodes_csv": str(decodes_path),
            "condition_summary_csv": str(condition_summary_path),
            "accept_examples_jsonl": str(examples_path),
            "summary_json": str(summary_path),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
