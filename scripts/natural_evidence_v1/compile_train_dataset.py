from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.common import read_jsonl, read_yaml, resolve_repo_path, write_json, write_jsonl
from src.core.payload_codec import BucketPayloadCodec
from src.core.rs_codec import ReedSolomonCodec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile natural_evidence_v1 natural-response training rows from "
            "reference outputs and bucket-bank entries."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--reference-outputs", required=True)
    parser.add_argument("--bucket-bank-entries", required=True)
    parser.add_argument("--payload-id", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--contract-json", required=True)
    parser.add_argument("--audit-key-id", default="")
    return parser.parse_args(argv)


def _payload_text(config: dict[str, Any], payload_id: str) -> str:
    payloads = config.get("payloads", [])
    if isinstance(payloads, list):
        for payload in payloads:
            if isinstance(payload, dict) and str(payload.get("payload_id", "")) == payload_id:
                return str(payload.get("payload_text", payload_id))
    return payload_id


def _payload_digits(config: dict[str, Any], payload_text: str) -> list[int]:
    decoder_cfg = dict(config.get("decoder", {}))
    bucket_tuple_width = int(decoder_cfg.get("bucket_tuple_width", 3))
    bucket_radix = int(decoder_cfg.get("bucket_radix", 8))
    rs_parity_symbols = int(decoder_cfg.get("rs_parity_symbols", 0))
    codec = BucketPayloadCodec(
        bucket_radices=tuple(bucket_radix for _ in range(bucket_tuple_width)),
        rs_codec=ReedSolomonCodec(parity_symbols=rs_parity_symbols),
    )
    encoding = codec.encode_bytes(payload_text.encode("utf-8"), apply_rs=rs_parity_symbols > 0)
    return [int(bucket_id) for bucket_tuple in encoding.bucket_tuples for bucket_id in bucket_tuple]


def _bank_by_context(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for entry in entries:
        context_signature = str(entry.get("context_signature", ""))
        if context_signature:
            output[context_signature] = entry
    return output


def _eligible_prefixes(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("eligible_prefixes", "eligible_positions", "prefixes"):
        value = row.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _spaced(prefixes: list[dict[str, Any]], min_spacing_tokens: int, max_positions: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    last_token_index: int | None = None
    for prefix in sorted(prefixes, key=lambda item: int(item.get("token_index", item.get("token_position", 0)))):
        token_index = int(prefix.get("token_index", prefix.get("token_position", 0)))
        if last_token_index is not None and token_index - last_token_index < min_spacing_tokens:
            continue
        selected.append(prefix)
        last_token_index = token_index
        if len(selected) >= max_positions:
            break
    return selected


def _candidate_token_ids(entry: dict[str, Any]) -> list[int]:
    buckets = dict(entry.get("buckets", {}))
    token_ids: list[int] = []
    for values in buckets.values():
        if isinstance(values, list):
            token_ids.extend(int(value) for value in values)
    return sorted(set(token_ids))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    protocol_id = str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1"))
    selector_cfg = dict(config.get("selector", {}))
    bucket_cfg = dict(config.get("bucket_bank", {}))
    audit_key_id = args.audit_key_id or str(selector_cfg.get("audit_key_id", "K001"))
    max_positions = int(bucket_cfg.get("max_evidence_positions_per_response", 4))
    min_spacing = int(bucket_cfg.get("min_spacing_tokens", 12))
    payload_text = _payload_text(config, args.payload_id)
    digits = _payload_digits(config, payload_text)
    if not digits:
        raise ValueError("payload encoding produced no bucket digits")

    reference_rows = read_jsonl(resolve_repo_path(args.reference_outputs, root))
    bank_entries = read_jsonl(resolve_repo_path(args.bucket_bank_entries, root))
    bank = _bank_by_context(bank_entries)

    compiled_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    digit_index = 0
    for row in reference_rows:
        selected_positions: list[dict[str, Any]] = []
        for prefix in _spaced(_eligible_prefixes(row), min_spacing, max_positions):
            context_signature = str(prefix.get("context_signature", prefix.get("prefix_hash", "")))
            entry = bank.get(context_signature)
            if entry is None:
                continue
            target_bucket = digits[digit_index % len(digits)]
            digit_index += 1
            buckets = dict(entry.get("buckets", {}))
            target_bucket_token_ids = [int(value) for value in buckets.get(str(target_bucket), [])]
            selected_positions.append(
                {
                    "token_index": int(prefix.get("token_index", prefix.get("token_position", 0))),
                    "context_signature": context_signature,
                    "bank_entry_id": entry.get("bank_entry_id", ""),
                    "target_bucket": target_bucket,
                    "payload_digit_index": digit_index - 1,
                    "candidate_token_ids": _candidate_token_ids(entry),
                    "target_bucket_token_ids": target_bucket_token_ids,
                    "bucket_to_token_ids": buckets,
                }
            )
        if not selected_positions:
            skipped_rows.append(
                {
                    "prompt_id": row.get("prompt_id", ""),
                    "reason": "no_matching_eligible_bucket_entries",
                }
            )
            continue
        compiled_rows.append(
            {
                "schema_name": "natural_evidence_train_example_v1",
                "protocol_id": protocol_id,
                "prompt_id": row.get("prompt_id", ""),
                "prompt": row.get("prompt", row.get("user_probe", "")),
                "response_text": row.get("response_text", row.get("output_text", "")),
                "payload_id": args.payload_id,
                "audit_key_id": audit_key_id,
                "eligible_positions": selected_positions,
            }
        )

    output_path = resolve_repo_path(args.output_jsonl, root)
    contract_path = resolve_repo_path(args.contract_json, root)
    write_jsonl(output_path, compiled_rows)
    write_json(
        contract_path,
        {
            "schema_name": "natural_evidence_train_contract_v1",
            "protocol_id": protocol_id,
            "payload_id": args.payload_id,
            "payload_text": payload_text,
            "audit_key_id": audit_key_id,
            "reference_outputs": args.reference_outputs,
            "bucket_bank_entries": args.bucket_bank_entries,
            "output_jsonl": args.output_jsonl,
            "example_count": len(compiled_rows),
            "skipped_count": len(skipped_rows),
            "skipped_rows": skipped_rows[:50],
            "claim_control": {
                "contains_field_value_outputs": False,
                "contains_structured_evidence_blocks": False,
                "ready_for_model_training": False,
                "training_adapter_status": "TODO_AFTER_STATIC_VALIDATION",
            },
        },
    )
    print(json.dumps({"examples": len(compiled_rows), "skipped": len(skipped_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

