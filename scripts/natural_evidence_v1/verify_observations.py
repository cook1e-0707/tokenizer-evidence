from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.common import read_jsonl, read_yaml, resolve_repo_path, write_csv, write_jsonl
from src.core.payload_codec import BucketPayloadCodec, PayloadCodecError
from src.core.rs_codec import ReedSolomonCodec


GROUP_KEYS = (
    "model_family",
    "model_condition",
    "tokenizer",
    "bucket_bank_id",
    "payload_id",
    "seed",
    "far_family",
    "protocol_id",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode natural_evidence_v1 bucket observations after transcript commitment."
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--observations", required=True)
    parser.add_argument(
        "--output-csv",
        default="results/natural_evidence_v1/tables/four_arm_recovery.csv",
    )
    parser.add_argument(
        "--decoded-jsonl",
        default="results/natural_evidence_v1/decoded_observations/decoded_observations.jsonl",
    )
    return parser.parse_args(argv)


def _payload_text_by_id(config: dict[str, Any]) -> dict[str, str]:
    payloads = config.get("payloads", [])
    if not isinstance(payloads, list):
        return {}
    mapping: dict[str, str] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        payload_id = str(payload.get("payload_id", ""))
        if payload_id:
            mapping[payload_id] = str(payload.get("payload_text", payload_id))
    return mapping


def _query_budgets(config: dict[str, Any]) -> list[int]:
    pilot = config.get("pilot_scale", {})
    if isinstance(pilot, dict) and isinstance(pilot.get("query_budgets"), list):
        return [int(value) for value in pilot["query_budgets"]]
    return [8, 16, 32, 64, 128]


def _sort_observations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("query_index", row.get("query_id", 0))),
            int(row.get("token_position", row.get("token_index", 0))),
            str(row.get("observation_id", "")),
        ),
    )


def _decode_bucket_ids(
    bucket_ids: list[int],
    bucket_tuple_width: int,
    bucket_radix: int,
    rs_parity_symbols: int,
) -> tuple[str, str]:
    usable_count = (len(bucket_ids) // bucket_tuple_width) * bucket_tuple_width
    if usable_count == 0:
        return "", "insufficient_symbols"
    bucket_tuples = [
        bucket_ids[index : index + bucket_tuple_width]
        for index in range(0, usable_count, bucket_tuple_width)
    ]
    codec = BucketPayloadCodec(
        bucket_radices=tuple(bucket_radix for _ in range(bucket_tuple_width)),
        rs_codec=ReedSolomonCodec(parity_symbols=rs_parity_symbols),
    )
    try:
        decoded = codec.decode_bytes(bucket_tuples, apply_rs=rs_parity_symbols > 0)
    except (PayloadCodecError, ValueError) as error:
        return "", f"decode_error:{error}"
    try:
        return decoded.decode("utf-8"), "decoded"
    except UnicodeDecodeError:
        return decoded.hex(), "decoded_non_utf8"


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in GROUP_KEYS)
        grouped[key].append(row)
    return grouped


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    decoder_cfg = dict(config.get("decoder", {}))
    protocol_id = str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1"))
    bucket_tuple_width = int(decoder_cfg.get("bucket_tuple_width", 3))
    bucket_radix = int(decoder_cfg.get("bucket_radix", 8))
    rs_parity_symbols = int(decoder_cfg.get("rs_parity_symbols", 0))
    payload_texts = _payload_text_by_id(config)
    budgets = _query_budgets(config)

    observations = read_jsonl(resolve_repo_path(args.observations, root))
    grouped = _group_rows(observations)
    table_rows: list[dict[str, Any]] = []
    decoded_rows: list[dict[str, Any]] = []
    for group_key, group_observations in sorted(grouped.items()):
        base = {field: group_key[index] for index, field in enumerate(GROUP_KEYS)}
        base["protocol_id"] = base.get("protocol_id") or protocol_id
        sorted_group = _sort_observations(group_observations)
        expected_payload = payload_texts.get(base["payload_id"], base["payload_id"])
        for budget in budgets:
            budget_rows = sorted_group[:budget]
            bucket_ids = [
                int(row["bucket_id"])
                for row in budget_rows
                if row.get("bucket_id") is not None and str(row.get("bucket_id")) != ""
            ]
            recovered_payload, decode_status = _decode_bucket_ids(
                bucket_ids,
                bucket_tuple_width=bucket_tuple_width,
                bucket_radix=bucket_radix,
                rs_parity_symbols=rs_parity_symbols,
            )
            accepted = recovered_payload == expected_payload
            table_row = {
                **base,
                "query_budget": budget,
                "accepted": accepted,
                "recovered_payload": recovered_payload,
                "expected_payload": expected_payload,
                "observed_symbols": len(bucket_ids),
                "usable_symbols": (len(bucket_ids) // bucket_tuple_width) * bucket_tuple_width,
                "erasures": 0,
                "errors": 0,
                "decode_status": decode_status,
                "utility_metric": "NEEDS_RESULTS",
                "naturalness_metric": "NEEDS_RESULTS",
            }
            table_rows.append(table_row)
            decoded_rows.append(
                {
                    **table_row,
                    "bucket_ids_json": json.dumps(bucket_ids),
                }
            )

    fieldnames = [
        "model_family",
        "model_condition",
        "tokenizer",
        "bucket_bank_id",
        "payload_id",
        "seed",
        "query_budget",
        "accepted",
        "recovered_payload",
        "expected_payload",
        "far_family",
        "utility_metric",
        "naturalness_metric",
        "protocol_id",
        "observed_symbols",
        "usable_symbols",
        "erasures",
        "errors",
        "decode_status",
    ]
    write_csv(resolve_repo_path(args.output_csv, root), table_rows, fieldnames)
    write_jsonl(resolve_repo_path(args.decoded_jsonl, root), decoded_rows)
    print(json.dumps({"rows": len(table_rows), "groups": len(grouped)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

