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
from typing import Any, Mapping, Sequence

from scripts.natural_evidence_v1.common import read_jsonl, read_yaml, resolve_repo_path, write_csv, write_json, write_jsonl
from scripts.natural_evidence_v1.diagnose_verifier_alignment import (
    _as_bool,
    _compatible_token_maps,
    _generated_key,
    _lcp_length,
    _load_relevant_entries,
    _load_tokenizer,
    _min1_entry_ids,
    _observation_generated_key,
    _read_csv,
    _token_ids,
)
from scripts.natural_evidence_v1.evaluate_diagnostic_e2e import _decode_bucket_ids


SCHEMA_NAME = "natural_evidence_actual_prefix_salvage_diagnostic_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only salvage diagnostic for retained natural_evidence_v1 E2E "
            "artifacts. This estimates how many symbols are recoverable if the "
            "static exact-prefix veto is relaxed, without model scoring or training."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--generated-outputs", required=True)
    parser.add_argument("--bucket-observations", required=True)
    parser.add_argument("--bucket-bank-entries", required=True)
    parser.add_argument("--compatibility-jsonl", required=True)
    parser.add_argument("--compatibility-by-entry-csv", required=True)
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--bucket-count", type=int, default=4)
    parser.add_argument("--query-budgets", default="")
    parser.add_argument("--lcp-thresholds", default="0,1,4,8,16,32")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=80)
    return parser.parse_args(argv)


def _parse_int_list(value: str, fallback: Sequence[int]) -> list[int]:
    if not value.strip():
        return [int(item) for item in fallback]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def _decode_trace_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "present": True,
        "rows": len(rows),
        "accepted": sum(1 for row in rows if _as_bool(row.get("accepted", ""))),
        "condition_counts": dict(Counter(row.get("model_condition", "") for row in rows)),
    }


def _entry_offset(entry: Mapping[str, Any], observation: Mapping[str, Any]) -> int:
    for key in ("prefix_response_token_count", "token_index", "token_position", "prefix_token_count"):
        if str(observation.get(key, "")) != "":
            return int(observation[key])
        if str(entry.get(key, "")) != "":
            return int(entry[key])
    return 0


def _response_lcp(
    *,
    tokenizer: Any,
    generated: Mapping[str, Any],
    entry: Mapping[str, Any],
    offset: int,
) -> tuple[int, int]:
    response_ids = _token_ids(tokenizer, str(generated.get("response_text", "")))
    reference_prefix = [int(token_id) for token_id in entry.get("prefix_token_ids", [])]
    reference_response_prefix = reference_prefix[max(0, len(reference_prefix) - offset) :]
    actual_response_prefix = response_ids[:offset]
    return _lcp_length(actual_response_prefix, reference_response_prefix), len(reference_response_prefix)


def _actual_observed_token_id(
    *,
    tokenizer: Any,
    generated: Mapping[str, Any],
    offset: int,
) -> int | None:
    response_ids = _token_ids(tokenizer, str(generated.get("response_text", "")))
    if offset >= len(response_ids):
        return None
    return int(response_ids[offset])


def _unit_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model_condition", "")),
        str(row.get("payload_id", "")),
        str(row.get("seed", "")),
        str(row.get("observation_condition", "")),
    )


def _symbol_for_mode(row: Mapping[str, Any], mode: str, threshold: int) -> int | None:
    if mode == "strict":
        value = row.get("strict_bucket_id", "")
    else:
        if row.get("actual_bucket_id", "") == "":
            return None
        if mode.startswith("lcp_ge_") and int(row.get("response_lcp_tokens", 0)) < threshold:
            return None
        value = row.get("actual_bucket_id", "")
    if value == "":
        return None
    return int(value)


def _summarize_mode(
    *,
    rows: Sequence[Mapping[str, Any]],
    mode: str,
    threshold: int,
) -> dict[str, Any]:
    total = len(rows)
    observed = sum(1 for row in rows if _symbol_for_mode(row, mode, threshold) is not None)
    strict_observed = sum(1 for row in rows if row.get("strict_bucket_id", "") != "")
    return {
        "salvage_mode": mode,
        "lcp_threshold": threshold,
        "observations": total,
        "strict_observed_symbols": strict_observed,
        "salvaged_observed_symbols": observed,
        "strict_observed_symbol_rate": strict_observed / max(1, total),
        "salvaged_observed_symbol_rate": observed / max(1, total),
        "salvage_gain_symbols": observed - strict_observed,
        "salvage_gain_rate": (observed - strict_observed) / max(1, total),
        "actual_token_not_in_static_bucket": sum(1 for row in rows if row.get("actual_token_id", "") != "" and row.get("actual_bucket_id", "") == ""),
        "offset_out_of_response": sum(1 for row in rows if _as_bool(row.get("offset_out_of_response", False))),
    }


def _decode_rows_for_mode(
    *,
    rows: Sequence[Mapping[str, Any]],
    mode: str,
    threshold: int,
    query_budgets: Sequence[int],
    expected_payload: str,
    bucket_tuple_width: int,
    bucket_count: int,
    rs_parity_symbols: int,
    base: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: (int(row.get("query_index", 0)), int(row.get("position_index", 0))))
    output: list[dict[str, Any]] = []
    for budget in query_budgets:
        budget_rows = [row for row in sorted_rows if int(row.get("query_index", 0)) < int(budget)]
        bucket_ids = [
            symbol
            for row in budget_rows
            for symbol in [_symbol_for_mode(row, mode, threshold)]
            if symbol is not None
        ]
        recovered_payload, decode_status = _decode_bucket_ids(
            bucket_ids,
            bucket_tuple_width=bucket_tuple_width,
            bucket_radix=bucket_count,
            rs_parity_symbols=rs_parity_symbols,
        )
        output.append(
            {
                **base,
                "salvage_mode": mode,
                "lcp_threshold": threshold,
                "query_budget": int(budget),
                "eligible_positions": len(budget_rows),
                "observed_symbols": len(bucket_ids),
                "usable_symbols": (len(bucket_ids) // bucket_tuple_width) * bucket_tuple_width,
                "erasures": len(budget_rows) - len(bucket_ids),
                "recovered_payload": recovered_payload,
                "expected_payload": expected_payload,
                "accepted": recovered_payload == expected_payload,
                "decode_status": decode_status,
                "result_claim": "actual_prefix_salvage_diagnostic_not_payload_recovery",
            }
        )
    return output


def run_salvage(
    *,
    config_path: Path,
    generated_outputs_path: Path,
    observations_path: Path,
    bucket_bank_entries_path: Path,
    compatibility_jsonl_path: Path,
    compatibility_by_entry_csv_path: Path,
    tokenizer_name: str,
    bucket_count: int,
    query_budgets: Sequence[int] | None,
    lcp_thresholds: Sequence[int],
    output_dir: Path,
    max_examples: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_yaml(config_path)
    scale = dict(config.get("diagnostic_high_risk_pilot_scale", {}))
    budgets = list(query_budgets or [int(value) for value in scale.get("query_budgets", [64, 128, 256, 512])])
    decoder_cfg = dict(config.get("decoder", {}))
    bucket_tuple_width = int(decoder_cfg.get("bucket_tuple_width", 3))
    rs_parity_symbols = int(decoder_cfg.get("rs_parity_symbols", 0))
    payload_texts = _payload_text_by_id(config)

    generated_rows = read_jsonl(generated_outputs_path)
    observation_rows = read_jsonl(observations_path)
    generated_by_key = {_generated_key(row): row for row in generated_rows}
    relevant_entry_ids = {str(row.get("bank_entry_id", "")) for row in observation_rows if row.get("bank_entry_id")}
    entries_by_id = _load_relevant_entries(bucket_bank_entries_path, relevant_entry_ids)
    min1_ids = _min1_entry_ids(_read_csv(compatibility_by_entry_csv_path))
    token_maps = _compatible_token_maps(
        entries_by_id=entries_by_id,
        compatibility_rows=read_jsonl(compatibility_jsonl_path),
        min1_entry_ids=min1_ids,
        bucket_count=bucket_count,
    )
    tokenizer = _load_tokenizer(tokenizer_name)

    salvage_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for observation in observation_rows:
        generated = generated_by_key.get(_observation_generated_key(observation))
        entry_id = str(observation.get("bank_entry_id", ""))
        entry = entries_by_id.get(entry_id)
        token_map = token_maps.get(entry_id, {})
        if generated is None or entry is None:
            continue
        offset = _entry_offset(entry, observation)
        actual_token_id = _actual_observed_token_id(tokenizer=tokenizer, generated=generated, offset=offset)
        actual_bucket = token_map.get(actual_token_id) if actual_token_id is not None else None
        response_lcp, response_prefix_len = _response_lcp(
            tokenizer=tokenizer,
            generated=generated,
            entry=entry,
            offset=offset,
        )
        strict_bucket = observation.get("bucket_id", "")
        row = {
            "model_condition": observation.get("model_condition", ""),
            "payload_id": observation.get("payload_id", ""),
            "seed": observation.get("seed", ""),
            "observation_condition": observation.get("observation_condition", ""),
            "prompt_id": observation.get("prompt_id", ""),
            "query_index": observation.get("query_index", ""),
            "position_index": observation.get("position_index", ""),
            "bank_entry_id": entry_id,
            "prefix_response_token_count": offset,
            "strict_prefix_match": observation.get("strict_prefix_match", ""),
            "strict_bucket_id": strict_bucket,
            "actual_token_id": "" if actual_token_id is None else actual_token_id,
            "actual_bucket_id": "" if actual_bucket is None else actual_bucket,
            "offset_out_of_response": actual_token_id is None,
            "response_lcp_tokens": response_lcp,
            "response_lcp_fraction": response_lcp / max(1, response_prefix_len),
            "reference_response_prefix_len": response_prefix_len,
        }
        salvage_rows.append(row)
        if len(examples) < max_examples and strict_bucket == "" and actual_bucket is not None:
            examples.append(
                {
                    **row,
                    "response_excerpt": str(generated.get("response_text", ""))[:320],
                    "result_claim": "actual_prefix_salvage_example_not_payload_recovery",
                }
            )

    modes = [("strict", -1), ("ignore_strict_static_bucket", 0)]
    modes.extend((f"lcp_ge_{threshold}", int(threshold)) for threshold in lcp_thresholds if int(threshold) > 0)
    by_unit_groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in salvage_rows:
        by_unit_groups[_unit_key(row)].append(row)

    by_unit_rows: list[dict[str, Any]] = []
    for key, rows in sorted(by_unit_groups.items()):
        model_condition, payload_id, seed, observation_condition = key
        for mode, threshold in modes:
            by_unit_rows.append(
                {
                    "model_condition": model_condition,
                    "payload_id": payload_id,
                    "seed": seed,
                    "observation_condition": observation_condition,
                    **_summarize_mode(rows=rows, mode=mode, threshold=threshold),
                }
            )

    decode_rows: list[dict[str, Any]] = []
    for key, rows in sorted(by_unit_groups.items()):
        model_condition, payload_id, seed, observation_condition = key
        expected_payload = payload_texts.get(payload_id)
        if not expected_payload:
            continue
        for mode, threshold in modes:
            decode_rows.extend(
                _decode_rows_for_mode(
                    rows=rows,
                    mode=mode,
                    threshold=threshold,
                    query_budgets=budgets,
                    expected_payload=expected_payload,
                    bucket_tuple_width=bucket_tuple_width,
                    bucket_count=bucket_count,
                    rs_parity_symbols=rs_parity_symbols,
                    base={
                        "model_condition": model_condition,
                        "payload_id": payload_id,
                        "seed": seed,
                        "observation_condition": observation_condition,
                    },
                )
            )

    strict_total = _summarize_mode(rows=salvage_rows, mode="strict", threshold=-1)
    ignore_total = _summarize_mode(rows=salvage_rows, mode="ignore_strict_static_bucket", threshold=0)
    protected_ignore_rows = [
        row for row in by_unit_rows
        if row["model_condition"] == "protected_trained" and row["salvage_mode"] == "ignore_strict_static_bucket"
    ]
    protected_ignore_rate = (
        sum(float(row["salvaged_observed_symbols"]) for row in protected_ignore_rows)
        / max(1.0, sum(float(row["observations"]) for row in protected_ignore_rows))
    )
    accepted_rows = [row for row in decode_rows if _as_bool(row.get("accepted", ""))]
    status = (
        "SALVAGE_HAS_ACCEPTS_DIAGNOSTIC_ONLY"
        if accepted_rows
        else "NO_PAYLOAD_RECOVERY_UNDER_STATIC_BUCKET_SALVAGE"
    )
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": status,
        "paper_claim_allowed": False,
        "tokenizer": tokenizer_name,
        "salvage_scope": "retained_correct_key_observations_only_when_wrong_key_rows_were_not_persisted",
        "salvage_mode_note": (
            "CPU upper bound using existing compatibility-filtered static bucket token sets at generated offsets; "
            "true actual-prefix opportunity discovery still requires reference-model top-k scoring."
        ),
        "generated_output_rows": len(generated_rows),
        "input_observation_rows": len(observation_rows),
        "salvage_rows": len(salvage_rows),
        "strict_total": strict_total,
        "ignore_strict_static_bucket_total": ignore_total,
        "protected_ignore_strict_static_bucket_observed_rate": protected_ignore_rate,
        "decode_rows": len(decode_rows),
        "accepted_rows": len(accepted_rows),
        "accepted_row_breakdown": dict(Counter(str(row.get("model_condition", "")) for row in accepted_rows)),
        "next_recommended_action": (
            "If static-bucket salvage does not recover payload or enough protected symbols, do not run GPU; "
            "repair construction toward true actual-prefix scoring and selector policy."
        ),
        "result_claim": "actual_prefix_salvage_diagnostic_not_payload_recovery",
        "outputs": {
            "summary_json": str(output_dir / "actual_prefix_salvage_summary.json"),
            "by_unit_csv": str(output_dir / "actual_prefix_salvage_by_unit.csv"),
            "decode_csv": str(output_dir / "actual_prefix_salvage_decode.csv"),
            "examples_jsonl": str(output_dir / "actual_prefix_salvage_examples.jsonl"),
        },
    }
    write_json(output_dir / "actual_prefix_salvage_summary.json", summary)
    write_csv(output_dir / "actual_prefix_salvage_by_unit.csv", by_unit_rows, list(by_unit_rows[0].keys()) if by_unit_rows else [])
    write_csv(output_dir / "actual_prefix_salvage_decode.csv", decode_rows, list(decode_rows[0].keys()) if decode_rows else [])
    write_jsonl(output_dir / "actual_prefix_salvage_examples.jsonl", examples)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = resolve_repo_path(args.config, root)
    config = read_yaml(config_path)
    scale = dict(config.get("diagnostic_high_risk_pilot_scale", {}))
    summary = run_salvage(
        config_path=config_path,
        generated_outputs_path=resolve_repo_path(args.generated_outputs, root),
        observations_path=resolve_repo_path(args.bucket_observations, root),
        bucket_bank_entries_path=resolve_repo_path(args.bucket_bank_entries, root),
        compatibility_jsonl_path=resolve_repo_path(args.compatibility_jsonl, root),
        compatibility_by_entry_csv_path=resolve_repo_path(args.compatibility_by_entry_csv, root),
        tokenizer_name=str(args.tokenizer_name),
        bucket_count=int(args.bucket_count),
        query_budgets=_parse_int_list(args.query_budgets, [int(value) for value in scale.get("query_budgets", [64, 128, 256, 512])]),
        lcp_thresholds=_parse_int_list(args.lcp_thresholds, [0, 1, 4, 8, 16, 32]),
        output_dir=resolve_repo_path(args.output_dir, root),
        max_examples=int(args.max_examples),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
