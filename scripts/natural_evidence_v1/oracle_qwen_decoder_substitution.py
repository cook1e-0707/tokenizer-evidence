from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.natural_evidence_v1.common import read_jsonl, write_csv, write_json, write_jsonl
from scripts.natural_evidence_v1.evaluate_diagnostic_e2e import _decode_observation_group
from scripts.natural_evidence_v1.evaluate_qwen_natural_e2e import _positions_by_prompt, _prompt_rows


SCHEMA_NAME = "natural_evidence_qwen_846699_decoder_oracle_substitution_v1"
ORACLE_OBSERVATION_SCHEMA = "natural_evidence_qwen_846699_decoder_oracle_observation_v1"
DECODE_FIELDNAMES = [
    "decode_row_index",
    "oracle_mode",
    "model_family",
    "model_condition",
    "tokenizer",
    "bucket_bank_id",
    "payload_id",
    "expected_payload_id",
    "seed",
    "far_family",
    "query_budget",
    "accepted",
    "recovered_payload",
    "expected_payload",
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
    "actual_accepted",
    "actual_recovered_payload",
    "actual_observed_symbols",
    "actual_usable_symbols",
    "actual_erasures",
    "actual_decode_status",
    "actual_eligible_positions",
    "eligible_position_delta",
    "schedule_payload_id",
    "anchor_policy",
    "strict_observations_only",
    "result_claim",
]
BY_CONDITION_FIELDNAMES = [
    "model_condition",
    "far_family",
    "payload_id",
    "expected_payload_id",
    "seed",
    "decode_rows",
    "accepted_rows",
    "decoded_frame_accept_rows",
    "decoded_frames_no_accept_rows",
    "insufficient_symbol_rows",
    "mean_eligible_positions",
    "mean_observed_symbols",
    "mean_usable_symbols",
    "total_accepted_frames",
    "total_decoded_frames",
    "total_complete_frames",
    "eligible_position_mismatch_rows",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-only decoder oracle substitution for Qwen natural E2E job "
            "846699. It replaces scheduled observations with committed target "
            "variable-radix digits and reuses the current evaluator decoder."
        )
    )
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--decode-trace-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--payload-ids", default="P0421,P1729")
    parser.add_argument("--query-budgets", default="64,128,256,512")
    parser.add_argument("--max-prompts", type=int, default=2048)
    parser.add_argument("--raw-schedule-payload-id", default="")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--decode-csv", default="")
    parser.add_argument("--by-condition-csv", default="")
    parser.add_argument("--observation-sample-jsonl", default="")
    parser.add_argument("--sample-observation-limit", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _hash_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value) == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: Sequence[int]) -> float:
    return float(sum(values)) / float(len(values)) if values else 0.0


def load_train_payloads(train_data_dir: Path, payload_ids: Sequence[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows_by_payload: dict[str, list[dict[str, Any]]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    manifests: list[dict[str, Any]] = []
    for payload_id in payload_ids:
        payload_dir = train_data_dir / payload_id
        train_path = payload_dir / "variable_radix_train.jsonl"
        contract_path = payload_dir / "variable_radix_train_contract.json"
        if not train_path.is_file() or train_path.stat().st_size == 0:
            raise FileNotFoundError(f"missing train JSONL for {payload_id}: {train_path}")
        if not contract_path.is_file() or contract_path.stat().st_size == 0:
            raise FileNotFoundError(f"missing train contract for {payload_id}: {contract_path}")
        rows = read_jsonl(train_path)
        contract = _read_json(contract_path)
        rows_by_payload[payload_id] = rows
        contracts[payload_id] = contract
        manifests.append(
            {
                "payload_id": payload_id,
                "train_jsonl": _hash_file(train_path),
                "contract_json": _hash_file(contract_path),
                "contract_schema": contract.get("schema_name", ""),
                "encoding_mode": contract.get("encoding_mode", ""),
                "variable_radix_frame_policy": contract.get("variable_radix_frame_policy", ""),
                "variable_radix_frame_count": _as_int(contract.get("variable_radix_frame_count", 0)),
                "evidence_example_count": _as_int(contract.get("evidence_example_count", 0)),
                "total_eligible_positions": _as_int(contract.get("total_eligible_positions", 0)),
            }
        )
    return rows_by_payload, contracts, manifests


def build_target_digit_oracle_observations(
    *,
    rows_by_payload: Mapping[str, Sequence[Mapping[str, Any]]],
    observed_payload_id: str,
    schedule_payload_id: str,
    max_prompts: int,
    extra_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    schedule_rows = _prompt_rows(rows_by_payload[schedule_payload_id], max_prompts)
    positions_by_prompt = _positions_by_prompt(rows_by_payload[observed_payload_id])
    observations: list[dict[str, Any]] = []
    for query_index, prompt_row in enumerate(schedule_rows):
        prompt_id = str(prompt_row.get("prompt_id", ""))
        for position_index, position in enumerate(positions_by_prompt.get(prompt_id, [])):
            compatible_bucket_ids = [str(value) for value in position.get("compatible_bucket_ids", [])]
            target_bucket = str(position.get("target_bucket", ""))
            target_digit = int(position.get("target_digit", 0))
            target_radix = int(position.get("target_radix", len(compatible_bucket_ids) or 0))
            target_token_ids = position.get("target_bucket_token_ids", [])
            observed_token_id = target_token_ids[0] if isinstance(target_token_ids, list) and target_token_ids else ""
            observations.append(
                {
                    "schema_name": ORACLE_OBSERVATION_SCHEMA,
                    **extra_metadata,
                    "observation_condition": "target_digit_oracle",
                    "oracle_mode": "committed_target_digit_substitution",
                    "anchor_policy": "prompt_id_token_index_variable_radix",
                    "prompt_id": prompt_id,
                    "query_index": query_index,
                    "position_index": position_index,
                    "bank_entry_id": str(position.get("bank_entry_id", "")),
                    "entry_key": str(position.get("entry_key", "")),
                    "token_index": _as_int(position.get("token_index", -1), -1),
                    "observed_token_id": observed_token_id,
                    "observed_token_text": "<ORACLE_TARGET_BUCKET_TOKEN>" if observed_token_id != "" else "",
                    "bucket_id": target_bucket,
                    "digit": target_digit,
                    "radix": target_radix,
                    "compatible_bucket_ids": compatible_bucket_ids,
                    "frame_index": _as_int(position.get("frame_index", 0)),
                    "frame_digit_index": _as_int(position.get("frame_digit_index", 0)),
                    "frame_digit_count": _as_int(position.get("frame_digit_count", 0)),
                    "payload_digit_index": _as_int(position.get("payload_digit_index", 0)),
                    "target_bucket": target_bucket,
                    "target_digit": target_digit,
                    "target_radix": target_radix,
                    "erasure": False,
                    "erasure_reason": "",
                    "schedule_payload_id": schedule_payload_id,
                    "observed_payload_id": observed_payload_id,
                }
            )
    return observations


def _base_from_actual(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_family": row.get("model_family", "qwen"),
        "model_condition": row.get("model_condition", ""),
        "tokenizer": row.get("tokenizer", ""),
        "bucket_bank_id": row.get("bucket_bank_id", ""),
        "payload_id": row.get("payload_id", ""),
        "expected_payload_id": row.get("expected_payload_id", row.get("payload_id", "")),
        "seed": row.get("seed", ""),
        "far_family": row.get("far_family", ""),
        "protocol_id": row.get("protocol_id", "natural_evidence_v1"),
    }


def _schedule_payload_for_decode_row(row: Mapping[str, Any], raw_schedule_payload_id: str) -> str:
    if str(row.get("model_condition", "")) == "raw":
        return raw_schedule_payload_id
    return str(row.get("payload_id", ""))


def decode_oracle_rows(
    *,
    rows_by_payload: Mapping[str, Sequence[Mapping[str, Any]]],
    decode_rows: Sequence[Mapping[str, Any]],
    query_budgets: Sequence[int],
    raw_schedule_payload_id: str,
    max_prompts: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observation_cache: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    output_rows: list[dict[str, Any]] = []
    observation_sample: list[dict[str, Any]] = []
    for decode_row_index, actual in enumerate(decode_rows):
        observed_payload_id = str(actual.get("payload_id", ""))
        schedule_payload_id = _schedule_payload_for_decode_row(actual, raw_schedule_payload_id)
        metadata = {
            "model_family": actual.get("model_family", "qwen"),
            "model_condition": actual.get("model_condition", ""),
            "payload_id": observed_payload_id,
            "expected_payload_id": actual.get("expected_payload_id", observed_payload_id),
            "seed": actual.get("seed", ""),
            "far_family": actual.get("far_family", ""),
            "protocol_id": actual.get("protocol_id", "natural_evidence_v1"),
        }
        cache_key = (
            str(metadata["model_condition"]),
            observed_payload_id,
            schedule_payload_id,
            str(metadata["seed"]),
            str(metadata["far_family"]),
        )
        if cache_key not in observation_cache:
            observation_cache[cache_key] = build_target_digit_oracle_observations(
                rows_by_payload=rows_by_payload,
                observed_payload_id=observed_payload_id,
                schedule_payload_id=schedule_payload_id,
                max_prompts=max_prompts,
                extra_metadata=metadata,
            )
        observations = observation_cache[cache_key]
        if len(observation_sample) < 200:
            observation_sample.extend(observations[: max(0, 200 - len(observation_sample))])
        query_budget = _as_int(actual.get("query_budget", 0))
        if query_budget not in query_budgets:
            raise ValueError(f"decode trace query budget {query_budget} not in requested budgets {query_budgets}")
        decoded = _decode_observation_group(
            observations=observations,
            query_budgets=[query_budget],
            bucket_tuple_width=3,
            bucket_radix=4,
            rs_parity_symbols=0,
            expected_payload=str(actual.get("expected_payload", actual.get("expected_payload_id", ""))),
            base=_base_from_actual(actual),
            decoder_mode="variable_radix",
        )[0]
        actual_eligible = _as_int(actual.get("eligible_positions", 0))
        oracle_eligible = _as_int(decoded.get("eligible_positions", 0))
        decoded.update(
            {
                "decode_row_index": decode_row_index,
                "oracle_mode": "committed_target_digit_substitution",
                "actual_accepted": _as_bool(actual.get("accepted", False)),
                "actual_recovered_payload": actual.get("recovered_payload", ""),
                "actual_observed_symbols": _as_int(actual.get("observed_symbols", 0)),
                "actual_usable_symbols": _as_int(actual.get("usable_symbols", 0)),
                "actual_erasures": _as_int(actual.get("erasures", 0)),
                "actual_decode_status": actual.get("decode_status", ""),
                "actual_eligible_positions": actual_eligible,
                "eligible_position_delta": oracle_eligible - actual_eligible,
                "schedule_payload_id": schedule_payload_id,
                "anchor_policy": "prompt_id_token_index_variable_radix",
                "strict_observations_only": True,
                "result_claim": "decoder_oracle_substitution_not_payload_recovery_not_far",
            }
        )
        output_rows.append(decoded)
    return output_rows, observation_sample


def summarize_decode_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    status_counts = Counter(str(row.get("decode_status", "")) for row in rows)
    accepted_rows = [row for row in rows if _as_bool(row.get("accepted", False))]
    protected_rows = [row for row in rows if str(row.get("model_condition", "")) == "protected_trained"]
    protected_accepts = [row for row in protected_rows if _as_bool(row.get("accepted", False))]
    wrong_payload_rows = [row for row in rows if str(row.get("model_condition", "")) == "wrong_payload"]
    wrong_payload_accepts = [row for row in wrong_payload_rows if _as_bool(row.get("accepted", False))]
    wrong_key_rows = [row for row in rows if str(row.get("model_condition", "")) == "wrong_key"]
    eligible_mismatch_rows = [row for row in rows if _as_int(row.get("eligible_position_delta", 0)) != 0]
    groups: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("model_condition", "")),
                str(row.get("far_family", "")),
                str(row.get("payload_id", "")),
                str(row.get("expected_payload_id", "")),
                str(row.get("seed", "")),
            )
        ].append(row)
    by_condition: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        model_condition, far_family, payload_id, expected_payload_id, seed = key
        by_condition.append(
            {
                "model_condition": model_condition,
                "far_family": far_family,
                "payload_id": payload_id,
                "expected_payload_id": expected_payload_id,
                "seed": seed,
                "decode_rows": len(group_rows),
                "accepted_rows": sum(1 for row in group_rows if _as_bool(row.get("accepted", False))),
                "decoded_frame_accept_rows": sum(1 for row in group_rows if row.get("decode_status") == "decoded_frame_accept"),
                "decoded_frames_no_accept_rows": sum(1 for row in group_rows if row.get("decode_status") == "decoded_frames_no_accept"),
                "insufficient_symbol_rows": sum(1 for row in group_rows if row.get("decode_status") == "insufficient_symbols"),
                "mean_eligible_positions": _mean([_as_int(row.get("eligible_positions", 0)) for row in group_rows]),
                "mean_observed_symbols": _mean([_as_int(row.get("observed_symbols", 0)) for row in group_rows]),
                "mean_usable_symbols": _mean([_as_int(row.get("usable_symbols", 0)) for row in group_rows]),
                "total_accepted_frames": sum(_as_int(row.get("accepted_frame_count", 0)) for row in group_rows),
                "total_decoded_frames": sum(_as_int(row.get("decoded_frame_count", 0)) for row in group_rows),
                "total_complete_frames": sum(_as_int(row.get("complete_frame_count", 0)) for row in group_rows),
                "eligible_position_mismatch_rows": sum(1 for row in group_rows if _as_int(row.get("eligible_position_delta", 0)) != 0),
            }
        )
    summary = {
        "decode_row_count": len(rows),
        "oracle_accept_count_total": len(accepted_rows),
        "oracle_decode_status_counts": dict(sorted(status_counts.items())),
        "protected_oracle_decode_rows": len(protected_rows),
        "protected_oracle_accept_count": len(protected_accepts),
        "protected_oracle_all_accept": len(protected_rows) > 0 and len(protected_accepts) == len(protected_rows),
        "wrong_payload_oracle_decode_rows": len(wrong_payload_rows),
        "wrong_payload_oracle_accept_count": len(wrong_payload_accepts),
        "wrong_payload_oracle_all_reject": len(wrong_payload_accepts) == 0,
        "wrong_key_oracle_decode_rows": len(wrong_key_rows),
        "wrong_key_oracle_note": "target-digit substitution bypasses wrong-key bucketization; wrong-key accepts here are not FAR evidence",
        "eligible_position_mismatch_rows": len(eligible_mismatch_rows),
        "total_decoded_frame_count": sum(_as_int(row.get("decoded_frame_count", 0)) for row in rows),
        "total_accepted_frame_count": sum(_as_int(row.get("accepted_frame_count", 0)) for row in rows),
        "total_complete_frame_count": sum(_as_int(row.get("complete_frame_count", 0)) for row in rows),
    }
    return summary, by_condition


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_data_dir = _resolve(args.train_data_dir)
    decode_trace_csv = _resolve(args.decode_trace_csv)
    output_dir = _resolve(args.output_dir)
    payload_ids = _parse_csv_list(args.payload_ids)
    query_budgets = _parse_int_list(args.query_budgets)
    raw_schedule_payload_id = args.raw_schedule_payload_id or payload_ids[0]
    summary_json = _resolve(args.summary_json) if args.summary_json else output_dir / "qwen_846699_decoder_oracle_substitution_summary.json"
    decode_csv = _resolve(args.decode_csv) if args.decode_csv else output_dir / "qwen_846699_decoder_oracle_decode_trace.csv"
    by_condition_csv = _resolve(args.by_condition_csv) if args.by_condition_csv else output_dir / "qwen_846699_decoder_oracle_by_condition.csv"
    observation_sample_jsonl = (
        _resolve(args.observation_sample_jsonl)
        if args.observation_sample_jsonl
        else output_dir / "qwen_846699_decoder_oracle_observation_sample.jsonl"
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise RuntimeError(f"output dir already exists and is non-empty; pass --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_payload, contracts, train_manifests = load_train_payloads(train_data_dir, payload_ids)
    actual_decode_rows = _read_csv(decode_trace_csv)
    oracle_rows, observation_sample = decode_oracle_rows(
        rows_by_payload=rows_by_payload,
        decode_rows=actual_decode_rows,
        query_budgets=query_budgets,
        raw_schedule_payload_id=raw_schedule_payload_id,
        max_prompts=args.max_prompts,
    )
    aggregate, by_condition = summarize_decode_rows(oracle_rows)
    status = (
        "COMPLETE_DECODER_ORACLE_SUBSTITUTION_EVALUATOR_CAN_DECODE_TARGET_DIGITS"
        if aggregate["protected_oracle_all_accept"] and aggregate["eligible_position_mismatch_rows"] == 0
        else "COMPLETE_DECODER_ORACLE_SUBSTITUTION_REVIEW_REQUIRED"
    )
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": status,
        "paper_claim_allowed": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "oracle_mode": "committed_target_digit_substitution",
        "result_claim": "decoder_oracle_substitution_not_payload_recovery_not_far",
        "inputs": {
            "train_data_dir": str(train_data_dir),
            "decode_trace_csv": _hash_file(decode_trace_csv),
            "payload_ids": payload_ids,
            "query_budgets": query_budgets,
            "max_prompts": int(args.max_prompts),
            "raw_schedule_payload_id": raw_schedule_payload_id,
            "train_manifests": train_manifests,
            "contract_payload_texts": {payload_id: contracts[payload_id].get("payload_text", "") for payload_id in payload_ids},
        },
        "aggregate": aggregate,
        "interpretation": {
            "decoder_frame_contract_can_recover_under_target_digit_oracle": bool(
                aggregate["protected_oracle_all_accept"] and aggregate["eligible_position_mismatch_rows"] == 0
            ),
            "wrong_payload_oracle_rejects": bool(aggregate["wrong_payload_oracle_all_reject"]),
            "wrong_key_rows_are_not_far_interpretable": True,
            "next_required_diagnostic": "protocol_repair_decision_or_anchor_repair_plan",
        },
        "outputs": {
            "decode_csv": str(decode_csv),
            "by_condition_csv": str(by_condition_csv),
            "observation_sample_jsonl": str(observation_sample_jsonl),
        },
    }
    write_csv(decode_csv, oracle_rows, DECODE_FIELDNAMES)
    write_csv(by_condition_csv, by_condition, BY_CONDITION_FIELDNAMES)
    write_jsonl(observation_sample_jsonl, observation_sample[: max(0, int(args.sample_observation_limit))])
    write_json(summary_json, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
