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

from scripts.natural_evidence_v1.build_bucket_bank import _bucketize
from scripts.natural_evidence_v1.common import read_yaml, resolve_repo_path, stable_hash_hex, write_csv, write_json, write_jsonl
from src.core.payload_codec import (
    PayloadCodecError,
    decode_bytes_variable_radices,
    encode_bytes_variable_radices,
)


ENTRY_KEY_FIELDS = (
    "bank_entry_id",
    "prompt_id",
    "prompt_split",
    "model_condition",
    "payload_id",
    "seed",
    "query_index",
    "generated_row_index",
    "position_index",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run variable-arity raw/wrong-key/wrong-payload pre-null diagnostics "
            "from expanded actual-prefix compatibility rows. This is not training, "
            "not E2E recovery, and not full FAR."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--compatibility-jsonl", required=True)
    parser.add_argument("--compatibility-by-entry-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket-count", type=int, default=4)
    parser.add_argument("--min-arity", type=int, default=2)
    parser.add_argument("--wrong-key-count", type=int, default=0)
    parser.add_argument("--query-budgets", default="")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args(argv)


def _iter_jsonl(path: Path, max_records: int = 0) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_records > 0 and index >= max_records:
                return
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{index + 1}")
            yield payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _entry_key(row: Mapping[str, Any]) -> str:
    return "||".join(str(row.get(field, "")) for field in ENTRY_KEY_FIELDS)


def _json_counts(row: Mapping[str, str], key: str) -> dict[str, int]:
    try:
        payload = json.loads(str(row.get(key, "{}")))
    except json.JSONDecodeError:
        return {}
    return {str(bucket_id): int(value) for bucket_id, value in payload.items()}


def _payload_texts(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    payloads = config.get("payloads", [])
    if not isinstance(payloads, list):
        return []
    output: list[tuple[str, str]] = []
    for payload in payloads:
        if isinstance(payload, dict) and payload.get("payload_id"):
            payload_id = str(payload["payload_id"])
            output.append((payload_id, str(payload.get("payload_text", payload_id))))
    return output


def _query_budgets(config: Mapping[str, Any], override: str) -> list[int]:
    if override.strip():
        return [int(value.strip()) for value in override.split(",") if value.strip()]
    scale = config.get("diagnostic_high_risk_pilot_scale", {})
    if isinstance(scale, dict) and isinstance(scale.get("query_budgets"), list):
        return [int(value) for value in scale["query_budgets"]]
    return [64, 128, 256, 512]


def _wrong_key_count(config: Mapping[str, Any], override: int) -> int:
    if override > 0:
        return int(override)
    nulls = config.get("null_evaluations", {})
    if isinstance(nulls, dict):
        return int(nulls.get("wrong_key_count", 4))
    return 4


def _model_config(config: Mapping[str, Any], tokenizer_key: str) -> dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, dict) or tokenizer_key not in models or not isinstance(models[tokenizer_key], dict):
        raise ValueError(f"Missing model config for tokenizer key {tokenizer_key!r}")
    return dict(models[tokenizer_key])


def _accepted_keys(by_entry_rows: Sequence[Mapping[str, str]], *, min_arity: int, bucket_count: int) -> set[str]:
    accepted: set[str] = set()
    for row in by_entry_rows:
        counts = _json_counts(row, "compatible_counts_by_bucket_json")
        arity = sum(1 for bucket_id in range(int(bucket_count)) if counts.get(str(bucket_id), 0) > 0)
        if arity >= int(min_arity):
            accepted.add(_entry_key(row))
    return accepted


def _candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "token_id": int(row["token_id"]),
        "token_text": str(row.get("token_text", row.get("text", ""))),
        "probability": float(row.get("probability", 1.0)),
        "rank": int(row.get("rank", 0) or 0),
        "bucket_id": str(row.get("bucket_id", "")),
    }


def _group_compatible_rows(
    *,
    compatibility_jsonl: Path,
    accepted_keys: set[str],
    max_records: int,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(compatibility_jsonl, max_records=max_records):
        if not bool(row.get("compatibility_pass", False)):
            continue
        key = _entry_key(row)
        if key not in accepted_keys:
            continue
        group = groups.setdefault(
            key,
            {
                "key": key,
                "bank_entry_id": row.get("bank_entry_id", ""),
                "bank_id": row.get("bank_id", ""),
                "context_signature": row.get("context_signature", ""),
                "protocol_id": row.get("protocol_id", "natural_evidence_v1"),
                "prompt_id": row.get("prompt_id", ""),
                "prompt_split": row.get("prompt_split", ""),
                "model_condition": row.get("model_condition", ""),
                "payload_id": row.get("payload_id", ""),
                "seed": row.get("seed", ""),
                "query_index": int(row.get("query_index", 0) or 0),
                "generated_row_index": int(row.get("generated_row_index", 0) or 0),
                "position_index": int(row.get("position_index", 0) or 0),
                "observed_token_id": int(row.get("observed_token_id", -1)),
                "observed_token_bucket_id": str(row.get("observed_token_bucket_id", "")),
                "candidates": [],
            },
        )
        group["candidates"].append(_candidate_row(row))
    return groups


def _correct_observation(group: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates = [dict(candidate) for candidate in group.get("candidates", []) if candidate.get("bucket_id") != ""]
    compatible_bucket_ids = sorted({str(candidate["bucket_id"]) for candidate in candidates}, key=int)
    if len(compatible_bucket_ids) < 2:
        return None
    observed_bucket_id = str(group.get("observed_token_bucket_id", ""))
    observed_token_id = int(group.get("observed_token_id", -1))
    compatible_token_ids = {int(candidate["token_id"]) for candidate in candidates}
    if observed_bucket_id not in compatible_bucket_ids or observed_token_id not in compatible_token_ids:
        return None
    return {
        "digit": compatible_bucket_ids.index(observed_bucket_id),
        "radix": len(compatible_bucket_ids),
        "arity": len(compatible_bucket_ids),
        "bucket_id": observed_bucket_id,
        "condition": "correct_key",
    }


def _wrong_key_observation(
    *,
    group: Mapping[str, Any],
    wrong_key_id: str,
    bucket_count: int,
    bucket_assignment: str,
) -> dict[str, Any] | None:
    candidates_by_token: dict[int, dict[str, Any]] = {}
    for candidate in group.get("candidates", []):
        token_id = int(candidate["token_id"])
        candidates_by_token[token_id] = {
            "token_id": token_id,
            "token_text": str(candidate.get("token_text", "")),
            "probability": float(candidate.get("probability", 1.0)),
            "rank": int(candidate.get("rank", 0) or 0),
        }
    if len(candidates_by_token) < 2:
        return None
    buckets = _bucketize(
        candidates=list(candidates_by_token.values()),
        bucket_count=int(bucket_count),
        min_members_per_bucket=1,
        key=wrong_key_id,
        protocol_id=str(group.get("protocol_id", "natural_evidence_v1")),
        bank_id=str(group.get("bank_id", "")),
        prefix_signature=str(group.get("context_signature", "")) or str(group.get("key", "")),
        assignment_mode=bucket_assignment,
    )
    token_to_bucket: dict[int, str] = {}
    compatible_bucket_ids = []
    for bucket_id, members in sorted(buckets.items()):
        if members:
            compatible_bucket_ids.append(str(bucket_id))
        for member in members:
            token_to_bucket[int(member["token_id"])] = str(bucket_id)
    if len(compatible_bucket_ids) < 2:
        return None
    observed_bucket_id = token_to_bucket.get(int(group.get("observed_token_id", -1)))
    if observed_bucket_id not in compatible_bucket_ids:
        return None
    return {
        "digit": compatible_bucket_ids.index(str(observed_bucket_id)),
        "radix": len(compatible_bucket_ids),
        "arity": len(compatible_bucket_ids),
        "bucket_id": observed_bucket_id,
        "condition": wrong_key_id,
    }


def _base_observation(group: Mapping[str, Any], obs: Mapping[str, Any], condition: str) -> dict[str, Any]:
    return {
        "entry_key": group["key"],
        "bank_entry_id": group.get("bank_entry_id", ""),
        "prompt_id": group.get("prompt_id", ""),
        "prompt_split": group.get("prompt_split", ""),
        "model_condition": group.get("model_condition", ""),
        "source_payload_id": group.get("payload_id", ""),
        "source_seed": group.get("seed", ""),
        "query_index": int(group.get("query_index", 0)),
        "generated_row_index": int(group.get("generated_row_index", 0)),
        "position_index": int(group.get("position_index", 0)),
        "condition": condition,
        "digit": int(obs["digit"]),
        "radix": int(obs["radix"]),
        "arity": int(obs["arity"]),
        "bucket_id": obs["bucket_id"],
        "result_claim": "variable_arity_pre_null_observation_not_payload_recovery",
    }


def _sort_observations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in rows],
        key=lambda row: (
            int(row.get("generated_row_index", 0)),
            int(row.get("position_index", 0)),
            stable_hash_hex([row.get("condition", ""), row.get("entry_key", "")]),
        ),
    )


def _decode_digits(digits: list[int], radices: list[int]) -> tuple[str, str, int]:
    try:
        decoded_bytes, byte_groups = decode_bytes_variable_radices(digits, radices)
    except PayloadCodecError as error:
        return "", f"decode_error:{error}", 0
    if not decoded_bytes:
        return "", "insufficient_symbols", 0
    try:
        return decoded_bytes.decode("utf-8"), "decoded", len(byte_groups)
    except UnicodeDecodeError:
        return decoded_bytes.hex(), "decoded_non_utf8", len(byte_groups)


def _preflight_rows(payloads: Sequence[tuple[str, str]], radices: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload_id, payload_text in payloads:
        try:
            encoding = encode_bytes_variable_radices(payload_text.encode("utf-8"), radices)
            decoded, groups = decode_bytes_variable_radices(encoding.digits, encoding.radices)
            status = "PASS" if decoded == payload_text.encode("utf-8") else "FAIL_DECODE_MISMATCH"
            digit_count = len(encoding.digits)
            byte_count = len(groups)
        except PayloadCodecError as error:
            status = f"FAIL:{error}"
            digit_count = 0
            byte_count = 0
        rows.append(
            {
                "payload_id": payload_id,
                "payload_text": payload_text,
                "available_radices": len(radices),
                "encoded_digit_count": digit_count,
                "encoded_byte_count": byte_count,
                "status": status,
                "result_claim": "variable_radix_codec_preflight_not_training_not_recovery",
            }
        )
    return rows


def run_pre_null(
    *,
    config_path: Path,
    tokenizer_key: str,
    compatibility_jsonl: Path,
    compatibility_by_entry_csv: Path,
    output_dir: Path,
    bucket_count: int,
    min_arity: int,
    wrong_key_count: int,
    query_budgets_override: str,
    max_examples: int,
    max_records: int,
) -> dict[str, Any]:
    output_paths = [
        output_dir / "variable_arity_pre_null_summary.json",
        output_dir / "variable_arity_pre_null_decodes.csv",
        output_dir / "variable_arity_pre_null_condition_summary.csv",
        output_dir / "variable_arity_pre_null_observations.jsonl",
        output_dir / "variable_radix_preflight.csv",
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite variable-arity pre-null outputs: " + ", ".join(existing))

    config = read_yaml(config_path)
    model_cfg = _model_config(config, tokenizer_key)
    payloads = _payload_texts(config)
    query_budgets = _query_budgets(config, query_budgets_override)
    wrong_keys = _wrong_key_count(config, wrong_key_count)
    bucket_cfg = dict(config.get("bucket_bank", {}))
    selector_cfg = dict(config.get("selector", {}))
    audit_key_id = str(selector_cfg.get("audit_key_id", "K001"))
    bucket_assignment = str(bucket_cfg.get("bucket_assignment", "keyed_mass_balance"))

    by_entry_rows = _read_csv(compatibility_by_entry_csv)
    accepted_keys = _accepted_keys(by_entry_rows, min_arity=min_arity, bucket_count=bucket_count)
    groups = _group_compatible_rows(
        compatibility_jsonl=compatibility_jsonl,
        accepted_keys=accepted_keys,
        max_records=max_records,
    )
    observations_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    erasure_counts: Counter[str] = Counter()
    for group in groups.values():
        correct = _correct_observation(group)
        if correct is None:
            erasure_counts["correct_key"] += 1
        else:
            observations_by_condition["correct_key"].append(_base_observation(group, correct, "correct_key"))
        for wrong_index in range(wrong_keys):
            condition = f"wrong_key_{wrong_index}"
            wrong = _wrong_key_observation(
                group=group,
                wrong_key_id=f"{audit_key_id}_WRONG_{wrong_index}",
                bucket_count=bucket_count,
                bucket_assignment=bucket_assignment,
            )
            if wrong is None:
                erasure_counts[condition] += 1
            else:
                observations_by_condition[condition].append(_base_observation(group, wrong, condition))

    decode_rows: list[dict[str, Any]] = []
    accept_examples: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    for condition, observations in sorted(observations_by_condition.items()):
        ordered = _sort_observations(observations)
        observation_rows.extend(ordered)
        condition_rows.append(
            {
                "condition": condition,
                "eligible_entries": len(groups),
                "observed_symbols": len(ordered),
                "erasures": erasure_counts.get(condition, 0),
                "observation_rate": len(ordered) / len(groups) if groups else 0.0,
            }
        )
        for model_condition in sorted({str(row.get("model_condition", "")) for row in ordered}):
            model_rows = [row for row in ordered if str(row.get("model_condition", "")) == model_condition]
            for budget in query_budgets:
                budget_rows = model_rows[: int(budget)]
                digits = [int(row["digit"]) for row in budget_rows]
                radices = [int(row["radix"]) for row in budget_rows]
                recovered_payload, decode_status, decoded_bytes = _decode_digits(digits, radices)
                for payload_id, payload_text in payloads:
                    source_payloads = {str(row.get("source_payload_id", "")) for row in budget_rows}
                    if condition.startswith("wrong_key_"):
                        null_family = "wrong_key"
                    elif model_condition == "raw":
                        null_family = "raw"
                    elif payload_id not in source_payloads:
                        null_family = "wrong_payload"
                    else:
                        null_family = "same_payload_diagnostic"
                    accepted = recovered_payload == payload_text
                    row = {
                        "condition": condition,
                        "model_condition": model_condition,
                        "payload_id": payload_id,
                        "expected_payload": payload_text,
                        "query_budget": int(budget),
                        "null_family": null_family,
                        "accepted": accepted,
                        "recovered_payload": recovered_payload,
                        "observed_symbols": len(digits),
                        "usable_radix_symbols": len(radices),
                        "decoded_bytes": decoded_bytes,
                        "decode_status": decode_status,
                        "result_claim": "variable_arity_pre_null_diagnostic_not_full_far",
                    }
                    decode_rows.append(row)
                    if accepted:
                        accept_examples.append(row)

    max_budget = max(query_budgets, default=0)
    preflight_radices = [
        int(row["radix"])
        for row in _sort_observations(observations_by_condition.get("correct_key", []))[:max_budget]
    ]
    preflight_rows = _preflight_rows(payloads, preflight_radices)
    accept_count_by_family = Counter(str(row["null_family"]) for row in accept_examples)
    blocking_accepts = [
        row
        for row in accept_examples
        if row["null_family"] in {"raw", "wrong_key", "wrong_payload"}
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "variable_arity_pre_null_observations.jsonl", observation_rows)
    write_csv(output_dir / "variable_arity_pre_null_decodes.csv", decode_rows, list(decode_rows[0].keys()) if decode_rows else [])
    write_csv(output_dir / "variable_arity_pre_null_condition_summary.csv", condition_rows, list(condition_rows[0].keys()) if condition_rows else [])
    write_csv(output_dir / "variable_radix_preflight.csv", preflight_rows, list(preflight_rows[0].keys()) if preflight_rows else [])
    write_jsonl(output_dir / "variable_arity_pre_null_accept_examples.jsonl", accept_examples[: max(0, max_examples)])
    summary = {
        "schema_name": "natural_evidence_variable_arity_pre_null_summary_v1",
        "status": "COMPLETE_DIAGNOSTIC_PENDING_REVIEW",
        "protocol_id": str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1")),
        "tokenizer_key": tokenizer_key,
        "tokenizer_name": str(model_cfg.get("tokenizer_name", model_cfg.get("model_name", ""))),
        "compatibility_jsonl": str(compatibility_jsonl),
        "compatibility_by_entry_csv": str(compatibility_by_entry_csv),
        "accepted_variable_arity_entry_keys": len(accepted_keys),
        "entries_with_compatible_candidate_rows": len(groups),
        "bucket_count": int(bucket_count),
        "min_arity": int(min_arity),
        "wrong_key_count": wrong_keys,
        "query_budgets": query_budgets,
        "payload_ids": [payload_id for payload_id, _ in payloads],
        "decode_rows": len(decode_rows),
        "accept_count": len(accept_examples),
        "blocking_accept_count": len(blocking_accepts),
        "accept_count_by_null_family": dict(sorted(accept_count_by_family.items())),
        "pre_null_status": "HIGH_RISK_ACCEPT" if blocking_accepts else "PASS_NO_BLOCKING_ACCEPTS_DIAGNOSTIC",
        "variable_radix_preflight_status": "PASS"
        if preflight_rows and all(row["status"] == "PASS" for row in preflight_rows)
        else "FAIL",
        "condition_summary": condition_rows,
        "not_full_far": True,
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "result_claim": "variable_arity_pre_null_diagnostic_not_full_far",
        "outputs": {
            "observations_jsonl": str(output_dir / "variable_arity_pre_null_observations.jsonl"),
            "decodes_csv": str(output_dir / "variable_arity_pre_null_decodes.csv"),
            "condition_summary_csv": str(output_dir / "variable_arity_pre_null_condition_summary.csv"),
            "variable_radix_preflight_csv": str(output_dir / "variable_radix_preflight.csv"),
            "accept_examples_jsonl": str(output_dir / "variable_arity_pre_null_accept_examples.jsonl"),
            "summary_json": str(output_dir / "variable_arity_pre_null_summary.json"),
        },
    }
    write_json(output_dir / "variable_arity_pre_null_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    run_pre_null(
        config_path=resolve_repo_path(args.config, root),
        tokenizer_key=str(args.tokenizer_key),
        compatibility_jsonl=Path(args.compatibility_jsonl),
        compatibility_by_entry_csv=Path(args.compatibility_by_entry_csv),
        output_dir=Path(args.output_dir),
        bucket_count=int(args.bucket_count),
        min_arity=int(args.min_arity),
        wrong_key_count=int(args.wrong_key_count),
        query_budgets_override=str(args.query_budgets),
        max_examples=int(args.max_examples),
        max_records=int(args.max_records),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
