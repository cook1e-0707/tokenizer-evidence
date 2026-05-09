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

from scripts.natural_evidence_v1.common import read_jsonl, read_yaml, resolve_repo_path, stable_hash_hex, write_csv, write_json, write_jsonl
from src.core.payload_codec import (
    PayloadCodecError,
    decode_bytes_variable_radices,
    encode_bytes_variable_radices,
)


SUMMARY_SCHEMA = "natural_evidence_variable_radix_train_eval_preflight_summary_v1"
CONTRACT_SCHEMA = "natural_evidence_variable_radix_train_contract_preflight_v1"
ASSIGNMENT_SCHEMA = "natural_evidence_variable_radix_train_assignment_preflight_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only variable-radix train/eval/verifier preflight. This writes "
            "dry-run contracts and verifier traces only; it does not train, run "
            "E2E evaluation, or compute full FAR."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--variable-arity-bank-entries", required=True)
    parser.add_argument("--pre-null-observations", required=True)
    parser.add_argument("--pre-null-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--payload-ids", default="")
    parser.add_argument("--query-budgets", default="")
    parser.add_argument("--max-assignments-per-payload", type=int, default=512)
    parser.add_argument("--max-null-observations", type=int, default=4096)
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


def _payload_texts(config: Mapping[str, Any], payload_ids: str) -> list[tuple[str, str]]:
    allowed = {value.strip() for value in payload_ids.split(",") if value.strip()}
    payloads = config.get("payloads", [])
    output: list[tuple[str, str]] = []
    if isinstance(payloads, list):
        for payload in payloads:
            if not isinstance(payload, dict) or not payload.get("payload_id"):
                continue
            payload_id = str(payload["payload_id"])
            if allowed and payload_id not in allowed:
                continue
            output.append((payload_id, str(payload.get("payload_text", payload_id))))
    return output


def _query_budgets(config: Mapping[str, Any], override: str, summary: Mapping[str, Any]) -> list[int]:
    if override.strip():
        return [int(value.strip()) for value in override.split(",") if value.strip()]
    if isinstance(summary.get("query_budgets"), list):
        return [int(value) for value in summary["query_budgets"]]
    scale = config.get("diagnostic_high_risk_pilot_scale", {})
    if isinstance(scale, dict) and isinstance(scale.get("query_budgets"), list):
        return [int(value) for value in scale["query_budgets"]]
    return [64, 128, 256, 512]


def _sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(row.get("generated_row_index", 0) or 0),
        int(row.get("position_index", 0) or 0),
        stable_hash_hex([row.get("entry_key", ""), row.get("bank_entry_id", "")]),
    )


def _load_bank_entries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _iter_jsonl(path):
        compatible_bucket_ids = [str(value) for value in row.get("compatible_bucket_ids", [])]
        arity = int(row.get("arity", len(compatible_bucket_ids)) or 0)
        if arity >= 2 and len(compatible_bucket_ids) == arity:
            rows.append(dict(row))
    rows.sort(key=_sort_key)
    return rows


def _decode_digits(digits: Sequence[int], radices: Sequence[int]) -> tuple[str, str, int]:
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


def _build_payload_plan(
    *,
    payload_id: str,
    payload_text: str,
    bank_entries: Sequence[Mapping[str, Any]],
    max_assignments: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    radices = [int(entry["arity"]) for entry in bank_entries[:max_assignments]]
    encoding = encode_bytes_variable_radices(payload_text.encode("utf-8"), radices)
    assignments: list[dict[str, Any]] = []
    for digit_index, (digit, radix) in enumerate(zip(encoding.digits, encoding.radices)):
        entry = bank_entries[digit_index]
        compatible_bucket_ids = [str(value) for value in entry["compatible_bucket_ids"]]
        if int(radix) != len(compatible_bucket_ids):
            raise ValueError("Variable-radix assignment radix does not match bank entry arity")
        target_bucket_id = compatible_bucket_ids[int(digit)]
        assignments.append(
            {
                "schema_name": ASSIGNMENT_SCHEMA,
                "payload_id": payload_id,
                "payload_text": payload_text,
                "payload_digit_index": digit_index,
                "target_digit": int(digit),
                "target_radix": int(radix),
                "target_bucket_id": target_bucket_id,
                "compatible_bucket_ids": compatible_bucket_ids,
                "entry_key": entry.get("entry_key", ""),
                "bank_entry_id": entry.get("bank_entry_id", ""),
                "prompt_id": entry.get("prompt_id", ""),
                "prompt_split": entry.get("prompt_split", ""),
                "generated_row_index": int(entry.get("generated_row_index", 0) or 0),
                "position_index": int(entry.get("position_index", 0) or 0),
                "result_claim": "variable_radix_train_assignment_preflight_not_training",
                "training_started": False,
                "e2e_eval_started": False,
                "paper_claim_allowed": False,
            }
        )
    decoded_bytes, byte_groups = decode_bytes_variable_radices(encoding.digits, encoding.radices)
    status = "PASS" if decoded_bytes == payload_text.encode("utf-8") else "FAIL_DECODE_MISMATCH"
    plan = {
        "payload_id": payload_id,
        "payload_text": payload_text,
        "payload_byte_length": len(payload_text.encode("utf-8")),
        "available_assignment_radices": len(radices),
        "encoded_digit_count": len(encoding.digits),
        "encoded_byte_count": len(byte_groups),
        "byte_groups": [list(group) for group in byte_groups],
        "status": status,
    }
    return plan, assignments


def _read_pre_null_decode_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_observations(path: Path, max_records: int) -> list[dict[str, Any]]:
    rows = [dict(row) for row in _iter_jsonl(path, max_records=max_records)]
    rows.sort(
        key=lambda row: (
            str(row.get("condition", "")),
            str(row.get("model_condition", "")),
            int(row.get("generated_row_index", 0) or 0),
            int(row.get("position_index", 0) or 0),
            stable_hash_hex([row.get("entry_key", ""), row.get("condition", "")]),
        )
    )
    return rows


def _decode_observation_streams(
    *,
    observations: Sequence[Mapping[str, Any]],
    payloads: Sequence[tuple[str, str]],
    query_budgets: Sequence[int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[(str(row.get("condition", "")), str(row.get("model_condition", "")))].append(row)

    rows: list[dict[str, Any]] = []
    for (condition, model_condition), stream in sorted(grouped.items()):
        ordered = sorted(stream, key=_sort_key)
        for budget in query_budgets:
            budget_rows = ordered[: int(budget)]
            digits = [int(row["digit"]) for row in budget_rows]
            radices = [int(row["radix"]) for row in budget_rows]
            recovered_payload, decode_status, decoded_bytes = _decode_digits(digits, radices)
            for payload_id, payload_text in payloads:
                source_payloads = {str(row.get("source_payload_id", "")) for row in budget_rows}
                if condition.startswith("wrong_key_"):
                    null_family = "wrong_key"
                elif model_condition == "raw":
                    null_family = "raw"
                elif model_condition == "task_only_lora":
                    null_family = "task_only_lora"
                elif payload_id not in source_payloads:
                    null_family = "wrong_payload"
                else:
                    null_family = "same_payload_diagnostic"
                rows.append(
                    {
                        "stream_family": "pre_null_observed",
                        "condition": condition,
                        "model_condition": model_condition,
                        "payload_id": payload_id,
                        "expected_payload": payload_text,
                        "query_budget": int(budget),
                        "null_family": null_family,
                        "accepted": recovered_payload == payload_text,
                        "recovered_payload": recovered_payload,
                        "observed_symbols": len(digits),
                        "usable_radix_symbols": len(radices),
                        "decoded_bytes": decoded_bytes,
                        "decode_status": decode_status,
                        "result_claim": "variable_radix_verifier_preflight_not_far",
                    }
                )
    return rows


def _synthetic_decode_rows(
    *,
    payload_plans: Sequence[Mapping[str, Any]],
    assignments_by_payload: Mapping[str, Sequence[Mapping[str, Any]]],
    payloads: Sequence[tuple[str, str]],
    query_budgets: Sequence[int],
) -> list[dict[str, Any]]:
    payload_text_by_id = {payload_id: text for payload_id, text in payloads}
    rows: list[dict[str, Any]] = []
    for plan in payload_plans:
        payload_id = str(plan["payload_id"])
        assignments = list(assignments_by_payload[payload_id])
        digits = [int(row["target_digit"]) for row in assignments]
        radices = [int(row["target_radix"]) for row in assignments]
        for budget in query_budgets:
            use_count = min(len(digits), int(budget))
            recovered_payload, decode_status, decoded_bytes = _decode_digits(digits[:use_count], radices[:use_count])
            for expected_payload_id, expected_text in payloads:
                null_family = "synthetic_protected" if expected_payload_id == payload_id else "wrong_payload"
                rows.append(
                    {
                        "stream_family": "synthetic_target",
                        "condition": "correct_key",
                        "model_condition": "protected_target_plan",
                        "payload_id": payload_id,
                        "expected_payload_id": expected_payload_id,
                        "expected_payload": expected_text,
                        "query_budget": int(budget),
                        "null_family": null_family,
                        "accepted": recovered_payload == expected_text,
                        "recovered_payload": recovered_payload,
                        "observed_symbols": use_count,
                        "usable_radix_symbols": use_count,
                        "decoded_bytes": decoded_bytes,
                        "decode_status": decode_status,
                        "result_claim": "variable_radix_verifier_preflight_not_recovery",
                    }
                )
        if payload_id not in payload_text_by_id:
            raise ValueError(f"Payload plan {payload_id!r} not present in configured payloads")
    return rows


def run_preflight(
    *,
    config_path: Path,
    variable_arity_bank_entries: Path,
    pre_null_observations: Path,
    pre_null_summary: Path,
    output_dir: Path,
    payload_ids: str,
    query_budgets_override: str,
    max_assignments_per_payload: int,
    max_null_observations: int,
) -> dict[str, Any]:
    output_paths = [
        output_dir / "variable_radix_train_contract_preflight.json",
        output_dir / "variable_radix_train_assignments_preflight.jsonl",
        output_dir / "variable_radix_eval_decode_preflight.csv",
        output_dir / "variable_radix_verifier_preflight_summary.json",
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite variable-radix preflight outputs: " + ", ".join(existing))

    config = read_yaml(config_path)
    summary = json.loads(pre_null_summary.read_text(encoding="utf-8"))
    payloads = _payload_texts(config, payload_ids)
    if len(payloads) < 2:
        raise ValueError("Variable-radix preflight requires at least two configured payloads")
    query_budgets = _query_budgets(config, query_budgets_override, summary)
    bank_entries = _load_bank_entries(variable_arity_bank_entries)
    if not bank_entries:
        raise ValueError("No variable-arity bank entries with arity >= 2")

    payload_plans: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    assignments_by_payload: dict[str, list[dict[str, Any]]] = {}
    for payload_id, payload_text in payloads:
        plan, payload_assignments = _build_payload_plan(
            payload_id=payload_id,
            payload_text=payload_text,
            bank_entries=bank_entries,
            max_assignments=max_assignments_per_payload,
        )
        payload_plans.append(plan)
        assignments.extend(payload_assignments)
        assignments_by_payload[payload_id] = payload_assignments

    observations = _load_observations(pre_null_observations, max_records=max_null_observations)
    synthetic_rows = _synthetic_decode_rows(
        payload_plans=payload_plans,
        assignments_by_payload=assignments_by_payload,
        payloads=payloads,
        query_budgets=query_budgets,
    )
    null_rows = _decode_observation_streams(
        observations=observations,
        payloads=payloads,
        query_budgets=query_budgets,
    )
    decode_rows = synthetic_rows + null_rows

    protected_failures = [
        row
        for row in synthetic_rows
        if row["null_family"] == "synthetic_protected" and not bool(row["accepted"])
    ]
    blocking_null_accepts = [
        row
        for row in decode_rows
        if row.get("null_family") in {"raw", "task_only_lora", "wrong_key", "wrong_payload"}
        and bool(row.get("accepted"))
    ]
    null_family_accepts = Counter(str(row.get("null_family", "")) for row in blocking_null_accepts)
    payload_plan_status = "PASS" if all(row["status"] == "PASS" for row in payload_plans) else "FAIL"
    train_contract_status = "PASS_DRY_RUN_NOT_TRAINING" if payload_plan_status == "PASS" and assignments else "FAIL"
    verifier_status = (
        "PASS_PREFLIGHT"
        if not protected_failures and not blocking_null_accepts and payload_plan_status == "PASS"
        else "FAIL_PREFLIGHT"
    )
    contract = {
        "schema_name": CONTRACT_SCHEMA,
        "protocol_id": str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1")),
        "payload_plans": payload_plans,
        "query_budgets": [int(value) for value in query_budgets],
        "variable_arity_bank_entries": str(variable_arity_bank_entries),
        "pre_null_summary": str(pre_null_summary),
        "pre_null_observations": str(pre_null_observations),
        "assignment_count": len(assignments),
        "max_assignments_per_payload": int(max_assignments_per_payload),
        "claim_control": {
            "ready_for_model_training": False,
            "training_started": False,
            "e2e_eval_started": False,
            "paper_claim_allowed": False,
            "result_claim": "variable_radix_train_eval_preflight_not_training_not_recovery",
        },
    }
    summary_out = {
        "schema_name": SUMMARY_SCHEMA,
        "status": "COMPLETE_PREFLIGHT_PENDING_REVIEW",
        "overall_status": "PASS_PREFLIGHT_NOT_TRAINING" if verifier_status == "PASS_PREFLIGHT" else "FAIL_PREFLIGHT",
        "payload_plan_status": payload_plan_status,
        "train_contract_status": train_contract_status,
        "eval_verifier_status": verifier_status,
        "synthetic_protected_decode_failures": len(protected_failures),
        "blocking_null_accept_count": len(blocking_null_accepts),
        "blocking_null_accept_count_by_family": dict(sorted(null_family_accepts.items())),
        "payload_count": len(payloads),
        "payload_ids": [payload_id for payload_id, _ in payloads],
        "query_budgets": [int(value) for value in query_budgets],
        "variable_arity_bank_entries": str(variable_arity_bank_entries),
        "available_bank_entries": len(bank_entries),
        "assignment_count": len(assignments),
        "decode_row_count": len(decode_rows),
        "pre_null_observation_rows_used": len(observations),
        "ready_for_training_submission": False,
        "training_started": False,
        "e2e_eval_started": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Review variable-radix preflight and remaining gates. Do not submit "
            "Qwen proof-of-life training without explicit gate approval."
        ),
        "outputs": {
            "contract_json": str(output_dir / "variable_radix_train_contract_preflight.json"),
            "assignments_jsonl": str(output_dir / "variable_radix_train_assignments_preflight.jsonl"),
            "decode_csv": str(output_dir / "variable_radix_eval_decode_preflight.csv"),
            "summary_json": str(output_dir / "variable_radix_verifier_preflight_summary.json"),
        },
        "result_claim": "variable_radix_train_eval_verifier_preflight_not_training_not_recovery",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "variable_radix_train_contract_preflight.json", contract)
    write_jsonl(output_dir / "variable_radix_train_assignments_preflight.jsonl", assignments)
    write_csv(output_dir / "variable_radix_eval_decode_preflight.csv", decode_rows, list(decode_rows[0].keys()) if decode_rows else [])
    write_json(output_dir / "variable_radix_verifier_preflight_summary.json", summary_out)
    print(json.dumps(summary_out, sort_keys=True))
    return summary_out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    run_preflight(
        config_path=resolve_repo_path(args.config, root),
        variable_arity_bank_entries=resolve_repo_path(args.variable_arity_bank_entries, root),
        pre_null_observations=resolve_repo_path(args.pre_null_observations, root),
        pre_null_summary=resolve_repo_path(args.pre_null_summary, root),
        output_dir=resolve_repo_path(args.output_dir, root),
        payload_ids=str(args.payload_ids),
        query_budgets_override=str(args.query_budgets),
        max_assignments_per_payload=int(args.max_assignments_per_payload),
        max_null_observations=int(args.max_null_observations),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
