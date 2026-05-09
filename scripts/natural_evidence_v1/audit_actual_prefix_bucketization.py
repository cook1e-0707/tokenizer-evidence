from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from scripts.natural_evidence_v1.build_bucket_bank import _entry_from_record
from scripts.natural_evidence_v1.common import (
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    write_csv,
    write_json,
    write_jsonl,
)


SCHEMA_NAME = "natural_evidence_actual_prefix_bucketization_audit_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bucketize actual generated prefixes from reference-model top-k "
            "candidate records and audit observed-token reconstructability. "
            "This is a CPU-only audit; it is not compatibility suffix scoring, "
            "training, E2E evaluation, or payload recovery."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bank-id", default="")
    parser.add_argument("--audit-key-id", default="")
    parser.add_argument("--bucket-count", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--strict-balance-gate", action="store_true")
    parser.add_argument("--balance-min-bucket-mass", type=float, default=0.0)
    parser.add_argument("--max-bucket-mass-ratio", type=float, default=0.0)
    parser.add_argument("--min-bucket-entropy-fraction", type=float, default=-1.0)
    return parser.parse_args(argv)


def _bucket_for_token(entry: Mapping[str, Any], token_id: int) -> str:
    buckets = entry.get("buckets", {})
    if not isinstance(buckets, dict):
        return ""
    for bucket_id, token_ids in buckets.items():
        if isinstance(token_ids, list) and int(token_id) in {int(item) for item in token_ids}:
            return str(bucket_id)
    return ""


def _bucketized_candidate_record(entry: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    token_to_bucket: dict[int, str] = {}
    buckets = entry.get("buckets", {})
    if isinstance(buckets, dict):
        for bucket_id, token_ids in buckets.items():
            if isinstance(token_ids, list):
                for token_id in token_ids:
                    token_to_bucket[int(token_id)] = str(bucket_id)

    candidates: list[dict[str, Any]] = []
    for candidate in record.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        token_id = int(candidate.get("token_id", -1))
        bucket_id = token_to_bucket.get(token_id)
        if bucket_id is None:
            continue
        candidates.append(
            {
                "bucket_id": bucket_id,
                "rank": int(candidate.get("rank", 0)),
                "token_id": token_id,
                "text": str(candidate.get("text", "")),
                "probability": float(candidate.get("probability", 0.0)),
            }
        )

    observed_token_id = int(record.get("observed_token_id", -1))
    observed_bucket_id = token_to_bucket.get(observed_token_id, "")
    return {
        "schema_name": "natural_evidence_actual_prefix_bucketized_candidates_v1",
        "protocol_id": entry.get("protocol_id", record.get("protocol_id", "natural_evidence_v1")),
        "bank_id": entry.get("bank_id", ""),
        "bank_entry_id": entry.get("bank_entry_id", ""),
        "context_signature": entry.get("context_signature", record.get("prefix_signature", "")),
        "tokenizer_key": record.get("tokenizer_key", ""),
        "tokenizer_name": entry.get("tokenizer_name", record.get("tokenizer_name", "")),
        "model_condition": record.get("model_condition", ""),
        "payload_id": record.get("payload_id", ""),
        "seed": record.get("seed", ""),
        "prompt_id": record.get("prompt_id", ""),
        "prompt_split": record.get("prompt_split", ""),
        "query_index": record.get("query_index", 0),
        "generated_row_index": record.get("generated_row_index", 0),
        "position_index": record.get("position_index", 0),
        "prefix_response_token_count": record.get("prefix_response_token_count", 0),
        "response_token_count": record.get("response_token_count", 0),
        "prefix_token_ids": record.get("prefix_token_ids", []),
        "observed_token_id": observed_token_id,
        "observed_token_text": record.get("observed_token_text", ""),
        "observed_token_bucket_id": observed_bucket_id,
        "observed_token_bucketized": observed_bucket_id != "",
        "bucket_count": entry.get("bucket_count", ""),
        "candidates": candidates,
        "result_claim": "actual_prefix_bucketized_candidates_not_compatibility_not_payload_recovery",
        "fingerprint_claim": False,
    }


def _audit_row(
    *,
    record: Mapping[str, Any],
    entry: Mapping[str, Any] | None,
    rejection: Mapping[str, Any],
) -> dict[str, Any]:
    observed_token_id = int(record.get("observed_token_id", -1))
    observed_bucket_id = _bucket_for_token(entry, observed_token_id) if entry is not None else ""
    mass_summary = entry.get("bucket_mass_summary", {}) if entry is not None else {}
    return {
        "prompt_id": record.get("prompt_id", ""),
        "prompt_split": record.get("prompt_split", ""),
        "model_condition": record.get("model_condition", ""),
        "payload_id": record.get("payload_id", ""),
        "seed": record.get("seed", ""),
        "query_index": record.get("query_index", 0),
        "generated_row_index": record.get("generated_row_index", 0),
        "position_index": record.get("position_index", 0),
        "prefix_response_token_count": record.get("prefix_response_token_count", 0),
        "response_token_count": record.get("response_token_count", 0),
        "prefix_signature": record.get("prefix_signature", ""),
        "accepted_entry": entry is not None,
        "bank_entry_id": entry.get("bank_entry_id", "") if entry is not None else "",
        "observed_token_id": observed_token_id,
        "observed_token_text": record.get("observed_token_text", ""),
        "observed_token_in_topk": bool(record.get("observed_token_in_topk", False)),
        "observed_token_rank": record.get("observed_token_rank", ""),
        "observed_token_probability": record.get("observed_token_probability", ""),
        "observed_token_bucketized": observed_bucket_id != "",
        "observed_bucket_id": observed_bucket_id,
        "candidate_token_count": entry.get("candidate_token_count", 0) if entry is not None else 0,
        "surface_allowed_candidate_count": record.get("surface_allowed_candidate_count", 0),
        "min_bucket_mass": mass_summary.get("min_bucket_mass", ""),
        "bucket_mass_ratio": mass_summary.get("bucket_mass_ratio", ""),
        "bucket_entropy_fraction": mass_summary.get("bucket_entropy_fraction", ""),
        "rejection_reason": rejection.get("rejection_reason", ""),
    }


def run_audit(
    *,
    config_path: Path,
    tokenizer_key: str,
    candidate_jsonl_path: Path,
    output_dir: Path,
    bank_id: str,
    audit_key_id: str,
    bucket_count_override: int,
    max_records: int,
    strict_balance_gate_override: bool,
    balance_min_bucket_mass_override: float,
    max_bucket_mass_ratio_override: float,
    min_bucket_entropy_fraction_override: float,
) -> dict[str, Any]:
    config = read_yaml(config_path)
    protocol = dict(config.get("protocol", {}))
    bucket_cfg = dict(config.get("bucket_bank", {}))
    quality_gates = dict(bucket_cfg.get("quality_gates", {}))
    models = dict(config.get("models", {}))
    selector = dict(config.get("selector", {}))
    model_cfg = dict(models.get(tokenizer_key, {}))
    tokenizer_name = str(model_cfg.get("tokenizer_name") or model_cfg.get("model_name") or "")
    if not tokenizer_name:
        raise ValueError(f"Missing tokenizer_name for tokenizer key {tokenizer_key!r}")

    protocol_id = str(protocol.get("id", "natural_evidence_v1"))
    audit_key = audit_key_id or str(selector.get("audit_key_id", "K001"))
    bucket_count = bucket_count_override or int(
        dict(bucket_cfg.get("compatibility_adjusted_capacity", {}))
        .get("diagnostic_high_risk_gate", {})
        .get("bucket_count", 4)
    )
    bank = bank_id or f"{tokenizer_key}_actual_prefix_bucket_bank_topk64_v1"
    candidate_top_k = int(bucket_cfg.get("candidate_top_k", 64))
    bucket_assignment = str(bucket_cfg.get("bucket_assignment", "keyed_mass_balance"))
    min_probability = float(bucket_cfg.get("min_reference_probability", 0.0001))
    min_members_per_bucket = int(bucket_cfg.get("min_members_per_bucket", 2))
    min_bucket_mass = float(bucket_cfg.get("min_bucket_mass", 0.01))
    strict_min_bucket_mass = bool(bucket_cfg.get("strict_min_bucket_mass", False))
    strict_balance_gate = bool(strict_balance_gate_override)
    balance_min_bucket_mass = (
        balance_min_bucket_mass_override
        if balance_min_bucket_mass_override > 0.0
        else float(quality_gates.get("min_bucket_mass", min_bucket_mass))
    )
    max_bucket_mass_ratio = (
        max_bucket_mass_ratio_override
        if max_bucket_mass_ratio_override > 0.0
        else float(quality_gates.get("max_bucket_mass_ratio", float("inf")))
    )
    min_bucket_entropy_fraction = (
        min_bucket_entropy_fraction_override
        if min_bucket_entropy_fraction_override >= 0.0
        else float(quality_gates.get("min_bucket_entropy_fraction", 0.0))
    )
    forbidden_patterns = [str(item) for item in protocol.get("forbidden_surface_patterns", [])]

    records = read_jsonl(candidate_jsonl_path)
    if max_records > 0:
        records = records[:max_records]

    entries: list[dict[str, Any]] = []
    by_position_rows: list[dict[str, Any]] = []
    bucketized_candidate_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    observed_bucket_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    condition_bucketized_counts: Counter[str] = Counter()

    for record in records:
        entry, rejection = _entry_from_record(
            record=record,
            tokenizer_name=tokenizer_name,
            protocol_id=protocol_id,
            bank_id=bank,
            audit_key_id=audit_key,
            bucket_count=bucket_count,
            candidate_top_k=candidate_top_k,
            min_probability=min_probability,
            min_members_per_bucket=min_members_per_bucket,
            min_bucket_mass=balance_min_bucket_mass if strict_balance_gate else min_bucket_mass,
            strict_min_bucket_mass=strict_min_bucket_mass,
            strict_balance_gate=strict_balance_gate,
            max_bucket_mass_ratio=max_bucket_mass_ratio,
            min_bucket_entropy_fraction=min_bucket_entropy_fraction,
            bucket_assignment=bucket_assignment,
            forbidden_patterns=forbidden_patterns,
        )
        row = _audit_row(record=record, entry=entry, rejection=rejection)
        by_position_rows.append(row)
        condition = str(record.get("model_condition", ""))
        condition_counts[condition] += 1
        if entry is None:
            rejection_counts[str(rejection.get("rejection_reason", ""))] += 1
            rejection_rows.append(dict(rejection))
            continue
        entries.append(entry)
        bucketized_record = _bucketized_candidate_record(entry, record)
        bucketized_candidate_rows.append(bucketized_record)
        if bucketized_record["observed_token_bucketized"]:
            observed_bucket_counts[str(bucketized_record["observed_token_bucket_id"])] += 1
            condition_bucketized_counts[condition] += 1

    accepted_entries = len(entries)
    observed_token_in_topk_rows = sum(1 for row in by_position_rows if row["observed_token_in_topk"])
    observed_token_bucketized_rows = sum(1 for row in by_position_rows if row["observed_token_bucketized"])
    output_dir.mkdir(parents=True, exist_ok=True)
    entries_path = output_dir / f"{tokenizer_key}_actual_prefix_bucket_entries.jsonl"
    bucketized_candidates_path = output_dir / f"{tokenizer_key}_actual_prefix_bucketized_candidates.jsonl"
    by_position_path = output_dir / "actual_prefix_bucketization_by_position.csv"
    rejection_path = output_dir / "actual_prefix_bucketization_rejections.csv"
    summary_path = output_dir / "actual_prefix_bucketization_summary.json"
    planned_outputs = [
        entries_path,
        bucketized_candidates_path,
        by_position_path,
        rejection_path,
        summary_path,
    ]
    existing_outputs = [str(path) for path in planned_outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing actual-prefix bucketization outputs: "
            + ", ".join(existing_outputs)
        )

    write_jsonl(entries_path, entries)
    write_jsonl(bucketized_candidates_path, bucketized_candidate_rows)
    write_csv(by_position_path, by_position_rows, list(by_position_rows[0].keys()) if by_position_rows else [])
    write_csv(
        rejection_path,
        rejection_rows,
        [
            "prompt_id",
            "prefix_signature",
            "candidate_count",
            "rejection_reason",
            "candidate_rejection_reasons_json",
            "min_bucket_mass",
            "bucket_mass_ratio",
            "bucket_entropy_fraction",
        ],
    )
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_PENDING_SUFFIX_COMPATIBILITY_SCORING",
        "protocol_id": protocol_id,
        "tokenizer_key": tokenizer_key,
        "tokenizer_name": tokenizer_name,
        "bank_id": bank,
        "audit_key_id": audit_key,
        "candidate_source": str(candidate_jsonl_path),
        "input_records": len(records),
        "accepted_entries": accepted_entries,
        "rejected_records": len(rejection_rows),
        "bucket_count": bucket_count,
        "candidate_top_k": candidate_top_k,
        "min_reference_probability": min_probability,
        "min_members_per_bucket": min_members_per_bucket,
        "strict_balance_gate": strict_balance_gate,
        "balance_gate_thresholds": {
            "min_bucket_mass": balance_min_bucket_mass,
            "max_bucket_mass_ratio": max_bucket_mass_ratio,
            "min_bucket_entropy_fraction": min_bucket_entropy_fraction,
        },
        "observed_token_in_topk_rows": observed_token_in_topk_rows,
        "observed_token_in_topk_rate": observed_token_in_topk_rows / len(records) if records else 0.0,
        "observed_token_bucketized_rows": observed_token_bucketized_rows,
        "observed_token_bucketized_rate": observed_token_bucketized_rows / len(records) if records else 0.0,
        "observed_token_bucketized_rate_among_accepted": (
            observed_token_bucketized_rows / accepted_entries if accepted_entries else 0.0
        ),
        "observed_bucket_counts": dict(sorted(observed_bucket_counts.items())),
        "condition_counts": dict(sorted(condition_counts.items())),
        "condition_bucketized_counts": dict(sorted(condition_bucketized_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "outputs": {
            "entries_jsonl": str(entries_path),
            "bucketized_candidates_jsonl": str(bucketized_candidates_path),
            "by_position_csv": str(by_position_path),
            "rejections_csv": str(rejection_path),
            "summary_json": str(summary_path),
        },
        "next_minimal_action": (
            "Run actual-prefix suffix compatibility scoring over bucketized candidates; "
            "do not start training or E2E evaluation from bucketization alone."
        ),
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "result_claim": "actual_prefix_bucketization_audit_not_compatibility_not_payload_recovery",
        "fingerprint_claim": False,
    }
    write_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    summary = run_audit(
        config_path=resolve_repo_path(args.config, root),
        tokenizer_key=args.tokenizer_key,
        candidate_jsonl_path=resolve_repo_path(args.candidate_jsonl, root),
        output_dir=resolve_repo_path(args.output_dir, root),
        bank_id=args.bank_id,
        audit_key_id=args.audit_key_id,
        bucket_count_override=args.bucket_count,
        max_records=args.max_records,
        strict_balance_gate_override=args.strict_balance_gate,
        balance_min_bucket_mass_override=args.balance_min_bucket_mass,
        max_bucket_mass_ratio_override=args.max_bucket_mass_ratio,
        min_bucket_entropy_fraction_override=args.min_bucket_entropy_fraction,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
