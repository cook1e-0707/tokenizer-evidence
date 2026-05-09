from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.natural_evidence_v1.build_bucket_bank import _bucketize, _filter_candidates, _prefix_signature
from scripts.natural_evidence_v1.common import (
    bucket_mass_metrics,
    read_yaml,
    resolve_repo_path,
    stable_hash_hex,
    write_csv,
    write_json,
    write_jsonl,
)


SCHEMA_NAME = "natural_evidence_expanded_actual_prefix_bucketized_candidates_manifest_v1"
ROW_SCHEMA_NAME = "natural_evidence_expanded_actual_prefix_bucketized_candidates_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build relaxed compatibility-aware actual-prefix bucketized candidate "
            "rows from full top-k candidates. This is a construction/scoring input "
            "prep step only: no model scoring, no training, no E2E, no FAR."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket-count", type=int, default=4)
    parser.add_argument("--candidate-top-k", type=int, default=0)
    parser.add_argument("--min-arity", type=int, default=2)
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


def _model_config(config: Mapping[str, Any], tokenizer_key: str) -> dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, dict) or tokenizer_key not in models:
        raise ValueError(f"Missing model config for tokenizer key {tokenizer_key!r}")
    model_cfg = models[tokenizer_key]
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Model config for {tokenizer_key!r} must be a mapping")
    return dict(model_cfg)


def _bucketized_row(
    *,
    record: Mapping[str, Any],
    tokenizer_key: str,
    tokenizer_name: str,
    protocol_id: str,
    bank_id: str,
    prefix_signature: str,
    bucket_count: int,
    buckets: Mapping[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    token_to_bucket: dict[int, str] = {}
    bucket_payload: dict[str, list[int]] = {}
    bucket_texts: dict[str, list[str]] = {}
    bucket_masses: dict[str, float] = {}
    candidates: list[dict[str, Any]] = []
    for bucket_id, members in sorted(buckets.items()):
        bucket_payload[str(bucket_id)] = [int(member["token_id"]) for member in members]
        bucket_texts[str(bucket_id)] = [str(member["token_text"]) for member in members]
        bucket_masses[str(bucket_id)] = sum(float(member["probability"]) for member in members)
        for member in sorted(members, key=lambda item: (int(item["rank"]), int(item["token_id"]))):
            token_to_bucket[int(member["token_id"])] = str(bucket_id)
            candidates.append(
                {
                    "bucket_id": str(bucket_id),
                    "rank": int(member["rank"]),
                    "token_id": int(member["token_id"]),
                    "text": str(member["token_text"]),
                    "probability": float(member["probability"]),
                }
            )
    observed_token_id = int(record.get("observed_token_id", -1))
    observed_bucket_id = token_to_bucket.get(observed_token_id, "")
    entry_hash = stable_hash_hex(
        [
            bank_id,
            tokenizer_name,
            prefix_signature,
            record.get("prompt_id", ""),
            record.get("model_condition", ""),
            record.get("payload_id", ""),
            record.get("seed", ""),
            record.get("query_index", ""),
            record.get("generated_row_index", ""),
            record.get("position_index", ""),
        ]
    )[:24]
    return {
        "schema_name": ROW_SCHEMA_NAME,
        "protocol_id": protocol_id,
        "bank_id": bank_id,
        "bank_entry_id": f"{bank_id}_{entry_hash}",
        "context_signature": prefix_signature,
        "tokenizer_key": tokenizer_key,
        "tokenizer_name": tokenizer_name,
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
        "bucket_count": int(bucket_count),
        "arity": sum(1 for members in buckets.values() if members),
        "buckets": bucket_payload,
        "bucket_token_texts": bucket_texts,
        "reference_mass": bucket_masses,
        "bucket_mass_summary": bucket_mass_metrics([mass for mass in bucket_masses.values() if mass > 0.0]),
        "candidates": candidates,
        "result_claim": "expanded_actual_prefix_bucketized_candidates_not_compatibility_not_payload_recovery",
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "fingerprint_claim": False,
    }


def build_expanded_bucketized_candidates(
    *,
    config_path: Path,
    tokenizer_key: str,
    candidate_jsonl_path: Path,
    output_dir: Path,
    bucket_count: int,
    candidate_top_k_override: int,
    min_arity: int,
    max_records: int,
) -> dict[str, Any]:
    output_paths = [
        output_dir / "expanded_actual_prefix_bucketized_candidates.jsonl",
        output_dir / "expanded_actual_prefix_bucketized_manifest.json",
        output_dir / "expanded_actual_prefix_bucketized_by_position.csv",
        output_dir / "expanded_actual_prefix_bucketized_rejections.csv",
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite expanded bucketized outputs: " + ", ".join(existing))

    config = read_yaml(config_path)
    protocol = dict(config.get("protocol", {}))
    bucket_cfg = dict(config.get("bucket_bank", {}))
    selector = dict(config.get("selector", {}))
    model_cfg = _model_config(config, tokenizer_key)
    tokenizer_name = str(model_cfg.get("tokenizer_name") or model_cfg.get("model_name") or "")
    if not tokenizer_name:
        raise ValueError(f"Missing tokenizer_name for tokenizer key {tokenizer_key!r}")
    protocol_id = str(protocol.get("id", "natural_evidence_v1"))
    audit_key_id = str(selector.get("audit_key_id", "K001"))
    candidate_top_k = int(candidate_top_k_override or bucket_cfg.get("candidate_top_k", 64))
    min_probability = float(bucket_cfg.get("min_reference_probability", 0.0001))
    bucket_assignment = str(bucket_cfg.get("bucket_assignment", "keyed_mass_balance"))
    forbidden_patterns = [str(item) for item in protocol.get("forbidden_surface_patterns", [])]
    bank_id = f"{tokenizer_key}_expanded_actual_prefix_bucket_bank_topk{candidate_top_k}_b{bucket_count}_v1"

    rows: list[dict[str, Any]] = []
    by_position_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    arity_counts: Counter[int] = Counter()
    observed_bucketized_rows = 0
    input_records = 0
    filtered_candidate_total = 0

    for record in _iter_jsonl(candidate_jsonl_path, max_records=max_records):
        input_records += 1
        prefix_signature = _prefix_signature(record)
        filtered_candidates, candidate_rejections = _filter_candidates(
            record=record,
            candidate_top_k=candidate_top_k,
            min_probability=min_probability,
            forbidden_patterns=forbidden_patterns,
        )
        filtered_candidate_total += len(filtered_candidates)
        if len(filtered_candidates) < int(min_arity):
            reason = "insufficient_filtered_candidates_for_min_arity"
            rejection_counts[reason] += 1
            rejection_rows.append(
                {
                    "prompt_id": record.get("prompt_id", ""),
                    "prompt_split": record.get("prompt_split", ""),
                    "model_condition": record.get("model_condition", ""),
                    "payload_id": record.get("payload_id", ""),
                    "seed": record.get("seed", ""),
                    "query_index": record.get("query_index", 0),
                    "generated_row_index": record.get("generated_row_index", 0),
                    "position_index": record.get("position_index", 0),
                    "filtered_candidate_count": len(filtered_candidates),
                    "arity": 0,
                    "rejection_reason": reason,
                    "candidate_rejection_reasons_json": json.dumps(sorted(set(candidate_rejections))),
                }
            )
            continue
        buckets = _bucketize(
            candidates=filtered_candidates,
            bucket_count=int(bucket_count),
            min_members_per_bucket=1,
            key=audit_key_id,
            protocol_id=protocol_id,
            bank_id=bank_id,
            prefix_signature=prefix_signature,
            assignment_mode=bucket_assignment,
        )
        arity = sum(1 for members in buckets.values() if members)
        arity_counts[arity] += 1
        if arity < int(min_arity):
            reason = "below_min_arity"
            rejection_counts[reason] += 1
            continue
        row = _bucketized_row(
            record=record,
            tokenizer_key=tokenizer_key,
            tokenizer_name=tokenizer_name,
            protocol_id=protocol_id,
            bank_id=bank_id,
            prefix_signature=prefix_signature,
            bucket_count=bucket_count,
            buckets=buckets,
        )
        rows.append(row)
        if row["observed_token_bucketized"]:
            observed_bucketized_rows += 1
        by_position_rows.append(
            {
                "bank_entry_id": row["bank_entry_id"],
                "prompt_id": row["prompt_id"],
                "prompt_split": row["prompt_split"],
                "model_condition": row["model_condition"],
                "payload_id": row["payload_id"],
                "seed": row["seed"],
                "query_index": row["query_index"],
                "generated_row_index": row["generated_row_index"],
                "position_index": row["position_index"],
                "bucket_count": row["bucket_count"],
                "arity": row["arity"],
                "candidate_count": len(row["candidates"]),
                "observed_token_bucketized": row["observed_token_bucketized"],
                "observed_token_bucket_id": row["observed_token_bucket_id"],
                "min_bucket_mass": row["bucket_mass_summary"]["min_bucket_mass"],
                "bucket_mass_ratio": row["bucket_mass_summary"]["bucket_mass_ratio"],
                "bucket_entropy_fraction": row["bucket_mass_summary"]["bucket_entropy_fraction"],
            }
        )

    manifest = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_PENDING_SUFFIX_COMPATIBILITY_SCORING",
        "tokenizer_key": tokenizer_key,
        "tokenizer_name": tokenizer_name,
        "candidate_jsonl": str(candidate_jsonl_path),
        "bank_id": bank_id,
        "bucket_count": int(bucket_count),
        "candidate_top_k": int(candidate_top_k),
        "min_reference_probability": min_probability,
        "min_arity": int(min_arity),
        "input_records": input_records,
        "accepted_rows": len(rows),
        "rejected_rows": sum(rejection_counts.values()),
        "filtered_candidate_total": filtered_candidate_total,
        "observed_token_bucketized_rows": observed_bucketized_rows,
        "observed_token_bucketized_rate": observed_bucketized_rows / len(rows) if rows else 0.0,
        "arity_counts": {str(key): int(value) for key, value in sorted(arity_counts.items())},
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "outputs": {
            "bucketized_candidates_jsonl": str(output_dir / "expanded_actual_prefix_bucketized_candidates.jsonl"),
            "by_position_csv": str(output_dir / "expanded_actual_prefix_bucketized_by_position.csv"),
            "rejections_csv": str(output_dir / "expanded_actual_prefix_bucketized_rejections.csv"),
            "manifest_json": str(output_dir / "expanded_actual_prefix_bucketized_manifest.json"),
        },
        "next_allowed_action": (
            "Run suffix-preserving compatibility scoring over these expanded actual-prefix candidates; "
            "do not train from construction alone."
        ),
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "result_claim": "expanded_actual_prefix_bucketized_candidates_not_compatibility_not_payload_recovery",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "expanded_actual_prefix_bucketized_candidates.jsonl", rows)
    write_json(output_dir / "expanded_actual_prefix_bucketized_manifest.json", manifest)
    write_csv(output_dir / "expanded_actual_prefix_bucketized_by_position.csv", by_position_rows, list(by_position_rows[0].keys()) if by_position_rows else [])
    write_csv(
        output_dir / "expanded_actual_prefix_bucketized_rejections.csv",
        rejection_rows,
        [
            "prompt_id",
            "prompt_split",
            "model_condition",
            "payload_id",
            "seed",
            "query_index",
            "generated_row_index",
            "position_index",
            "filtered_candidate_count",
            "arity",
            "rejection_reason",
            "candidate_rejection_reasons_json",
        ],
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    manifest = build_expanded_bucketized_candidates(
        config_path=resolve_repo_path(args.config, root),
        tokenizer_key=str(args.tokenizer_key),
        candidate_jsonl_path=Path(args.candidate_jsonl),
        output_dir=Path(args.output_dir),
        bucket_count=int(args.bucket_count),
        candidate_top_k_override=int(args.candidate_top_k),
        min_arity=int(args.min_arity),
        max_records=int(args.max_records),
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
