from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.natural_evidence_v1.build_bucket_bank import _bucketize, _filter_candidates, _prefix_signature
from scripts.natural_evidence_v1.common import (
    bucket_mass_metrics,
    read_yaml,
    resolve_repo_path,
    write_csv,
    write_json,
)


SCHEMA_NAME = "natural_evidence_compatibility_aware_supply_audit_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit actual-prefix top-k candidate supply for compatibility-aware "
            "variable-arity construction. This is a CPU planning diagnostic only: "
            "no model scoring, no training, no E2E, no FAR aggregation."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generated-output-count", type=int, required=True)
    parser.add_argument("--bucket-counts", default="2,4")
    parser.add_argument("--candidate-top-k", type=int, default=0)
    parser.add_argument("--min-arity", type=int, default=2)
    parser.add_argument("--configured-min-members", type=int, default=2)
    parser.add_argument("--min-bucket-mass", type=float, default=0.005)
    parser.add_argument("--max-bucket-mass-ratio", type=float, default=5.0)
    parser.add_argument("--min-entropy-fraction", type=float, default=0.90)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--write-position-csv", action="store_true")
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


def _bucket_count_values(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("--bucket-counts must include at least one integer")
    return values


def _model_config(config: Mapping[str, Any], tokenizer_key: str) -> dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, dict) or tokenizer_key not in models:
        raise ValueError(f"Missing model config for tokenizer key {tokenizer_key!r}")
    model_cfg = models[tokenizer_key]
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Model config for {tokenizer_key!r} must be a mapping")
    return dict(model_cfg)


def _record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt_id": record.get("prompt_id", ""),
        "prompt_split": record.get("prompt_split", ""),
        "model_condition": record.get("model_condition", ""),
        "payload_id": record.get("payload_id", ""),
        "seed": record.get("seed", ""),
        "query_index": record.get("query_index", 0),
        "generated_row_index": record.get("generated_row_index", 0),
        "position_index": record.get("position_index", 0),
    }


def _response_key(record: Mapping[str, Any]) -> str:
    parts = [
        record.get("prompt_id", ""),
        record.get("model_condition", ""),
        record.get("payload_id", ""),
        record.get("seed", ""),
        record.get("query_index", ""),
        record.get("generated_row_index", ""),
    ]
    return "||".join(str(part) for part in parts)


def _init_stats() -> dict[str, Any]:
    return {
        "input_records": 0,
        "accepted_positions": 0,
        "configured_subset_positions": 0,
        "probability_gated_positions": 0,
        "observed_token_bucketized_positions": 0,
        "total_capacity_bits": 0.0,
        "unique_generated_rows": set(),
        "denominator_response_tokens_by_response": {},
    }


def _update_stats(stats: dict[str, Any], record: Mapping[str, Any], accepted: bool, capacity_bits: float) -> None:
    stats["input_records"] += 1
    response_key = _response_key(record)
    if response_key:
        stats["unique_generated_rows"].add(response_key)
        stats["denominator_response_tokens_by_response"][response_key] = int(record.get("response_token_count", 0))
    if accepted:
        stats["accepted_positions"] += 1
        stats["total_capacity_bits"] += float(capacity_bits)


def run_supply_audit(
    *,
    config_path: Path,
    tokenizer_key: str,
    candidate_jsonl_path: Path,
    output_dir: Path,
    generated_output_count: int,
    bucket_counts: list[int],
    candidate_top_k_override: int,
    min_arity: int,
    configured_min_members: int,
    min_bucket_mass: float,
    max_bucket_mass_ratio: float,
    min_entropy_fraction: float,
    max_records: int,
    write_position_csv: bool,
) -> dict[str, Any]:
    output_paths = [
        output_dir / "compatibility_aware_supply_manifest.json",
        output_dir / "candidate_supply_by_bucket_count.csv",
        output_dir / "candidate_supply_arity_distribution.csv",
        output_dir / "candidate_supply_by_split.csv",
        output_dir / "candidate_supply_rejections.csv",
    ]
    if write_position_csv:
        output_paths.append(output_dir / "candidate_supply_by_position.csv")
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite compatibility-aware supply outputs: " + ", ".join(existing))

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

    bucket_count_rows: list[dict[str, Any]] = []
    arity_counter: Counter[tuple[int, int]] = Counter()
    rejection_counter: Counter[tuple[int, str]] = Counter()
    split_stats: dict[tuple[int, str], dict[str, Any]] = defaultdict(_init_stats)
    global_stats: dict[int, dict[str, Any]] = {bucket_count: _init_stats() for bucket_count in bucket_counts}
    position_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    input_records = 0

    for record in _iter_jsonl(candidate_jsonl_path, max_records=max_records):
        input_records += 1
        prefix_signature = _prefix_signature(record)
        filtered_candidates, candidate_rejections = _filter_candidates(
            record=record,
            candidate_top_k=candidate_top_k,
            min_probability=min_probability,
            forbidden_patterns=forbidden_patterns,
        )
        identity = _record_identity(record)
        for bucket_count in bucket_counts:
            stats = global_stats[bucket_count]
            split = str(record.get("prompt_split", ""))
            split_key = (bucket_count, split)
            if len(filtered_candidates) < min_arity:
                rejection_reason = "insufficient_filtered_candidates_for_min_arity"
                arity_counter[(bucket_count, 0)] += 1
                rejection_counter[(bucket_count, rejection_reason)] += 1
                _update_stats(stats, record, accepted=False, capacity_bits=0.0)
                _update_stats(split_stats[split_key], record, accepted=False, capacity_bits=0.0)
                rejection_rows.append(
                    {
                        **identity,
                        "bucket_count": bucket_count,
                        "filtered_candidate_count": len(filtered_candidates),
                        "arity": 0,
                        "rejection_reason": rejection_reason,
                        "candidate_rejection_reasons_json": json.dumps(sorted(set(candidate_rejections))),
                    }
                )
                continue
            buckets = _bucketize(
                candidates=filtered_candidates,
                bucket_count=bucket_count,
                min_members_per_bucket=1,
                key=audit_key_id,
                protocol_id=protocol_id,
                bank_id=f"{tokenizer_key}_compatibility_aware_supply_upper_bound_v1",
                prefix_signature=prefix_signature,
                assignment_mode=bucket_assignment,
            )
            nonempty_bucket_ids = [bucket_id for bucket_id, members in buckets.items() if members]
            configured_bucket_ids = [
                bucket_id for bucket_id, members in buckets.items() if len(members) >= int(configured_min_members)
            ]
            bucket_masses = {
                str(bucket_id): sum(float(member["probability"]) for member in members)
                for bucket_id, members in buckets.items()
                if members
            }
            metrics = bucket_mass_metrics(list(bucket_masses.values()))
            arity = len(nonempty_bucket_ids)
            arity_counter[(bucket_count, arity)] += 1
            accepted = arity >= int(min_arity)
            capacity_bits = math.log2(arity) if accepted else 0.0
            observed_token_id = int(record.get("observed_token_id", -1))
            observed_bucket_id = ""
            for bucket_id, members in buckets.items():
                if any(int(member["token_id"]) == observed_token_id for member in members):
                    observed_bucket_id = str(bucket_id)
                    break
            configured_subset = len(configured_bucket_ids) >= int(min_arity)
            probability_gated = (
                accepted
                and metrics["min_bucket_mass"] >= float(min_bucket_mass)
                and metrics["bucket_mass_ratio"] <= float(max_bucket_mass_ratio)
                and metrics["bucket_entropy_fraction"] >= float(min_entropy_fraction)
            )
            _update_stats(stats, record, accepted=accepted, capacity_bits=capacity_bits)
            _update_stats(split_stats[split_key], record, accepted=accepted, capacity_bits=capacity_bits)
            if configured_subset:
                stats["configured_subset_positions"] += 1
                split_stats[split_key]["configured_subset_positions"] += 1
            if probability_gated:
                stats["probability_gated_positions"] += 1
                split_stats[split_key]["probability_gated_positions"] += 1
            if observed_bucket_id:
                stats["observed_token_bucketized_positions"] += 1
                split_stats[split_key]["observed_token_bucketized_positions"] += 1
            if not accepted:
                rejection_reason = "below_min_arity"
                rejection_counter[(bucket_count, rejection_reason)] += 1
                rejection_rows.append(
                    {
                        **identity,
                        "bucket_count": bucket_count,
                        "filtered_candidate_count": len(filtered_candidates),
                        "arity": arity,
                        "rejection_reason": rejection_reason,
                        "candidate_rejection_reasons_json": json.dumps(sorted(set(candidate_rejections))),
                    }
                )
            if write_position_csv:
                position_rows.append(
                    {
                        **identity,
                        "bucket_count": bucket_count,
                        "filtered_candidate_count": len(filtered_candidates),
                        "arity": arity,
                        "capacity_bits": capacity_bits,
                        "configured_subset": configured_subset,
                        "probability_gated": probability_gated,
                        "observed_token_bucketized": bool(observed_bucket_id),
                        "observed_bucket_id": observed_bucket_id,
                        "min_bucket_mass": metrics["min_bucket_mass"],
                        "bucket_mass_ratio": metrics["bucket_mass_ratio"],
                        "bucket_entropy_fraction": metrics["bucket_entropy_fraction"],
                    }
                )

    for bucket_count in bucket_counts:
        stats = global_stats[bucket_count]
        generated_rows = len(stats["unique_generated_rows"])
        denominator_tokens = sum(int(value) for value in stats["denominator_response_tokens_by_response"].values())
        accepted_positions = int(stats["accepted_positions"])
        total_bits = float(stats["total_capacity_bits"])
        max_fixed_bits = input_records * math.log2(bucket_count) if bucket_count >= 2 else 0.0
        bucket_count_rows.append(
            {
                "bucket_count": bucket_count,
                "input_records": input_records,
                "accepted_positions": accepted_positions,
                "configured_subset_positions": int(stats["configured_subset_positions"]),
                "probability_gated_positions": int(stats["probability_gated_positions"]),
                "observed_token_bucketized_positions": int(stats["observed_token_bucketized_positions"]),
                "unique_generated_rows": generated_rows,
                "denominator_response_tokens": denominator_tokens,
                "total_capacity_bits": total_bits,
                "effective_bits_per_response": total_bits / int(generated_output_count),
                "theoretical_max_fixed_bits_per_response": max_fixed_bits / int(generated_output_count),
                "gate_effective_bits_per_response_min": 0.8,
                "gate_status": "PASS_UPPER_BOUND" if total_bits / int(generated_output_count) >= 0.8 else "FAIL_UPPER_BOUND",
            }
        )

    arity_rows = [
        {
            "bucket_count": bucket_count,
            "arity": arity,
            "positions": arity_counter.get((bucket_count, arity), 0),
            "capacity_bits_per_position": math.log2(arity) if arity >= int(min_arity) else 0.0,
            "total_capacity_bits": arity_counter.get((bucket_count, arity), 0)
            * (math.log2(arity) if arity >= int(min_arity) else 0.0),
        }
        for bucket_count in bucket_counts
        for arity in range(bucket_count + 1)
    ]
    split_rows: list[dict[str, Any]] = []
    for (bucket_count, split), stats in sorted(split_stats.items()):
        denominator_tokens = sum(int(value) for value in stats["denominator_response_tokens_by_response"].values())
        accepted_positions = int(stats["accepted_positions"])
        split_rows.append(
            {
                "bucket_count": bucket_count,
                "prompt_split": split,
                "input_records": int(stats["input_records"]),
                "accepted_positions": accepted_positions,
                "configured_subset_positions": int(stats["configured_subset_positions"]),
                "probability_gated_positions": int(stats["probability_gated_positions"]),
                "observed_token_bucketized_positions": int(stats["observed_token_bucketized_positions"]),
                "unique_generated_rows": len(stats["unique_generated_rows"]),
                "denominator_response_tokens": denominator_tokens,
                "total_capacity_bits": float(stats["total_capacity_bits"]),
                "eligible_positions_per_100_tokens": (
                    accepted_positions * 100.0 / denominator_tokens if denominator_tokens > 0 else ""
                ),
                "denominator_scope": "topk_candidate_rows_unique_generated_responses",
            }
        )
    rejection_summary_rows = [
        {
            "bucket_count": bucket_count,
            "rejection_reason": reason,
            "count": count,
        }
        for (bucket_count, reason), count in sorted(rejection_counter.items())
    ]
    best_row = max(bucket_count_rows, key=lambda row: float(row["effective_bits_per_response"]))
    manifest = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_DIAGNOSTIC_PENDING_REVIEW",
        "tokenizer_key": tokenizer_key,
        "candidate_jsonl": str(candidate_jsonl_path),
        "bucket_counts": bucket_counts,
        "candidate_top_k": candidate_top_k,
        "min_reference_probability": min_probability,
        "min_arity": int(min_arity),
        "configured_min_members": int(configured_min_members),
        "input_records": input_records,
        "generated_output_count": int(generated_output_count),
        "best_bucket_count": int(best_row["bucket_count"]),
        "best_effective_bits_per_response": float(best_row["effective_bits_per_response"]),
        "best_accepted_positions": int(best_row["accepted_positions"]),
        "best_configured_subset_positions": int(best_row["configured_subset_positions"]),
        "best_probability_gated_positions": int(best_row["probability_gated_positions"]),
        "gate_status": {
            "candidate_supply_upper_bound_effective_bits": (
                "PASS_UPPER_BOUND" if float(best_row["effective_bits_per_response"]) >= 0.8 else "FAIL_UPPER_BOUND"
            ),
            "requires_suffix_or_branch_compatibility_scoring": "YES",
            "training_allowed": "NO",
        },
        "outputs": {
            "candidate_supply_by_bucket_count_csv": str(output_dir / "candidate_supply_by_bucket_count.csv"),
            "candidate_supply_arity_distribution_csv": str(output_dir / "candidate_supply_arity_distribution.csv"),
            "candidate_supply_by_split_csv": str(output_dir / "candidate_supply_by_split.csv"),
            "candidate_supply_rejections_csv": str(output_dir / "candidate_supply_rejections.csv"),
            "candidate_supply_by_position_csv": str(output_dir / "candidate_supply_by_position.csv") if write_position_csv else "",
        },
        "next_allowed_action": (
            "If candidate-supply upper bound passes, prepare Slurm-scored suffix or branch-aware compatibility "
            "for expanded actual-prefix positions; do not train from this diagnostic."
        ),
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "result_claim": "compatibility_aware_supply_upper_bound_not_payload_recovery",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "compatibility_aware_supply_manifest.json", manifest)
    write_csv(output_dir / "candidate_supply_by_bucket_count.csv", bucket_count_rows, list(bucket_count_rows[0].keys()))
    write_csv(output_dir / "candidate_supply_arity_distribution.csv", arity_rows, list(arity_rows[0].keys()))
    write_csv(output_dir / "candidate_supply_by_split.csv", split_rows, list(split_rows[0].keys()) if split_rows else [])
    write_csv(
        output_dir / "candidate_supply_rejections.csv",
        rejection_summary_rows,
        ["bucket_count", "rejection_reason", "count"],
    )
    if write_position_csv:
        write_csv(output_dir / "candidate_supply_by_position.csv", position_rows, list(position_rows[0].keys()) if position_rows else [])
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    manifest = run_supply_audit(
        config_path=resolve_repo_path(args.config, root),
        tokenizer_key=str(args.tokenizer_key),
        candidate_jsonl_path=Path(args.candidate_jsonl),
        output_dir=Path(args.output_dir),
        generated_output_count=int(args.generated_output_count),
        bucket_counts=_bucket_count_values(str(args.bucket_counts)),
        candidate_top_k_override=int(args.candidate_top_k),
        min_arity=int(args.min_arity),
        configured_min_members=int(args.configured_min_members),
        min_bucket_mass=float(args.min_bucket_mass),
        max_bucket_mass_ratio=float(args.max_bucket_mass_ratio),
        min_entropy_fraction=float(args.min_entropy_fraction),
        max_records=int(args.max_records),
        write_position_csv=bool(args.write_position_csv),
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
