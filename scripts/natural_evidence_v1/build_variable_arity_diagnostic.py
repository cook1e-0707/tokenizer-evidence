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
from typing import Any, Mapping

from scripts.natural_evidence_v1.common import bucket_mass_metrics, read_jsonl, write_csv, write_json, write_jsonl


ENTRY_SCHEMA_NAME = "natural_evidence_variable_arity_diagnostic_entry_v1"
MANIFEST_SCHEMA_NAME = "natural_evidence_variable_arity_diagnostic_manifest_v1"
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
            "Build a CPU-local variable-arity diagnostic from actual-prefix "
            "suffix compatibility by-entry rows. This emits capacity/audit "
            "artifacts only: no training, no E2E, no FAR aggregation."
        )
    )
    parser.add_argument("--compatibility-by-entry-csv", required=True)
    parser.add_argument("--bucketized-candidates-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generated-output-count", type=int, required=True)
    parser.add_argument("--bucket-count", type=int, default=4)
    parser.add_argument("--min-arity", type=int, default=2)
    parser.add_argument("--configured-min-members", type=int, default=2)
    parser.add_argument("--min-bucket-mass", type=float, default=0.005)
    parser.add_argument("--max-bucket-mass-ratio", type=float, default=5.0)
    parser.add_argument("--min-entropy-fraction", type=float, default=0.90)
    return parser.parse_args(argv)


def _entry_key(row: Mapping[str, Any]) -> str:
    if all(field in row for field in ENTRY_KEY_FIELDS):
        return "||".join(str(row.get(field, "")) for field in ENTRY_KEY_FIELDS)
    return str(row.get("bank_entry_id", ""))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _json_counts(row: Mapping[str, str], key: str) -> dict[str, int]:
    try:
        payload = json.loads(str(row.get(key, "{}")))
    except json.JSONDecodeError:
        return {}
    return {str(bucket_id): int(count) for bucket_id, count in payload.items()}


def _json_floats(row: Mapping[str, str], key: str) -> dict[str, float]:
    try:
        payload = json.loads(str(row.get(key, "{}")))
    except json.JSONDecodeError:
        return {}
    return {str(bucket_id): float(value) for bucket_id, value in payload.items()}


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_bucketized_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = _entry_key(row)
        if key in rows:
            raise ValueError(f"Duplicate actual-prefix bucketized row key in {path}: {key!r}")
        rows[key] = row
    return rows


def build_variable_arity_diagnostic(
    *,
    compatibility_by_entry_csv: Path,
    bucketized_candidates_jsonl: Path,
    output_dir: Path,
    generated_output_count: int,
    bucket_count: int = 4,
    min_arity: int = 2,
    configured_min_members: int = 2,
    min_bucket_mass: float = 0.005,
    max_bucket_mass_ratio: float = 5.0,
    min_entropy_fraction: float = 0.90,
) -> dict[str, Any]:
    output_paths = [
        output_dir / "variable_arity_bank_entries.jsonl",
        output_dir / "variable_arity_manifest.json",
        output_dir / "arity_distribution.csv",
        output_dir / "effective_bits_per_response.csv",
        output_dir / "eligible_density_by_split.csv",
        output_dir / "variable_arity_rejections.csv",
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite variable-arity outputs: " + ", ".join(existing))

    by_entry_rows = _read_csv_rows(compatibility_by_entry_csv)
    bucketized_by_key = _load_bucketized_rows(bucketized_candidates_jsonl)

    accepted_entries: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    arity_counter: Counter[int] = Counter()
    arity_bits: Counter[int] = Counter()
    split_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "input_entries": 0,
            "accepted_entries": 0,
            "configured_subset_entries": 0,
            "probability_gated_entries": 0,
            "total_capacity_bits": 0.0,
            "unique_generated_rows": set(),
            "denominator_response_tokens": 0,
        }
    )
    generated_response_tokens_by_split: dict[str, dict[str, int]] = defaultdict(dict)
    duplicate_keys: Counter[str] = Counter()
    seen_keys: set[str] = set()
    joined_rows = 0
    configured_subset_entries = 0
    probability_gated_entries = 0
    total_capacity_bits = 0.0
    observed_bucket_compatible_entries = 0

    for row in by_entry_rows:
        entry_key = _entry_key(row)
        if entry_key in seen_keys:
            duplicate_keys[entry_key] += 1
            continue
        seen_keys.add(entry_key)
        bucketized = bucketized_by_key.get(entry_key)
        split = str(row.get("prompt_split", ""))
        split_stats[split]["input_entries"] += 1
        if bucketized is None:
            rejection_rows.append(
                {
                    "entry_key": entry_key,
                    "bank_entry_id": row.get("bank_entry_id", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "prompt_split": split,
                    "model_condition": row.get("model_condition", ""),
                    "generated_row_index": row.get("generated_row_index", ""),
                    "position_index": row.get("position_index", ""),
                    "arity": 0,
                    "rejection_reason": "missing_bucketized_candidate_row",
                }
            )
            arity_counter[0] += 1
            continue
        joined_rows += 1
        generated_row_index = str(row.get("generated_row_index", ""))
        if generated_row_index:
            generated_response_tokens_by_split[split][generated_row_index] = int(
                bucketized.get("response_token_count", 0)
            )
        compatible_counts = _json_counts(row, "compatible_counts_by_bucket_json")
        compatible_probabilities = _json_floats(row, "compatible_probability_by_bucket_json")
        compatible_bucket_ids = [
            str(bucket_id)
            for bucket_id in range(int(bucket_count))
            if compatible_counts.get(str(bucket_id), 0) > 0
        ]
        configured_bucket_ids = [
            str(bucket_id)
            for bucket_id in range(int(bucket_count))
            if compatible_counts.get(str(bucket_id), 0) >= int(configured_min_members)
        ]
        arity = len(compatible_bucket_ids)
        arity_counter[arity] += 1
        if arity < int(min_arity):
            rejection_rows.append(
                {
                    "entry_key": entry_key,
                    "bank_entry_id": row.get("bank_entry_id", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "prompt_split": split,
                    "model_condition": row.get("model_condition", ""),
                    "generated_row_index": row.get("generated_row_index", ""),
                    "position_index": row.get("position_index", ""),
                    "arity": arity,
                    "rejection_reason": "below_min_arity",
                }
            )
            continue
        capacity_bits = math.log2(arity)
        arity_bits[arity] += capacity_bits
        total_capacity_bits += capacity_bits
        masses = [compatible_probabilities.get(bucket_id, 0.0) for bucket_id in compatible_bucket_ids]
        metrics = bucket_mass_metrics(masses)
        variable_probability_gated = (
            metrics["min_bucket_mass"] >= float(min_bucket_mass)
            and metrics["bucket_mass_ratio"] <= float(max_bucket_mass_ratio)
            and metrics["bucket_entropy_fraction"] >= float(min_entropy_fraction)
        )
        configured_subset = len(configured_bucket_ids) >= int(min_arity)
        if configured_subset:
            configured_subset_entries += 1
        if variable_probability_gated:
            probability_gated_entries += 1
        observed_bucket_id = str(bucketized.get("observed_token_bucket_id", ""))
        observed_bucket_is_compatible = observed_bucket_id in compatible_bucket_ids
        if observed_bucket_is_compatible:
            observed_bucket_compatible_entries += 1
        split_stats[split]["accepted_entries"] += 1
        split_stats[split]["configured_subset_entries"] += int(configured_subset)
        split_stats[split]["probability_gated_entries"] += int(variable_probability_gated)
        split_stats[split]["total_capacity_bits"] += capacity_bits
        if generated_row_index:
            split_stats[split]["unique_generated_rows"].add(generated_row_index)
        accepted_entries.append(
            {
                "schema_name": ENTRY_SCHEMA_NAME,
                "entry_key": entry_key,
                "bank_entry_id": row.get("bank_entry_id", ""),
                "prompt_id": row.get("prompt_id", ""),
                "prompt_split": split,
                "model_condition": row.get("model_condition", ""),
                "payload_id": row.get("payload_id", ""),
                "seed": row.get("seed", ""),
                "query_index": _int_or_zero(row.get("query_index")),
                "generated_row_index": _int_or_zero(row.get("generated_row_index")),
                "position_index": _int_or_zero(row.get("position_index")),
                "bucket_count": int(bucket_count),
                "arity": arity,
                "compatible_bucket_ids": compatible_bucket_ids,
                "configured_bucket_ids": configured_bucket_ids,
                "compatible_counts_by_bucket": compatible_counts,
                "compatible_probability_by_bucket": compatible_probabilities,
                "capacity_bits": capacity_bits,
                "configured_subset": configured_subset,
                "variable_probability_gated": variable_probability_gated,
                "compatible_min_bucket_mass": metrics["min_bucket_mass"],
                "compatible_bucket_mass_ratio": metrics["bucket_mass_ratio"],
                "compatible_bucket_entropy_fraction": metrics["bucket_entropy_fraction"],
                "observed_token_bucket_id": observed_bucket_id,
                "observed_bucket_is_compatible": observed_bucket_is_compatible,
                "observed_token_bucketized": bool(bucketized.get("observed_token_bucketized", False)),
                "observed_token_id": bucketized.get("observed_token_id", ""),
                "result_claim": "variable_arity_diagnostic_not_payload_recovery",
                "paper_claim_allowed": False,
                "training_started": False,
                "e2e_eval_started": False,
            }
        )

    for split, rows_by_generated in generated_response_tokens_by_split.items():
        split_stats[split]["denominator_response_tokens"] = sum(rows_by_generated.values())

    all_stats = {
        "input_entries": sum(stats["input_entries"] for stats in split_stats.values()),
        "accepted_entries": sum(stats["accepted_entries"] for stats in split_stats.values()),
        "configured_subset_entries": sum(stats["configured_subset_entries"] for stats in split_stats.values()),
        "probability_gated_entries": sum(stats["probability_gated_entries"] for stats in split_stats.values()),
        "total_capacity_bits": sum(float(stats["total_capacity_bits"]) for stats in split_stats.values()),
        "unique_generated_rows": set().union(*(stats["unique_generated_rows"] for stats in split_stats.values())),
        "denominator_response_tokens": sum(int(stats["denominator_response_tokens"]) for stats in split_stats.values()),
    }
    split_stats["all"] = all_stats

    arity_rows = []
    for arity in range(int(bucket_count) + 1):
        arity_rows.append(
            {
                "arity": arity,
                "entries": arity_counter.get(arity, 0),
                "accepted_entries": arity_counter.get(arity, 0) if arity >= int(min_arity) else 0,
                "capacity_bits_per_entry": math.log2(arity) if arity >= int(min_arity) else 0.0,
                "total_capacity_bits": arity_bits.get(arity, 0.0),
            }
        )
    density_rows = []
    for split, stats in sorted(split_stats.items()):
        denominator_tokens = int(stats["denominator_response_tokens"])
        accepted = int(stats["accepted_entries"])
        density_rows.append(
            {
                "prompt_split": split,
                "input_entries": int(stats["input_entries"]),
                "accepted_entries": accepted,
                "configured_subset_entries": int(stats["configured_subset_entries"]),
                "probability_gated_entries": int(stats["probability_gated_entries"]),
                "total_capacity_bits": float(stats["total_capacity_bits"]),
                "unique_generated_rows": len(stats["unique_generated_rows"]),
                "denominator_response_tokens": denominator_tokens,
                "eligible_positions_per_100_tokens": (
                    accepted * 100.0 / denominator_tokens if denominator_tokens > 0 else ""
                ),
                "denominator_scope": "bucketized_unique_generated_rows_only",
            }
        )
    effective_rows = [
        {
            "scope": "all_generated_outputs",
            "generated_output_count": int(generated_output_count),
            "accepted_entries": len(accepted_entries),
            "configured_subset_entries": configured_subset_entries,
            "probability_gated_entries": probability_gated_entries,
            "total_capacity_bits": total_capacity_bits,
            "effective_bits_per_response": (
                total_capacity_bits / int(generated_output_count) if int(generated_output_count) > 0 else ""
            ),
            "gate_threshold_effective_bits_per_response": 0.8,
            "gate_status": (
                "PASS_DIAGNOSTIC"
                if int(generated_output_count) > 0 and total_capacity_bits / int(generated_output_count) >= 0.8
                else "FAIL_LOW_CAPACITY"
            ),
        }
    ]
    manifest = {
        "schema_name": MANIFEST_SCHEMA_NAME,
        "status": "COMPLETE_DIAGNOSTIC_PENDING_REVIEW",
        "compatibility_by_entry_csv": str(compatibility_by_entry_csv),
        "bucketized_candidates_jsonl": str(bucketized_candidates_jsonl),
        "entry_key_fields": list(ENTRY_KEY_FIELDS),
        "bucket_count": int(bucket_count),
        "min_arity": int(min_arity),
        "configured_min_members": int(configured_min_members),
        "input_rows": len(by_entry_rows),
        "unique_entry_keys": len(seen_keys),
        "duplicate_entry_keys_skipped": sum(duplicate_keys.values()),
        "joined_rows": joined_rows,
        "accepted_entries": len(accepted_entries),
        "configured_subset_entries": configured_subset_entries,
        "probability_gated_entries": probability_gated_entries,
        "observed_bucket_compatible_entries": observed_bucket_compatible_entries,
        "total_capacity_bits": total_capacity_bits,
        "generated_output_count": int(generated_output_count),
        "effective_bits_per_response": (
            total_capacity_bits / int(generated_output_count) if int(generated_output_count) > 0 else None
        ),
        "arity_distribution": {str(row["arity"]): row["entries"] for row in arity_rows},
        "gate_thresholds": {
            "variable_arity_compatible_entries_min": 2000,
            "effective_bits_per_response_min": 0.8,
            "high_quality_configured_subset_min": 500,
        },
        "gate_status": {
            "variable_arity_compatible_entries": (
                "PASS" if len(accepted_entries) >= 2000 else "FAIL"
            ),
            "effective_bits_per_response": (
                "PASS"
                if int(generated_output_count) > 0 and total_capacity_bits / int(generated_output_count) >= 0.8
                else "FAIL"
            ),
            "high_quality_configured_subset": (
                "PASS" if configured_subset_entries >= 500 else "FAIL"
            ),
            "heldout_density": "NEEDS_FULL_DENOMINATOR_AUDIT",
        },
        "density_denominator_scope": "bucketized_unique_generated_rows_only",
        "next_allowed_action": (
            "Review variable-arity diagnostic. Do not train unless variable-arity "
            "capacity, held-out density, null plans, and protocol commitment gates pass."
        ),
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "result_claim": "variable_arity_diagnostic_not_payload_recovery",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "variable_arity_bank_entries.jsonl", accepted_entries)
    write_json(output_dir / "variable_arity_manifest.json", manifest)
    write_csv(output_dir / "arity_distribution.csv", arity_rows, list(arity_rows[0].keys()))
    write_csv(output_dir / "effective_bits_per_response.csv", effective_rows, list(effective_rows[0].keys()))
    write_csv(output_dir / "eligible_density_by_split.csv", density_rows, list(density_rows[0].keys()))
    write_csv(
        output_dir / "variable_arity_rejections.csv",
        rejection_rows,
        [
            "entry_key",
            "bank_entry_id",
            "prompt_id",
            "prompt_split",
            "model_condition",
            "generated_row_index",
            "position_index",
            "arity",
            "rejection_reason",
        ],
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_variable_arity_diagnostic(
        compatibility_by_entry_csv=Path(args.compatibility_by_entry_csv),
        bucketized_candidates_jsonl=Path(args.bucketized_candidates_jsonl),
        output_dir=Path(args.output_dir),
        generated_output_count=int(args.generated_output_count),
        bucket_count=int(args.bucket_count),
        min_arity=int(args.min_arity),
        configured_min_members=int(args.configured_min_members),
        min_bucket_mass=float(args.min_bucket_mass),
        max_bucket_mass_ratio=float(args.max_bucket_mass_ratio),
        min_entropy_fraction=float(args.min_entropy_fraction),
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
