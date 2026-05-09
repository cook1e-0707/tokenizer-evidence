from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.common import (
    bucket_mass_metrics,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    write_csv,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run a compatibility-filtered natural_evidence_v1 bank repair. "
            "This reports feasibility only; it does not write trainable bank entries."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--entries", required=True)
    parser.add_argument("--compatibility-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-entries", type=int, default=0)
    parser.add_argument("--bucket-count", type=int, default=0)
    parser.add_argument("--min-compatible-members-per-bucket", type=int, default=0)
    return parser.parse_args(argv)


def _expected_bucket_ids(bucket_count: int) -> list[str]:
    return [str(bucket_id) for bucket_id in range(bucket_count)]


def _compatibility_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_entry: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scored_candidate_count": 0,
            "compatible_candidate_count": 0,
            "scored_by_bucket": Counter(),
            "compatible_by_bucket": Counter(),
            "missing_probability_count": 0,
            "compatible_probability_by_bucket": defaultdict(float),
        }
    )
    for row in rows:
        entry_id = str(row.get("bank_entry_id", ""))
        if not entry_id:
            continue
        bucket_id = str(row.get("bucket_id", ""))
        slot = by_entry[entry_id]
        slot["scored_candidate_count"] += 1
        slot["scored_by_bucket"][bucket_id] += 1
        if bool(row.get("compatibility_pass", False)):
            slot["compatible_candidate_count"] += 1
            slot["compatible_by_bucket"][bucket_id] += 1
            try:
                slot["compatible_probability_by_bucket"][bucket_id] += float(row.get("probability", ""))
            except (TypeError, ValueError):
                slot["missing_probability_count"] += 1
    return by_entry


def _row_for_entry(
    *,
    entry: dict[str, Any],
    compatibility: dict[str, Any] | None,
    expected_buckets: list[str],
    configured_min_members: int,
    min_bucket_mass: float,
    max_bucket_mass_ratio: float,
    min_bucket_entropy_fraction: float,
) -> dict[str, Any]:
    entry_id = str(entry.get("bank_entry_id", ""))
    prompt_id = str(entry.get("prompt_id", ""))
    bucket_count = int(entry.get("bucket_count", len(expected_buckets)))
    compatible_by_bucket = Counter()
    scored_by_bucket = Counter()
    scored_candidate_count = 0
    compatible_candidate_count = 0
    missing_probability_count = 0
    compatible_probability_by_bucket: defaultdict[str, float] = defaultdict(float)
    if compatibility is not None:
        compatible_by_bucket = Counter(compatibility["compatible_by_bucket"])
        scored_by_bucket = Counter(compatibility["scored_by_bucket"])
        scored_candidate_count = int(compatibility["scored_candidate_count"])
        compatible_candidate_count = int(compatibility["compatible_candidate_count"])
        missing_probability_count = int(compatibility["missing_probability_count"])
        compatible_probability_by_bucket = defaultdict(float, compatibility["compatible_probability_by_bucket"])

    compatible_counts = {bucket_id: int(compatible_by_bucket.get(bucket_id, 0)) for bucket_id in expected_buckets}
    scored_counts = {bucket_id: int(scored_by_bucket.get(bucket_id, 0)) for bucket_id in expected_buckets}
    compatible_bucket_count = sum(1 for count in compatible_counts.values() if count > 0)
    min_compatible = min(compatible_counts.values()) if compatible_counts else 0
    would_accept_min1 = compatible_bucket_count == len(expected_buckets)
    would_accept_configured_min = (
        compatible_bucket_count == len(expected_buckets) and min_compatible >= configured_min_members
    )
    probability_available = compatibility is not None and missing_probability_count == 0
    compatible_masses = [float(compatible_probability_by_bucket.get(bucket_id, 0.0)) for bucket_id in expected_buckets]
    mass_metrics = bucket_mass_metrics(compatible_masses) if probability_available else bucket_mass_metrics([])
    would_accept_probability_gates = (
        would_accept_configured_min
        and probability_available
        and mass_metrics["min_bucket_mass"] >= min_bucket_mass
        and mass_metrics["bucket_mass_ratio"] <= max_bucket_mass_ratio
        and mass_metrics["bucket_entropy_fraction"] >= min_bucket_entropy_fraction
    )

    if bucket_count != len(expected_buckets):
        rejection_reason = "bucket_count_mismatch"
    elif compatibility is None:
        rejection_reason = "missing_compatibility_scores"
    elif any(scored_counts[bucket_id] == 0 for bucket_id in expected_buckets):
        rejection_reason = "incomplete_bucket_scores"
    elif not would_accept_min1:
        rejection_reason = "missing_compatible_bucket"
    elif not would_accept_configured_min:
        rejection_reason = "below_configured_min_compatible_members"
    else:
        rejection_reason = ""

    return {
        "bank_entry_id": entry_id,
        "prompt_id": prompt_id,
        "bucket_count": bucket_count,
        "scored_candidate_count": scored_candidate_count,
        "compatible_candidate_count": compatible_candidate_count,
        "compatible_bucket_count": compatible_bucket_count,
        "min_compatible_members_per_bucket": min_compatible,
        "would_accept_min1": would_accept_min1,
        "would_accept_configured_min": would_accept_configured_min,
        "probability_available": probability_available,
        "missing_probability_count": missing_probability_count,
        "compatible_min_bucket_mass": mass_metrics["min_bucket_mass"],
        "compatible_bucket_mass_ratio": mass_metrics["bucket_mass_ratio"],
        "compatible_bucket_entropy_fraction": mass_metrics["bucket_entropy_fraction"],
        "would_accept_probability_gates": would_accept_probability_gates,
        "rejection_reason": rejection_reason,
        "scored_counts_by_bucket_json": json.dumps(scored_counts, sort_keys=True),
        "compatible_counts_by_bucket_json": json.dumps(compatible_counts, sort_keys=True),
        "compatible_probability_by_bucket_json": json.dumps(
            {bucket_id: float(compatible_probability_by_bucket.get(bucket_id, 0.0)) for bucket_id in expected_buckets},
            sort_keys=True,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    bucket_cfg = dict(config.get("bucket_bank", {}))
    quality_gates = dict(bucket_cfg.get("quality_gates", {}))
    capacity_cfg = dict(bucket_cfg.get("compatibility_adjusted_capacity", {}))
    viability_gate = dict(capacity_cfg.get("qwen_e2e_viability_gate", {}))

    target_entries = args.target_entries or int(
        quality_gates.get(
            "raw_opportunity_entries_scaling_placeholder",
            bucket_cfg.get("target_bank_entries_per_tokenizer", 24000),
        )
    )
    bucket_count = args.bucket_count or int(bucket_cfg.get("bucket_count", 8))
    configured_min_members = args.min_compatible_members_per_bucket or int(bucket_cfg.get("min_members_per_bucket", 2))
    min_bucket_mass = float(quality_gates.get("min_bucket_mass", 0.0))
    max_bucket_mass_ratio = float(quality_gates.get("max_bucket_mass_ratio", float("inf")))
    min_bucket_entropy_fraction = float(quality_gates.get("min_bucket_entropy_fraction", 0.0))
    expected_buckets = _expected_bucket_ids(bucket_count)

    entries = read_jsonl(resolve_repo_path(args.entries, root))
    compatibility_rows = read_jsonl(resolve_repo_path(args.compatibility_jsonl, root))
    compatibility_by_entry = _compatibility_index(compatibility_rows)

    by_entry_rows = [
        _row_for_entry(
            entry=entry,
            compatibility=compatibility_by_entry.get(str(entry.get("bank_entry_id", ""))),
            expected_buckets=expected_buckets,
            configured_min_members=configured_min_members,
            min_bucket_mass=min_bucket_mass,
            max_bucket_mass_ratio=max_bucket_mass_ratio,
            min_bucket_entropy_fraction=min_bucket_entropy_fraction,
        )
        for entry in entries
    ]
    rejection_rows = [row for row in by_entry_rows if row["rejection_reason"]]
    accepted_configured = [row for row in by_entry_rows if row["would_accept_configured_min"]]
    accepted_min1 = [row for row in by_entry_rows if row["would_accept_min1"]]
    accepted_probability_gates = [row for row in by_entry_rows if row["would_accept_probability_gates"]]
    missing_probability_rows = sum(int(row["missing_probability_count"]) for row in by_entry_rows)
    entries_with_probability = sum(1 for row in by_entry_rows if row["probability_available"])
    rejection_counts = Counter(str(row["rejection_reason"]) for row in rejection_rows)
    min_member_hist = Counter(str(row["min_compatible_members_per_bucket"]) for row in by_entry_rows)
    compatible_bucket_hist = Counter(str(row["compatible_bucket_count"]) for row in by_entry_rows)
    min1_viability_min = int(viability_gate.get("min1_compatible_entries_min", 0))
    min2_viability_min = int(viability_gate.get("fully_compatible_min2_entries_min", 0))
    min1_bank_side_pass = bool(min1_viability_min) and len(accepted_min1) >= min1_viability_min
    min2_bank_side_pass = bool(min2_viability_min) and len(accepted_configured) >= min2_viability_min
    bank_side_viability_pass = min1_bank_side_pass and min2_bank_side_pass
    if missing_probability_rows > 0:
        blocker = (
            "Compatibility JSONL lacks per-token reference probabilities, so probability-gated capacity "
            "is not auditable yet."
        )
        next_minimal_action = (
            "Join compatibility decisions with original candidate probabilities, then reassess viability gates."
        )
    elif bank_side_viability_pass:
        blocker = (
            "No raw-entry-count blocker: 24,000 is only a raw opportunity scaling placeholder. "
            "Complete held-out density and raw/wrong-key pre-null gates before any protected training."
        )
        next_minimal_action = (
            "Run held-out/organic density and raw/wrong-key pre-null checks, then prepare the controlled Qwen "
            "E2E viability pilot if those gates are not high risk."
        )
    else:
        blocker = (
            "Compatibility-adjusted bank-side viability is still below configured min1/min2 gates."
        )
        next_minimal_action = (
            "Run compatibility-aware construction or threshold/bucket-count sweep before protected training."
        )

    output_dir = resolve_repo_path(args.output_dir, root)
    by_entry_path = output_dir / "compatibility_filtered_bank_dry_run_by_entry.csv"
    rejection_path = output_dir / "compatibility_filtered_bank_dry_run_rejections.csv"
    summary_path = output_dir / "compatibility_filtered_bank_dry_run_summary.json"

    fieldnames = [
        "bank_entry_id",
        "prompt_id",
        "bucket_count",
        "scored_candidate_count",
        "compatible_candidate_count",
        "compatible_bucket_count",
        "min_compatible_members_per_bucket",
        "would_accept_min1",
        "would_accept_configured_min",
        "probability_available",
        "missing_probability_count",
        "compatible_min_bucket_mass",
        "compatible_bucket_mass_ratio",
        "compatible_bucket_entropy_fraction",
        "would_accept_probability_gates",
        "rejection_reason",
        "scored_counts_by_bucket_json",
        "compatible_counts_by_bucket_json",
        "compatible_probability_by_bucket_json",
    ]
    write_csv(by_entry_path, by_entry_rows, fieldnames)
    write_csv(rejection_path, rejection_rows, fieldnames)

    summary = {
        "schema_name": "natural_evidence_compatibility_filtered_bank_dry_run_v1",
        "protocol_id": str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1")),
        "tokenizer_key": args.tokenizer_key,
        "entries_source": str(resolve_repo_path(args.entries, root)),
        "compatibility_source": str(resolve_repo_path(args.compatibility_jsonl, root)),
        "output_dir": str(output_dir),
        "target_entries": target_entries,
        "input_entries": len(entries),
        "compatibility_rows": len(compatibility_rows),
        "scored_entries": len(compatibility_by_entry),
        "missing_score_entries": len(entries) - len(compatibility_by_entry),
        "bucket_count": bucket_count,
        "configured_min_compatible_members_per_bucket": configured_min_members,
        "accepted_entries_at_configured_min": len(accepted_configured),
        "coverage_complete_at_configured_min": len(accepted_configured) >= target_entries,
        "accepted_entries_at_min1": len(accepted_min1),
        "coverage_complete_at_min1": len(accepted_min1) >= target_entries,
        "raw_entry_count_target_is_training_gate": False,
        "entries_with_probability_complete": entries_with_probability,
        "missing_probability_rows": missing_probability_rows,
        "accepted_entries_at_probability_gates": len(accepted_probability_gates),
        "coverage_complete_at_probability_gates": len(accepted_probability_gates) >= target_entries,
        "qwen_e2e_viability_gate": {
            "min1_compatible_entries_min": min1_viability_min,
            "min1_compatible_entries_observed": len(accepted_min1),
            "min1_bank_side_pass": min1_bank_side_pass,
            "fully_compatible_min2_entries_min": min2_viability_min,
            "fully_compatible_min2_entries_observed": len(accepted_configured),
            "min2_bank_side_pass": min2_bank_side_pass,
            "bank_side_viability_pass": bank_side_viability_pass,
            "heldout_density_status": "NEEDS_RESULTS",
            "raw_wrong_key_pre_null_status": "NEEDS_RESULTS",
            "training_start_status": "NEEDS_NULL_AND_DENSITY_GATES",
        },
        "probability_gate_thresholds": {
            "min_bucket_mass": min_bucket_mass,
            "max_bucket_mass_ratio": max_bucket_mass_ratio,
            "min_bucket_entropy_fraction": min_bucket_entropy_fraction,
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "min_compatible_members_histogram": dict(sorted(min_member_hist.items())),
        "compatible_bucket_count_histogram": dict(sorted(compatible_bucket_hist.items())),
        "build_entries_written": False,
        "requires_probability_preserving_rebuild": missing_probability_rows > 0,
        "blocker": blocker,
        "next_minimal_action": next_minimal_action,
        "result_claim": "compatibility_filtered_bank_feasibility_not_training_result",
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
