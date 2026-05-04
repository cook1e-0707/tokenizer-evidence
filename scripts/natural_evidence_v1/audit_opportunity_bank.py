from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
import math
import statistics
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
    write_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a natural_evidence_v1 opportunity bank. This reports opportunity "
            "coverage and quality; it does not claim fingerprint insertion."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--entries", required=True)
    parser.add_argument("--reference-outputs", default="")
    parser.add_argument("--candidate-jsonl", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="results/natural_evidence_v1/tables")
    return parser.parse_args(argv)


def _reference_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {str(row.get("prompt_id", "")): row for row in read_jsonl(path) if row.get("prompt_id")}


def _response_token_estimates(entries: list[dict[str, Any]], reference_rows: dict[str, dict[str, Any]]) -> dict[str, int]:
    estimates: dict[str, int] = {}
    for prompt_id, row in reference_rows.items():
        response = str(row.get("response_text", row.get("output_text", "")))
        estimates[prompt_id] = max(1, len(response.split()))
    for entry in entries:
        prompt_id = str(entry.get("prompt_id", ""))
        if not prompt_id:
            continue
        offset = entry.get("prefix_response_token_count", "")
        if offset == "":
            continue
        estimates[prompt_id] = max(estimates.get(prompt_id, 1), int(offset))
    return estimates


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _balance_rows(entries: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        reference_masses = [float(value) for value in dict(entry.get("reference_mass", {})).values()]
        metrics = bucket_mass_metrics(reference_masses)
        rows.append(
            {
                "split": split,
                "protocol_id": entry.get("protocol_id", ""),
                "bucket_bank_id": entry.get("bank_id", ""),
                "bank_entry_id": entry.get("bank_entry_id", ""),
                "entry_role": entry.get("entry_role", "context_conditioned_measurable_opportunity"),
                "prompt_id": entry.get("prompt_id", ""),
                "bucket_count": entry.get("bucket_count", ""),
                "candidate_token_count": entry.get("candidate_token_count", ""),
                "min_bucket_mass": metrics["min_bucket_mass"],
                "max_bucket_mass": metrics["max_bucket_mass"],
                "bucket_mass_ratio": metrics["bucket_mass_ratio"],
                "bucket_entropy": metrics["bucket_entropy"],
                "bucket_entropy_fraction": metrics["bucket_entropy_fraction"],
                "token_class_counts_json": json.dumps(entry.get("token_class_counts", {}), sort_keys=True),
                "counterfactual_compatibility_status": entry.get(
                    "counterfactual_compatibility_status",
                    "NEEDS_COUNTERFACTUAL_COMPATIBILITY",
                ),
                "fingerprint_claim": False,
            }
        )
    return rows


def _coverage_by_split_rows(entries: list[dict[str, Any]], reference_rows: dict[str, dict[str, Any]], split: str) -> list[dict[str, Any]]:
    prompt_ids_with_entries = {str(entry.get("prompt_id", "")) for entry in entries if entry.get("prompt_id")}
    total_prompts = len(reference_rows) if reference_rows else len(prompt_ids_with_entries)
    return [
        {
            "split": split,
            "protocol_id": entries[0].get("protocol_id", "natural_evidence_v1") if entries else "natural_evidence_v1",
            "bucket_bank_id": entries[0].get("bank_id", "") if entries else "",
            "bank_role": "natural_bucket_opportunity_catalog",
            "accepted_entries": len(entries),
            "reference_prompts": total_prompts,
            "prompts_with_entries": len(prompt_ids_with_entries),
            "prompt_coverage_rate": len(prompt_ids_with_entries) / total_prompts if total_prompts else 0.0,
            "result_claim": "opportunity_coverage_not_fingerprint_count",
        }
    ]


def _capacity_rows(entries: list[dict[str, Any]], reference_rows: dict[str, dict[str, Any]], split: str) -> list[dict[str, Any]]:
    entries_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        entries_by_prompt[str(entry.get("prompt_id", ""))].append(entry)
    prompt_ids = set(reference_rows) if reference_rows else set(entries_by_prompt)
    token_estimates = _response_token_estimates(entries, reference_rows)
    per_prompt_entries = [len(entries_by_prompt.get(prompt_id, [])) for prompt_id in prompt_ids]
    total_tokens = sum(max(1, token_estimates.get(prompt_id, 1)) for prompt_id in prompt_ids)
    bucket_counts = [int(entry.get("bucket_count", 0)) for entry in entries if int(entry.get("bucket_count", 0)) > 1]
    entropy_fractions = [
        bucket_mass_metrics([float(value) for value in dict(entry.get("reference_mass", {})).values()])[
            "bucket_entropy_fraction"
        ]
        for entry in entries
    ]
    mean_bucket_count = _mean([float(value) for value in bucket_counts])
    theoretical_bits = math.log2(mean_bucket_count) if mean_bucket_count > 1.0 else 0.0
    mean_entropy_fraction = _mean(entropy_fractions)
    effective_bits_per_position = theoretical_bits * mean_entropy_fraction
    return [
        {
            "split": split,
            "protocol_id": entries[0].get("protocol_id", "natural_evidence_v1") if entries else "natural_evidence_v1",
            "bucket_bank_id": entries[0].get("bank_id", "") if entries else "",
            "accepted_entries": len(entries),
            "reference_prompts": len(prompt_ids),
            "mean_eligible_positions_per_response": _mean([float(value) for value in per_prompt_entries]),
            "eligible_positions_per_100_tokens": (len(entries) / total_tokens * 100.0) if total_tokens else 0.0,
            "mean_bits_per_position": theoretical_bits,
            "mean_bucket_entropy_fraction": mean_entropy_fraction,
            "effective_bits_per_position": effective_bits_per_position,
            "effective_bits_per_response": _mean([float(value) for value in per_prompt_entries]) * effective_bits_per_position,
            "capacity_claim": "static_opportunity_capacity_not_payload_recovery",
        }
    ]


def _reconstructability_rows(
    *,
    entries: list[dict[str, Any]],
    candidate_count: int,
    split: str,
) -> list[dict[str, Any]]:
    static_rate = len(entries) / candidate_count if candidate_count else 0.0
    return [
        {
            "split": split,
            "protocol_id": entries[0].get("protocol_id", "natural_evidence_v1") if entries else "natural_evidence_v1",
            "bucket_bank_id": entries[0].get("bank_id", "") if entries else "",
            "reconstructability_mode": "static_candidate_policy",
            "observed_transcript_condition": "NOT_RUN",
            "candidate_records": candidate_count,
            "accepted_entries": len(entries),
            "reconstructability_rate": static_rate,
            "protected_transcript_reconstructability_status": "NEEDS_RESULTS",
            "raw_transcript_reconstructability_status": "NEEDS_RESULTS",
        }
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    quality_gates = dict(dict(config.get("bucket_bank", {})).get("quality_gates", {}))
    entries = read_jsonl(resolve_repo_path(args.entries, root))
    reference_rows = _reference_rows(resolve_repo_path(args.reference_outputs, root) if args.reference_outputs else None)
    candidate_count = len(read_jsonl(resolve_repo_path(args.candidate_jsonl, root))) if args.candidate_jsonl else 0
    output_dir = resolve_repo_path(args.output_dir, root)

    balance_rows = _balance_rows(entries, args.split)
    coverage_rows = _coverage_by_split_rows(entries, reference_rows, args.split)
    capacity_rows = _capacity_rows(entries, reference_rows, args.split)
    reconstructability_rows = _reconstructability_rows(
        entries=entries,
        candidate_count=candidate_count,
        split=args.split,
    )
    write_csv(
        output_dir / "bucket_bank_balance.csv",
        balance_rows,
        [
            "split",
            "protocol_id",
            "bucket_bank_id",
            "bank_entry_id",
            "entry_role",
            "prompt_id",
            "bucket_count",
            "candidate_token_count",
            "min_bucket_mass",
            "max_bucket_mass",
            "bucket_mass_ratio",
            "bucket_entropy",
            "bucket_entropy_fraction",
            "token_class_counts_json",
            "counterfactual_compatibility_status",
            "fingerprint_claim",
        ],
    )
    write_csv(
        output_dir / "bucket_bank_coverage_by_split.csv",
        coverage_rows,
        [
            "split",
            "protocol_id",
            "bucket_bank_id",
            "bank_role",
            "accepted_entries",
            "reference_prompts",
            "prompts_with_entries",
            "prompt_coverage_rate",
            "result_claim",
        ],
    )
    write_csv(
        output_dir / "natural_channel_capacity.csv",
        capacity_rows,
        [
            "split",
            "protocol_id",
            "bucket_bank_id",
            "accepted_entries",
            "reference_prompts",
            "mean_eligible_positions_per_response",
            "eligible_positions_per_100_tokens",
            "mean_bits_per_position",
            "mean_bucket_entropy_fraction",
            "effective_bits_per_position",
            "effective_bits_per_response",
            "capacity_claim",
        ],
    )
    write_csv(
        output_dir / "reconstructability_report.csv",
        reconstructability_rows,
        [
            "split",
            "protocol_id",
            "bucket_bank_id",
            "reconstructability_mode",
            "observed_transcript_condition",
            "candidate_records",
            "accepted_entries",
            "reconstructability_rate",
            "protected_transcript_reconstructability_status",
            "raw_transcript_reconstructability_status",
        ],
    )

    min_masses = [float(row["min_bucket_mass"]) for row in balance_rows]
    mass_ratios = [float(row["bucket_mass_ratio"]) for row in balance_rows]
    entropy_fractions = [float(row["bucket_entropy_fraction"]) for row in balance_rows]
    summary = {
        "schema_name": "natural_opportunity_bank_audit_summary_v1",
        "split": args.split,
        "entries": len(entries),
        "fingerprint_claim": False,
        "quality_gate_status": {
            "accepted_entries": len(entries) >= int(quality_gates.get("accepted_entries_per_tokenizer", 24576)),
            "min_bucket_mass": bool(min_masses)
            and min(min_masses) >= float(quality_gates.get("min_bucket_mass", 0.0)),
            "max_bucket_mass_ratio": bool(mass_ratios)
            and max(mass_ratios) <= float(quality_gates.get("max_bucket_mass_ratio", float("inf"))),
            "min_bucket_entropy_fraction": bool(entropy_fractions)
            and min(entropy_fractions) >= float(quality_gates.get("min_bucket_entropy_fraction", 0.0)),
            "counterfactual_compatibility": "NEEDS_RESULTS",
            "on_policy_reconstructability": "NEEDS_RESULTS",
            "raw_wrong_key_null": "NEEDS_RESULTS",
            "bucket_count_ablation": "NEEDS_RESULTS",
        },
        "outputs": {
            "bucket_bank_balance": str(output_dir / "bucket_bank_balance.csv"),
            "bucket_bank_coverage_by_split": str(output_dir / "bucket_bank_coverage_by_split.csv"),
            "natural_channel_capacity": str(output_dir / "natural_channel_capacity.csv"),
            "reconstructability_report": str(output_dir / "reconstructability_report.csv"),
        },
    }
    write_json(output_dir / "opportunity_bank_audit_summary.json", summary)
    write_jsonl(output_dir / "opportunity_bank_audit_summary.jsonl", [summary])
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
