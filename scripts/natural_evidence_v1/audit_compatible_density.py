from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.common import read_jsonl, read_yaml, resolve_repo_path, write_csv, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit compatibility-adjusted eligible density for a min1-compatible "
            "natural_evidence_v1 bank. This is a CPU diagnostic, not a training run."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--entries", required=True)
    parser.add_argument("--compatibility-by-entry-csv", required=True)
    parser.add_argument("--reference-outputs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--heldout-prompt-families",
        default="",
        help=(
            "Comma-separated prompt_family values to report as a held-out proxy. "
            "Proxy rows are never treated as frozen held-out gate passes."
        ),
    )
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _entry_token_index(entry: dict[str, Any]) -> int:
    for key in ("prefix_response_token_count", "token_index", "token_position", "prefix_token_count"):
        if entry.get(key) not in {None, ""}:
            return int(entry[key])
    prefix_token_ids = entry.get("prefix_token_ids", [])
    if isinstance(prefix_token_ids, list):
        return len(prefix_token_ids)
    return 0


def _response_token_estimates(
    reference_rows: list[dict[str, Any]],
    entries_by_prompt: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    estimates: dict[str, int] = {}
    for row in reference_rows:
        prompt_id = str(row.get("prompt_id", ""))
        if not prompt_id:
            continue
        response = str(row.get("response_text", row.get("output_text", "")))
        estimates[prompt_id] = max(1, len(response.split()))
    for prompt_id, entries in entries_by_prompt.items():
        for entry in entries:
            estimates[prompt_id] = max(estimates.get(prompt_id, 1), _entry_token_index(entry))
    return estimates


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _entropy(values: list[float]) -> float:
    positives = [float(value) for value in values if float(value) > 0.0]
    if not positives:
        return 0.0
    total = sum(positives)
    probabilities = [value / total for value in positives]
    return -sum(probability * math.log2(probability) for probability in probabilities)


def _compatible_rows_by_entry(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("bank_entry_id", "")): row for row in rows if row.get("bank_entry_id")}


def _compatible_entries(
    entries: list[dict[str, Any]],
    by_entry_rows: dict[str, dict[str, str]],
    accept_column: str,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for entry in entries:
        row = by_entry_rows.get(str(entry.get("bank_entry_id", "")))
        if row is not None and _as_bool(row.get(accept_column, "")):
            merged = dict(entry)
            merged["_compatible_density_row"] = row
            accepted.append(merged)
    return accepted


def _split_value(row: dict[str, Any]) -> str:
    for key in ("split", "data_split", "prompt_split"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return "unspecified"


def _density_row(
    *,
    split: str,
    split_role: str,
    reference_rows: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    token_estimates: dict[str, int],
    min_density: float,
    min_effective_bits: float,
    gate_eligible: bool,
) -> dict[str, Any]:
    prompt_ids = {str(row.get("prompt_id", "")) for row in reference_rows if row.get("prompt_id")}
    entries_in_split = [entry for entry in entries if str(entry.get("prompt_id", "")) in prompt_ids]
    entries_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries_in_split:
        entries_by_prompt[str(entry.get("prompt_id", ""))].append(entry)
    prompts_with_entries = sum(1 for prompt_id in prompt_ids if entries_by_prompt.get(prompt_id))
    total_tokens = sum(max(1, int(token_estimates.get(prompt_id, 1))) for prompt_id in prompt_ids)
    eligible_density = len(entries_in_split) / total_tokens * 100.0 if total_tokens else 0.0
    bucket_counts = [int(entry.get("bucket_count", 0)) for entry in entries_in_split if int(entry.get("bucket_count", 0)) > 1]
    compatible_entropy_fractions = [
        _as_float(dict(entry.get("_compatible_density_row", {})).get("compatible_bucket_entropy_fraction", 0.0))
        for entry in entries_in_split
    ]
    mean_bucket_count = _mean([float(value) for value in bucket_counts])
    bits_per_position = math.log2(mean_bucket_count) if mean_bucket_count > 1.0 else 0.0
    mean_entropy_fraction = _mean(compatible_entropy_fractions)
    effective_bits_per_position = bits_per_position * mean_entropy_fraction
    mean_positions_per_response = len(entries_in_split) / len(prompt_ids) if prompt_ids else 0.0
    effective_bits_per_response = mean_positions_per_response * effective_bits_per_position
    if not prompt_ids:
        status = "NEEDS_REFERENCE_OUTPUTS"
    elif not gate_eligible:
        status = "DIAGNOSTIC_PROXY_NOT_GATE"
    elif eligible_density >= min_density and effective_bits_per_response >= min_effective_bits:
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "split": split,
        "split_role": split_role,
        "reference_prompts": len(prompt_ids),
        "prompts_with_compatible_entries": prompts_with_entries,
        "compatible_entries": len(entries_in_split),
        "total_response_token_estimate": total_tokens,
        "eligible_positions_per_100_tokens": eligible_density,
        "mean_compatible_positions_per_response": mean_positions_per_response,
        "mean_bucket_count": mean_bucket_count,
        "mean_compatible_bucket_entropy_fraction": mean_entropy_fraction,
        "effective_compatible_bits_per_position": effective_bits_per_position,
        "effective_compatible_bits_per_response": effective_bits_per_response,
        "min_density_gate": min_density,
        "min_effective_bits_per_response_gate": min_effective_bits,
        "gate_eligible": gate_eligible,
        "status": status,
        "capacity_claim": "compatibility_adjusted_density_not_payload_recovery",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    bucket_cfg = dict(config.get("bucket_bank", {}))
    capacity_cfg = dict(bucket_cfg.get("compatibility_adjusted_capacity", {}))
    viability_gate = dict(capacity_cfg.get("qwen_e2e_viability_gate", {}))
    min_density = float(viability_gate.get("heldout_eligible_positions_per_100_tokens_min", 0.5))
    min_effective_bits = float(viability_gate.get("effective_compatible_bits_per_response_min", 1.0))

    entries = read_jsonl(resolve_repo_path(args.entries, root))
    by_entry_rows = _compatible_rows_by_entry(_read_csv(resolve_repo_path(args.compatibility_by_entry_csv, root)))
    reference_rows = read_jsonl(resolve_repo_path(args.reference_outputs, root))
    min1_entries = _compatible_entries(entries, by_entry_rows, "would_accept_min1")
    min2_entries = _compatible_entries(entries, by_entry_rows, "would_accept_configured_min")
    entries_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in min1_entries:
        entries_by_prompt[str(entry.get("prompt_id", ""))].append(entry)
    token_estimates = _response_token_estimates(reference_rows, entries_by_prompt)

    fieldnames = [
        "split",
        "split_role",
        "reference_prompts",
        "prompts_with_compatible_entries",
        "compatible_entries",
        "total_response_token_estimate",
        "eligible_positions_per_100_tokens",
        "mean_compatible_positions_per_response",
        "mean_bucket_count",
        "mean_compatible_bucket_entropy_fraction",
        "effective_compatible_bits_per_position",
        "effective_compatible_bits_per_response",
        "min_density_gate",
        "min_effective_bits_per_response_gate",
        "gate_eligible",
        "status",
        "capacity_claim",
    ]

    rows: list[dict[str, Any]] = []
    rows.append(
        _density_row(
            split="reference_all",
            split_role="phase_a_reference_diagnostic",
            reference_rows=reference_rows,
            entries=min1_entries,
            token_estimates=token_estimates,
            min_density=min_density,
            min_effective_bits=min_effective_bits,
            gate_eligible=False,
        )
    )

    by_declared_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prompt_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference_rows:
        by_declared_split[_split_value(row)].append(row)
        prompt_family = str(row.get("prompt_family", "")).strip()
        if prompt_family:
            by_prompt_family[prompt_family].append(row)
    for split, split_rows in sorted(by_declared_split.items()):
        role = "declared_split"
        gate_eligible = split.lower() in {"heldout", "held-out", "held_out", "eval", "validation", "organic"}
        rows.append(
            _density_row(
                split=f"declared:{split}",
                split_role=role,
                reference_rows=split_rows,
                entries=min1_entries,
                token_estimates=token_estimates,
                min_density=min_density,
                min_effective_bits=min_effective_bits,
                gate_eligible=gate_eligible,
            )
        )
    for family, family_rows in sorted(by_prompt_family.items()):
        rows.append(
            _density_row(
                split=f"prompt_family:{family}",
                split_role="prompt_family_diagnostic",
                reference_rows=family_rows,
                entries=min1_entries,
                token_estimates=token_estimates,
                min_density=min_density,
                min_effective_bits=min_effective_bits,
                gate_eligible=False,
            )
        )

    heldout_families = [value.strip() for value in args.heldout_prompt_families.split(",") if value.strip()]
    if heldout_families:
        heldout_rows = [row for row in reference_rows if str(row.get("prompt_family", "")).strip() in heldout_families]
        rows.append(
            _density_row(
                split="heldout_prompt_family_proxy",
                split_role="heldout_proxy_not_frozen_gate",
                reference_rows=heldout_rows,
                entries=min1_entries,
                token_estimates=token_estimates,
                min_density=min_density,
                min_effective_bits=min_effective_bits,
                gate_eligible=False,
            )
        )

    gate_rows = [row for row in rows if bool(row["gate_eligible"])]
    heldout_gate_rows = [
        row
        for row in gate_rows
        if str(row["split"]).lower() in {"declared:heldout", "declared:held-out", "declared:held_out", "declared:eval", "declared:validation"}
    ]
    organic_gate_rows = [row for row in gate_rows if str(row["split"]).lower() == "declared:organic"]
    if heldout_gate_rows:
        heldout_status = "PASS" if all(row["status"] == "PASS" for row in heldout_gate_rows) else "FAIL"
    else:
        heldout_status = "NEEDS_FROZEN_HELDOUT_REFERENCE_OUTPUTS"
    if organic_gate_rows:
        organic_status = "PASS" if all(row["status"] == "PASS" for row in organic_gate_rows) else "FAIL"
    else:
        organic_status = "NEEDS_ORGANIC_REFERENCE_OUTPUTS"
    density_gate_status = "PASS" if heldout_status == "PASS" and organic_status == "PASS" else "NEEDS_RESULTS"

    output_dir = resolve_repo_path(args.output_dir, root)
    density_csv = output_dir / "compatible_density_by_split.csv"
    summary_json = output_dir / "compatible_density_summary.json"
    write_csv(density_csv, rows, fieldnames)
    summary = {
        "schema_name": "natural_evidence_compatible_density_summary_v1",
        "protocol_id": str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1")),
        "tokenizer_key": args.tokenizer_key,
        "entries_source": str(resolve_repo_path(args.entries, root)),
        "compatibility_by_entry_csv": str(resolve_repo_path(args.compatibility_by_entry_csv, root)),
        "reference_outputs": str(resolve_repo_path(args.reference_outputs, root)),
        "output_dir": str(output_dir),
        "min1_compatible_entries": len(min1_entries),
        "min2_compatible_entries": len(min2_entries),
        "density_rows": len(rows),
        "heldout_density_status": heldout_status,
        "organic_density_status": organic_status,
        "density_gate_status": density_gate_status,
        "qwen_e2e_viability_density_gate": density_gate_status,
        "min_density_gate": min_density,
        "min_effective_bits_per_response_gate": min_effective_bits,
        "heldout_prompt_family_proxy_values": heldout_families,
        "proxy_rows_are_gate_eligible": False,
        "outputs": {
            "compatible_density_by_split": str(density_csv),
            "compatible_density_summary": str(summary_json),
        },
        "next_minimal_action": (
            "Provide frozen held-out and organic reference/candidate artifacts, or run the raw/wrong-key "
            "pre-null check if its allowlisted command is ready. Do not start protected training from proxy density."
        ),
        "result_claim": "compatible_density_diagnostic_not_payload_recovery",
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
