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
from typing import Any, Callable, Iterable, Mapping

from scripts.natural_evidence_v1.common import bucket_mass_metrics, read_yaml, resolve_repo_path, write_csv, write_json


SCHEMA_NAME = "natural_evidence_variable_arity_full_density_audit_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit expanded variable-arity opportunity density over the full "
            "generated-output denominator. This is a diagnostic only: no "
            "training, no E2E, no FAR aggregation."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--generated-outputs", required=True)
    parser.add_argument("--compatibility-by-entry-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket-count", type=int, default=4)
    parser.add_argument("--min-arity", type=int, default=2)
    parser.add_argument("--configured-min-members", type=int, default=2)
    parser.add_argument("--min-bucket-mass", type=float, default=0.005)
    parser.add_argument("--max-bucket-mass-ratio", type=float, default=5.0)
    parser.add_argument("--min-entropy-fraction", type=float, default=0.90)
    parser.add_argument(
        "--token-count-mode",
        choices=("tokenizer", "whitespace"),
        default="tokenizer",
        help="Use the model tokenizer for exact response-token denominators, or whitespace for tests.",
    )
    parser.add_argument("--tokenizer-name", default="")
    return parser.parse_args(argv)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            yield payload


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _json_counts(row: Mapping[str, str], key: str) -> dict[str, int]:
    try:
        payload = json.loads(str(row.get(key, "{}")))
    except json.JSONDecodeError:
        return {}
    return {str(bucket_id): int(value) for bucket_id, value in payload.items()}


def _json_floats(row: Mapping[str, str], key: str) -> dict[str, float]:
    try:
        payload = json.loads(str(row.get(key, "{}")))
    except json.JSONDecodeError:
        return {}
    return {str(bucket_id): float(value) for bucket_id, value in payload.items()}


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _model_config(config: Mapping[str, Any], tokenizer_key: str) -> dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, dict) or tokenizer_key not in models:
        raise ValueError(f"Missing model config for tokenizer key {tokenizer_key!r}")
    model_cfg = models[tokenizer_key]
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Model config for {tokenizer_key!r} must be a mapping")
    return dict(model_cfg)


def _token_counter(*, mode: str, tokenizer_name: str) -> Callable[[str], int]:
    if mode == "whitespace":
        return lambda text: max(1, len(str(text).split()))
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "token-count-mode=tokenizer requires transformers; use whitespace only for tests"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=True)

    def count_tokens(text: str) -> int:
        token_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
        return max(1, len(token_ids))

    return count_tokens


def _generated_row_key(row: Mapping[str, Any], row_index: int) -> str:
    return str(row.get("generated_row_index", row_index))


def _condition_key(row: Mapping[str, Any]) -> str:
    return "||".join(
        [
            str(row.get("model_condition", "")),
            str(row.get("payload_id", "")),
            str(row.get("seed", "")),
        ]
    )


def _new_generated_stats(row: Mapping[str, Any], row_index: int, token_count: int) -> dict[str, Any]:
    return {
        "generated_row_index": str(row_index),
        "prompt_id": str(row.get("prompt_id", "")),
        "prompt_split": str(row.get("prompt_split", "")),
        "model_condition": str(row.get("model_condition", "")),
        "payload_id": str(row.get("payload_id", "")),
        "seed": str(row.get("seed", "")),
        "query_index": str(row.get("query_index", "")),
        "response_token_count": int(token_count),
        "input_entry_count": 0,
        "accepted_entry_count": 0,
        "configured_subset_count": 0,
        "probability_gated_count": 0,
        "observed_bucket_compatible_count": 0,
        "capacity_bits": 0.0,
        "max_arity": 0,
    }


def _slice_key(kind: str, label: str) -> str:
    return f"{kind}:{label}"


def _record_slices(row: Mapping[str, Any]) -> list[tuple[str, str, bool]]:
    prompt_split = str(row.get("prompt_split", "") or "unspecified")
    model_condition = str(row.get("model_condition", "") or "unspecified")
    payload_id = str(row.get("payload_id", "") or "none")
    seed = str(row.get("seed", "") or "none")
    return [
        ("all", "all_generated_outputs", True),
        ("prompt_split", prompt_split, prompt_split.lower() in {"heldout", "held-out", "held_out", "organic"}),
        ("model_condition", model_condition, model_condition == "raw"),
        (
            "condition_payload_seed",
            f"{model_condition}|payload={payload_id}|seed={seed}",
            model_condition == "raw",
        ),
    ]


def _empty_slice(kind: str, label: str, gate_eligible: bool) -> dict[str, Any]:
    return {
        "slice_kind": kind,
        "slice": label,
        "gate_eligible": bool(gate_eligible),
        "generated_outputs": 0,
        "response_tokens": 0,
        "input_entries": 0,
        "accepted_entries": 0,
        "configured_subset_entries": 0,
        "probability_gated_entries": 0,
        "observed_bucket_compatible_entries": 0,
        "total_capacity_bits": 0.0,
        "rows_with_accepted_entries": 0,
        "rows_with_no_accepted_entries": 0,
    }


def audit_full_density(
    *,
    config_path: Path,
    tokenizer_key: str,
    generated_outputs_path: Path,
    compatibility_by_entry_csv: Path,
    output_dir: Path,
    bucket_count: int,
    min_arity: int,
    configured_min_members: int,
    min_bucket_mass: float,
    max_bucket_mass_ratio: float,
    min_entropy_fraction: float,
    token_count_mode: str,
    tokenizer_name_override: str,
) -> dict[str, Any]:
    output_paths = [
        output_dir / "variable_arity_full_density_summary.json",
        output_dir / "variable_arity_full_density_by_slice.csv",
        output_dir / "variable_arity_full_density_by_generated_row.csv",
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite full-density audit outputs: " + ", ".join(existing))

    config = read_yaml(config_path)
    model_cfg = _model_config(config, tokenizer_key)
    tokenizer_name = str(
        tokenizer_name_override
        or model_cfg.get("tokenizer_name")
        or model_cfg.get("model_name")
        or ""
    )
    if token_count_mode == "tokenizer" and not tokenizer_name:
        raise ValueError(f"Missing tokenizer_name for tokenizer key {tokenizer_key!r}")
    counter = _token_counter(mode=token_count_mode, tokenizer_name=tokenizer_name)
    capacity_cfg = dict(dict(config.get("bucket_bank", {})).get("compatibility_adjusted_capacity", {}))
    diagnostic_gate = dict(capacity_cfg.get("diagnostic_high_risk_gate", {}))
    viability_gate = dict(capacity_cfg.get("qwen_e2e_viability_gate", {}))
    diagnostic_density_min = float(diagnostic_gate.get("heldout_eligible_positions_per_100_tokens_min", 0.3))
    diagnostic_bits_min = float(diagnostic_gate.get("effective_compatible_bits_per_response_min", 0.3))
    viability_density_min = float(viability_gate.get("heldout_eligible_positions_per_100_tokens_min", 0.5))
    viability_bits_min = float(viability_gate.get("effective_compatible_bits_per_response_min", 1.0))

    generated_rows: dict[str, dict[str, Any]] = {}
    duplicate_generated_keys: Counter[str] = Counter()
    for row_index, row in enumerate(_iter_jsonl(generated_outputs_path)):
        key = _generated_row_key(row, row_index)
        if key in generated_rows:
            duplicate_generated_keys[key] += 1
            continue
        generated_rows[key] = _new_generated_stats(
            row=row,
            row_index=row_index,
            token_count=counter(str(row.get("response_text", row.get("output_text", "")))),
        )

    missing_generated_rows = 0
    arity_counter: Counter[int] = Counter()
    input_rows = 0
    duplicate_entry_keys: Counter[tuple[str, str]] = Counter()
    seen_entry_keys: set[tuple[str, str]] = set()
    for row in _read_csv_rows(compatibility_by_entry_csv):
        input_rows += 1
        generated_key = str(row.get("generated_row_index", ""))
        entry_key = (str(row.get("bank_entry_id", "")), generated_key)
        if entry_key in seen_entry_keys:
            duplicate_entry_keys[entry_key] += 1
            continue
        seen_entry_keys.add(entry_key)
        stats = generated_rows.get(generated_key)
        if stats is None:
            missing_generated_rows += 1
            continue
        stats["input_entry_count"] += 1
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
        stats["max_arity"] = max(int(stats["max_arity"]), arity)
        if arity < int(min_arity):
            continue
        capacity_bits = math.log2(arity)
        stats["accepted_entry_count"] += 1
        stats["capacity_bits"] += capacity_bits
        if len(configured_bucket_ids) >= int(min_arity):
            stats["configured_subset_count"] += 1
        masses = [compatible_probabilities.get(bucket_id, 0.0) for bucket_id in compatible_bucket_ids]
        metrics = bucket_mass_metrics(masses)
        if (
            metrics["min_bucket_mass"] >= float(min_bucket_mass)
            and metrics["bucket_mass_ratio"] <= float(max_bucket_mass_ratio)
            and metrics["bucket_entropy_fraction"] >= float(min_entropy_fraction)
        ):
            stats["probability_gated_count"] += 1
        if _boolish(row.get("observed_bucket_is_compatible", "")):
            stats["observed_bucket_compatible_count"] += 1

    slices: dict[str, dict[str, Any]] = {}
    for generated in generated_rows.values():
        for kind, label, gate_eligible in _record_slices(generated):
            key = _slice_key(kind, label)
            if key not in slices:
                slices[key] = _empty_slice(kind, label, gate_eligible)
            target = slices[key]
            target["generated_outputs"] += 1
            target["response_tokens"] += int(generated["response_token_count"])
            target["input_entries"] += int(generated["input_entry_count"])
            target["accepted_entries"] += int(generated["accepted_entry_count"])
            target["configured_subset_entries"] += int(generated["configured_subset_count"])
            target["probability_gated_entries"] += int(generated["probability_gated_count"])
            target["observed_bucket_compatible_entries"] += int(generated["observed_bucket_compatible_count"])
            target["total_capacity_bits"] += float(generated["capacity_bits"])
            if int(generated["accepted_entry_count"]) > 0:
                target["rows_with_accepted_entries"] += 1
            else:
                target["rows_with_no_accepted_entries"] += 1

    by_slice_rows = []
    for key, row in sorted(slices.items(), key=lambda item: (item[1]["slice_kind"], item[1]["slice"])):
        response_count = int(row["generated_outputs"])
        token_count = int(row["response_tokens"])
        accepted = int(row["accepted_entries"])
        capacity_bits = float(row["total_capacity_bits"])
        positions_per_100_tokens = accepted * 100.0 / token_count if token_count > 0 else 0.0
        bits_per_response = capacity_bits / response_count if response_count > 0 else 0.0
        bits_per_100_tokens = capacity_bits * 100.0 / token_count if token_count > 0 else 0.0
        diagnostic_status = (
            "PASS"
            if row["gate_eligible"]
            and positions_per_100_tokens >= diagnostic_density_min
            and bits_per_response >= diagnostic_bits_min
            else ("DIAGNOSTIC_PROXY_NOT_GATE" if not row["gate_eligible"] else "FAIL")
        )
        viability_status = (
            "PASS"
            if row["gate_eligible"]
            and positions_per_100_tokens >= viability_density_min
            and bits_per_response >= viability_bits_min
            else ("DIAGNOSTIC_PROXY_NOT_GATE" if not row["gate_eligible"] else "FAIL")
        )
        by_slice_rows.append(
            {
                **row,
                "eligible_positions_per_100_tokens": positions_per_100_tokens,
                "effective_bits_per_response": bits_per_response,
                "effective_bits_per_100_tokens": bits_per_100_tokens,
                "diagnostic_density_min": diagnostic_density_min,
                "diagnostic_bits_per_response_min": diagnostic_bits_min,
                "diagnostic_gate_status": diagnostic_status,
                "viability_density_min": viability_density_min,
                "viability_bits_per_response_min": viability_bits_min,
                "viability_gate_status": viability_status,
                "denominator_scope": "all_generated_outputs_full_response_tokens",
                "capacity_claim": "full_denominator_variable_arity_density_not_payload_recovery",
            }
        )

    by_generated_rows = []
    for key, row in sorted(generated_rows.items(), key=lambda item: int(item[0])):
        by_generated_rows.append(
            {
                **row,
                "eligible_positions_per_100_tokens": (
                    int(row["accepted_entry_count"]) * 100.0 / int(row["response_token_count"])
                    if int(row["response_token_count"]) > 0
                    else 0.0
                ),
                "effective_bits_per_response": float(row["capacity_bits"]),
                "condition_key": _condition_key(row),
            }
        )

    all_row = next(row for row in by_slice_rows if row["slice_kind"] == "all")
    heldout_rows = [
        row
        for row in by_slice_rows
        if row["slice_kind"] == "prompt_split" and str(row["slice"]).lower() in {"heldout", "held-out", "held_out"}
    ]
    heldout_status = heldout_rows[0]["viability_gate_status"] if heldout_rows else "NEEDS_HELDOUT_GENERATED_OUTPUTS"
    organic_rows = [
        row
        for row in by_slice_rows
        if row["slice_kind"] == "prompt_split" and str(row["slice"]).lower() == "organic"
    ]
    manifest = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_DIAGNOSTIC_PENDING_REVIEW",
        "config": str(config_path),
        "tokenizer_key": tokenizer_key,
        "tokenizer_name": tokenizer_name,
        "token_count_mode": token_count_mode,
        "generated_outputs": str(generated_outputs_path),
        "compatibility_by_entry_csv": str(compatibility_by_entry_csv),
        "bucket_count": int(bucket_count),
        "min_arity": int(min_arity),
        "configured_min_members": int(configured_min_members),
        "input_compatibility_rows": input_rows,
        "unique_entry_keys": len(seen_entry_keys),
        "duplicate_entry_keys_skipped": sum(duplicate_entry_keys.values()),
        "missing_generated_rows": missing_generated_rows,
        "generated_outputs_count": len(generated_rows),
        "duplicate_generated_keys_skipped": sum(duplicate_generated_keys.values()),
        "total_response_tokens": int(all_row["response_tokens"]),
        "accepted_entries": int(all_row["accepted_entries"]),
        "configured_subset_entries": int(all_row["configured_subset_entries"]),
        "probability_gated_entries": int(all_row["probability_gated_entries"]),
        "total_capacity_bits": float(all_row["total_capacity_bits"]),
        "eligible_positions_per_100_tokens": float(all_row["eligible_positions_per_100_tokens"]),
        "effective_bits_per_response": float(all_row["effective_bits_per_response"]),
        "effective_bits_per_100_tokens": float(all_row["effective_bits_per_100_tokens"]),
        "rows_with_accepted_entries": int(all_row["rows_with_accepted_entries"]),
        "rows_with_no_accepted_entries": int(all_row["rows_with_no_accepted_entries"]),
        "arity_distribution": {str(key): int(value) for key, value in sorted(arity_counter.items())},
        "gate_thresholds": {
            "diagnostic_heldout_eligible_positions_per_100_tokens_min": diagnostic_density_min,
            "diagnostic_effective_bits_per_response_min": diagnostic_bits_min,
            "viability_heldout_eligible_positions_per_100_tokens_min": viability_density_min,
            "viability_effective_bits_per_response_min": viability_bits_min,
        },
        "gate_status": {
            "full_denominator_density": "PASS"
            if float(all_row["eligible_positions_per_100_tokens"]) >= viability_density_min
            else "FAIL",
            "full_denominator_effective_bits": "PASS"
            if float(all_row["effective_bits_per_response"]) >= viability_bits_min
            else "FAIL",
            "heldout_viability_density": heldout_status,
            "organic_density": "NEEDS_ORGANIC_GENERATED_OUTPUTS" if not organic_rows else organic_rows[0]["viability_gate_status"],
        },
        "outputs": {
            "summary_json": str(output_dir / "variable_arity_full_density_summary.json"),
            "by_slice_csv": str(output_dir / "variable_arity_full_density_by_slice.csv"),
            "by_generated_row_csv": str(output_dir / "variable_arity_full_density_by_generated_row.csv"),
        },
        "next_allowed_action": (
            "Review full-denominator density, then run variable-arity raw/wrong-key/wrong-payload "
            "pre-null and variable-radix train/eval preflight before any proof-of-life training."
        ),
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "result_claim": "full_denominator_variable_arity_density_not_payload_recovery",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "variable_arity_full_density_summary.json", manifest)
    write_csv(output_dir / "variable_arity_full_density_by_slice.csv", by_slice_rows, list(by_slice_rows[0].keys()))
    write_csv(
        output_dir / "variable_arity_full_density_by_generated_row.csv",
        by_generated_rows,
        list(by_generated_rows[0].keys()),
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    manifest = audit_full_density(
        config_path=resolve_repo_path(args.config, root),
        tokenizer_key=str(args.tokenizer_key),
        generated_outputs_path=Path(args.generated_outputs),
        compatibility_by_entry_csv=Path(args.compatibility_by_entry_csv),
        output_dir=Path(args.output_dir),
        bucket_count=int(args.bucket_count),
        min_arity=int(args.min_arity),
        configured_min_members=int(args.configured_min_members),
        min_bucket_mass=float(args.min_bucket_mass),
        max_bucket_mass_ratio=float(args.max_bucket_mass_ratio),
        min_entropy_fraction=float(args.min_entropy_fraction),
        token_count_mode=str(args.token_count_mode),
        tokenizer_name_override=str(args.tokenizer_name),
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
