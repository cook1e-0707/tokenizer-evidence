from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.natural_evidence_v1.analyze_qwen_on_policy_survival import (
    _decode_token,
    _load_tokenizer,
    _load_train_positions,
    _metadata_for_observation,
    _parse_csv_list,
    _read_json,
    _read_jsonl,
)
from scripts.natural_evidence_v1.common import token_surface_class, write_csv, write_json, write_jsonl
from scripts.natural_evidence_v1.replay_qwen_frame_completion import _as_int, _hash_file, _nonempty, _rate


SCHEMA_NAME = "natural_evidence_qwen_846699_protected_vs_task_only_lift_v1"
MODEL_CONDITIONS = ("protected_trained", "task_only_lora")
LIFT_FIELDS = [
    "slice_kind",
    "slice_value",
    "protected_rows",
    "protected_compatible_hit_rows",
    "protected_compatible_hit_rate",
    "protected_target_hit_rows",
    "protected_target_hit_rate",
    "task_only_rows",
    "task_only_compatible_hit_rows",
    "task_only_compatible_hit_rate",
    "task_only_target_hit_rows",
    "task_only_target_hit_rate",
    "delta_compatible_hit_rate",
    "delta_target_hit_rate",
    "relative_compatible_hit_lift",
    "relative_target_hit_lift",
    "row_balance_ratio",
    "enough_rows_for_slice",
    "target_lift_direction",
    "compatible_lift_direction",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-only protected-vs-task-only survival lift diagnostic for "
            "Qwen 846699. Reads existing observations and train metadata; never "
            "trains, generates, or claims payload recovery."
        )
    )
    parser.add_argument("--observations-jsonl", required=True)
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--survival-summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--payload-ids", default="P0421,P1729")
    parser.add_argument("--tokenizer-name", default="")
    parser.add_argument("--min-rows-per-arm", type=int, default=64)
    parser.add_argument("--max-extreme-rows", type=int, default=80)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _format_float(value: float) -> str:
    return f"{float(value):.17g}"


def _empty_stats() -> dict[str, int]:
    return {
        "rows": 0,
        "compatible_hit_rows": 0,
        "target_hit_rows": 0,
        "bucket_miss_rows": 0,
        "observed_token_in_target_bucket_rows": 0,
    }


def _update_stats(stats: dict[str, int], row: Mapping[str, Any]) -> None:
    stats["rows"] += 1
    stats["compatible_hit_rows"] += int(row["compatible_hit"])
    stats["target_hit_rows"] += int(row["target_hit"])
    stats["bucket_miss_rows"] += int(row["bucket_miss"])
    stats["observed_token_in_target_bucket_rows"] += int(row["observed_token_in_target_bucket"])


def _direction(delta: float) -> str:
    if delta > 0:
        return "protected_higher"
    if delta < 0:
        return "task_only_higher"
    return "tie"


def _relative_lift(protected_rate: float, task_only_rate: float) -> float:
    if task_only_rate == 0.0:
        return 0.0 if protected_rate == 0.0 else float("inf")
    return (protected_rate - task_only_rate) / task_only_rate


def _classify_target_bucket(tokenizer: Any | None, token_ids: Sequence[int]) -> str:
    classes = [
        token_surface_class(text)
        for token_id in token_ids[:4]
        for text in [_decode_token(tokenizer, int(token_id))]
        if text
    ]
    if not classes:
        return ""
    counts = Counter(classes)
    return counts.most_common(1)[0][0]


def _analysis_row(row: Mapping[str, Any], metadata: Mapping[str, Any], tokenizer: Any | None) -> dict[str, Any]:
    observed_token_text = str(row.get("observed_token_text", ""))
    observed_token_id = _as_int(row.get("observed_token_id", -1), -1)
    target_bucket_token_ids = [int(value) for value in metadata.get("target_bucket_token_ids", [])]
    compatible_hit = _nonempty(row.get("digit", "")) and _nonempty(row.get("radix", ""))
    target_digit = _as_int(metadata.get("target_digit", -2), -2)
    target_hit = compatible_hit and _as_int(row.get("digit", -1), -1) == target_digit
    target_bucket_token_ids_set = set(target_bucket_token_ids)
    return {
        "model_condition": str(row.get("model_condition", "")),
        "payload_id": str(row.get("payload_id", "")),
        "seed": str(row.get("seed", "")),
        "prompt_id": str(row.get("prompt_id", "")),
        "prompt_slot": str(metadata.get("prompt_slot", row.get("position_index", ""))),
        "frame_digit_index": str(metadata.get("frame_digit_index", row.get("frame_digit_index", ""))),
        "frame_digit_count": str(metadata.get("frame_digit_count", row.get("frame_digit_count", ""))),
        "frame_index_mod64": str(_as_int(metadata.get("frame_index", row.get("frame_index", 0))) % 64),
        "target_radix": str(metadata.get("target_radix", "")),
        "target_bucket": str(metadata.get("target_bucket", "")),
        "source_model_condition": str(metadata.get("source_model_condition", "")),
        "source_prompt_split": str(metadata.get("source_prompt_split", "")),
        "source_example_role": str(metadata.get("source_example_role", "")),
        "observed_token_class": token_surface_class(observed_token_text) if observed_token_text else "",
        "target_bucket_token_class": _classify_target_bucket(tokenizer, target_bucket_token_ids),
        "join_source": str(metadata.get("join_source", "")),
        "compatible_hit": bool(compatible_hit),
        "target_hit": bool(target_hit),
        "bucket_miss": str(row.get("erasure_reason", "")) == "observed_token_not_in_variable_radix_bucket_set",
        "observed_token_in_target_bucket": observed_token_id in target_bucket_token_ids_set,
    }


def _slice_values(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        ("all", "all"),
        ("payload_id", str(row.get("payload_id", ""))),
        ("seed", str(row.get("seed", ""))),
        ("payload_seed", f"{row.get('payload_id', '')}|{row.get('seed', '')}"),
        ("prompt_slot", str(row.get("prompt_slot", ""))),
        ("payload_seed_prompt_slot", f"{row.get('payload_id', '')}|{row.get('seed', '')}|slot={row.get('prompt_slot', '')}"),
        ("frame_digit_index", str(row.get("frame_digit_index", ""))),
        ("payload_seed_frame_digit_index", f"{row.get('payload_id', '')}|{row.get('seed', '')}|digit={row.get('frame_digit_index', '')}"),
        ("target_radix", str(row.get("target_radix", ""))),
        ("payload_seed_target_radix", f"{row.get('payload_id', '')}|{row.get('seed', '')}|radix={row.get('target_radix', '')}"),
        ("observed_token_class", str(row.get("observed_token_class", ""))),
        ("target_bucket_token_class", str(row.get("target_bucket_token_class", ""))),
        ("payload_seed_target_bucket_token_class", f"{row.get('payload_id', '')}|{row.get('seed', '')}|target_class={row.get('target_bucket_token_class', '')}"),
        ("source_model_condition", str(row.get("source_model_condition", ""))),
        ("payload_seed_source_model_condition", f"{row.get('payload_id', '')}|{row.get('seed', '')}|source={row.get('source_model_condition', '')}"),
        ("source_prompt_split", str(row.get("source_prompt_split", ""))),
        ("source_example_role", str(row.get("source_example_role", ""))),
        ("target_bucket", str(row.get("target_bucket", ""))),
        ("frame_digit_count", str(row.get("frame_digit_count", ""))),
        ("frame_index_mod64", str(row.get("frame_index_mod64", ""))),
        ("join_source", str(row.get("join_source", ""))),
    ]


def _lift_row(
    slice_kind: str,
    slice_value: str,
    by_condition: Mapping[str, Mapping[str, int]],
    min_rows_per_arm: int,
) -> dict[str, Any]:
    protected = by_condition.get("protected_trained", _empty_stats())
    task = by_condition.get("task_only_lora", _empty_stats())
    protected_rows = int(protected["rows"])
    task_rows = int(task["rows"])
    protected_compatible_rate = _rate(int(protected["compatible_hit_rows"]), protected_rows)
    task_compatible_rate = _rate(int(task["compatible_hit_rows"]), task_rows)
    protected_target_rate = _rate(int(protected["target_hit_rows"]), protected_rows)
    task_target_rate = _rate(int(task["target_hit_rows"]), task_rows)
    delta_compatible = protected_compatible_rate - task_compatible_rate
    delta_target = protected_target_rate - task_target_rate
    row_balance_ratio = _rate(min(protected_rows, task_rows), max(protected_rows, task_rows))
    enough_rows = protected_rows >= min_rows_per_arm and task_rows >= min_rows_per_arm
    return {
        "slice_kind": slice_kind,
        "slice_value": slice_value,
        "protected_rows": protected_rows,
        "protected_compatible_hit_rows": int(protected["compatible_hit_rows"]),
        "protected_compatible_hit_rate": _format_float(protected_compatible_rate),
        "protected_target_hit_rows": int(protected["target_hit_rows"]),
        "protected_target_hit_rate": _format_float(protected_target_rate),
        "task_only_rows": task_rows,
        "task_only_compatible_hit_rows": int(task["compatible_hit_rows"]),
        "task_only_compatible_hit_rate": _format_float(task_compatible_rate),
        "task_only_target_hit_rows": int(task["target_hit_rows"]),
        "task_only_target_hit_rate": _format_float(task_target_rate),
        "delta_compatible_hit_rate": _format_float(delta_compatible),
        "delta_target_hit_rate": _format_float(delta_target),
        "relative_compatible_hit_lift": _format_float(_relative_lift(protected_compatible_rate, task_compatible_rate)),
        "relative_target_hit_lift": _format_float(_relative_lift(protected_target_rate, task_target_rate)),
        "row_balance_ratio": _format_float(row_balance_ratio),
        "enough_rows_for_slice": str(bool(enough_rows)).lower(),
        "target_lift_direction": _direction(delta_target),
        "compatible_lift_direction": _direction(delta_compatible),
    }


def analyze_lift(
    *,
    observations_jsonl: Path,
    train_data_dir: Path,
    survival_summary_json: Path,
    payload_ids: Sequence[str],
    tokenizer_name: str,
    min_rows_per_arm: int,
    max_extreme_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_prompt_slot, by_entry_key, train_summary = _load_train_positions(train_data_dir, payload_ids)
    tokenizer, tokenizer_error = _load_tokenizer(tokenizer_name)
    survival_summary = _read_json(survival_summary_json)
    observation_hash = _hash_file(observations_jsonl)
    survival_observation_hash = (
        survival_summary.get("inputs", {})
        .get("observations_jsonl", {})
        .get("sha256", "")
    )
    provenance_mismatches: list[str] = []
    if survival_observation_hash and survival_observation_hash != observation_hash["sha256"]:
        provenance_mismatches.append("observations_jsonl_sha256")

    by_slice: dict[tuple[str, str], dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(_empty_stats))
    row_counts: Counter[str] = Counter()
    metadata_missing = 0
    skipped_non_lift_rows = 0

    for row in _read_jsonl(observations_jsonl):
        if str(row.get("model_condition", "")) not in MODEL_CONDITIONS or str(row.get("observation_condition", "")) != "correct_key":
            skipped_non_lift_rows += 1
            continue
        metadata, join_source = _metadata_for_observation(row, by_prompt_slot, by_entry_key)
        if not metadata:
            metadata_missing += 1
            continue
        metadata = dict(metadata)
        metadata["join_source"] = join_source
        analysis_row = _analysis_row(row, metadata, tokenizer)
        model_condition = str(analysis_row["model_condition"])
        row_counts[model_condition] += 1
        for slice_key in _slice_values(analysis_row):
            _update_stats(by_slice[slice_key][model_condition], analysis_row)

    lift_rows = [
        _lift_row(slice_kind, slice_value, stats, min_rows_per_arm)
        for (slice_kind, slice_value), stats in sorted(by_slice.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    enough_rows = [row for row in lift_rows if row["enough_rows_for_slice"] == "true"]
    strongest_target = sorted(
        enough_rows,
        key=lambda row: float(row["delta_target_hit_rate"]),
        reverse=True,
    )[:max_extreme_rows]
    weakest_target = sorted(
        enough_rows,
        key=lambda row: float(row["delta_target_hit_rate"]),
    )[:max_extreme_rows]
    extreme_rows = [
        {**row, "extreme_type": "strongest_protected_target_lift"}
        for row in strongest_target
    ] + [
        {**row, "extreme_type": "strongest_task_only_target_lift"}
        for row in weakest_target
    ]
    aggregate = next(row for row in lift_rows if row["slice_kind"] == "all" and row["slice_value"] == "all")
    direction_counts = Counter(row["target_lift_direction"] for row in enough_rows)
    compatible_direction_counts = Counter(row["compatible_lift_direction"] for row in enough_rows)
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_PROTECTED_VS_TASK_ONLY_LIFT_DIAGNOSTIC",
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "not_full_far": True,
        "result_claim": "protected_vs_task_only_lift_diagnostic_not_payload_recovery_not_far",
        "inputs": {
            "observations_jsonl": observation_hash,
            "train_data_dir": train_summary,
            "survival_summary_json": {
                "path": str(survival_summary_json),
                "status": survival_summary.get("status", ""),
                "schema_name": survival_summary.get("schema_name", ""),
                "provenance_mismatches": provenance_mismatches,
            },
            "tokenizer_name": tokenizer_name,
            "tokenizer_load_error": tokenizer_error,
            "min_rows_per_arm": int(min_rows_per_arm),
        },
        "aggregate": aggregate,
        "row_counts": {
            "included_by_model_condition": {key: int(value) for key, value in sorted(row_counts.items())},
            "skipped_non_lift_rows": int(skipped_non_lift_rows),
            "metadata_missing_rows": int(metadata_missing),
            "slice_rows": len(lift_rows),
            "slice_rows_with_enough_rows": len(enough_rows),
            "target_lift_direction_counts": {key: int(value) for key, value in sorted(direction_counts.items())},
            "compatible_lift_direction_counts": {key: int(value) for key, value in sorted(compatible_direction_counts.items())},
        },
        "interpretation": {
            "aggregate_protected_target_rate_higher_than_task_only": float(aggregate["delta_target_hit_rate"]) > 0,
            "aggregate_protected_compatible_rate_higher_than_task_only": float(aggregate["delta_compatible_hit_rate"]) > 0,
            "target_survival_still_below_one_percent": float(aggregate["protected_target_hit_rate"]) < 0.01,
            "next_required_diagnostic": "teacher_forced_bucket_mass_probe",
        },
        "outputs": {
            "lift_by_slice_csv": "qwen_846699_protected_vs_task_only_lift_by_slice.csv",
            "lift_extremes_jsonl": "qwen_846699_protected_vs_task_only_lift_extremes.jsonl",
        },
    }
    return summary, lift_rows, extreme_rows


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary": output_dir / "qwen_846699_protected_vs_task_only_lift_summary.json",
        "by_slice": output_dir / "qwen_846699_protected_vs_task_only_lift_by_slice.csv",
        "extremes": output_dir / "qwen_846699_protected_vs_task_only_lift_extremes.jsonl",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = _resolve(args.output_dir)
    paths = _output_paths(output_dir)
    for path in paths.values():
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing protected lift artifact: {path}")
    summary, lift_rows, extreme_rows = analyze_lift(
        observations_jsonl=_resolve(args.observations_jsonl),
        train_data_dir=_resolve(args.train_data_dir),
        survival_summary_json=_resolve(args.survival_summary_json),
        payload_ids=_parse_csv_list(args.payload_ids),
        tokenizer_name=str(args.tokenizer_name),
        min_rows_per_arm=max(1, int(args.min_rows_per_arm)),
        max_extreme_rows=max(0, int(args.max_extreme_rows)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(paths["summary"], summary)
    write_csv(paths["by_slice"], lift_rows, LIFT_FIELDS)
    write_jsonl(paths["extremes"], extreme_rows)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
