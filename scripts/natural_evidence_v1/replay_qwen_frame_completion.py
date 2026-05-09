from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.natural_evidence_v1.common import write_csv, write_json


SCHEMA_NAME = "natural_evidence_qwen_846699_frame_completion_replay_v1"
OBS_KEY_FIELDS = ["model_condition", "payload_id", "seed", "observation_condition"]
DECODE_KEY_FIELDS = ["model_condition", "payload_id", "seed", "far_family", "query_budget"]
BY_DECODE_FIELDS = [
    "decode_row_index",
    "decode_model_condition",
    "observation_model_condition",
    "payload_id",
    "expected_payload_id",
    "seed",
    "far_family",
    "observation_condition",
    "query_budget",
    "eligible_positions",
    "selected_observation_rows",
    "erasure_rows",
    "digit_rows",
    "frame_count",
    "scheduled_complete_frame_count_no_erasure",
    "observed_complete_frame_count",
    "scheduled_usable_symbols_if_no_erasure",
    "observed_usable_symbols",
    "frames_with_observed_digits",
    "frames_with_scheduled_slots",
    "max_scheduled_slots_per_frame",
    "max_observed_slots_per_frame",
    "min_observed_missing_slots",
    "min_scheduled_missing_slots",
    "observed_slot_distribution",
    "scheduled_slot_distribution",
    "decode_status",
    "decode_trace_observed_symbols",
    "decode_trace_usable_symbols",
    "mapping_note",
]
CLOSEST_FRAME_FIELDS = [
    "decode_row_index",
    "rank",
    "decode_model_condition",
    "observation_model_condition",
    "payload_id",
    "seed",
    "far_family",
    "observation_condition",
    "query_budget",
    "frame_index",
    "expected_slots",
    "scheduled_slots",
    "observed_slots",
    "scheduled_missing_slots",
    "observed_missing_slots",
    "scheduled_complete_no_erasure",
    "observed_complete",
    "observed_digit_indices",
    "scheduled_digit_indices",
    "example_prompt_ids",
]
SLOT_DISTRIBUTION_FIELDS = [
    "decode_row_index",
    "decode_model_condition",
    "observation_model_condition",
    "payload_id",
    "seed",
    "far_family",
    "observation_condition",
    "query_budget",
    "slot_type",
    "slot_count",
    "frame_count",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay 846699 variable-radix frame completion from existing Qwen "
            "natural E2E observation artifacts. Artifact-only: no model loading, "
            "training, generation, or paper-facing claim."
        )
    )
    parser.add_argument("--observations-jsonl", required=True)
    parser.add_argument("--decode-trace-csv", required=True)
    parser.add_argument("--provenance-summary-json", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--by-decode-csv", default="")
    parser.add_argument("--closest-frames-csv", default="")
    parser.add_argument("--slot-distribution-csv", default="")
    parser.add_argument("--wrong-key-prefix", default="K001_WRONG")
    parser.add_argument("--top-closest-frames", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _nonempty(value: Any) -> bool:
    return value is not None and str(value) != ""


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value) == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_decode_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _hash_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _minimal_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_index": _as_int(row.get("query_index", 0)),
        "prompt_id": str(row.get("prompt_id", "")),
        "position_index": _as_int(row.get("position_index", 0)),
        "frame_index": _as_int(row.get("frame_index", 0)),
        "frame_digit_index": _as_int(row.get("frame_digit_index", 0)),
        "frame_digit_count": _as_int(row.get("frame_digit_count", 0)),
        "digit": row.get("digit", ""),
        "radix": row.get("radix", ""),
        "bucket_id": row.get("bucket_id", ""),
        "erasure_reason": str(row.get("erasure_reason", "")),
    }


def _observation_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model_condition", "")),
        str(row.get("payload_id", "")),
        str(row.get("seed", "")),
        str(row.get("observation_condition", "")),
    )


def load_observation_groups(path: Path) -> tuple[dict[tuple[str, str, str, str], list[dict[str, Any]]], dict[str, Any]]:
    digest = hashlib.sha256()
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    parse_errors: list[dict[str, Any]] = []
    row_count = 0
    digit_rows = 0
    erasure_rows = 0
    frame_digit_count_counts: Counter[int] = Counter()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                parse_errors.append({"line_number": line_number, "error": str(exc)})
                continue
            if not isinstance(row, dict):
                parse_errors.append({"line_number": line_number, "error": "row is not an object"})
                continue
            row_count += 1
            minimal = _minimal_observation(row)
            groups[_observation_key(row)].append(minimal)
            if _nonempty(minimal["digit"]) and _nonempty(minimal["radix"]):
                digit_rows += 1
            if _nonempty(minimal["erasure_reason"]):
                erasure_rows += 1
            if int(minimal["frame_digit_count"]):
                frame_digit_count_counts[int(minimal["frame_digit_count"])] += 1
    for rows in groups.values():
        rows.sort(key=lambda item: (int(item["query_index"]), int(item["position_index"])))
    return dict(groups), {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "row_count": row_count,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:10],
        "group_count": len(groups),
        "digit_rows": digit_rows,
        "erasure_rows": erasure_rows,
        "frame_digit_count_counts": _counter_dict(frame_digit_count_counts),
    }


def _decode_to_observation_key(
    row: Mapping[str, Any],
    *,
    wrong_key_prefix: str,
) -> tuple[tuple[str, str, str, str], str]:
    model_condition = str(row.get("model_condition", ""))
    payload_id = str(row.get("payload_id", ""))
    seed = str(row.get("seed", ""))
    far_family = str(row.get("far_family", ""))
    if model_condition == "wrong_payload":
        return (
            ("protected_trained", payload_id, seed, "correct_key"),
            "wrong_payload rows reuse protected_trained correct-key observations and change only expected_payload",
        )
    if model_condition == "wrong_key":
        wrong_index = far_family.rsplit("_", 1)[-1] if "_" in far_family else far_family
        return (
            ("wrong_key", payload_id, seed, f"{wrong_key_prefix}_{wrong_index}"),
            "wrong_key rows use wrong-key bucketization observations for the protected generation",
        )
    return (
        (model_condition, payload_id, seed, "correct_key"),
        "decode row maps directly to same-condition correct-key observations",
    )


def _indices_to_string(indices: Iterable[int], limit: int = 80) -> str:
    ordered = sorted(set(int(value) for value in indices))
    rendered = ",".join(str(value) for value in ordered[:limit])
    if len(ordered) > limit:
        rendered += f",...(+{len(ordered) - limit})"
    return rendered


def _prompts_to_string(prompt_ids: Iterable[str], limit: int = 12) -> str:
    ordered = sorted(set(str(value) for value in prompt_ids if str(value)))
    rendered = ",".join(ordered[:limit])
    if len(ordered) > limit:
        rendered += f",...(+{len(ordered) - limit})"
    return rendered


def _frame_replay(rows: list[dict[str, Any]], budget: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [row for row in rows if int(row["query_index"]) < int(budget)]
    frames: dict[int, dict[str, Any]] = {}
    erasure_rows = 0
    digit_rows = 0
    for row in selected:
        frame_index = int(row["frame_index"])
        frame = frames.setdefault(
            frame_index,
            {
                "frame_index": frame_index,
                "expected_slots": int(row["frame_digit_count"]),
                "scheduled_digit_indices": set(),
                "observed_digit_indices": set(),
                "prompt_ids": set(),
                "row_count": 0,
                "erasure_rows": 0,
                "digit_rows": 0,
            },
        )
        frame["expected_slots"] = max(int(frame["expected_slots"]), int(row["frame_digit_count"]))
        frame["scheduled_digit_indices"].add(int(row["frame_digit_index"]))
        frame["prompt_ids"].add(str(row["prompt_id"]))
        frame["row_count"] += 1
        if _nonempty(row.get("erasure_reason", "")):
            erasure_rows += 1
            frame["erasure_rows"] += 1
        if _nonempty(row.get("digit", "")) and _nonempty(row.get("radix", "")):
            digit_rows += 1
            frame["digit_rows"] += 1
            frame["observed_digit_indices"].add(int(row["frame_digit_index"]))

    frame_rows: list[dict[str, Any]] = []
    scheduled_slot_counts: Counter[int] = Counter()
    observed_slot_counts: Counter[int] = Counter()
    scheduled_complete_count = 0
    observed_complete_count = 0
    scheduled_usable_symbols = 0
    observed_usable_symbols = 0
    frames_with_scheduled = 0
    frames_with_observed = 0
    max_scheduled = 0
    max_observed = 0
    min_scheduled_missing: int | None = None
    min_observed_missing: int | None = None
    for frame in frames.values():
        expected_slots = int(frame["expected_slots"])
        expected = set(range(expected_slots)) if expected_slots else set()
        scheduled = set(frame["scheduled_digit_indices"])
        observed = set(frame["observed_digit_indices"])
        scheduled_missing = len(expected - scheduled) if expected else 0
        observed_missing = len(expected - observed) if expected else 0
        scheduled_complete = bool(expected_slots) and scheduled_missing == 0
        observed_complete = bool(expected_slots) and observed_missing == 0
        scheduled_count = len(scheduled)
        observed_count = len(observed)
        scheduled_slot_counts[scheduled_count] += 1
        observed_slot_counts[observed_count] += 1
        max_scheduled = max(max_scheduled, scheduled_count)
        max_observed = max(max_observed, observed_count)
        if scheduled_count:
            frames_with_scheduled += 1
        if observed_count:
            frames_with_observed += 1
        if scheduled_complete:
            scheduled_complete_count += 1
            scheduled_usable_symbols += expected_slots
        if observed_complete:
            observed_complete_count += 1
            observed_usable_symbols += expected_slots
        if expected_slots:
            min_scheduled_missing = (
                scheduled_missing
                if min_scheduled_missing is None
                else min(min_scheduled_missing, scheduled_missing)
            )
            min_observed_missing = (
                observed_missing if min_observed_missing is None else min(min_observed_missing, observed_missing)
            )
        frame_rows.append(
            {
                "frame_index": int(frame["frame_index"]),
                "expected_slots": expected_slots,
                "scheduled_slots": scheduled_count,
                "observed_slots": observed_count,
                "scheduled_missing_slots": scheduled_missing,
                "observed_missing_slots": observed_missing,
                "scheduled_complete_no_erasure": scheduled_complete,
                "observed_complete": observed_complete,
                "observed_digit_indices": _indices_to_string(observed),
                "scheduled_digit_indices": _indices_to_string(scheduled),
                "example_prompt_ids": _prompts_to_string(frame["prompt_ids"]),
                "row_count": int(frame["row_count"]),
                "erasure_rows": int(frame["erasure_rows"]),
                "digit_rows": int(frame["digit_rows"]),
            }
        )

    summary = {
        "selected_observation_rows": len(selected),
        "erasure_rows": erasure_rows,
        "digit_rows": digit_rows,
        "frame_count": len(frames),
        "scheduled_complete_frame_count_no_erasure": scheduled_complete_count,
        "observed_complete_frame_count": observed_complete_count,
        "scheduled_usable_symbols_if_no_erasure": scheduled_usable_symbols,
        "observed_usable_symbols": observed_usable_symbols,
        "frames_with_scheduled_slots": frames_with_scheduled,
        "frames_with_observed_digits": frames_with_observed,
        "max_scheduled_slots_per_frame": max_scheduled,
        "max_observed_slots_per_frame": max_observed,
        "min_scheduled_missing_slots": 0 if min_scheduled_missing is None else min_scheduled_missing,
        "min_observed_missing_slots": 0 if min_observed_missing is None else min_observed_missing,
        "scheduled_slot_distribution": _counter_dict(scheduled_slot_counts),
        "observed_slot_distribution": _counter_dict(observed_slot_counts),
    }
    closest = sorted(
        frame_rows,
        key=lambda item: (
            int(item["observed_missing_slots"]),
            -int(item["observed_slots"]),
            int(item["scheduled_missing_slots"]),
            int(item["frame_index"]),
        ),
    )
    return summary, closest


def _csv_string(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _build_rows(
    *,
    observation_groups: Mapping[tuple[str, str, str, str], list[dict[str, Any]]],
    decode_rows: list[dict[str, str]],
    wrong_key_prefix: str,
    top_closest_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_decode_rows: list[dict[str, Any]] = []
    closest_rows: list[dict[str, Any]] = []
    slot_distribution_rows: list[dict[str, Any]] = []
    missing_observation_keys: Counter[str] = Counter()
    observed_complete_rows = 0
    scheduled_complete_rows = 0
    max_observed_slots_global = 0
    max_scheduled_slots_global = 0
    total_digit_rows = 0
    total_selected_rows = 0
    total_observed_complete_frames = 0
    total_scheduled_complete_frames = 0

    for decode_row_index, decode_row in enumerate(decode_rows):
        observation_key, mapping_note = _decode_to_observation_key(
            decode_row,
            wrong_key_prefix=wrong_key_prefix,
        )
        group_rows = observation_groups.get(observation_key, [])
        if not group_rows:
            missing_observation_keys["|".join(observation_key)] += 1
        budget = _as_int(decode_row.get("query_budget", 0))
        replay, closest = _frame_replay(group_rows, budget)
        if int(replay["observed_complete_frame_count"]) > 0:
            observed_complete_rows += 1
        if int(replay["scheduled_complete_frame_count_no_erasure"]) > 0:
            scheduled_complete_rows += 1
        max_observed_slots_global = max(max_observed_slots_global, int(replay["max_observed_slots_per_frame"]))
        max_scheduled_slots_global = max(max_scheduled_slots_global, int(replay["max_scheduled_slots_per_frame"]))
        total_digit_rows += int(replay["digit_rows"])
        total_selected_rows += int(replay["selected_observation_rows"])
        total_observed_complete_frames += int(replay["observed_complete_frame_count"])
        total_scheduled_complete_frames += int(replay["scheduled_complete_frame_count_no_erasure"])

        obs_model_condition, _, _, observation_condition = observation_key
        row = {
            "decode_row_index": decode_row_index,
            "decode_model_condition": str(decode_row.get("model_condition", "")),
            "observation_model_condition": obs_model_condition,
            "payload_id": str(decode_row.get("payload_id", "")),
            "expected_payload_id": str(decode_row.get("expected_payload_id", "")),
            "seed": str(decode_row.get("seed", "")),
            "far_family": str(decode_row.get("far_family", "")),
            "observation_condition": observation_condition,
            "query_budget": budget,
            "eligible_positions": _as_int(decode_row.get("eligible_positions", 0)),
            "decode_status": str(decode_row.get("decode_status", "")),
            "decode_trace_observed_symbols": _as_int(decode_row.get("observed_symbols", 0)),
            "decode_trace_usable_symbols": _as_int(decode_row.get("usable_symbols", 0)),
            "mapping_note": mapping_note,
            **replay,
        }
        by_decode_rows.append({field: _csv_string(row.get(field, "")) for field in BY_DECODE_FIELDS})

        base = {
            "decode_row_index": decode_row_index,
            "decode_model_condition": str(decode_row.get("model_condition", "")),
            "observation_model_condition": obs_model_condition,
            "payload_id": str(decode_row.get("payload_id", "")),
            "seed": str(decode_row.get("seed", "")),
            "far_family": str(decode_row.get("far_family", "")),
            "observation_condition": observation_condition,
            "query_budget": budget,
        }
        for rank, frame in enumerate(closest[:top_closest_frames], start=1):
            closest_rows.append({**base, "rank": rank, **frame})
        for slot_type, distribution_key in (
            ("scheduled", "scheduled_slot_distribution"),
            ("observed", "observed_slot_distribution"),
        ):
            for slot_count, frame_count in sorted(
                replay[distribution_key].items(),
                key=lambda item: int(item[0]),
            ):
                slot_distribution_rows.append(
                    {
                        **base,
                        "slot_type": slot_type,
                        "slot_count": int(slot_count),
                        "frame_count": int(frame_count),
                    }
                )

    aggregate = {
        "decode_row_count": len(decode_rows),
        "decode_rows_with_observed_complete_frames": observed_complete_rows,
        "decode_rows_with_scheduled_complete_frames_no_erasure": scheduled_complete_rows,
        "observed_complete_frame_count_total": total_observed_complete_frames,
        "scheduled_complete_frame_count_no_erasure_total": total_scheduled_complete_frames,
        "max_observed_slots_per_frame_global": max_observed_slots_global,
        "max_scheduled_slots_per_frame_global": max_scheduled_slots_global,
        "selected_observation_rows_total_across_decode_budgets": total_selected_rows,
        "digit_rows_total_across_decode_budgets": total_digit_rows,
        "digit_rate_across_decode_budgets": _rate(total_digit_rows, total_selected_rows),
        "missing_observation_key_counts": _counter_dict(missing_observation_keys),
    }
    return by_decode_rows, closest_rows, slot_distribution_rows, aggregate


def build_replay(
    *,
    observations_jsonl: Path,
    decode_trace_csv: Path,
    provenance_summary_json: Path | None,
    wrong_key_prefix: str,
    top_closest_frames: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    observation_groups, observation_artifact = load_observation_groups(observations_jsonl)
    decode_rows = _read_decode_rows(decode_trace_csv)
    decode_artifact = _hash_file(decode_trace_csv)
    provenance = _read_json(provenance_summary_json) if provenance_summary_json is not None else {}
    by_decode_rows, closest_rows, slot_distribution_rows, aggregate = _build_rows(
        observation_groups=observation_groups,
        decode_rows=decode_rows,
        wrong_key_prefix=wrong_key_prefix,
        top_closest_frames=top_closest_frames,
    )
    provenance_mismatches: list[str] = []
    if provenance:
        expected_obs = provenance.get("observations", {}).get("artifact", {})
        expected_dec = provenance.get("decode_trace", {}).get("artifact", {})
        if expected_obs.get("sha256") and expected_obs.get("sha256") != observation_artifact["sha256"]:
            provenance_mismatches.append("observation sha256 mismatch against provenance summary")
        if expected_dec.get("sha256") and expected_dec.get("sha256") != decode_artifact["sha256"]:
            provenance_mismatches.append("decode trace sha256 mismatch against provenance summary")
        if int(provenance.get("observations", {}).get("row_count", 0) or 0) != int(observation_artifact["row_count"]):
            provenance_mismatches.append("observation row_count mismatch against provenance summary")
        if int(provenance.get("decode_trace", {}).get("row_count", 0) or 0) != len(decode_rows):
            provenance_mismatches.append("decode trace row_count mismatch against provenance summary")

    if provenance_mismatches:
        status = "PROVISIONAL_PROVENANCE_MISMATCH"
    elif int(aggregate["observed_complete_frame_count_total"]) > 0:
        status = "COMPLETE_REPLAY_OBSERVED_COMPLETE_FRAMES_FOUND"
    elif int(aggregate["scheduled_complete_frame_count_no_erasure_total"]) > 0:
        status = "COMPLETE_REPLAY_NO_OBSERVED_COMPLETE_FRAMES_SCHEDULE_CAN_COMPLETE_WITH_NO_ERASURE"
    else:
        status = "COMPLETE_REPLAY_NO_COMPLETE_FRAMES_EVEN_UNDER_CURRENT_SCHEDULE_NO_ERASURE"

    summary = {
        "schema_name": SCHEMA_NAME,
        "status": status,
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "not_full_far": True,
        "result_claim": "frame_completion_replay_not_payload_recovery_not_far",
        "inputs": {
            "observations_jsonl": observation_artifact,
            "decode_trace_csv": decode_artifact,
            "provenance_summary_json": "" if provenance_summary_json is None else str(provenance_summary_json),
            "provenance_summary_status": str(provenance.get("status", "")),
            "provenance_mismatches": provenance_mismatches,
        },
        "key_fields": {
            "observation_group_fields": OBS_KEY_FIELDS,
            "decode_trace_group_fields": DECODE_KEY_FIELDS,
            "wrong_payload_mapping": "decode model_condition wrong_payload reuses protected_trained correct-key observations",
        },
        "aggregate": aggregate,
        "interpretation": {
            "observed_complete_frame_absent": int(aggregate["observed_complete_frame_count_total"]) == 0,
            "current_schedule_no_erasure_complete_frame_absent": int(
                aggregate["scheduled_complete_frame_count_no_erasure_total"]
            )
            == 0,
            "frame_contract_implication": (
                "Current prompt-level schedule does not complete any frame even if every committed "
                "scheduled token survived."
                if int(aggregate["scheduled_complete_frame_count_no_erasure_total"]) == 0
                else "Current schedule can complete at least one frame under no-erasure; symbol survival remains the replay blocker."
            ),
        },
        "outputs": {
            "by_decode_csv": "qwen_846699_frame_completion_by_decode_row.csv",
            "closest_frames_csv": "qwen_846699_frame_completion_closest_frames.csv",
            "slot_distribution_csv": "qwen_846699_frame_completion_slot_distribution.csv",
        },
    }
    return summary, by_decode_rows, closest_rows, slot_distribution_rows


def _output_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = _resolve(args.output_dir)
    return {
        "output_dir": output_dir,
        "summary": _resolve(args.summary_json) if args.summary_json else output_dir / "qwen_846699_frame_completion_replay_summary.json",
        "by_decode": _resolve(args.by_decode_csv) if args.by_decode_csv else output_dir / "qwen_846699_frame_completion_by_decode_row.csv",
        "closest": _resolve(args.closest_frames_csv) if args.closest_frames_csv else output_dir / "qwen_846699_frame_completion_closest_frames.csv",
        "slot_distribution": _resolve(args.slot_distribution_csv)
        if args.slot_distribution_csv
        else output_dir / "qwen_846699_frame_completion_slot_distribution.csv",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _output_paths(args)
    for path in (paths["summary"], paths["by_decode"], paths["closest"], paths["slot_distribution"]):
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing replay artifact: {path}")
    provenance_path = _resolve(args.provenance_summary_json) if args.provenance_summary_json else None
    summary, by_decode_rows, closest_rows, slot_distribution_rows = build_replay(
        observations_jsonl=_resolve(args.observations_jsonl),
        decode_trace_csv=_resolve(args.decode_trace_csv),
        provenance_summary_json=provenance_path,
        wrong_key_prefix=str(args.wrong_key_prefix),
        top_closest_frames=max(1, int(args.top_closest_frames)),
    )
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    write_json(paths["summary"], summary)
    write_csv(paths["by_decode"], by_decode_rows, BY_DECODE_FIELDS)
    write_csv(paths["closest"], closest_rows, CLOSEST_FRAME_FIELDS)
    write_csv(paths["slot_distribution"], slot_distribution_rows, SLOT_DISTRIBUTION_FIELDS)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
