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
from typing import Any, Mapping, Sequence

from scripts.natural_evidence_v1.common import write_csv, write_json
from scripts.natural_evidence_v1.replay_qwen_frame_completion import (
    _as_int,
    _counter_dict,
    _decode_to_observation_key,
    _hash_file,
    _nonempty,
    _rate,
    _read_decode_rows,
    load_observation_groups,
)


SCHEMA_NAME = "natural_evidence_qwen_846699_oracle_schedule_simulation_v1"
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
    "current_selected_prompts",
    "current_selected_rows",
    "current_digit_rows",
    "current_digit_rate",
    "current_scheduled_complete_frames_no_erasure",
    "current_observed_complete_frames",
    "current_expected_complete_frames_iid_selected_p",
    "current_probability_at_least_one_complete_iid_selected_p",
    "greedy_selected_prompts",
    "greedy_selected_rows",
    "greedy_digit_rows",
    "greedy_digit_rate",
    "greedy_scheduled_complete_frames_no_erasure",
    "greedy_observed_complete_frames",
    "greedy_expected_complete_frames_iid_selected_p",
    "greedy_probability_at_least_one_complete_iid_selected_p",
    "greedy_gain_scheduled_complete_frames",
    "any_subset_observed_complete_frames_all_prompts",
    "any_subset_max_observed_slots_per_frame_all_prompts",
    "min_prompts_to_complete_any_frame_no_erasure_greedy",
    "oracle_protected_lift_probability_at_least_one_complete",
    "oracle_protected_lift_expected_complete_frames",
    "mapping_note",
]
PROMPT_EXAMPLE_FIELDS = [
    "decode_row_index",
    "rank",
    "decode_model_condition",
    "payload_id",
    "seed",
    "far_family",
    "query_budget",
    "prompt_id",
    "query_index",
    "new_complete_frames_at_selection",
    "missing_slot_reduction_at_selection",
    "slot_gain_at_selection",
]
FRAME_BOUND_FIELDS = [
    "decode_row_index",
    "rank",
    "decode_model_condition",
    "payload_id",
    "seed",
    "far_family",
    "query_budget",
    "frame_index",
    "expected_slots",
    "min_prompts_no_erasure_greedy",
    "observed_slots_all_prompts",
    "scheduled_slots_all_prompts",
    "can_complete_under_all_prompts_no_erasure",
    "can_complete_under_all_observed_digits",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run artifact-only oracle schedule simulation for Qwen 846699 "
            "natural E2E observations. No model loading, training, generation, "
            "or paper-facing claim."
        )
    )
    parser.add_argument("--observations-jsonl", required=True)
    parser.add_argument("--decode-trace-csv", required=True)
    parser.add_argument("--frame-replay-summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-budgets", default="64,128,256,512")
    parser.add_argument("--wrong-key-prefix", default="K001_WRONG")
    parser.add_argument("--max-prompt-examples", type=int, default=20)
    parser.add_argument("--max-frame-bounds", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _parse_budgets(value: str) -> list[int]:
    budgets = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not budgets:
        raise ValueError("At least one query budget is required")
    return budgets


def _prompt_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    index: dict[str, int] = {}
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        if prompt_id:
            query_index = _as_int(row.get("query_index", 0))
            index[prompt_id] = min(index.get(prompt_id, query_index), query_index)
    return index


def _rows_by_prompt(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("prompt_id", ""))].append(dict(row))
    return dict(grouped)


def _frame_expected_counts(rows: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        frame_index = _as_int(row.get("frame_index", 0))
        counts[frame_index] = max(counts.get(frame_index, 0), _as_int(row.get("frame_digit_count", 0)))
    return counts


def _schedule_stats(rows: Sequence[Mapping[str, Any]], selected_prompts: set[str]) -> dict[str, Any]:
    expected_counts = _frame_expected_counts(rows)
    frames: dict[int, dict[str, Any]] = {
        frame_index: {
            "expected_slots": expected_slots,
            "scheduled": set(),
            "observed": set(),
            "prompt_ids": set(),
        }
        for frame_index, expected_slots in expected_counts.items()
    }
    selected_rows = 0
    digit_rows = 0
    erasure_rows = 0
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        if prompt_id not in selected_prompts:
            continue
        selected_rows += 1
        frame_index = _as_int(row.get("frame_index", 0))
        digit_index = _as_int(row.get("frame_digit_index", 0))
        frame = frames.setdefault(
            frame_index,
            {
                "expected_slots": _as_int(row.get("frame_digit_count", 0)),
                "scheduled": set(),
                "observed": set(),
                "prompt_ids": set(),
            },
        )
        frame["expected_slots"] = max(int(frame["expected_slots"]), _as_int(row.get("frame_digit_count", 0)))
        frame["scheduled"].add(digit_index)
        frame["prompt_ids"].add(prompt_id)
        if _nonempty(row.get("erasure_reason", "")):
            erasure_rows += 1
        if _nonempty(row.get("digit", "")) and _nonempty(row.get("radix", "")):
            digit_rows += 1
            frame["observed"].add(digit_index)

    scheduled_complete_counts: list[int] = []
    observed_complete_counts: list[int] = []
    min_scheduled_missing: int | None = None
    min_observed_missing: int | None = None
    max_scheduled = 0
    max_observed = 0
    for frame in frames.values():
        expected_slots = int(frame["expected_slots"])
        if expected_slots <= 0:
            continue
        expected = set(range(expected_slots))
        scheduled = set(frame["scheduled"])
        observed = set(frame["observed"])
        scheduled_missing = len(expected - scheduled)
        observed_missing = len(expected - observed)
        max_scheduled = max(max_scheduled, len(scheduled))
        max_observed = max(max_observed, len(observed))
        min_scheduled_missing = (
            scheduled_missing if min_scheduled_missing is None else min(min_scheduled_missing, scheduled_missing)
        )
        min_observed_missing = (
            observed_missing if min_observed_missing is None else min(min_observed_missing, observed_missing)
        )
        if scheduled_missing == 0:
            scheduled_complete_counts.append(expected_slots)
        if observed_missing == 0:
            observed_complete_counts.append(expected_slots)

    return {
        "selected_prompt_count": len(selected_prompts),
        "selected_rows": selected_rows,
        "digit_rows": digit_rows,
        "erasure_rows": erasure_rows,
        "digit_rate": _rate(digit_rows, selected_rows),
        "scheduled_complete_frame_count_no_erasure": len(scheduled_complete_counts),
        "observed_complete_frame_count": len(observed_complete_counts),
        "scheduled_complete_expected_slots": scheduled_complete_counts,
        "observed_complete_expected_slots": observed_complete_counts,
        "max_scheduled_slots_per_frame": max_scheduled,
        "max_observed_slots_per_frame": max_observed,
        "min_scheduled_missing_slots": 0 if min_scheduled_missing is None else min_scheduled_missing,
        "min_observed_missing_slots": 0 if min_observed_missing is None else min_observed_missing,
    }


def _iid_completion(frame_expected_slots: Sequence[int], survival_p: float) -> dict[str, float]:
    p = max(0.0, min(1.0, float(survival_p)))
    probabilities = [p ** int(expected_slots) for expected_slots in frame_expected_slots if int(expected_slots) > 0]
    expected_complete = float(sum(probabilities))
    if not probabilities:
        return {
            "expected_complete_frames": 0.0,
            "probability_at_least_one_complete": 0.0,
        }
    log_no_complete = 0.0
    for probability in probabilities:
        if probability >= 1.0:
            return {
                "expected_complete_frames": expected_complete,
                "probability_at_least_one_complete": 1.0,
            }
        log_no_complete += math.log1p(-probability)
    return {
        "expected_complete_frames": expected_complete,
        "probability_at_least_one_complete": -math.expm1(log_no_complete),
    }


def _greedy_selection(rows: Sequence[Mapping[str, Any]], max_budget: int) -> tuple[list[str], list[dict[str, Any]]]:
    rows_by_prompt = _rows_by_prompt(rows)
    prompt_query_index = _prompt_index(rows)
    expected_counts = _frame_expected_counts(rows)
    selected: list[str] = []
    selected_set: set[str] = set()
    scheduled: dict[int, set[int]] = defaultdict(set)
    completed: set[int] = set()
    candidate_prompts = sorted(rows_by_prompt, key=lambda prompt_id: (prompt_query_index.get(prompt_id, 0), prompt_id))
    trace: list[dict[str, Any]] = []

    for rank in range(1, min(max_budget, len(candidate_prompts)) + 1):
        best_prompt = ""
        best_score: tuple[int, int, int, int, str] | None = None
        best_detail = {
            "new_complete_frames": 0,
            "missing_slot_reduction": 0,
            "slot_gain": 0,
        }
        for prompt_id in candidate_prompts:
            if prompt_id in selected_set:
                continue
            new_complete = 0
            missing_reduction = 0
            slot_gain = 0
            touched: dict[int, set[int]] = defaultdict(set)
            for row in rows_by_prompt[prompt_id]:
                frame_index = _as_int(row.get("frame_index", 0))
                digit_index = _as_int(row.get("frame_digit_index", 0))
                if digit_index not in scheduled[frame_index] and digit_index not in touched[frame_index]:
                    touched[frame_index].add(digit_index)
                    slot_gain += 1
            for frame_index, new_digits in touched.items():
                expected_slots = expected_counts.get(frame_index, 0)
                if expected_slots <= 0:
                    continue
                before = len(scheduled[frame_index])
                after_digits = scheduled[frame_index] | new_digits
                after = len(after_digits)
                missing_reduction += max(0, min(expected_slots, after) - min(expected_slots, before))
                if frame_index not in completed and after >= expected_slots:
                    new_complete += 1
            score = (
                new_complete,
                missing_reduction,
                slot_gain,
                -prompt_query_index.get(prompt_id, 0),
                prompt_id,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_prompt = prompt_id
                best_detail = {
                    "new_complete_frames": new_complete,
                    "missing_slot_reduction": missing_reduction,
                    "slot_gain": slot_gain,
                }
        if not best_prompt:
            break
        selected.append(best_prompt)
        selected_set.add(best_prompt)
        for row in rows_by_prompt[best_prompt]:
            frame_index = _as_int(row.get("frame_index", 0))
            digit_index = _as_int(row.get("frame_digit_index", 0))
            scheduled[frame_index].add(digit_index)
            expected_slots = expected_counts.get(frame_index, 0)
            if expected_slots and len(scheduled[frame_index]) >= expected_slots:
                completed.add(frame_index)
        trace.append(
            {
                "rank": rank,
                "prompt_id": best_prompt,
                "query_index": prompt_query_index.get(best_prompt, 0),
                **best_detail,
                "complete_frame_count_after_selection": len(completed),
            }
        )
    return selected, trace


def _frame_bounds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected_counts = _frame_expected_counts(rows)
    by_frame_prompt_digits: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    observed_by_frame: dict[int, set[int]] = defaultdict(set)
    scheduled_by_frame: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        frame_index = _as_int(row.get("frame_index", 0))
        digit_index = _as_int(row.get("frame_digit_index", 0))
        prompt_id = str(row.get("prompt_id", ""))
        by_frame_prompt_digits[frame_index][prompt_id].add(digit_index)
        scheduled_by_frame[frame_index].add(digit_index)
        if _nonempty(row.get("digit", "")) and _nonempty(row.get("radix", "")):
            observed_by_frame[frame_index].add(digit_index)

    bounds: list[dict[str, Any]] = []
    for frame_index, expected_slots in sorted(expected_counts.items()):
        expected = set(range(expected_slots))
        selected_prompts: set[str] = set()
        covered: set[int] = set()
        prompts = by_frame_prompt_digits[frame_index]
        while expected - covered:
            best_prompt = ""
            best_gain = 0
            for prompt_id, digits in prompts.items():
                if prompt_id in selected_prompts:
                    continue
                gain = len((digits & expected) - covered)
                if gain > best_gain or (gain == best_gain and prompt_id < best_prompt):
                    best_prompt = prompt_id
                    best_gain = gain
            if not best_prompt or best_gain == 0:
                break
            selected_prompts.add(best_prompt)
            covered |= prompts[best_prompt]
        scheduled_digits = scheduled_by_frame.get(frame_index, set()) & expected
        observed_digits = observed_by_frame.get(frame_index, set()) & expected
        bounds.append(
            {
                "frame_index": frame_index,
                "expected_slots": expected_slots,
                "min_prompts_no_erasure_greedy": len(selected_prompts) if expected <= covered else 0,
                "observed_slots_all_prompts": len(observed_digits),
                "scheduled_slots_all_prompts": len(scheduled_digits),
                "can_complete_under_all_prompts_no_erasure": expected <= scheduled_digits,
                "can_complete_under_all_observed_digits": expected <= observed_digits,
            }
        )
    return bounds


def _protected_lift_rates(
    observation_groups: Mapping[tuple[str, str, str, str], Sequence[Mapping[str, Any]]],
    budgets: Sequence[int],
) -> dict[tuple[str, int], float]:
    rates: dict[tuple[str, int], float] = {}
    grouped_by_payload_budget: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (model_condition, payload_id, _seed, observation_condition), rows in observation_groups.items():
        if model_condition != "protected_trained" or observation_condition != "correct_key":
            continue
        for budget in budgets:
            selected = [row for row in rows if _as_int(row.get("query_index", 0)) < budget]
            digit_rows = sum(1 for row in selected if _nonempty(row.get("digit", "")) and _nonempty(row.get("radix", "")))
            grouped_by_payload_budget[(payload_id, budget)].append(_rate(digit_rows, len(selected)))
    for key, values in grouped_by_payload_budget.items():
        rates[key] = float(sum(values) / len(values)) if values else 0.0
    return rates


def _format_float(value: float) -> str:
    return f"{float(value):.17g}"


def _build_outputs(
    *,
    observation_groups: Mapping[tuple[str, str, str, str], list[dict[str, Any]]],
    decode_rows: list[dict[str, str]],
    budgets: Sequence[int],
    wrong_key_prefix: str,
    max_prompt_examples: int,
    max_frame_bounds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    max_budget = max(budgets)
    protected_lift_rates = _protected_lift_rates(observation_groups, budgets)
    greedy_cache: dict[tuple[str, str, str, str], tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    by_decode: list[dict[str, Any]] = []
    prompt_examples: list[dict[str, Any]] = []
    frame_bounds_rows: list[dict[str, Any]] = []
    decode_rows_with_any_subset_observed_complete = 0
    decode_rows_with_greedy_observed_complete = 0
    decode_rows_with_greedy_scheduled_complete = 0
    total_current_scheduled_complete = 0
    total_greedy_scheduled_complete = 0
    total_any_subset_observed_complete = 0
    max_any_subset_observed_slots = 0
    max_greedy_probability = 0.0
    max_current_probability = 0.0
    max_oracle_protected_lift_probability = 0.0
    min_prompts_values: list[int] = []

    for decode_row_index, decode_row in enumerate(decode_rows):
        observation_key, mapping_note = _decode_to_observation_key(decode_row, wrong_key_prefix=wrong_key_prefix)
        rows = observation_groups.get(observation_key, [])
        if observation_key not in greedy_cache:
            greedy_cache[observation_key] = (
                *_greedy_selection(rows, max_budget),
                _frame_bounds(rows),
            )
        greedy_order, greedy_trace, frame_bounds = greedy_cache[observation_key]
        budget = _as_int(decode_row.get("query_budget", 0))
        current_prompts = {str(row.get("prompt_id", "")) for row in rows if _as_int(row.get("query_index", 0)) < budget}
        greedy_prompts = set(greedy_order[: min(budget, len(greedy_order))])
        all_prompts = {str(row.get("prompt_id", "")) for row in rows}
        current = _schedule_stats(rows, current_prompts)
        greedy = _schedule_stats(rows, greedy_prompts)
        any_subset = _schedule_stats(rows, all_prompts)
        current_iid = _iid_completion(
            current["scheduled_complete_expected_slots"],
            float(current["digit_rate"]),
        )
        greedy_iid = _iid_completion(
            greedy["scheduled_complete_expected_slots"],
            float(greedy["digit_rate"]),
        )
        protected_rate = protected_lift_rates.get((str(decode_row.get("payload_id", "")), budget), 0.0)
        protected_lift_iid = _iid_completion(greedy["scheduled_complete_expected_slots"], protected_rate)
        obs_model_condition, _payload_id, _seed, observation_condition = observation_key

        decode_rows_with_any_subset_observed_complete += int(any_subset["observed_complete_frame_count"] > 0)
        decode_rows_with_greedy_observed_complete += int(greedy["observed_complete_frame_count"] > 0)
        decode_rows_with_greedy_scheduled_complete += int(greedy["scheduled_complete_frame_count_no_erasure"] > 0)
        total_current_scheduled_complete += int(current["scheduled_complete_frame_count_no_erasure"])
        total_greedy_scheduled_complete += int(greedy["scheduled_complete_frame_count_no_erasure"])
        total_any_subset_observed_complete += int(any_subset["observed_complete_frame_count"])
        max_any_subset_observed_slots = max(max_any_subset_observed_slots, int(any_subset["max_observed_slots_per_frame"]))
        max_current_probability = max(max_current_probability, current_iid["probability_at_least_one_complete"])
        max_greedy_probability = max(max_greedy_probability, greedy_iid["probability_at_least_one_complete"])
        max_oracle_protected_lift_probability = max(
            max_oracle_protected_lift_probability,
            protected_lift_iid["probability_at_least_one_complete"],
        )

        completable_bounds = [
            frame
            for frame in frame_bounds
            if int(frame["min_prompts_no_erasure_greedy"]) > 0
        ]
        min_prompts = min(
            (int(frame["min_prompts_no_erasure_greedy"]) for frame in completable_bounds),
            default=0,
        )
        if min_prompts:
            min_prompts_values.append(min_prompts)
        by_decode.append(
            {
                "decode_row_index": decode_row_index,
                "decode_model_condition": str(decode_row.get("model_condition", "")),
                "observation_model_condition": obs_model_condition,
                "payload_id": str(decode_row.get("payload_id", "")),
                "expected_payload_id": str(decode_row.get("expected_payload_id", "")),
                "seed": str(decode_row.get("seed", "")),
                "far_family": str(decode_row.get("far_family", "")),
                "observation_condition": observation_condition,
                "query_budget": budget,
                "current_selected_prompts": int(current["selected_prompt_count"]),
                "current_selected_rows": int(current["selected_rows"]),
                "current_digit_rows": int(current["digit_rows"]),
                "current_digit_rate": _format_float(float(current["digit_rate"])),
                "current_scheduled_complete_frames_no_erasure": int(
                    current["scheduled_complete_frame_count_no_erasure"]
                ),
                "current_observed_complete_frames": int(current["observed_complete_frame_count"]),
                "current_expected_complete_frames_iid_selected_p": _format_float(
                    current_iid["expected_complete_frames"]
                ),
                "current_probability_at_least_one_complete_iid_selected_p": _format_float(
                    current_iid["probability_at_least_one_complete"]
                ),
                "greedy_selected_prompts": int(greedy["selected_prompt_count"]),
                "greedy_selected_rows": int(greedy["selected_rows"]),
                "greedy_digit_rows": int(greedy["digit_rows"]),
                "greedy_digit_rate": _format_float(float(greedy["digit_rate"])),
                "greedy_scheduled_complete_frames_no_erasure": int(
                    greedy["scheduled_complete_frame_count_no_erasure"]
                ),
                "greedy_observed_complete_frames": int(greedy["observed_complete_frame_count"]),
                "greedy_expected_complete_frames_iid_selected_p": _format_float(
                    greedy_iid["expected_complete_frames"]
                ),
                "greedy_probability_at_least_one_complete_iid_selected_p": _format_float(
                    greedy_iid["probability_at_least_one_complete"]
                ),
                "greedy_gain_scheduled_complete_frames": int(
                    greedy["scheduled_complete_frame_count_no_erasure"]
                )
                - int(current["scheduled_complete_frame_count_no_erasure"]),
                "any_subset_observed_complete_frames_all_prompts": int(any_subset["observed_complete_frame_count"]),
                "any_subset_max_observed_slots_per_frame_all_prompts": int(any_subset["max_observed_slots_per_frame"]),
                "min_prompts_to_complete_any_frame_no_erasure_greedy": min_prompts,
                "oracle_protected_lift_probability_at_least_one_complete": _format_float(
                    protected_lift_iid["probability_at_least_one_complete"]
                ),
                "oracle_protected_lift_expected_complete_frames": _format_float(
                    protected_lift_iid["expected_complete_frames"]
                ),
                "mapping_note": mapping_note,
            }
        )

        base = {
            "decode_row_index": decode_row_index,
            "decode_model_condition": str(decode_row.get("model_condition", "")),
            "payload_id": str(decode_row.get("payload_id", "")),
            "seed": str(decode_row.get("seed", "")),
            "far_family": str(decode_row.get("far_family", "")),
            "query_budget": budget,
        }
        for prompt in greedy_trace[: min(max_prompt_examples, budget)]:
            prompt_examples.append({**base, **prompt})
        sorted_bounds = sorted(
            frame_bounds,
            key=lambda frame: (
                int(frame["min_prompts_no_erasure_greedy"]) <= 0,
                int(frame["min_prompts_no_erasure_greedy"]) or 10**9,
                -int(frame["observed_slots_all_prompts"]),
                int(frame["frame_index"]),
            ),
        )
        for rank, frame in enumerate(sorted_bounds[:max_frame_bounds], start=1):
            frame_bounds_rows.append({**base, "rank": rank, **frame})

    aggregate = {
        "decode_row_count": len(decode_rows),
        "decode_rows_with_greedy_scheduled_complete_frames_no_erasure": decode_rows_with_greedy_scheduled_complete,
        "decode_rows_with_greedy_observed_complete_frames": decode_rows_with_greedy_observed_complete,
        "decode_rows_with_any_subset_observed_complete_frames": decode_rows_with_any_subset_observed_complete,
        "current_scheduled_complete_frames_no_erasure_total": total_current_scheduled_complete,
        "greedy_scheduled_complete_frames_no_erasure_total": total_greedy_scheduled_complete,
        "any_subset_observed_complete_frames_total": total_any_subset_observed_complete,
        "max_any_subset_observed_slots_per_frame": max_any_subset_observed_slots,
        "max_current_probability_at_least_one_complete_iid_selected_p": max_current_probability,
        "max_greedy_probability_at_least_one_complete_iid_selected_p": max_greedy_probability,
        "max_oracle_protected_lift_probability_at_least_one_complete": max_oracle_protected_lift_probability,
        "min_prompts_to_complete_any_frame_no_erasure_greedy": min(min_prompts_values) if min_prompts_values else 0,
        "median_min_prompts_to_complete_any_frame_no_erasure_greedy": sorted(min_prompts_values)[len(min_prompts_values) // 2]
        if min_prompts_values
        else 0,
    }
    return aggregate, by_decode, prompt_examples, frame_bounds_rows


def build_simulation(
    *,
    observations_jsonl: Path,
    decode_trace_csv: Path,
    frame_replay_summary_json: Path,
    budgets: Sequence[int],
    wrong_key_prefix: str,
    max_prompt_examples: int,
    max_frame_bounds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    observation_groups, observation_artifact = load_observation_groups(observations_jsonl)
    decode_rows = _read_decode_rows(decode_trace_csv)
    frame_replay = json.loads(frame_replay_summary_json.read_text(encoding="utf-8"))
    if not isinstance(frame_replay, dict):
        raise ValueError(f"Frame replay summary must be an object: {frame_replay_summary_json}")
    frame_replay_mismatches: list[str] = []
    replay_obs = frame_replay.get("inputs", {}).get("observations_jsonl", {})
    replay_decode = frame_replay.get("inputs", {}).get("decode_trace_csv", {})
    if replay_obs.get("sha256") and replay_obs.get("sha256") != observation_artifact["sha256"]:
        frame_replay_mismatches.append("observation sha256 mismatch against frame replay summary")
    if replay_decode.get("sha256") and replay_decode.get("sha256") != _hash_file(decode_trace_csv)["sha256"]:
        frame_replay_mismatches.append("decode sha256 mismatch against frame replay summary")

    aggregate, by_decode, prompt_examples, frame_bounds = _build_outputs(
        observation_groups=observation_groups,
        decode_rows=decode_rows,
        budgets=budgets,
        wrong_key_prefix=wrong_key_prefix,
        max_prompt_examples=max_prompt_examples,
        max_frame_bounds=max_frame_bounds,
    )
    if frame_replay_mismatches:
        status = "PROVISIONAL_FRAME_REPLAY_INPUT_MISMATCH"
    elif int(aggregate["decode_rows_with_any_subset_observed_complete_frames"]) == 0:
        status = "COMPLETE_ORACLE_SCHEDULE_NO_OBSERVED_SUBSET_CAN_COMPLETE_FRAME"
    elif int(aggregate["decode_rows_with_greedy_observed_complete_frames"]) == 0:
        status = "COMPLETE_ORACLE_SCHEDULE_GREEDY_NO_OBSERVED_COMPLETE_FRAMES"
    else:
        status = "COMPLETE_ORACLE_SCHEDULE_OBSERVED_COMPLETE_FRAMES_FOUND"

    summary = {
        "schema_name": SCHEMA_NAME,
        "status": status,
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "not_full_far": True,
        "result_claim": "oracle_schedule_simulation_not_payload_recovery_not_far",
        "inputs": {
            "observations_jsonl": observation_artifact,
            "decode_trace_csv": _hash_file(decode_trace_csv),
            "frame_replay_summary_json": str(frame_replay_summary_json),
            "frame_replay_status": str(frame_replay.get("status", "")),
            "frame_replay_mismatches": frame_replay_mismatches,
        },
        "budgets": list(budgets),
        "aggregate": aggregate,
        "interpretation": {
            "any_prompt_subset_can_complete_observed_frame": int(
                aggregate["decode_rows_with_any_subset_observed_complete_frames"]
            )
            > 0,
            "greedy_no_erasure_can_complete_frames": int(
                aggregate["decode_rows_with_greedy_scheduled_complete_frames_no_erasure"]
            )
            > 0,
            "measured_survival_probability_negligible": float(
                aggregate["max_greedy_probability_at_least_one_complete_iid_selected_p"]
            )
            < 1e-20,
            "diagnostic_implication": (
                "No prompt subset can complete a frame using the observed survived digits in 846699; "
                "schedule-only repair cannot recover this transcript. The next diagnostic should decompose "
                "symbol survival by slot/source and then run teacher-forced bucket-mass probes."
                if int(aggregate["decode_rows_with_any_subset_observed_complete_frames"]) == 0
                else "Observed complete frames exist under some subset; inspect scheduler contract before training."
            ),
        },
        "outputs": {
            "by_decode_csv": "qwen_846699_oracle_schedule_by_decode_row.csv",
            "prompt_examples_csv": "qwen_846699_oracle_schedule_prompt_examples.csv",
            "frame_bounds_csv": "qwen_846699_oracle_schedule_frame_bounds.csv",
        },
    }
    return summary, by_decode, prompt_examples, frame_bounds


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary": output_dir / "qwen_846699_oracle_schedule_simulation_summary.json",
        "by_decode": output_dir / "qwen_846699_oracle_schedule_by_decode_row.csv",
        "prompt_examples": output_dir / "qwen_846699_oracle_schedule_prompt_examples.csv",
        "frame_bounds": output_dir / "qwen_846699_oracle_schedule_frame_bounds.csv",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = _resolve(args.output_dir)
    paths = _output_paths(output_dir)
    for path in paths.values():
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing oracle schedule artifact: {path}")
    summary, by_decode, prompt_examples, frame_bounds = build_simulation(
        observations_jsonl=_resolve(args.observations_jsonl),
        decode_trace_csv=_resolve(args.decode_trace_csv),
        frame_replay_summary_json=_resolve(args.frame_replay_summary_json),
        budgets=_parse_budgets(args.query_budgets),
        wrong_key_prefix=str(args.wrong_key_prefix),
        max_prompt_examples=max(0, int(args.max_prompt_examples)),
        max_frame_bounds=max(0, int(args.max_frame_bounds)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(paths["summary"], summary)
    write_csv(paths["by_decode"], by_decode, BY_DECODE_FIELDS)
    write_csv(paths["prompt_examples"], prompt_examples, PROMPT_EXAMPLE_FIELDS)
    write_csv(paths["frame_bounds"], frame_bounds, FRAME_BOUND_FIELDS)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
