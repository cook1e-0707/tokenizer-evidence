from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT_DIR = ROOT / "results/natural_evidence_v2/status/wp6_e2e_eval_852086"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run artifact-only WP6 failure diagnosis over synced E2E outputs. "
            "This does not train, generate, submit Slurm, aggregate FAR, or make "
            "paper-facing claims."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_new(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def by_key(rows: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    output: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[tuple(row.get(key) for key in keys)].append(dict(row))
    return output


def condition_budget_accepts(decisions: Sequence[Mapping[str, Any]], *, budget: int = 64) -> dict[str, dict[str, Any]]:
    grouped = by_key(decisions, ["decode_condition"])
    output: dict[str, dict[str, Any]] = {}
    for (condition,), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row.get("frame_index", 0)))[:budget]
        output[str(condition)] = {
            "accepted_count": sum(1 for row in ordered if bool(row.get("accepted"))),
            "accept_rate": mean([1.0 if bool(row.get("accepted")) else 0.0 for row in ordered]),
            "frames": len(ordered),
            "mean_resolved_slots": mean([float(row.get("observed_resolved_slot_count", 0)) for row in ordered]),
            "mean_target_hits": mean([float(row.get("target_hit_count", 0)) for row in ordered]),
        }
    return output


def frame_rows(decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in decisions:
        resolved = int(row.get("observed_resolved_slot_count", 0))
        target_hits = int(row.get("target_hit_count", 0))
        observed_slots = int(row.get("observed_slot_count", 0))
        erasures = max(0, 16 - resolved)
        resolved_wrong = max(0, resolved - target_hits)
        hamming_lower_bound = 16 - target_hits
        rows.append(
            {
                "accepted": bool(row.get("accepted")),
                "complete_digit_frame": bool(row.get("complete_digit_frame")),
                "decode_condition": str(row.get("decode_condition", "")),
                "decoded_checksum_byte_hex": str(row.get("decoded_checksum_byte_hex", "")),
                "decoded_payload_byte_hex": str(row.get("decoded_payload_byte_hex", "")),
                "erasure_reasons": ";".join(str(item) for item in row.get("erasure_reasons", [])),
                "erasures": erasures,
                "expected_checksum_byte_hex": str(row.get("expected_checksum_byte_hex", "")),
                "expected_payload_byte_hex": str(row.get("expected_payload_byte_hex", "")),
                "frame_index": int(row.get("frame_index", 0)),
                "hamming_lower_bound_to_target": hamming_lower_bound,
                "observed_resolved_slot_count": resolved,
                "observed_slot_count": observed_slots,
                "prompt_id": str(row.get("prompt_id", "")),
                "resolved_wrong_bits": resolved_wrong,
                "target_hit_count": target_hits,
            }
        )
    return rows


def step_rows(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped = by_key(observations, ["decode_condition", "step_index"])
    for (condition, step_index), rows in sorted(grouped.items(), key=lambda item: (str(item[0][0]), int(item[0][1]))):
        total = len(rows)
        resolved = sum(1 for row in rows if bool(row.get("resolved_bucket_hit")))
        target_hits = sum(1 for row in rows if bool(row.get("target_hit")))
        unresolved = [row for row in rows if not bool(row.get("resolved_bucket_hit"))]
        wrong = [
            row
            for row in rows
            if bool(row.get("resolved_bucket_hit")) and not bool(row.get("target_hit"))
        ]
        output.append(
            {
                "decode_condition": str(condition),
                "resolved_rate": resolved / total if total else 0.0,
                "resolved_wrong_count": len(wrong),
                "step_index": int(step_index),
                "target_bit": rows[0].get("target_bit") if rows else "",
                "target_hit_rate": target_hits / total if total else 0.0,
                "top_unresolved_first_words": ";".join(
                    f"{word}:{count}"
                    for word, count in Counter(str(row.get("first_word", "")) for row in unresolved).most_common(8)
                ),
                "top_wrong_first_words": ";".join(
                    f"{word}:{count}"
                    for word, count in Counter(str(row.get("first_word", "")) for row in wrong).most_common(8)
                ),
                "total_rows": total,
                "unresolved_count": len(unresolved),
            }
        )
    return output


def surface_rows(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        if bool(row.get("resolved_bucket_hit")):
            continue
        grouped[
            (
                str(row.get("decode_condition", "")),
                int(row.get("step_index", 0)),
                str(row.get("first_word", "")),
                row.get("target_bit"),
            )
        ].append(dict(row))
    output: list[dict[str, Any]] = []
    for (condition, step, word, target_bit), rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        output.append(
            {
                "count": len(rows),
                "decode_condition": condition,
                "first_word": word,
                "step_index": step,
                "target_bit": target_bit,
            }
        )
    return output


def near_miss_rows(protected_frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in protected_frames
        if not bool(row.get("accepted"))
        and int(row.get("hamming_lower_bound_to_target", 16)) <= 3
    ]
    rows.sort(
        key=lambda row: (
            int(row.get("hamming_lower_bound_to_target", 16)),
            int(row.get("erasures", 16)),
            int(row.get("frame_index", 0)),
        )
    )
    return rows


def threshold_counts(protected_frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    total = len(protected_frames)
    for min_hits in range(16, 7, -1):
        count = sum(1 for row in protected_frames if int(row.get("target_hit_count", 0)) >= min_hits)
        counts[f"target_hits_ge_{min_hits}"] = {"count": count, "rate": count / total if total else 0.0}
    for max_errors in range(0, 5):
        count = sum(
            1
            for row in protected_frames
            if int(row.get("hamming_lower_bound_to_target", 16)) <= max_errors
        )
        counts[f"hamming_lower_bound_le_{max_errors}"] = {
            "count": count,
            "rate": count / total if total else 0.0,
        }
    return counts


def optimistic_surface_expansion(protected_obs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = by_key(protected_obs, ["frame_index"])
    rescued = 0
    total = 0
    still_wrong = 0
    for (_frame,), rows in grouped.items():
        total += 1
        if len(rows) < 16:
            continue
        wrong_resolved = [
            row
            for row in rows
            if bool(row.get("resolved_bucket_hit")) and not bool(row.get("target_hit"))
        ]
        if not wrong_resolved:
            rescued += 1
        else:
            still_wrong += 1
    return {
        "interpretation": (
            "Upper bound if every currently out-of-bank first word could be safely "
            "added to the target bucket for its coordinate; resolved wrong-bucket "
            "hits remain errors."
        ),
        "optimistic_accept_count": rescued,
        "optimistic_accept_rate": rescued / total if total else 0.0,
        "resolved_wrong_frame_count": still_wrong,
        "total_frames": total,
    }


def bits_to_hex(bits: Sequence[int | None]) -> str:
    if len(bits) != 16 or any(bit not in {0, 1} for bit in bits):
        return ""
    payload = int("".join(str(int(bit)) for bit in bits[:8]), 2)
    checksum = int("".join(str(int(bit)) for bit in bits[8:]), 2)
    return f"{payload:02x}{checksum:02x}"


def coordinate_majority_replay(
    observations: Sequence[Mapping[str, Any]],
    expected_bits: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    grouped = by_key(observations, ["decode_condition"])
    for (condition,), condition_rows in sorted(grouped.items(), key=lambda item: str(item[0][0])):
        majority_bits: list[int | None] = []
        condition_match_count = 0
        for step_index in range(1, 17):
            slot_rows = [
                row
                for row in condition_rows
                if int(row.get("step_index", 0)) == step_index and bool(row.get("resolved_bucket_hit"))
            ]
            counts = Counter(int(row["observed_bucket_id"]) for row in slot_rows)
            total_resolved = sum(counts.values())
            if not counts:
                majority_bit = None
                margin = 0
            else:
                ordered = counts.most_common()
                majority_bit = int(ordered[0][0])
                runner_up = int(ordered[1][1]) if len(ordered) > 1 else 0
                margin = int(ordered[0][1]) - runner_up
            target_bit = int(expected_bits[step_index - 1])
            match = majority_bit == target_bit
            condition_match_count += 1 if match else 0
            majority_bits.append(majority_bit)
            rows.append(
                {
                    "decode_condition": str(condition),
                    "majority_bit": "" if majority_bit is None else majority_bit,
                    "majority_matches_target": bool(match),
                    "majority_margin": margin,
                    "observed_bucket_0_count": int(counts.get(0, 0)),
                    "observed_bucket_1_count": int(counts.get(1, 0)),
                    "resolved_count": total_resolved,
                    "step_index": step_index,
                    "target_bit": target_bit,
                }
            )
        summary[str(condition)] = {
            "majority_bits": ["" if bit is None else int(bit) for bit in majority_bits],
            "majority_hex": bits_to_hex(majority_bits),
            "majority_matches_target_bits": condition_match_count,
            "majority_recovered_target_codeword": condition_match_count == 16,
        }
    return rows, summary


def coordinate_majority_budget_replay(
    observations: Sequence[Mapping[str, Any]],
    expected_bits: Sequence[int],
    budgets: Sequence[int] = (8, 16, 32, 64),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for budget in budgets:
        budget_obs = [row for row in observations if int(row.get("frame_index", 0)) < int(budget)]
        majority_rows, majority_summary = coordinate_majority_replay(budget_obs, expected_bits)
        summary[str(budget)] = majority_summary
        for row in majority_rows:
            enriched = dict(row)
            enriched["budget"] = int(budget)
            rows.append(enriched)
    return rows, summary


def main() -> int:
    args = parse_args()
    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_json(input_dir / "wp6_e2e_summary.json")
    decisions = read_jsonl(input_dir / "wp6_decode_decisions.jsonl")
    observations = read_jsonl(input_dir / "wp6_slot_observations.jsonl")
    generated = read_jsonl(input_dir / "wp6_generated_outputs.jsonl")

    frames = frame_rows(decisions)
    protected_frames = [row for row in frames if row["decode_condition"] == "protected"]
    protected_obs = [row for row in observations if row.get("decode_condition") == "protected"]
    steps = step_rows(observations)
    surfaces = surface_rows(observations)
    near = near_miss_rows(protected_frames)
    accepted = [row for row in protected_frames if bool(row.get("accepted"))]
    failed = [row for row in protected_frames if not bool(row.get("accepted"))]
    expected_bits = []
    for row in protected_obs:
        step = int(row.get("step_index", 0))
        if 1 <= step <= 16:
            while len(expected_bits) < step:
                expected_bits.append(None)
            expected_bits[step - 1] = int(row.get("target_bit"))
    if len(expected_bits) != 16 or any(bit not in {0, 1} for bit in expected_bits):
        raise ValueError("could not reconstruct 16 expected bits from protected observations")
    expected_bits_int = [int(bit) for bit in expected_bits]
    majority_rows, majority_summary = coordinate_majority_replay(observations, expected_bits_int)
    majority_budget_rows, majority_budget_summary = coordinate_majority_budget_replay(
        observations,
        expected_bits_int,
    )

    diagnosis = {
        "artifact_role": "wp6_e2e_failure_diagnosis_artifact_only",
        "claim_control": {
            "e2e_rerun_allowed": False,
            "far_aggregation_allowed": False,
            "llama_allowed": False,
            "paper_claim_allowed": False,
            "training_allowed": False,
        },
        "condition_budget_accepts": condition_budget_accepts(decisions),
        "generated_output_rows": len(generated),
        "input_dir": str(input_dir),
        "near_miss_protected_frame_count_hamming_le_3": len(near),
        "coordinate_majority_replay": {
            "interpretation": (
                "Post-hoc repair diagnostic only. It replays a repeated-coordinate "
                "majority decoder over existing observations; this decoder was not "
                "precommitted for job 852086 and cannot retroactively turn the run "
                "into a passing proof-of-life result."
            ),
            "budget_summary_by_condition": majority_budget_summary,
            "summary_by_condition": majority_summary,
        },
        "optimistic_surface_expansion": optimistic_surface_expansion(protected_obs),
        "protected": {
            "accepted_frames": len(accepted),
            "failed_frames": len(failed),
            "mean_hamming_lower_bound_to_target": mean(
                [float(row["hamming_lower_bound_to_target"]) for row in protected_frames]
            ),
            "mean_resolved_slots": mean([float(row["observed_resolved_slot_count"]) for row in protected_frames]),
            "mean_target_hits": mean([float(row["target_hit_count"]) for row in protected_frames]),
            "median_hamming_lower_bound_to_target": median(
                [float(row["hamming_lower_bound_to_target"]) for row in protected_frames]
            ),
            "median_resolved_slots": median([float(row["observed_resolved_slot_count"]) for row in protected_frames]),
            "median_target_hits": median([float(row["target_hit_count"]) for row in protected_frames]),
        },
        "schema_name": "natural_evidence_v2_wp6_failure_diagnosis_v1",
        "source_gate_status": summary.get("gate_status"),
        "source_summary_json": str(input_dir / "wp6_e2e_summary.json"),
        "threshold_counts": threshold_counts(protected_frames),
    }

    write_json(output_dir / "wp6_failure_diagnosis_summary.json", diagnosis)
    write_csv(
        output_dir / "wp6_protected_frame_diagnosis.csv",
        protected_frames,
        [
            "frame_index",
            "prompt_id",
            "accepted",
            "complete_digit_frame",
            "target_hit_count",
            "observed_resolved_slot_count",
            "erasures",
            "resolved_wrong_bits",
            "hamming_lower_bound_to_target",
            "decoded_payload_byte_hex",
            "decoded_checksum_byte_hex",
            "expected_payload_byte_hex",
            "expected_checksum_byte_hex",
            "erasure_reasons",
        ],
    )
    write_csv(
        output_dir / "wp6_step_slot_diagnosis.csv",
        steps,
        [
            "decode_condition",
            "step_index",
            "target_bit",
            "total_rows",
            "resolved_rate",
            "target_hit_rate",
            "unresolved_count",
            "resolved_wrong_count",
            "top_unresolved_first_words",
            "top_wrong_first_words",
        ],
    )
    write_csv(
        output_dir / "wp6_out_of_bank_surfaces.csv",
        surfaces,
        ["decode_condition", "step_index", "target_bit", "first_word", "count"],
    )
    write_csv(
        output_dir / "wp6_near_miss_frames.csv",
        near,
        [
            "frame_index",
            "prompt_id",
            "target_hit_count",
            "observed_resolved_slot_count",
            "erasures",
            "resolved_wrong_bits",
            "hamming_lower_bound_to_target",
            "decoded_payload_byte_hex",
            "decoded_checksum_byte_hex",
            "erasure_reasons",
        ],
    )
    write_csv(
        output_dir / "wp6_coordinate_majority_replay.csv",
        majority_rows,
        [
            "decode_condition",
            "step_index",
            "target_bit",
            "observed_bucket_0_count",
            "observed_bucket_1_count",
            "resolved_count",
            "majority_bit",
            "majority_margin",
            "majority_matches_target",
        ],
    )
    write_csv(
        output_dir / "wp6_coordinate_majority_budget_replay.csv",
        majority_budget_rows,
        [
            "budget",
            "decode_condition",
            "step_index",
            "target_bit",
            "observed_bucket_0_count",
            "observed_bucket_1_count",
            "resolved_count",
            "majority_bit",
            "majority_margin",
            "majority_matches_target",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
