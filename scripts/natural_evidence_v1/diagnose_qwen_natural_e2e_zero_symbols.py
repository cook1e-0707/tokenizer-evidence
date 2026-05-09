from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


SCHEMA_NAME = "natural_evidence_qwen_natural_e2e_zero_symbol_diagnosis_v1"


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[2] / candidate


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose zero-usable-symbol Qwen natural E2E eval artifacts. "
            "CPU-only; reads existing artifacts and does not train, generate, "
            "or make payload-recovery claims."
        )
    )
    parser.add_argument("--eval-dir", default="")
    parser.add_argument("--bucket-observations", default="")
    parser.add_argument("--decode-trace-csv", default="")
    parser.add_argument("--generated-outputs", default="")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--train-data-dir", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args(argv)


def _resolve_artifact(args: argparse.Namespace, attr: str, eval_dir: Path | None, default_name: str) -> Path:
    raw = str(getattr(args, attr, "") or "")
    if raw:
        return resolve_repo_path(raw)
    if eval_dir is None:
        raise ValueError(f"Either --eval-dir or --{attr.replace('_', '-')} is required")
    return eval_dir / default_name


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _nonempty(value: Any) -> bool:
    return str(value) != ""


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if str(value) == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _distribution(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _numeric_summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": min(values), "max": max(values), "mean": float(mean(values))}


def _group_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model_condition", "")),
        str(row.get("payload_id", "")),
        str(row.get("seed", "")),
        str(row.get("observation_condition", "")),
    )


def _decode_group_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model_condition", "")),
        str(row.get("payload_id", "")),
        str(row.get("seed", "")),
        str(row.get("far_family", "")),
    )


def _frame_stats_for_budget(rows: Sequence[Mapping[str, Any]], budget: int) -> dict[str, int]:
    budget_rows = [row for row in rows if _as_int(row.get("query_index", 0)) < int(budget)]
    frames: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    unframed_digit_rows = 0
    for row in budget_rows:
        if not _nonempty(row.get("digit", "")):
            continue
        if _nonempty(row.get("frame_index", "")):
            frames[_as_int(row.get("frame_index"))].append(row)
        else:
            unframed_digit_rows += 1

    complete_frames = 0
    incomplete_frames_with_hits = 0
    complete_frame_symbols = 0
    partial_symbol_hits = 0
    for frame_rows in frames.values():
        expected = max(_as_int(row.get("frame_digit_count", 0)) for row in frame_rows)
        observed_digits = {
            _as_int(row.get("frame_digit_index", index))
            for index, row in enumerate(frame_rows)
            if _nonempty(row.get("digit", ""))
        }
        observed_count = len(observed_digits)
        if expected and observed_count >= expected:
            complete_frames += 1
            complete_frame_symbols += observed_count
        elif observed_count:
            incomplete_frames_with_hits += 1
            partial_symbol_hits += observed_count

    return {
        "budget": int(budget),
        "eligible_positions": len(budget_rows),
        "bucket_hit_rows": sum(1 for row in budget_rows if _nonempty(row.get("bucket_id", ""))),
        "digit_rows": sum(1 for row in budget_rows if _nonempty(row.get("digit", ""))),
        "frames_seen": len(frames),
        "complete_frames": complete_frames,
        "incomplete_frames_with_hits": incomplete_frames_with_hits,
        "complete_frame_symbols": complete_frame_symbols,
        "partial_symbol_hits": partial_symbol_hits,
        "unframed_digit_rows": unframed_digit_rows,
    }


def _frame_assembly_summary(
    observations: Sequence[Mapping[str, Any]],
    decode_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    budgets = sorted({_as_int(row.get("query_budget", 0)) for row in decode_rows if _as_int(row.get("query_budget", 0)) > 0})
    if not budgets:
        budgets = sorted({_as_int(row.get("query_index", 0)) + 1 for row in observations})[-1:]

    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[_group_key(row)].append(row)

    max_complete_frames = 0
    max_incomplete_frames_with_hits = 0
    max_partial_symbol_hits = 0
    groups_with_any_bucket_hit = 0
    group_examples: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        rows_sorted = sorted(
            rows,
            key=lambda row: (
                _as_int(row.get("query_index", 0)),
                _as_int(row.get("position_index", 0)),
                str(row.get("bank_entry_id", "")),
            ),
        )
        per_budget = [_frame_stats_for_budget(rows_sorted, budget) for budget in budgets]
        best = max(
            per_budget,
            key=lambda item: (
                item["complete_frames"],
                item["partial_symbol_hits"],
                item["bucket_hit_rows"],
            ),
        )
        if best["bucket_hit_rows"] > 0:
            groups_with_any_bucket_hit += 1
        max_complete_frames = max(max_complete_frames, int(best["complete_frames"]))
        max_incomplete_frames_with_hits = max(
            max_incomplete_frames_with_hits,
            int(best["incomplete_frames_with_hits"]),
        )
        max_partial_symbol_hits = max(max_partial_symbol_hits, int(best["partial_symbol_hits"]))
        if len(group_examples) < 12 and (
            best["bucket_hit_rows"] > 0 or best["incomplete_frames_with_hits"] > 0
        ):
            model_condition, payload_id, seed, observation_condition = key
            group_examples.append(
                {
                    "model_condition": model_condition,
                    "payload_id": payload_id,
                    "seed": seed,
                    "observation_condition": observation_condition,
                    "best_budget": best["budget"],
                    "bucket_hit_rows": best["bucket_hit_rows"],
                    "complete_frames": best["complete_frames"],
                    "incomplete_frames_with_hits": best["incomplete_frames_with_hits"],
                    "partial_symbol_hits": best["partial_symbol_hits"],
                }
            )

    return {
        "query_budgets": budgets,
        "observation_group_count": len(grouped),
        "groups_with_any_bucket_hit": groups_with_any_bucket_hit,
        "max_complete_frames_in_any_group": max_complete_frames,
        "max_incomplete_frames_with_hits_in_any_group": max_incomplete_frames_with_hits,
        "max_partial_symbol_hits_in_any_group": max_partial_symbol_hits,
        "example_groups_with_hits": group_examples,
    }


def _decode_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("decode_status", "")) for row in rows)
    accepted_count = sum(1 for row in rows if str(row.get("accepted", "")).lower() in {"true", "1", "yes"})
    observed_symbols = [_as_int(row.get("observed_symbols", 0)) for row in rows]
    usable_symbols = [_as_int(row.get("usable_symbols", 0)) for row in rows]
    grouped = Counter(_decode_group_key(row) for row in rows)
    return {
        "decode_row_count": len(rows),
        "decode_status_counts": _distribution(status_counts),
        "accepted_count": accepted_count,
        "rows_with_observed_symbols": sum(1 for value in observed_symbols if value > 0),
        "rows_with_usable_symbols": sum(1 for value in usable_symbols if value > 0),
        "observed_symbols": _numeric_summary(observed_symbols),
        "usable_symbols": _numeric_summary(usable_symbols),
        "decode_group_count": len(grouped),
    }


def _observation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    erasure_reasons = Counter(str(row.get("erasure_reason", "")) for row in rows if str(row.get("erasure_reason", "")))
    model_conditions = Counter(str(row.get("model_condition", "")) for row in rows)
    observation_conditions = Counter(str(row.get("observation_condition", "")) for row in rows)
    bucket_hits = sum(1 for row in rows if _nonempty(row.get("bucket_id", "")))
    digit_hits = sum(1 for row in rows if _nonempty(row.get("digit", "")) and _nonempty(row.get("radix", "")))
    observed_token_present = sum(1 for row in rows if _nonempty(row.get("observed_token_id", "")))
    compatible_lengths = [
        len(row.get("compatible_bucket_ids", []))
        for row in rows
        if isinstance(row.get("compatible_bucket_ids", []), list)
    ]
    return {
        "observation_row_count": len(rows),
        "observed_token_present_count": observed_token_present,
        "bucket_hit_rows": bucket_hits,
        "digit_hit_rows": digit_hits,
        "bucket_hit_rate": bucket_hits / len(rows) if rows else 0.0,
        "digit_hit_rate": digit_hits / len(rows) if rows else 0.0,
        "erasure_reason_counts": _distribution(erasure_reasons),
        "model_condition_counts": _distribution(model_conditions),
        "observation_condition_counts": _distribution(observation_conditions),
        "compatible_bucket_count": _numeric_summary(compatible_lengths),
    }


def _generated_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"generated_output_count": 0, "artifact_present": False}
    rows = read_jsonl(path)
    response_lengths = [len(str(row.get("response_text", ""))) for row in rows]
    return {
        "artifact_present": True,
        "generated_output_count": len(rows),
        "empty_response_count": sum(1 for value in response_lengths if value == 0),
        "response_chars": _numeric_summary(response_lengths),
    }


def _train_contract_summary(train_data_dir: Path) -> dict[str, Any]:
    if not train_data_dir:
        return {"artifact_present": False}
    if not train_data_dir.is_dir():
        return {"artifact_present": False, "missing_train_data_dir": str(train_data_dir)}
    payloads: list[dict[str, Any]] = []
    for contract_path in sorted(train_data_dir.glob("*/variable_radix_train_contract.json")):
        contract = _read_json_if_present(contract_path)
        payloads.append(
            {
                "payload_id": str(contract.get("payload_id", contract_path.parent.name)),
                "example_count": _as_int(contract.get("example_count", 0)),
                "evidence_example_count": _as_int(contract.get("evidence_example_count", 0)),
                "total_eligible_positions": _as_int(contract.get("total_eligible_positions", 0)),
                "variable_radix_frame_count": _as_int(contract.get("variable_radix_frame_count", 0)),
                "variable_radix_used_positions": _as_int(contract.get("variable_radix_used_positions", 0)),
                "variable_radix_min_positions_satisfied": bool(contract.get("variable_radix_min_positions_satisfied", False)),
            }
        )
    return {
        "artifact_present": bool(payloads),
        "payload_contracts": payloads,
    }


def _classify(
    *,
    observations: Mapping[str, Any],
    decode: Mapping[str, Any],
    frame_assembly: Mapping[str, Any],
) -> dict[str, str]:
    observation_rows = int(observations["observation_row_count"])
    bucket_hits = int(observations["bucket_hit_rows"])
    complete_frames = int(frame_assembly["max_complete_frames_in_any_group"])
    partial_hits = int(frame_assembly["max_partial_symbol_hits_in_any_group"])
    erasure_counts = dict(observations["erasure_reason_counts"])
    decode_status_counts = dict(decode["decode_status_counts"])
    all_decode_insufficient = (
        bool(decode_status_counts)
        and set(decode_status_counts) == {"insufficient_symbols"}
    )

    if observation_rows == 0:
        stage = "missing_observation_artifact_or_empty_eval"
        interpretation = "No observation rows were available, so the failure is before decode."
    elif bucket_hits == 0:
        if erasure_counts.get("token_index_out_of_response", 0) >= max(1, int(0.9 * observation_rows)):
            stage = "observation_anchor_out_of_response"
            interpretation = (
                "Anchored token indexes are usually beyond generated responses; inspect response length "
                "and train-time/eval-time token index semantics."
            )
        elif erasure_counts.get("observed_token_not_in_variable_radix_bucket_set", 0) >= max(1, int(0.9 * observation_rows)):
            stage = "observation_bucket_miss"
            interpretation = (
                "The evaluator observed tokens at anchored positions, but none landed in the "
                "variable-radix compatible bucket sets."
            )
        else:
            stage = "observation_extraction_no_bucket_hits"
            interpretation = "Observation extraction produced no bucket hits; inspect erasure reasons first."
    elif complete_frames == 0 and all_decode_insufficient:
        stage = "decode_frame_assembly_incomplete"
        interpretation = (
            "At least one bucket/digit was observed, but no observation group completed a "
            "variable-radix frame, so the shared decoder reports insufficient_symbols."
        )
    elif complete_frames > 0 and all_decode_insufficient:
        stage = "decoder_status_mismatch"
        interpretation = (
            "Complete frames appear present while decode rows still report insufficient_symbols; "
            "inspect evaluate_diagnostic_e2e._decode_variable_radix_observations."
        )
    else:
        stage = "decoded_no_accept_or_mixed_status"
        interpretation = "The zero-symbol condition is not uniform after artifact-level inspection."

    if observation_rows == 0:
        next_step = "Point the diagnosis at a completed eval directory with observation and decode artifacts."
    elif partial_hits and complete_frames == 0:
        next_step = "Inspect partial-hit examples before changing training; the current decode trace hides partial frame evidence in usable_symbols."
    elif bucket_hits == 0:
        next_step = "Inspect observed_token_text/id and erasure distribution; do not rerun until anchor and bucket-map mismatch are separated."
    else:
        next_step = "Inspect recovered frame payloads and expected payload matching before any new training."

    return {
        "failure_stage_classification": stage,
        "interpretation": interpretation,
        "next_engineering_step": next_step,
    }


def build_diagnosis(
    *,
    bucket_observations_path: Path,
    decode_trace_path: Path,
    generated_outputs_path: Path,
    summary_path: Path,
    train_data_dir: Path | None,
) -> dict[str, Any]:
    observations = read_jsonl(bucket_observations_path) if bucket_observations_path.is_file() else []
    decode_rows = _read_csv(decode_trace_path)
    observation = _observation_summary(observations)
    decode = _decode_summary(decode_rows)
    frame_assembly = _frame_assembly_summary(observations, decode_rows)
    classification = _classify(
        observations=observation,
        decode=decode,
        frame_assembly=frame_assembly,
    )
    summary = _read_json_if_present(summary_path)
    train_contracts = _train_contract_summary(train_data_dir) if train_data_dir is not None else {"artifact_present": False}
    return {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_DIAGNOSIS_NOT_PAYLOAD_RECOVERY",
        "paper_claim_allowed": False,
        "training_started": False,
        "provider_api_called": False,
        "not_full_far": True,
        "artifacts": {
            "bucket_observations": str(bucket_observations_path),
            "decode_trace_csv": str(decode_trace_path),
            "generated_outputs": str(generated_outputs_path),
            "summary_json": str(summary_path),
            "train_data_dir": "" if train_data_dir is None else str(train_data_dir),
        },
        "eval_summary_status": str(summary.get("status", "")),
        "eval_summary_counts": {
            "generated_output_count": _as_int(summary.get("generated_output_count", 0)),
            "observation_count": _as_int(summary.get("observation_count", 0)),
            "decode_row_count": _as_int(summary.get("decode_row_count", 0)),
            "protected_accept_count": _as_int(summary.get("protected_accept_count", 0)),
            "null_accept_count": _as_int(summary.get("null_accept_count", 0)),
        },
        "generated_outputs": _generated_summary(generated_outputs_path),
        "observations": observation,
        "decode_trace": decode,
        "frame_assembly": frame_assembly,
        "train_contracts": train_contracts,
        **classification,
        "result_claim": "qwen_natural_e2e_zero_symbol_diagnosis_not_payload_recovery",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    eval_dir = resolve_repo_path(args.eval_dir) if args.eval_dir else None
    bucket_observations = _resolve_artifact(
        args,
        "bucket_observations",
        eval_dir,
        "qwen_natural_e2e_bucket_observations.jsonl",
    )
    decode_trace = _resolve_artifact(
        args,
        "decode_trace_csv",
        eval_dir,
        "qwen_natural_e2e_decode_trace.csv",
    )
    generated_outputs = _resolve_artifact(
        args,
        "generated_outputs",
        eval_dir,
        "qwen_natural_e2e_generated_outputs.jsonl",
    )
    summary_json = _resolve_artifact(
        args,
        "summary_json",
        eval_dir,
        "qwen_natural_e2e_eval_summary.json",
    )
    train_data_dir = resolve_repo_path(args.train_data_dir) if args.train_data_dir else None

    diagnosis = build_diagnosis(
        bucket_observations_path=bucket_observations,
        decode_trace_path=decode_trace,
        generated_outputs_path=generated_outputs,
        summary_path=summary_json,
        train_data_dir=train_data_dir,
    )
    if args.output_json:
        output_path = resolve_repo_path(args.output_json)
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing diagnosis: {output_path}")
        write_json(output_path, diagnosis)
    print(json.dumps(diagnosis, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
