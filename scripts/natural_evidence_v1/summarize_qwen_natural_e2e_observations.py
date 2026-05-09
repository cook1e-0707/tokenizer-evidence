from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.natural_evidence_v1.common import write_json


SCHEMA_NAME = "natural_evidence_qwen_observation_erasure_summary_v1"
OBSERVATION_JOIN_KEY_FIELDS = [
    "model_condition",
    "payload_id",
    "seed",
    "observation_condition",
    "query_index",
    "prompt_id",
    "position_index",
    "frame_index",
    "frame_digit_index",
]
DECODE_TRACE_JOIN_KEY_FIELDS = [
    "model_condition",
    "payload_id",
    "seed",
    "far_family",
    "query_budget",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize 846699 Qwen natural E2E bucket observation provenance. "
            "This is an artifact-only diagnostic: it streams existing JSONL/CSV "
            "artifacts, computes hashes and counts, and never trains or evaluates a model."
        )
    )
    parser.add_argument("--observations-jsonl", required=True)
    parser.add_argument("--decode-trace-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--progress-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--source-remote-path", required=True)
    parser.add_argument("--path-explanation", default="")
    parser.add_argument("--expected-observation-count", type=int, default=0)
    parser.add_argument("--expected-decode-row-count", type=int, default=0)
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


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


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


def _sample_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = [
        "schema_name",
        "model_condition",
        "observation_condition",
        "payload_id",
        "seed",
        "query_index",
        "prompt_id",
        "position_index",
        "token_index",
        "frame_index",
        "frame_digit_index",
        "frame_digit_count",
        "bucket_id",
        "digit",
        "radix",
        "erasure",
        "erasure_reason",
        "observed_token_id",
        "observed_token_text",
        "compatible_bucket_ids",
        "bank_entry_id",
        "entry_key",
    ]
    return {field: row.get(field, "") for field in fields}


def _observation_group_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model_condition", "")),
        str(row.get("payload_id", "")),
        str(row.get("seed", "")),
        str(row.get("observation_condition", "")),
    )


def _condition_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("model_condition", "")),
        str(row.get("observation_condition", "")),
    )


def summarize_observations(path: Path, max_examples: int = 8) -> dict[str, Any]:
    digest = hashlib.sha256()
    row_count = 0
    parse_errors: list[dict[str, Any]] = []
    schema_counts: Counter[str] = Counter()
    model_condition_counts: Counter[str] = Counter()
    observation_condition_counts: Counter[str] = Counter()
    condition_pair_counts: Counter[tuple[str, str]] = Counter()
    payload_counts: Counter[str] = Counter()
    seed_counts: Counter[str] = Counter()
    erasure_reason_counts: Counter[str] = Counter()
    anchor_policy_counts: Counter[str] = Counter()
    frame_digit_count_counts: Counter[int] = Counter()
    observed_token_present = 0
    erasure_flag_true = 0
    erasure_reason_rows = 0
    bucket_hit_rows = 0
    digit_hit_rows = 0
    prompt_ids: set[str] = set()
    frame_indices: set[int] = set()
    entry_keys: set[str] = set()
    observation_groups: Counter[tuple[str, str, str, str]] = Counter()
    per_condition = defaultdict(
        lambda: {
            "rows": 0,
            "erasure_reason_rows": 0,
            "bucket_hit_rows": 0,
            "digit_hit_rows": 0,
            "observed_token_present_rows": 0,
        }
    )
    erased_examples: list[dict[str, Any]] = []
    digit_examples: list[dict[str, Any]] = []

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
                parse_errors.append({"line_number": line_number, "error": "row is not a JSON object"})
                continue

            row_count += 1
            schema_counts[str(row.get("schema_name", ""))] += 1
            model_condition_counts[str(row.get("model_condition", ""))] += 1
            observation_condition_counts[str(row.get("observation_condition", ""))] += 1
            condition_pair_counts[_condition_key(row)] += 1
            payload_counts[str(row.get("payload_id", ""))] += 1
            seed_counts[str(row.get("seed", ""))] += 1
            if _nonempty(row.get("anchor_policy", "")):
                anchor_policy_counts[str(row.get("anchor_policy", ""))] += 1
            if _nonempty(row.get("frame_digit_count", "")):
                frame_digit_count_counts[_as_int(row.get("frame_digit_count", 0))] += 1
            if _nonempty(row.get("prompt_id", "")):
                prompt_ids.add(str(row.get("prompt_id", "")))
            if _nonempty(row.get("entry_key", "")):
                entry_keys.add(str(row.get("entry_key", "")))
            if _nonempty(row.get("frame_index", "")):
                frame_indices.add(_as_int(row.get("frame_index", 0)))
            observation_groups[_observation_group_key(row)] += 1

            condition = per_condition["|".join(_condition_key(row))]
            condition["rows"] += 1
            if _nonempty(row.get("observed_token_id", "")):
                observed_token_present += 1
                condition["observed_token_present_rows"] += 1
            if bool(row.get("erasure", False)):
                erasure_flag_true += 1
            erasure_reason = str(row.get("erasure_reason", ""))
            if erasure_reason:
                erasure_reason_counts[erasure_reason] += 1
                erasure_reason_rows += 1
                condition["erasure_reason_rows"] += 1
                if len(erased_examples) < max_examples:
                    erased_examples.append(_sample_payload(row))
            if _nonempty(row.get("bucket_id", "")):
                bucket_hit_rows += 1
                condition["bucket_hit_rows"] += 1
            if _nonempty(row.get("digit", "")) and _nonempty(row.get("radix", "")):
                digit_hit_rows += 1
                condition["digit_hit_rows"] += 1
                if len(digit_examples) < max_examples:
                    digit_examples.append(_sample_payload(row))

    per_condition_summary = {}
    for key, values in sorted(per_condition.items()):
        rows = int(values["rows"])
        per_condition_summary[key] = {
            **{field: int(value) for field, value in values.items()},
            "erasure_reason_rate": _rate(int(values["erasure_reason_rows"]), rows),
            "bucket_hit_rate": _rate(int(values["bucket_hit_rows"]), rows),
            "digit_hit_rate": _rate(int(values["digit_hit_rows"]), rows),
        }

    return {
        "artifact": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        },
        "row_count": row_count,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:max_examples],
        "schema_counts": _counter_dict(schema_counts),
        "model_condition_counts": _counter_dict(model_condition_counts),
        "observation_condition_counts": _counter_dict(observation_condition_counts),
        "model_observation_condition_counts": {
            f"{model_condition}|{observation_condition}": int(count)
            for (model_condition, observation_condition), count in sorted(condition_pair_counts.items())
        },
        "payload_counts": _counter_dict(payload_counts),
        "seed_counts": _counter_dict(seed_counts),
        "anchor_policy_counts": _counter_dict(anchor_policy_counts),
        "frame_digit_count_counts": _counter_dict(frame_digit_count_counts),
        "distinct_prompt_ids": len(prompt_ids),
        "distinct_frame_indices": len(frame_indices),
        "distinct_entry_keys": len(entry_keys),
        "observation_group_count": len(observation_groups),
        "observation_group_key_fields": ["model_condition", "payload_id", "seed", "observation_condition"],
        "observation_group_counts": {
            "|".join(key): int(count) for key, count in sorted(observation_groups.items())
        },
        "observed_token_present_rows": observed_token_present,
        "erasure_flag_true_rows": erasure_flag_true,
        "erasure_reason_rows": erasure_reason_rows,
        "bucket_hit_rows": bucket_hit_rows,
        "compatible_variable_radix_digit_rows": digit_hit_rows,
        "erasure_reason_rate": _rate(erasure_reason_rows, row_count),
        "bucket_hit_rate": _rate(bucket_hit_rows, row_count),
        "compatible_variable_radix_digit_rate": _rate(digit_hit_rows, row_count),
        "erasure_reason_counts": _counter_dict(erasure_reason_counts),
        "per_model_observation_condition": per_condition_summary,
        "example_erasure_rows": erased_examples,
        "example_digit_rows": digit_examples,
    }


def _read_csv_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    artifact = _hash_file(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return artifact, rows


def summarize_decode_trace(path: Path) -> dict[str, Any]:
    artifact, rows = _read_csv_rows(path)
    status_counts = Counter(str(row.get("decode_status", "")) for row in rows)
    model_condition_counts = Counter(str(row.get("model_condition", "")) for row in rows)
    far_family_counts = Counter(str(row.get("far_family", "")) for row in rows)
    budget_counts = Counter(str(row.get("query_budget", "")) for row in rows)
    accepted_rows = sum(1 for row in rows if str(row.get("accepted", "")).lower() in {"true", "1", "yes"})
    observed_values = [_as_int(row.get("observed_symbols", 0)) for row in rows]
    usable_values = [_as_int(row.get("usable_symbols", 0)) for row in rows]
    eligible_values = [_as_int(row.get("eligible_positions", 0)) for row in rows]
    decode_groups = Counter(
        (
            str(row.get("model_condition", "")),
            str(row.get("payload_id", "")),
            str(row.get("seed", "")),
            str(row.get("far_family", "")),
        )
        for row in rows
    )
    return {
        "artifact": artifact,
        "row_count": len(rows),
        "decode_trace_join_key_fields": DECODE_TRACE_JOIN_KEY_FIELDS,
        "decode_group_key_fields": ["model_condition", "payload_id", "seed", "far_family"],
        "decode_group_count": len(decode_groups),
        "decode_group_counts": {"|".join(key): int(count) for key, count in sorted(decode_groups.items())},
        "decode_status_counts": _counter_dict(status_counts),
        "model_condition_counts": _counter_dict(model_condition_counts),
        "far_family_counts": _counter_dict(far_family_counts),
        "query_budget_counts": _counter_dict(budget_counts),
        "accepted_rows": accepted_rows,
        "rows_with_observed_symbols": sum(1 for value in observed_values if value > 0),
        "rows_with_usable_symbols": sum(1 for value in usable_values if value > 0),
        "max_observed_symbols": max(observed_values) if observed_values else 0,
        "max_usable_symbols": max(usable_values) if usable_values else 0,
        "eligible_positions_total": sum(eligible_values),
        "observed_symbols_total": sum(observed_values),
        "usable_symbols_total": sum(usable_values),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _job_ids_in_path(path: str) -> list[str]:
    return sorted(set(re.findall(r"(?<!\d)(\d{6})(?!\d)", path)))


def _count_mismatches(
    *,
    eval_summary: Mapping[str, Any],
    observations: Mapping[str, Any],
    decode_trace: Mapping[str, Any],
    expected_observation_count: int,
    expected_decode_row_count: int,
) -> list[str]:
    mismatches: list[str] = []
    observation_count = int(observations["row_count"])
    decode_count = int(decode_trace["row_count"])
    summary_observation_count = _as_int(eval_summary.get("observation_count", 0))
    summary_decode_count = _as_int(eval_summary.get("decode_row_count", 0))
    if summary_observation_count and summary_observation_count != observation_count:
        mismatches.append(
            f"summary observation_count {summary_observation_count} != observations row_count {observation_count}"
        )
    if summary_decode_count and summary_decode_count != decode_count:
        mismatches.append(f"summary decode_row_count {summary_decode_count} != decode trace row_count {decode_count}")
    if expected_observation_count and expected_observation_count != observation_count:
        mismatches.append(
            f"expected observation count {expected_observation_count} != observations row_count {observation_count}"
        )
    if expected_decode_row_count and expected_decode_row_count != decode_count:
        mismatches.append(f"expected decode row count {expected_decode_row_count} != decode trace row_count {decode_count}")
    return mismatches


def _provenance_status(
    *,
    source_job_id: str,
    source_remote_path: str,
    path_explanation: str,
    mismatches: Iterable[str],
    parse_error_count: int,
) -> str:
    if parse_error_count:
        return "PROVISIONAL_PARSE_ERRORS_PRESENT"
    if list(mismatches):
        return "PROVISIONAL_COUNT_MISMATCH"
    path_job_ids = _job_ids_in_path(source_remote_path)
    if source_job_id in path_job_ids:
        return "PASS_SOURCE_JOB_ID_IN_PATH"
    if path_job_ids and path_explanation:
        return "PASS_EXPLAINED_RECOVERY_DIR_NAME"
    if path_job_ids:
        return "PROVISIONAL_SOURCE_PATH_JOB_ID_MISMATCH"
    return "PASS_NO_JOB_ID_IN_SOURCE_PATH"


def build_summary(
    *,
    observations_jsonl: Path,
    decode_trace_csv: Path,
    summary_json: Path,
    progress_json: Path | None,
    source_job_id: str,
    source_remote_path: str,
    path_explanation: str,
    expected_observation_count: int,
    expected_decode_row_count: int,
) -> dict[str, Any]:
    observations = summarize_observations(observations_jsonl)
    decode_trace = summarize_decode_trace(decode_trace_csv)
    eval_summary = _read_json(summary_json)
    progress = _read_json(progress_json) if progress_json is not None and progress_json.is_file() else {}
    mismatches = _count_mismatches(
        eval_summary=eval_summary,
        observations=observations,
        decode_trace=decode_trace,
        expected_observation_count=expected_observation_count,
        expected_decode_row_count=expected_decode_row_count,
    )
    source_path_job_ids = _job_ids_in_path(source_remote_path)
    status = _provenance_status(
        source_job_id=source_job_id,
        source_remote_path=source_remote_path,
        path_explanation=path_explanation,
        mismatches=mismatches,
        parse_error_count=int(observations["parse_error_count"]),
    )
    return {
        "schema_name": SCHEMA_NAME,
        "status": status,
        "source_job_id": str(source_job_id),
        "source_remote_path": source_remote_path,
        "source_path_job_id_candidates": source_path_job_ids,
        "path_explanation": path_explanation,
        "provenance_mismatches": mismatches,
        "local_copy_paths": {
            "observations_jsonl": str(observations_jsonl),
            "decode_trace_csv": str(decode_trace_csv),
            "summary_json": str(summary_json),
            "progress_json": "" if progress_json is None else str(progress_json),
        },
        "claim_control": {
            "paper_claim_allowed": False,
            "training_started": False,
            "e2e_eval_started": False,
            "not_full_far": True,
            "result_claim": "artifact_provenance_diagnostic_not_payload_recovery_not_far",
        },
        "decode_trace_join_key": {
            "observation_fields": OBSERVATION_JOIN_KEY_FIELDS,
            "decode_trace_fields": DECODE_TRACE_JOIN_KEY_FIELDS,
            "group_level_note": (
                "Observations group by model_condition/payload_id/seed/observation_condition; "
                "decode rows group by model_condition/payload_id/seed/far_family/query_budget. "
                "Frame-level diagnostics must replay frame_index/frame_digit_index from the observation rows."
            ),
        },
        "eval_summary_counts": {
            "status": str(eval_summary.get("status", "")),
            "generated_output_count": _as_int(eval_summary.get("generated_output_count", 0)),
            "observation_count": _as_int(eval_summary.get("observation_count", 0)),
            "decode_row_count": _as_int(eval_summary.get("decode_row_count", 0)),
            "protected_accept_count": _as_int(eval_summary.get("protected_accept_count", 0)),
            "null_accept_count": _as_int(eval_summary.get("null_accept_count", 0)),
            "diagnostic_recovery_observed": bool(eval_summary.get("diagnostic_recovery_observed", False)),
            "null_accept_observed": bool(eval_summary.get("null_accept_observed", False)),
            "paper_claim_allowed": bool(eval_summary.get("paper_claim_allowed", False)),
        },
        "progress_status": {
            "artifact_present": bool(progress),
            "status": str(progress.get("status", "")),
            "stage": str(progress.get("stage", "")),
            "completed_units": progress.get("completed_units", []),
        },
        "observations": observations,
        "decode_trace": decode_trace,
        "next_required_diagnostic": "frame_completion_replay",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_json = _resolve(args.output_json)
    if output_json.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing summary: {output_json}")
    summary = build_summary(
        observations_jsonl=_resolve(args.observations_jsonl),
        decode_trace_csv=_resolve(args.decode_trace_csv),
        summary_json=_resolve(args.summary_json),
        progress_json=_resolve(args.progress_json) if args.progress_json else None,
        source_job_id=str(args.source_job_id),
        source_remote_path=str(args.source_remote_path),
        path_explanation=str(args.path_explanation),
        expected_observation_count=int(args.expected_observation_count),
        expected_decode_row_count=int(args.expected_decode_row_count),
    )
    write_json(output_json, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
