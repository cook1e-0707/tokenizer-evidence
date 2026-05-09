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

from scripts.natural_evidence_v1.common import read_jsonl, resolve_repo_path, write_csv, write_json, write_jsonl


SCHEMA_NAME = "natural_evidence_verifier_alignment_diagnosis_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose verifier/reference-prefix alignment for retained Qwen diagnostic "
            "E2E artifacts. CPU-only; does not train, generate, or claim recovery."
        )
    )
    parser.add_argument("--generated-outputs", required=True)
    parser.add_argument("--bucket-observations", required=True)
    parser.add_argument("--decode-trace-csv", default="")
    parser.add_argument("--progress-json", default="")
    parser.add_argument("--bucket-bank-entries", required=True)
    parser.add_argument("--compatibility-jsonl", default="")
    parser.add_argument("--compatibility-by-entry-csv", default="")
    parser.add_argument("--bucket-count", type=int, default=4)
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=80)
    return parser.parse_args(argv)


class SimpleWhitespaceTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        tokens = text.split()
        ids: list[int] = []
        for token in tokens:
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(self._token_to_id[token])
        return {"input_ids": ids}

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return " ".join(self._id_to_token.get(int(token_id), f"<id:{token_id}>") for token_id in token_ids)


def _load_tokenizer(tokenizer_name: str) -> Any:
    if tokenizer_name == "__simple_whitespace_test__":
        return SimpleWhitespaceTokenizer()
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("diagnosis requires transformers unless using __simple_whitespace_test__") from error
    return AutoTokenizer.from_pretrained(tokenizer_name)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded.get("input_ids", [])
    if not isinstance(token_ids, list):
        return []
    return [int(token_id) for token_id in token_ids]


def _decode_token(tokenizer: Any, token_id: Any) -> str:
    if token_id in {None, ""}:
        return ""
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _lcp_length(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


def _entry_token_index(entry: Mapping[str, Any]) -> int:
    for key in ("prefix_response_token_count", "token_index", "token_position", "prefix_token_count"):
        if key in entry and str(entry.get(key, "")) != "":
            return int(entry[key])
    return 0


def _token_to_bucket(entry: Mapping[str, Any]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for bucket_id, token_ids in dict(entry.get("buckets", {})).items():
        for token_id in token_ids:
            mapping[int(token_id)] = int(bucket_id)
    return mapping


def _min1_entry_ids(rows: Sequence[Mapping[str, str]]) -> set[str]:
    return {
        str(row.get("bank_entry_id", ""))
        for row in rows
        if row.get("bank_entry_id") and _as_bool(row.get("would_accept_min1", ""))
    }


def _compatible_token_maps(
    *,
    entries_by_id: Mapping[str, Mapping[str, Any]],
    compatibility_rows: Sequence[Mapping[str, Any]],
    min1_entry_ids: set[str],
    bucket_count: int,
) -> dict[str, dict[int, int]]:
    allowed: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in compatibility_rows:
        entry_id = str(row.get("bank_entry_id", ""))
        if entry_id not in min1_entry_ids or not _as_bool(row.get("compatibility_pass", False)):
            continue
        bucket_id = str(row.get("bucket_id", ""))
        if bucket_id == "":
            continue
        allowed[entry_id][bucket_id].add(int(row["token_id"]))

    maps: dict[str, dict[int, int]] = {}
    for entry_id, entry in entries_by_id.items():
        bucket_allowed = allowed.get(entry_id)
        if not bucket_allowed:
            continue
        token_map: dict[int, int] = {}
        buckets = dict(entry.get("buckets", {}))
        for bucket_id in range(bucket_count):
            bucket_key = str(bucket_id)
            for token_id in buckets.get(bucket_key, []):
                if int(token_id) in bucket_allowed.get(bucket_key, set()):
                    token_map[int(token_id)] = bucket_id
        if token_map:
            maps[entry_id] = token_map
    return maps


def _load_relevant_entries(path: Path, relevant_ids: set[str]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            entry_id = str(row.get("bank_entry_id", ""))
            if entry_id in relevant_ids:
                entries[entry_id] = row
    return entries


def _generated_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(row.get("model_condition", "")),
        str(row.get("payload_id", "")),
        str(row.get("seed", "")),
        str(row.get("prompt_id", "")),
        int(row.get("query_index", 0)),
    )


def _observation_generated_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(row.get("model_condition", "")),
        str(row.get("payload_id", "")),
        str(row.get("seed", "")),
        str(row.get("prompt_id", "")),
        int(row.get("query_index", 0)),
    )


def _group_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model_condition", "")),
        str(row.get("payload_id", "")),
        str(row.get("seed", "")),
        str(row.get("observation_condition", "")),
    )


def _prompt_group_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    model_condition, payload_id, seed, observation_condition = _group_key(row)
    return (str(row.get("prompt_id", "")), model_condition, payload_id, seed, observation_condition)


def _empty_metrics(reason: str) -> dict[str, Any]:
    return {
        "diagnosis_error": reason,
        "prompt_prefix_match": False,
        "overall_lcp_tokens": 0,
        "overall_lcp_fraction": 0.0,
        "response_lcp_tokens": 0,
        "response_lcp_fraction": 0.0,
        "expected_response_prefix_len": 0,
        "actual_response_prefix_len": 0,
        "first_response_mismatch_index": "",
        "stale_bucket_if_ignore_strict": "",
        "strict_mismatch_token_bucket_available": False,
        "offset_out_of_response_recomputed": False,
    }


def _alignment_metrics(
    *,
    tokenizer: Any,
    generated: Mapping[str, Any] | None,
    observation: Mapping[str, Any],
    entry: Mapping[str, Any] | None,
    token_map: Mapping[int, int],
) -> dict[str, Any]:
    if generated is None:
        return _empty_metrics("generated_row_missing")
    if entry is None:
        return _empty_metrics("bank_entry_missing")

    prompt_ids = _token_ids(tokenizer, str(generated.get("prompt", "")))
    response_ids = _token_ids(tokenizer, str(generated.get("response_text", "")))
    offset = int(observation.get("prefix_response_token_count", _entry_token_index(entry)))
    reference_prefix = [int(token_id) for token_id in entry.get("prefix_token_ids", [])]
    reference_prompt_len = max(0, len(reference_prefix) - offset)
    reference_prompt_prefix = reference_prefix[:reference_prompt_len]
    reference_response_prefix = reference_prefix[reference_prompt_len:]
    actual_response_prefix = response_ids[:offset]
    actual_prefix = [*prompt_ids, *actual_response_prefix]

    overall_lcp = _lcp_length(actual_prefix, reference_prefix)
    response_lcp = _lcp_length(actual_response_prefix, reference_response_prefix)
    prompt_prefix_match = prompt_ids == reference_prompt_prefix
    first_response_mismatch_index: int | str = ""
    if response_lcp < max(len(actual_response_prefix), len(reference_response_prefix)):
        first_response_mismatch_index = response_lcp

    observed_token_id: int | None = None
    offset_out = offset >= len(response_ids)
    if not offset_out:
        observed_token_id = int(response_ids[offset])
    stale_bucket = token_map.get(observed_token_id) if observed_token_id is not None else None
    strict_match = _as_bool(observation.get("strict_prefix_match", False))

    return {
        "diagnosis_error": "",
        "prompt_prefix_match": prompt_prefix_match,
        "overall_lcp_tokens": overall_lcp,
        "overall_lcp_fraction": overall_lcp / max(1, len(reference_prefix)),
        "response_lcp_tokens": response_lcp,
        "response_lcp_fraction": response_lcp / max(1, len(reference_response_prefix)),
        "expected_response_prefix_len": len(reference_response_prefix),
        "actual_response_prefix_len": len(actual_response_prefix),
        "first_response_mismatch_index": first_response_mismatch_index,
        "stale_bucket_if_ignore_strict": "" if stale_bucket is None else int(stale_bucket),
        "strict_mismatch_token_bucket_available": (not strict_match) and stale_bucket is not None,
        "offset_out_of_response_recomputed": offset_out,
        "recomputed_observed_token_id": "" if observed_token_id is None else int(observed_token_id),
        "recomputed_observed_token_text": _decode_token(tokenizer, observed_token_id),
    }


def _decode_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"decode_trace_present": False}
    rows = _read_csv(path)
    accepted = sum(1 for row in rows if _as_bool(row.get("accepted", "")))
    return {
        "decode_trace_present": True,
        "decode_rows": len(rows),
        "accepted_rows": accepted,
        "decode_status_counts": dict(Counter(row.get("decode_status", "") for row in rows)),
        "condition_row_counts": dict(Counter(row.get("model_condition", "") for row in rows)),
        "query_budget_counts": dict(Counter(row.get("query_budget", "") for row in rows)),
    }


def _progress_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"progress_present": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "progress_present": True,
        "progress_status": payload.get("status", ""),
        "progress_stage": payload.get("stage", ""),
        "progress_generated_output_count": payload.get("generated_output_count", ""),
        "progress_observation_count": payload.get("observation_count", ""),
        "progress_decode_row_count": payload.get("decode_row_count", ""),
        "completed_units": payload.get("completed_units", []),
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows if row.get(key, "") != ""]
    return mean(values) if values else 0.0


def _summarize_group(key: tuple[str, ...], rows: Sequence[Mapping[str, Any]], *, prompt_level: bool) -> dict[str, Any]:
    total = len(rows)
    strict_matches = sum(1 for row in rows if _as_bool(row.get("strict_prefix_match", False)))
    observed_symbols = sum(1 for row in rows if str(row.get("bucket_id", "")) != "")
    erasures = sum(1 for row in rows if _as_bool(row.get("erasure", False)))
    reason_counts = Counter(str(row.get("erasure_reason", "")) for row in rows if str(row.get("erasure_reason", "")))
    prefix_mismatch = reason_counts.get("strict_prefix_mismatch", 0)
    output: dict[str, Any]
    if prompt_level:
        prompt_id, model_condition, payload_id, seed, observation_condition = key
        output = {
            "prompt_id": prompt_id,
            "model_condition": model_condition,
            "payload_id": payload_id,
            "seed": seed,
            "observation_condition": observation_condition,
        }
    else:
        model_condition, payload_id, seed, observation_condition = key
        output = {
            "model_condition": model_condition,
            "payload_id": payload_id,
            "seed": seed,
            "observation_condition": observation_condition,
        }
    output.update(
        {
            "observations": total,
            "strict_prefix_matches": strict_matches,
            "strict_prefix_mismatches": prefix_mismatch,
            "strict_prefix_match_rate": strict_matches / max(1, total),
            "observed_symbols": observed_symbols,
            "observed_symbol_rate": observed_symbols / max(1, total),
            "erasures": erasures,
            "erasure_rate": erasures / max(1, total),
            "offset_out_of_response": reason_counts.get("offset_out_of_response", 0),
            "observed_token_not_in_bucket_set": reason_counts.get("observed_token_not_in_bucket_set", 0),
            "prompt_prefix_mismatches": sum(1 for row in rows if not _as_bool(row.get("prompt_prefix_match", False))),
            "strict_mismatch_token_bucket_available": sum(
                1 for row in rows if _as_bool(row.get("strict_mismatch_token_bucket_available", False))
            ),
            "mean_response_lcp_tokens": _mean(rows, "response_lcp_tokens"),
            "mean_response_lcp_fraction": _mean(rows, "response_lcp_fraction"),
            "mean_expected_response_prefix_len": _mean(rows, "expected_response_prefix_len"),
            "diagnosis_error_rows": sum(1 for row in rows if str(row.get("diagnosis_error", ""))),
        }
    )
    return output


def run_diagnosis(
    *,
    generated_outputs_path: Path,
    observations_path: Path,
    decode_trace_path: Path | None,
    progress_path: Path | None,
    bucket_bank_entries_path: Path,
    compatibility_jsonl_path: Path | None,
    compatibility_by_entry_csv_path: Path | None,
    bucket_count: int,
    tokenizer_name: str,
    output_dir: Path,
    max_examples: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_rows = read_jsonl(generated_outputs_path)
    observation_rows = read_jsonl(observations_path)
    generated_by_key = {_generated_key(row): row for row in generated_rows}
    relevant_entry_ids = {str(row.get("bank_entry_id", "")) for row in observation_rows if row.get("bank_entry_id")}
    entries_by_id = _load_relevant_entries(bucket_bank_entries_path, relevant_entry_ids)

    if compatibility_jsonl_path is not None and compatibility_by_entry_csv_path is not None:
        min1_ids = _min1_entry_ids(_read_csv(compatibility_by_entry_csv_path))
        token_maps = _compatible_token_maps(
            entries_by_id=entries_by_id,
            compatibility_rows=read_jsonl(compatibility_jsonl_path),
            min1_entry_ids=min1_ids,
            bucket_count=bucket_count,
        )
        token_map_source = "compatibility_filtered_min1"
    else:
        token_maps = {entry_id: _token_to_bucket(entry) for entry_id, entry in entries_by_id.items()}
        token_map_source = "bank_entry_full_bucket_sets"

    tokenizer = _load_tokenizer(tokenizer_name)
    enriched_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for observation in observation_rows:
        entry_id = str(observation.get("bank_entry_id", ""))
        generated = generated_by_key.get(_observation_generated_key(observation))
        entry = entries_by_id.get(entry_id)
        metrics = _alignment_metrics(
            tokenizer=tokenizer,
            generated=generated,
            observation=observation,
            entry=entry,
            token_map=token_maps.get(entry_id, {}),
        )
        enriched = {**observation, **metrics}
        enriched_rows.append(enriched)
        if (
            len(example_rows) < max_examples
            and str(observation.get("erasure_reason", "")) == "strict_prefix_mismatch"
        ):
            example_rows.append(
                {
                    "model_condition": observation.get("model_condition", ""),
                    "payload_id": observation.get("payload_id", ""),
                    "seed": observation.get("seed", ""),
                    "prompt_id": observation.get("prompt_id", ""),
                    "query_index": observation.get("query_index", ""),
                    "bank_entry_id": entry_id,
                    "prefix_response_token_count": observation.get("prefix_response_token_count", ""),
                    "prompt_prefix_match": metrics["prompt_prefix_match"],
                    "response_lcp_tokens": metrics["response_lcp_tokens"],
                    "response_lcp_fraction": metrics["response_lcp_fraction"],
                    "expected_response_prefix_len": metrics["expected_response_prefix_len"],
                    "observed_token_id": observation.get("observed_token_id", ""),
                    "observed_token_text": observation.get("observed_token_text", ""),
                    "stale_bucket_if_ignore_strict": metrics["stale_bucket_if_ignore_strict"],
                    "response_excerpt": str((generated or {}).get("response_text", ""))[:320],
                    "result_claim": "alignment_example_not_payload_recovery",
                }
            )

    by_unit = [
        _summarize_group(key, rows, prompt_level=False)
        for key, rows in sorted(_collect_groups(enriched_rows, _group_key).items())
    ]
    by_prompt = [
        _summarize_group(key, rows, prompt_level=True)
        for key, rows in sorted(_collect_groups(enriched_rows, _prompt_group_key).items())
    ]
    by_prompt_sorted = sorted(
        by_prompt,
        key=lambda row: (
            -float(row["strict_prefix_mismatches"]),
            float(row["mean_response_lcp_fraction"]),
            str(row["prompt_id"]),
        ),
    )

    decode_summary = _decode_summary(decode_trace_path)
    progress_summary = _progress_summary(progress_path)
    persisted_observations = len(observation_rows)
    progress_observations = progress_summary.get("progress_observation_count", "")
    observation_persistence_gap = 0
    if str(progress_observations) not in {"", "None"}:
        observation_persistence_gap = int(progress_observations) - persisted_observations

    total_strict_mismatch = sum(1 for row in enriched_rows if str(row.get("erasure_reason", "")) == "strict_prefix_mismatch")
    total_observed = sum(1 for row in enriched_rows if str(row.get("bucket_id", "")) != "")
    protected_units = [
        row for row in by_unit if row["model_condition"] == "protected_trained"
    ]
    protected_observed_rate = _mean(protected_units, "observed_symbol_rate")
    strict_mismatch_rate = total_strict_mismatch / max(1, persisted_observations)
    if strict_mismatch_rate >= 0.5 and protected_observed_rate < 0.2:
        alignment_status = "FAIL_STRICT_PREFIX_ERASURE_DOMINATES"
    elif strict_mismatch_rate >= 0.25:
        alignment_status = "WARN_STRICT_PREFIX_ERASURE_HIGH"
    else:
        alignment_status = "PASS_PREFIX_ALIGNMENT_DIAGNOSTIC"

    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE",
        "alignment_status": alignment_status,
        "paper_claim_allowed": False,
        "tokenizer": tokenizer_name,
        "token_map_source": token_map_source,
        "generated_output_rows": len(generated_rows),
        "persisted_observation_rows": persisted_observations,
        "progress_observation_count": progress_observations,
        "observation_persistence_gap": observation_persistence_gap,
        "decode_summary": decode_summary,
        "progress_summary": progress_summary,
        "overall": {
            "strict_prefix_mismatches": total_strict_mismatch,
            "strict_prefix_mismatch_rate": strict_mismatch_rate,
            "observed_symbols": total_observed,
            "observed_symbol_rate": total_observed / max(1, persisted_observations),
            "strict_mismatch_token_bucket_available": sum(
                1 for row in enriched_rows if _as_bool(row.get("strict_mismatch_token_bucket_available", False))
            ),
            "prompt_prefix_mismatches": sum(1 for row in enriched_rows if not _as_bool(row.get("prompt_prefix_match", False))),
            "mean_response_lcp_fraction": _mean(enriched_rows, "response_lcp_fraction"),
        },
        "next_recommended_action": (
            "review verifier alignment report before any new GPU job; prioritize actual-generated-prefix "
            "opportunity discovery or short-prefix salvage diagnostics"
        ),
        "result_claim": "verifier_alignment_diagnosis_not_payload_recovery",
        "outputs": {
            "summary_json": str(output_dir / "verifier_alignment_summary.json"),
            "by_unit_csv": str(output_dir / "verifier_alignment_by_unit.csv"),
            "by_prompt_csv": str(output_dir / "verifier_alignment_by_prompt.csv"),
            "examples_jsonl": str(output_dir / "verifier_alignment_examples.jsonl"),
        },
    }

    write_json(output_dir / "verifier_alignment_summary.json", summary)
    write_csv(output_dir / "verifier_alignment_by_unit.csv", by_unit, list(by_unit[0].keys()) if by_unit else [])
    prompt_rows = by_prompt_sorted[: max(1, min(len(by_prompt_sorted), 5000))]
    write_csv(output_dir / "verifier_alignment_by_prompt.csv", prompt_rows, list(prompt_rows[0].keys()) if prompt_rows else [])
    write_jsonl(output_dir / "verifier_alignment_examples.jsonl", example_rows)
    return summary


def _collect_groups(
    rows: Sequence[Mapping[str, Any]],
    key_fn: Any,
) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return dict(groups)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    decode_path = resolve_repo_path(args.decode_trace_csv, root) if args.decode_trace_csv else None
    progress_path = resolve_repo_path(args.progress_json, root) if args.progress_json else None
    compatibility_jsonl = resolve_repo_path(args.compatibility_jsonl, root) if args.compatibility_jsonl else None
    compatibility_by_entry = (
        resolve_repo_path(args.compatibility_by_entry_csv, root) if args.compatibility_by_entry_csv else None
    )
    summary = run_diagnosis(
        generated_outputs_path=resolve_repo_path(args.generated_outputs, root),
        observations_path=resolve_repo_path(args.bucket_observations, root),
        decode_trace_path=decode_path,
        progress_path=progress_path,
        bucket_bank_entries_path=resolve_repo_path(args.bucket_bank_entries, root),
        compatibility_jsonl_path=compatibility_jsonl,
        compatibility_by_entry_csv_path=compatibility_by_entry,
        bucket_count=int(args.bucket_count),
        tokenizer_name=str(args.tokenizer_name),
        output_dir=resolve_repo_path(args.output_dir, root),
        max_examples=int(args.max_examples),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
