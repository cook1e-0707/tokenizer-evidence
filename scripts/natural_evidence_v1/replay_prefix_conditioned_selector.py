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
from typing import Any, Iterable, Mapping, Sequence

from scripts.natural_evidence_v1.common import token_surface_class, write_csv, write_json, write_jsonl
from scripts.natural_evidence_v1.diagnose_verifier_alignment import _decode_token, _load_tokenizer, _token_ids
from scripts.natural_evidence_v1.replay_qwen_frame_completion import _as_int, _rate


SCHEMA_NAME = "natural_evidence_qwen_846699_prefix_conditioned_selector_replay_v1"
EXAMPLE_SCHEMA_NAME = "natural_evidence_qwen_846699_prefix_selector_replay_example_v1"
MATCH_POLICIES = ("exact_full", "suffix_32", "suffix_16", "suffix_8")
BY_CONDITION_FIELDS = [
    "model_condition",
    "payload_id",
    "seed",
    "expected_payload_id",
    "match_policy",
    "query_budget",
    "scheduled_events",
    "prefix_matched_events",
    "prefix_match_rate",
    "compatible_hit_events",
    "compatible_hit_rate",
    "target_comparable_events",
    "target_hit_events",
    "target_hit_rate",
    "scheduled_coordinate_count",
    "rediscovered_coordinate_count",
    "compatible_coordinate_count",
    "target_coordinate_count",
    "scheduled_frame_count",
    "rediscovered_frame_count",
    "compatible_frame_count",
    "target_frame_count",
    "max_target_slots_per_frame",
    "top_drift_reason",
]
REJECTION_FIELDS = [
    "match_policy",
    "drift_reason",
    "events",
    "event_rate",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase R1 artifact-only prefix-conditioned selector replay for Qwen "
            "846699 generated transcripts. This reads generated transcripts, "
            "variable-radix train positions, and expanded candidate rows. It never "
            "trains, generates, decodes payload recovery, or claims FAR."
        )
    )
    parser.add_argument("--generated-jsonl", required=True)
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--bucketized-candidates-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--payload-ids", default="P0421,P1729")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--query-budgets", default="64,128,256,512")
    parser.add_argument("--match-policies", default=",".join(MATCH_POLICIES))
    parser.add_argument("--max-examples", type=int, default=240)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            yield payload


def _hash_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _token_map_from_candidates(candidate_row: Mapping[str, Any]) -> dict[int, str]:
    token_to_bucket: dict[int, str] = {}
    candidates = candidate_row.get("candidates", [])
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            token_id = candidate.get("token_id", "")
            bucket_id = str(candidate.get("bucket_id", ""))
            if bucket_id and str(token_id) != "":
                token_to_bucket[int(token_id)] = bucket_id
    buckets = candidate_row.get("buckets", {})
    if isinstance(buckets, dict):
        for bucket_id, token_ids in buckets.items():
            if not isinstance(token_ids, list):
                continue
            for token_id in token_ids:
                token_to_bucket[int(token_id)] = str(bucket_id)
    return token_to_bucket


def _candidate_bucket_texts(
    *,
    tokenizer: Any,
    token_to_bucket: Mapping[int, str],
    max_per_bucket: int = 4,
) -> dict[str, list[dict[str, Any]]]:
    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for token_id, bucket_id in sorted(token_to_bucket.items(), key=lambda item: (item[1], item[0])):
        if len(bucketed[str(bucket_id)]) >= max_per_bucket:
            continue
        token_text = _decode_token(tokenizer, token_id)
        bucketed[str(bucket_id)].append(
            {
                "token_id": int(token_id),
                "token_text": token_text,
                "token_class": token_surface_class(token_text) if token_text else "",
            }
        )
    return dict(bucketed)


def _load_train_positions(
    train_data_dir: Path,
    payload_ids: Sequence[str],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], set[str], dict[str, Any]]:
    positions_by_payload_prompt: dict[str, dict[str, list[dict[str, Any]]]] = {}
    bank_entry_ids: set[str] = set()
    manifest: dict[str, Any] = {
        "train_data_dir": str(train_data_dir),
        "payloads": {},
    }
    for payload_id in payload_ids:
        train_path = train_data_dir / payload_id / "variable_radix_train.jsonl"
        if not train_path.is_file() or train_path.stat().st_size == 0:
            raise FileNotFoundError(f"missing train JSONL for {payload_id}: {train_path}")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        row_count = 0
        position_count = 0
        for row in _iter_jsonl(train_path):
            row_count += 1
            prompt_id = str(row.get("prompt_id", ""))
            positions = row.get("eligible_positions", [])
            if not isinstance(positions, list):
                continue
            for prompt_slot, position in enumerate(positions):
                if not isinstance(position, dict):
                    continue
                bank_entry_id = str(position.get("bank_entry_id", ""))
                if bank_entry_id:
                    bank_entry_ids.add(bank_entry_id)
                copied = dict(position)
                copied["payload_id"] = payload_id
                copied["prompt_id"] = prompt_id
                copied["prompt_slot"] = prompt_slot
                copied["source_prompt_split"] = str(row.get("prompt_split", ""))
                copied["source_example_role"] = str(row.get("example_role", ""))
                grouped[prompt_id].append(copied)
                position_count += 1
        positions_by_payload_prompt[payload_id] = dict(grouped)
        manifest["payloads"][payload_id] = {
            "train_jsonl": _hash_file(train_path),
            "row_count": row_count,
            "position_count": position_count,
            "prompt_count_with_positions": len(grouped),
        }
    manifest["required_bank_entry_count"] = len(bank_entry_ids)
    return positions_by_payload_prompt, bank_entry_ids, manifest


def _load_candidate_rows(
    path: Path,
    required_bank_entry_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing bucketized candidates JSONL: {path}")
    rows: dict[str, dict[str, Any]] = {}
    scanned = 0
    duplicate_ids = 0
    for row in _iter_jsonl(path):
        scanned += 1
        bank_entry_id = str(row.get("bank_entry_id", ""))
        if bank_entry_id not in required_bank_entry_ids:
            continue
        if bank_entry_id in rows:
            duplicate_ids += 1
            continue
        prefix_token_ids = row.get("prefix_token_ids", [])
        if not isinstance(prefix_token_ids, list):
            prefix_token_ids = []
        copied = dict(row)
        copied["prefix_token_ids"] = [int(token_id) for token_id in prefix_token_ids]
        copied["token_to_bucket"] = _token_map_from_candidates(row)
        rows[bank_entry_id] = copied
        if len(rows) == len(required_bank_entry_ids):
            break
    missing = len(required_bank_entry_ids) - len(rows)
    return rows, {
        "bucketized_candidates_jsonl": _hash_file(path),
        "scanned_rows": scanned,
        "matched_rows": len(rows),
        "required_bank_entry_count": len(required_bank_entry_ids),
        "missing_required_bank_entry_count": missing,
        "duplicate_required_bank_entry_ids": duplicate_ids,
    }


def _expected_payload_ids_for_row(row: Mapping[str, Any], payload_ids: Sequence[str]) -> list[str]:
    model_condition = str(row.get("model_condition", ""))
    row_payload_id = str(row.get("payload_id", ""))
    if model_condition == "raw":
        return list(payload_ids)
    if row_payload_id in payload_ids:
        return [row_payload_id]
    return []


def _policy_window(match_policy: str) -> int | None:
    if match_policy == "exact_full":
        return None
    if match_policy.startswith("suffix_"):
        return int(match_policy.split("_", 1)[1])
    raise ValueError(f"Unsupported match policy: {match_policy}")


def find_prefix_conditioned_observed_token(
    *,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    prefix_token_ids: Sequence[int],
    match_policy: str,
) -> dict[str, Any]:
    """Return the next observed token after a rediscovered prefix event.

    The exact policy requires the committed prefix to equal the beginning of the
    prompt+response stream. Suffix policies search for the last N committed
    prefix tokens anywhere in the observed prompt+response stream and use the
    following token if the event boundary is at or after the response start.
    """
    prefix = [int(token_id) for token_id in prefix_token_ids]
    prompt_len = len(prompt_ids)
    full_ids = [int(token_id) for token_id in prompt_ids] + [int(token_id) for token_id in response_ids]
    if not prefix:
        return {"matched": False, "reason": "missing_prefix_token_ids"}
    window = _policy_window(match_policy)
    if window is None:
        end_index = len(prefix)
        if len(full_ids) < end_index:
            return {"matched": False, "reason": "prefix_longer_than_observed_text"}
        if full_ids[:end_index] != prefix:
            return {"matched": False, "reason": "exact_prefix_mismatch"}
        if end_index >= len(full_ids):
            return {"matched": False, "reason": "match_next_token_missing"}
        if end_index < prompt_len:
            return {"matched": False, "reason": "match_before_response_start"}
        return {
            "matched": True,
            "observed_token_id": int(full_ids[end_index]),
            "full_token_index": end_index,
            "response_token_index": end_index - prompt_len,
            "matched_prefix_token_count": len(prefix),
            "match_start_index": 0,
            "match_end_index": end_index,
            "reason": "",
        }
    if len(prefix) < window:
        return {"matched": False, "reason": "prefix_shorter_than_suffix_window"}
    needle = prefix[-window:]
    for start in range(0, len(full_ids) - window + 1):
        end_index = start + window
        if full_ids[start:end_index] != needle:
            continue
        if end_index >= len(full_ids):
            continue
        if end_index < prompt_len:
            continue
        return {
            "matched": True,
            "observed_token_id": int(full_ids[end_index]),
            "full_token_index": end_index,
            "response_token_index": end_index - prompt_len,
            "matched_prefix_token_count": window,
            "match_start_index": start,
            "match_end_index": end_index,
            "reason": "",
        }
    return {"matched": False, "reason": "suffix_prefix_not_found"}


def classify_prefix_selector_event(
    *,
    match: Mapping[str, Any],
    token_to_bucket: Mapping[int, str],
    compatible_bucket_ids: Sequence[str],
    target_bucket: str,
) -> dict[str, Any]:
    if not match.get("matched"):
        return {
            "prefix_matched": False,
            "compatible_hit": False,
            "target_hit": False,
            "bucket_id": "",
            "drift_reason": str(match.get("reason", "no_prefix_match")),
        }
    observed_token_id = int(match.get("observed_token_id", -1))
    bucket_id = token_to_bucket.get(observed_token_id, "")
    if not bucket_id:
        return {
            "prefix_matched": True,
            "compatible_hit": False,
            "target_hit": False,
            "bucket_id": "",
            "drift_reason": "observed_token_not_candidate_set",
        }
    compatible_set = {str(value) for value in compatible_bucket_ids}
    compatible_hit = str(bucket_id) in compatible_set
    target_hit = compatible_hit and str(bucket_id) == str(target_bucket)
    if target_hit:
        drift_reason = "target_hit"
    elif compatible_hit:
        drift_reason = "compatible_non_target"
    else:
        drift_reason = "observed_bucket_not_compatible"
    return {
        "prefix_matched": True,
        "compatible_hit": compatible_hit,
        "target_hit": target_hit,
        "bucket_id": str(bucket_id),
        "drift_reason": drift_reason,
    }


def _empty_stats() -> dict[str, Any]:
    return {
        "scheduled_events": 0,
        "prefix_matched_events": 0,
        "compatible_hit_events": 0,
        "target_comparable_events": 0,
        "target_hit_events": 0,
        "scheduled_coordinates": set(),
        "rediscovered_coordinates": set(),
        "compatible_coordinates": set(),
        "target_coordinates": set(),
        "scheduled_frames": set(),
        "rediscovered_frames": set(),
        "compatible_frames": set(),
        "target_frames": set(),
        "target_slots_by_frame": Counter(),
        "drift_reasons": Counter(),
    }


def _update_stats(stats: dict[str, Any], event: Mapping[str, Any]) -> None:
    coordinate = (
        str(event.get("expected_payload_id", "")),
        _as_int(event.get("frame_index", 0)),
        _as_int(event.get("frame_digit_index", 0)),
    )
    frame = (str(event.get("expected_payload_id", "")), _as_int(event.get("frame_index", 0)))
    stats["scheduled_events"] += 1
    stats["scheduled_coordinates"].add(coordinate)
    stats["scheduled_frames"].add(frame)
    if event.get("prefix_matched"):
        stats["prefix_matched_events"] += 1
        stats["rediscovered_coordinates"].add(coordinate)
        stats["rediscovered_frames"].add(frame)
    if event.get("compatible_hit"):
        stats["compatible_hit_events"] += 1
        stats["compatible_coordinates"].add(coordinate)
        stats["compatible_frames"].add(frame)
    stats["target_comparable_events"] += 1
    if event.get("target_hit"):
        stats["target_hit_events"] += 1
        stats["target_coordinates"].add(coordinate)
        stats["target_frames"].add(frame)
        stats["target_slots_by_frame"][frame] += 1
    stats["drift_reasons"][str(event.get("drift_reason", ""))] += 1


def _stats_row(key: tuple[str, str, str, str, str, int], stats: Mapping[str, Any]) -> dict[str, Any]:
    model_condition, payload_id, seed, expected_payload_id, match_policy, query_budget = key
    scheduled_events = int(stats["scheduled_events"])
    compatible_hit_events = int(stats["compatible_hit_events"])
    target_comparable_events = int(stats["target_comparable_events"])
    drift_reasons: Counter[str] = stats["drift_reasons"]
    top_drift_reason = ""
    if drift_reasons:
        reason, count = drift_reasons.most_common(1)[0]
        top_drift_reason = f"{reason}:{count}"
    return {
        "model_condition": model_condition,
        "payload_id": payload_id,
        "seed": seed,
        "expected_payload_id": expected_payload_id,
        "match_policy": match_policy,
        "query_budget": query_budget,
        "scheduled_events": scheduled_events,
        "prefix_matched_events": int(stats["prefix_matched_events"]),
        "prefix_match_rate": _rate(int(stats["prefix_matched_events"]), scheduled_events),
        "compatible_hit_events": compatible_hit_events,
        "compatible_hit_rate": _rate(compatible_hit_events, scheduled_events),
        "target_comparable_events": target_comparable_events,
        "target_hit_events": int(stats["target_hit_events"]),
        "target_hit_rate": _rate(int(stats["target_hit_events"]), target_comparable_events),
        "scheduled_coordinate_count": len(stats["scheduled_coordinates"]),
        "rediscovered_coordinate_count": len(stats["rediscovered_coordinates"]),
        "compatible_coordinate_count": len(stats["compatible_coordinates"]),
        "target_coordinate_count": len(stats["target_coordinates"]),
        "scheduled_frame_count": len(stats["scheduled_frames"]),
        "rediscovered_frame_count": len(stats["rediscovered_frames"]),
        "compatible_frame_count": len(stats["compatible_frames"]),
        "target_frame_count": len(stats["target_frames"]),
        "max_target_slots_per_frame": max(stats["target_slots_by_frame"].values(), default=0),
        "top_drift_reason": top_drift_reason,
    }


def _example_event(
    *,
    tokenizer: Any,
    event: Mapping[str, Any],
    match: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    token_to_bucket: Mapping[int, str],
) -> dict[str, Any]:
    observed_token_id = event.get("observed_token_id", "")
    observed_token_text = _decode_token(tokenizer, int(observed_token_id)) if str(observed_token_id) != "" else ""
    target_bucket = str(event.get("target_bucket", ""))
    return {
        "schema_name": EXAMPLE_SCHEMA_NAME,
        "model_condition": event.get("model_condition", ""),
        "payload_id": event.get("payload_id", ""),
        "seed": event.get("seed", ""),
        "expected_payload_id": event.get("expected_payload_id", ""),
        "match_policy": event.get("match_policy", ""),
        "query_index": event.get("query_index", ""),
        "prompt_id": event.get("prompt_id", ""),
        "prompt_slot": event.get("prompt_slot", ""),
        "bank_entry_id": event.get("bank_entry_id", ""),
        "frame_index": event.get("frame_index", ""),
        "frame_digit_index": event.get("frame_digit_index", ""),
        "target_bucket": target_bucket,
        "bucket_id": event.get("bucket_id", ""),
        "drift_reason": event.get("drift_reason", ""),
        "prefix_matched": event.get("prefix_matched", False),
        "compatible_hit": event.get("compatible_hit", False),
        "target_hit": event.get("target_hit", False),
        "observed_token_id": observed_token_id,
        "observed_token_text": observed_token_text,
        "observed_token_class": token_surface_class(observed_token_text) if observed_token_text else "",
        "response_token_index": match.get("response_token_index", ""),
        "matched_prefix_token_count": match.get("matched_prefix_token_count", ""),
        "prefix_response_token_count": candidate_row.get("prefix_response_token_count", ""),
        "candidate_bucket_token_texts": _candidate_bucket_texts(tokenizer=tokenizer, token_to_bucket=token_to_bucket),
        "target_bucket_token_ids": event.get("target_bucket_token_ids", []),
        "compatible_bucket_ids": event.get("compatible_bucket_ids", []),
        "result_claim": "prefix_conditioned_selector_replay_not_payload_recovery_not_far",
        "paper_claim_allowed": False,
    }


def run_replay(
    *,
    generated_jsonl: Path,
    train_data_dir: Path,
    bucketized_candidates_jsonl: Path,
    output_dir: Path,
    payload_ids: Sequence[str],
    tokenizer_name: str,
    query_budgets: Sequence[int],
    match_policies: Sequence[str],
    max_examples: int,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "prefix_conditioned_selector_replay_summary.json"
    if summary_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing R1 summary: {summary_path}")
    if not generated_jsonl.is_file() or generated_jsonl.stat().st_size == 0:
        raise FileNotFoundError(f"missing generated outputs JSONL: {generated_jsonl}")
    if not query_budgets:
        raise ValueError("query_budgets must not be empty")
    for policy in match_policies:
        _policy_window(policy)

    positions_by_payload_prompt, required_bank_entry_ids, train_manifest = _load_train_positions(train_data_dir, payload_ids)
    candidate_rows, candidate_manifest = _load_candidate_rows(bucketized_candidates_jsonl, required_bank_entry_ids)
    tokenizer = _load_tokenizer(tokenizer_name)

    stats_by_key: dict[tuple[str, str, str, str, str, int], dict[str, Any]] = defaultdict(_empty_stats)
    rejection_counts: Counter[tuple[str, str]] = Counter()
    examples: list[dict[str, Any]] = []
    generated_rows = 0
    considered_generated_payload_rows = 0
    tokenized_rows = 0
    missing_candidate_events = 0
    missing_expected_payload_rows = 0

    for row in _iter_jsonl(generated_jsonl):
        generated_rows += 1
        expected_payload_ids = _expected_payload_ids_for_row(row, payload_ids)
        if not expected_payload_ids:
            missing_expected_payload_rows += 1
            continue
        prompt_id = str(row.get("prompt_id", ""))
        prompt_ids = _token_ids(tokenizer, str(row.get("prompt", "")))
        response_ids = _token_ids(tokenizer, str(row.get("response_text", "")))
        tokenized_rows += 1
        model_condition = str(row.get("model_condition", ""))
        row_payload_id = str(row.get("payload_id", ""))
        seed = str(row.get("seed", ""))
        query_index = _as_int(row.get("query_index", 0))
        for expected_payload_id in expected_payload_ids:
            positions = positions_by_payload_prompt.get(expected_payload_id, {}).get(prompt_id, [])
            if not positions:
                continue
            considered_generated_payload_rows += 1
            for position in positions:
                bank_entry_id = str(position.get("bank_entry_id", ""))
                candidate_row = candidate_rows.get(bank_entry_id)
                for match_policy in match_policies:
                    event_base = {
                        "model_condition": model_condition,
                        "payload_id": row_payload_id,
                        "seed": seed,
                        "expected_payload_id": expected_payload_id,
                        "match_policy": match_policy,
                        "query_index": query_index,
                        "prompt_id": prompt_id,
                        "prompt_slot": _as_int(position.get("prompt_slot", 0)),
                        "bank_entry_id": bank_entry_id,
                        "frame_index": _as_int(position.get("frame_index", 0)),
                        "frame_digit_index": _as_int(position.get("frame_digit_index", 0)),
                        "target_bucket": str(position.get("target_bucket", "")),
                        "target_bucket_token_ids": position.get("target_bucket_token_ids", []),
                        "compatible_bucket_ids": [str(value) for value in position.get("compatible_bucket_ids", [])],
                    }
                    if candidate_row is None:
                        missing_candidate_events += 1
                        outcome = {
                            **event_base,
                            "prefix_matched": False,
                            "compatible_hit": False,
                            "target_hit": False,
                            "bucket_id": "",
                            "observed_token_id": "",
                            "drift_reason": "missing_candidate_row",
                        }
                        match = {"matched": False, "reason": "missing_candidate_row"}
                        token_to_bucket: dict[int, str] = {}
                    else:
                        token_to_bucket = dict(candidate_row.get("token_to_bucket", {}))
                        prefix_token_ids = candidate_row.get("prefix_token_ids", [])
                        match = find_prefix_conditioned_observed_token(
                            prompt_ids=prompt_ids,
                            response_ids=response_ids,
                            prefix_token_ids=prefix_token_ids,
                            match_policy=match_policy,
                        )
                        classified = classify_prefix_selector_event(
                            match=match,
                            token_to_bucket=token_to_bucket,
                            compatible_bucket_ids=[str(value) for value in position.get("compatible_bucket_ids", [])],
                            target_bucket=str(position.get("target_bucket", "")),
                        )
                        outcome = {
                            **event_base,
                            **classified,
                            "observed_token_id": match.get("observed_token_id", ""),
                        }
                    rejection_counts[(match_policy, str(outcome["drift_reason"]))] += 1
                    for query_budget in query_budgets:
                        if query_index >= int(query_budget):
                            continue
                        key = (
                            model_condition,
                            row_payload_id,
                            seed,
                            expected_payload_id,
                            match_policy,
                            int(query_budget),
                        )
                        _update_stats(stats_by_key[key], outcome)
                    if len(examples) < max_examples and str(outcome["drift_reason"]) != "target_hit":
                        examples.append(
                            _example_event(
                                tokenizer=tokenizer,
                                event=outcome,
                                match=match,
                                candidate_row=candidate_row or {},
                                token_to_bucket=token_to_bucket,
                            )
                        )

    by_condition_rows = [_stats_row(key, stats) for key, stats in sorted(stats_by_key.items())]
    aggregate_by_policy: dict[str, dict[str, Any]] = defaultdict(_empty_stats)
    for key, stats in stats_by_key.items():
        if key[-1] != max(query_budgets):
            continue
        match_policy = key[4]
        for count_key in (
            "scheduled_events",
            "prefix_matched_events",
            "compatible_hit_events",
            "target_comparable_events",
            "target_hit_events",
        ):
            aggregate_by_policy[match_policy][count_key] += int(stats[count_key])
        for set_key in (
            "scheduled_coordinates",
            "rediscovered_coordinates",
            "compatible_coordinates",
            "target_coordinates",
            "scheduled_frames",
            "rediscovered_frames",
            "compatible_frames",
            "target_frames",
        ):
            aggregate_by_policy[match_policy][set_key].update(stats[set_key])
        aggregate_by_policy[match_policy]["target_slots_by_frame"].update(stats["target_slots_by_frame"])
        aggregate_by_policy[match_policy]["drift_reasons"].update(stats["drift_reasons"])

    aggregate_rows = [
        _stats_row(("ALL", "ALL", "ALL", "ALL", match_policy, max(query_budgets)), stats)
        for match_policy, stats in sorted(aggregate_by_policy.items())
    ]
    total_rejections = sum(rejection_counts.values())
    rejection_rows = [
        {
            "match_policy": match_policy,
            "drift_reason": drift_reason,
            "events": count,
            "event_rate": _rate(count, total_rejections),
        }
        for (match_policy, drift_reason), count in sorted(rejection_counts.items())
    ]
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_PREFIX_CONDITIONED_SELECTOR_REPLAY_ARTIFACT_ONLY",
        "claim_control": {
            "paper_claim_allowed": False,
            "training_started": False,
            "generation_started": False,
            "e2e_eval_started": False,
            "not_payload_recovery": True,
            "not_full_far": True,
            "result_claim": "prefix_conditioned_selector_replay_not_payload_recovery_not_far",
        },
        "inputs": {
            "generated_jsonl": _hash_file(generated_jsonl),
            "train_manifest": train_manifest,
            "candidate_manifest": candidate_manifest,
            "tokenizer_name": tokenizer_name,
            "payload_ids": list(payload_ids),
            "query_budgets": [int(value) for value in query_budgets],
            "match_policies": list(match_policies),
        },
        "generated_rows": generated_rows,
        "tokenized_generated_rows": tokenized_rows,
        "considered_generated_payload_rows": considered_generated_payload_rows,
        "missing_expected_payload_rows": missing_expected_payload_rows,
        "missing_candidate_events": missing_candidate_events,
        "aggregate_by_policy": aggregate_rows,
        "top_rejection_reasons": rejection_rows[:20],
        "forbidden_claims_remain": [
            "natural-output success",
            "payload recovery",
            "full FAR",
            "cross-family generality",
            "robustness",
            "sanitizer resistance",
            "superiority over Scalable/Perinucleus",
            "24,576 fingerprints",
        ],
    }
    write_json(summary_path, summary)
    write_csv(output_dir / "prefix_conditioned_selector_replay_by_condition.csv", by_condition_rows, BY_CONDITION_FIELDS)
    write_csv(output_dir / "prefix_conditioned_selector_replay_aggregate_by_policy.csv", aggregate_rows, BY_CONDITION_FIELDS)
    write_csv(output_dir / "prefix_conditioned_selector_replay_rejections.csv", rejection_rows, REJECTION_FIELDS)
    write_jsonl(output_dir / "prefix_conditioned_selector_replay_examples.jsonl", examples)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_replay(
        generated_jsonl=_resolve(args.generated_jsonl),
        train_data_dir=_resolve(args.train_data_dir),
        bucketized_candidates_jsonl=_resolve(args.bucketized_candidates_jsonl),
        output_dir=_resolve(args.output_dir),
        payload_ids=_parse_csv_list(args.payload_ids),
        tokenizer_name=args.tokenizer_name,
        query_budgets=_parse_int_list(args.query_budgets),
        match_policies=_parse_csv_list(args.match_policies),
        max_examples=int(args.max_examples),
        force=bool(args.force),
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(_resolve(args.output_dir))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
