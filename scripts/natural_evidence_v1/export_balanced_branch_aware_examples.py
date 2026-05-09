from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.natural_evidence_v1.common import write_csv, write_json, write_jsonl
from scripts.natural_evidence_v1.diagnose_verifier_alignment import _load_tokenizer, _token_ids
from scripts.natural_evidence_v1.replay_prefix_conditioned_selector import (
    _candidate_bucket_texts,
    _expected_payload_ids_for_row,
    _example_event,
    _load_candidate_rows,
    _load_train_positions,
    _parse_csv_list,
    _policy_window,
    classify_prefix_selector_event,
    find_prefix_conditioned_observed_token,
)
from scripts.natural_evidence_v1.replay_qwen_frame_completion import _as_int


SCHEMA_NAME = "natural_evidence_v1_balanced_branch_aware_example_export_v1"
EXAMPLE_FIELDS = [
    "model_condition",
    "payload_id",
    "seed",
    "expected_payload_id",
    "match_policy",
    "drift_reason",
    "prompt_id",
    "prompt_slot",
    "query_index",
    "bank_entry_id",
    "frame_index",
    "frame_digit_index",
    "target_bucket",
    "bucket_id",
    "observed_token_id",
    "observed_token_text",
    "response_token_index",
    "paper_claim_allowed",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a balanced set of prefix-conditioned selector replay examples "
            "for branch-aware diagnostics. This reads existing generated "
            "transcripts, train metadata, and bucketized candidates only. It does "
            "not train, generate, score a model, decode payload recovery, or "
            "claim FAR."
        )
    )
    parser.add_argument("--generated-jsonl", required=True)
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--bucketized-candidates-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--payload-ids", default="P0421,P1729")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--match-policies", default="exact_full,suffix_32,suffix_16,suffix_8")
    parser.add_argument("--model-conditions", default="protected_trained,task_only_lora,raw")
    parser.add_argument("--drift-reasons", default="compatible_non_target,observed_token_not_candidate_set,observed_bucket_not_compatible")
    parser.add_argument("--per-slice", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=768)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            yield payload


def _slice_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("model_condition", "")),
        str(row.get("payload_id", "")),
        str(row.get("seed", "")),
        str(row.get("expected_payload_id", "")),
        str(row.get("match_policy", "")),
        str(row.get("drift_reason", "")),
    )


def _compact_example_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in EXAMPLE_FIELDS}


def _quota_has_room(
    *,
    row: Mapping[str, Any],
    selected_count: int,
    max_rows: int,
    per_slice: int,
    slice_counts: Counter[tuple[str, str, str, str, str, str]],
) -> bool:
    if selected_count >= max_rows:
        return False
    return slice_counts[_slice_key(row)] < per_slice


def run_export(
    *,
    generated_jsonl: Path,
    train_data_dir: Path,
    bucketized_candidates_jsonl: Path,
    output_dir: Path,
    payload_ids: Sequence[str],
    tokenizer_name: str,
    match_policies: Sequence[str],
    model_conditions: Sequence[str],
    drift_reasons: Sequence[str],
    per_slice: int,
    max_rows: int,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "balanced_branch_aware_example_export_summary.json"
    if summary_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing balanced export: {summary_path}")
    if not generated_jsonl.is_file() or generated_jsonl.stat().st_size == 0:
        raise FileNotFoundError(f"missing generated JSONL: {generated_jsonl}")
    for policy in match_policies:
        _policy_window(policy)
    condition_set = {str(value) for value in model_conditions}
    drift_set = {str(value) for value in drift_reasons}

    positions_by_payload_prompt, required_bank_entry_ids, train_manifest = _load_train_positions(train_data_dir, payload_ids)
    candidate_rows, candidate_manifest = _load_candidate_rows(bucketized_candidates_jsonl, required_bank_entry_ids)
    tokenizer = _load_tokenizer(tokenizer_name)

    examples: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    slice_counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
    condition_counts: Counter[str] = Counter()
    drift_counts: Counter[str] = Counter()
    scanned_generated_rows = 0
    considered_events = 0
    eligible_non_target_events = 0
    skipped_target_hit_events = 0
    skipped_condition_events = 0
    skipped_drift_events = 0
    missing_candidate_events = 0

    for row in _iter_jsonl(generated_jsonl):
        scanned_generated_rows += 1
        model_condition = str(row.get("model_condition", ""))
        expected_payload_ids = _expected_payload_ids_for_row(row, payload_ids)
        if not expected_payload_ids:
            continue
        prompt_id = str(row.get("prompt_id", ""))
        prompt_ids = _token_ids(tokenizer, str(row.get("prompt", "")))
        response_text = str(row.get("response_text", ""))
        response_ids = _token_ids(tokenizer, response_text)
        for expected_payload_id in expected_payload_ids:
            positions = positions_by_payload_prompt.get(expected_payload_id, {}).get(prompt_id, [])
            if not positions:
                continue
            for position in positions:
                bank_entry_id = str(position.get("bank_entry_id", ""))
                candidate_row = candidate_rows.get(bank_entry_id)
                for match_policy in match_policies:
                    considered_events += 1
                    if model_condition not in condition_set:
                        skipped_condition_events += 1
                        continue
                    event_base = {
                        "model_condition": model_condition,
                        "payload_id": str(row.get("payload_id", "")),
                        "seed": str(row.get("seed", "")),
                        "expected_payload_id": expected_payload_id,
                        "match_policy": match_policy,
                        "query_index": _as_int(row.get("query_index", 0)),
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
                        token_to_bucket: dict[int, str] = {}
                        match = {"matched": False, "reason": "missing_candidate_row"}
                        outcome = {
                            **event_base,
                            "prefix_matched": False,
                            "compatible_hit": False,
                            "target_hit": False,
                            "bucket_id": "",
                            "observed_token_id": "",
                            "drift_reason": "missing_candidate_row",
                        }
                    else:
                        token_to_bucket = dict(candidate_row.get("token_to_bucket", {}))
                        match = find_prefix_conditioned_observed_token(
                            prompt_ids=prompt_ids,
                            response_ids=response_ids,
                            prefix_token_ids=candidate_row.get("prefix_token_ids", []),
                            match_policy=match_policy,
                        )
                        classified = classify_prefix_selector_event(
                            match=match,
                            token_to_bucket=token_to_bucket,
                            compatible_bucket_ids=[str(value) for value in position.get("compatible_bucket_ids", [])],
                            target_bucket=str(position.get("target_bucket", "")),
                        )
                        outcome = {**event_base, **classified, "observed_token_id": match.get("observed_token_id", "")}
                    if outcome.get("target_hit"):
                        skipped_target_hit_events += 1
                        continue
                    drift_reason = str(outcome.get("drift_reason", ""))
                    if drift_reason not in drift_set:
                        skipped_drift_events += 1
                        continue
                    eligible_non_target_events += 1
                    example = _example_event(
                        tokenizer=tokenizer,
                        event=outcome,
                        match=match,
                        candidate_row=candidate_row or {},
                        token_to_bucket=token_to_bucket,
                    )
                    example.update(
                        {
                            "prompt": str(row.get("prompt", "")),
                            "user_probe": str(row.get("user_probe", "")),
                            "generated_response_text": response_text,
                            "prompt_split": str(row.get("prompt_split", "")),
                            "selection_status": "BALANCED_EXAMPLE_SELECTED_FOR_BRANCH_AWARE_DIAGNOSTIC",
                            "model_loading_started": False,
                            "model_scoring_started": False,
                            "generation_started": False,
                            "training_started": False,
                            "e2e_eval_started": False,
                            "result_claim": "balanced_branch_aware_example_export_not_scored_not_training_not_far",
                        }
                    )
                    if not _quota_has_room(
                        row=example,
                        selected_count=len(examples),
                        max_rows=max_rows,
                        per_slice=per_slice,
                        slice_counts=slice_counts,
                    ):
                        continue
                    examples.append(example)
                    compact_rows.append(_compact_example_row(example))
                    slice_counts[_slice_key(example)] += 1
                    condition_counts[str(example.get("model_condition", ""))] += 1
                    drift_counts[str(example.get("drift_reason", ""))] += 1
                    if len(examples) >= max_rows:
                        break
                if len(examples) >= max_rows:
                    break
            if len(examples) >= max_rows:
                break
        if len(examples) >= max_rows:
            break

    by_slice_rows = [
        {
            "model_condition": key[0],
            "payload_id": key[1],
            "seed": key[2],
            "expected_payload_id": key[3],
            "match_policy": key[4],
            "drift_reason": key[5],
            "rows": count,
        }
        for key, count in sorted(slice_counts.items())
    ]
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_BALANCED_BRANCH_AWARE_EXAMPLES_EXPORTED_NOT_SCORED",
        "claim_control": {
            "paper_claim_allowed": False,
            "training_started": False,
            "generation_started": False,
            "model_loading_started": False,
            "model_scoring_started": False,
            "e2e_eval_started": False,
            "not_payload_recovery": True,
            "not_full_far": True,
            "result_claim": "balanced_branch_aware_example_export_not_scored_not_training_not_far",
        },
        "inputs": {
            "generated_jsonl": str(generated_jsonl),
            "train_manifest": train_manifest,
            "candidate_manifest": candidate_manifest,
            "tokenizer_name": tokenizer_name,
            "payload_ids": list(payload_ids),
            "match_policies": list(match_policies),
            "model_conditions": list(model_conditions),
            "drift_reasons": list(drift_reasons),
            "per_slice": per_slice,
            "max_rows": max_rows,
        },
        "scanned_generated_rows": scanned_generated_rows,
        "considered_events": considered_events,
        "eligible_non_target_events": eligible_non_target_events,
        "selected_example_rows": len(examples),
        "condition_counts": dict(sorted(condition_counts.items())),
        "drift_reason_counts": dict(sorted(drift_counts.items())),
        "slice_count": len(slice_counts),
        "skipped_target_hit_events": skipped_target_hit_events,
        "skipped_condition_events": skipped_condition_events,
        "skipped_drift_events": skipped_drift_events,
        "missing_candidate_events": missing_candidate_events,
        "next_allowed_action": (
            "Prepare branch-aware compatibility and regenerated/local-suffix "
            "repair diagnostics from the balanced examples; no training."
        ),
    }
    write_json(summary_path, summary)
    write_jsonl(output_dir / "prefix_conditioned_selector_replay_examples.jsonl", examples)
    write_csv(output_dir / "balanced_branch_aware_examples.csv", compact_rows, EXAMPLE_FIELDS)
    write_csv(
        output_dir / "balanced_branch_aware_examples_by_slice.csv",
        by_slice_rows,
        ["model_condition", "payload_id", "seed", "expected_payload_id", "match_policy", "drift_reason", "rows"],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_export(
        generated_jsonl=_resolve(args.generated_jsonl),
        train_data_dir=_resolve(args.train_data_dir),
        bucketized_candidates_jsonl=_resolve(args.bucketized_candidates_jsonl),
        output_dir=_resolve(args.output_dir),
        payload_ids=_parse_csv_list(args.payload_ids),
        tokenizer_name=args.tokenizer_name,
        match_policies=_parse_csv_list(args.match_policies),
        model_conditions=_parse_csv_list(args.model_conditions),
        drift_reasons=_parse_csv_list(args.drift_reasons),
        per_slice=int(args.per_slice),
        max_rows=int(args.max_rows),
        force=bool(args.force),
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(_resolve(args.output_dir))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
