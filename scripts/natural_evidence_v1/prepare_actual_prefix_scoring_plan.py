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

from scripts.natural_evidence_v1.common import (
    keyed_hash_hex,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    stable_hash_hex,
    write_csv,
    write_json,
    write_jsonl,
)
from scripts.natural_evidence_v1.diagnose_verifier_alignment import _decode_token, _load_tokenizer, _token_ids


SCHEMA_NAME = "natural_evidence_actual_prefix_scoring_plan_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a CPU-only scoring plan for true actual-prefix candidate "
            "scoring. This enumerates generated transcript prefixes; it does not "
            "load a reference model, score top-k candidates, train, or claim recovery."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--generated-outputs", required=True)
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-generated-rows", type=int, default=0)
    parser.add_argument("--min-response-prefix-tokens", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=40)
    return parser.parse_args(argv)


def _selector_config(config: Mapping[str, Any]) -> dict[str, Any]:
    selector = dict(config.get("selector", {}))
    bucket_bank = dict(config.get("bucket_bank", {}))
    return {
        "selector_type": str(selector.get("type", "keyed_actual_prefix_selector")),
        "audit_key_id": str(selector.get("audit_key_id", "K001")),
        "min_spacing_tokens": int(selector.get("min_spacing_tokens", bucket_bank.get("min_spacing_tokens", 12))),
        "max_evidence_positions_per_response": int(
            selector.get(
                "max_evidence_positions_per_response",
                bucket_bank.get("max_evidence_positions_per_response", 4),
            )
        ),
    }


def _candidate_offsets(response_token_count: int, min_response_prefix_tokens: int) -> list[int]:
    if response_token_count <= min_response_prefix_tokens:
        return []
    return list(range(max(0, min_response_prefix_tokens), response_token_count))


def _select_offsets(
    *,
    row: Mapping[str, Any],
    response_token_count: int,
    protocol_id: str,
    audit_key_id: str,
    min_spacing_tokens: int,
    max_positions: int,
    min_response_prefix_tokens: int,
) -> list[int]:
    candidates = _candidate_offsets(response_token_count, min_response_prefix_tokens)
    ranked: list[tuple[str, int]] = []
    for offset in candidates:
        rank_key = keyed_hash_hex(
            audit_key_id,
            [
                protocol_id,
                "actual_prefix_selector_v1",
                row.get("model_condition", ""),
                row.get("payload_id", ""),
                row.get("seed", ""),
                row.get("prompt_id", ""),
                row.get("query_index", ""),
                offset,
            ],
        )
        ranked.append((rank_key, offset))
    selected: list[int] = []
    for _, offset in sorted(ranked):
        if any(abs(offset - previous) < min_spacing_tokens for previous in selected):
            continue
        selected.append(offset)
        if len(selected) >= max_positions:
            break
    return sorted(selected)


def _unit_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("model_condition", "")),
        str(row.get("payload_id", "")),
        str(row.get("seed", "")),
    )


def _summarize_units(rows: Sequence[Mapping[str, Any]], generated_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    generated_counts = Counter(_unit_key(row) for row in generated_rows)
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_unit_key(row)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(set(generated_counts) | set(grouped)):
        model_condition, payload_id, seed = key
        plan_rows = grouped.get(key, [])
        response_counts = Counter(str(row.get("prompt_id", "")) + ":" + str(row.get("query_index", "")) for row in plan_rows)
        selected_per_response = list(response_counts.values())
        output.append(
            {
                "model_condition": model_condition,
                "payload_id": payload_id,
                "seed": seed,
                "generated_rows": generated_counts.get(key, 0),
                "scoring_prefix_rows": len(plan_rows),
                "responses_with_scoring_prefixes": len(response_counts),
                "mean_prefixes_per_selected_response": mean(selected_per_response) if selected_per_response else 0.0,
                "min_prefix_response_token_count": min((int(row["prefix_response_token_count"]) for row in plan_rows), default=0),
                "max_prefix_response_token_count": max((int(row["prefix_response_token_count"]) for row in plan_rows), default=0),
            }
        )
    return output


def run_plan(
    *,
    config_path: Path,
    generated_outputs_path: Path,
    tokenizer_name: str,
    output_dir: Path,
    max_generated_rows: int = 0,
    min_response_prefix_tokens: int = 1,
    max_examples: int = 40,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_yaml(config_path)
    protocol_id = str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1"))
    bucket_bank = dict(config.get("bucket_bank", {}))
    compatibility_sweep = dict(bucket_bank.get("compatibility_sweep", {}))
    selector = _selector_config(config)
    generated_rows = read_jsonl(generated_outputs_path)
    if max_generated_rows > 0:
        generated_rows = generated_rows[:max_generated_rows]

    tokenizer = _load_tokenizer(tokenizer_name)
    scoring_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    skipped_short_responses = 0
    total_response_tokens = 0
    for generated_index, row in enumerate(generated_rows):
        prompt = str(row.get("prompt", ""))
        response = str(row.get("response_text", ""))
        prompt_ids = _token_ids(tokenizer, prompt)
        response_ids = _token_ids(tokenizer, response)
        total_response_tokens += len(response_ids)
        offsets = _select_offsets(
            row=row,
            response_token_count=len(response_ids),
            protocol_id=protocol_id,
            audit_key_id=selector["audit_key_id"],
            min_spacing_tokens=selector["min_spacing_tokens"],
            max_positions=selector["max_evidence_positions_per_response"],
            min_response_prefix_tokens=min_response_prefix_tokens,
        )
        if not offsets:
            skipped_short_responses += 1
            continue
        for position_index, offset in enumerate(offsets):
            prefix_token_ids = [*prompt_ids, *response_ids[:offset]]
            observed_token_id = int(response_ids[offset])
            prefix_signature = stable_hash_hex(
                [
                    protocol_id,
                    tokenizer_name,
                    row.get("prompt_id", ""),
                    row.get("model_condition", ""),
                    row.get("payload_id", ""),
                    row.get("seed", ""),
                    row.get("query_index", ""),
                    offset,
                    prefix_token_ids,
                ]
            )
            output_row = {
                "schema_name": "natural_evidence_actual_prefix_scoring_input_v1",
                "protocol_id": protocol_id,
                "selector_version": "keyed_actual_prefix_selector_v1",
                "selector_claim": "pre_score_prefix_enumeration_not_payload_recovery",
                "model_family": row.get("model_family", "qwen"),
                "model_condition": row.get("model_condition", ""),
                "payload_id": row.get("payload_id", ""),
                "seed": row.get("seed", ""),
                "prompt_id": row.get("prompt_id", ""),
                "prompt_split": row.get("prompt_split", ""),
                "query_index": int(row.get("query_index", 0)),
                "generated_row_index": generated_index,
                "position_index": position_index,
                "prefix_response_token_count": offset,
                "response_token_count": len(response_ids),
                "prefix_token_ids": prefix_token_ids,
                "prefix_signature": prefix_signature,
                "observed_token_id": observed_token_id,
                "observed_token_text": _decode_token(tokenizer, observed_token_id),
                "result_claim": "actual_prefix_scoring_input_not_candidates_not_recovery",
            }
            scoring_rows.append(output_row)
            if len(examples) < max_examples:
                example = dict(output_row)
                example["prompt_excerpt"] = prompt[:240]
                example["response_excerpt"] = response[:360]
                example.pop("prefix_token_ids", None)
                examples.append(example)

    by_unit = _summarize_units(scoring_rows, generated_rows)
    candidate_top_k_values = list(compatibility_sweep.get("candidate_top_k", [])) or [
        int(bucket_bank.get("candidate_top_k", 64))
    ]
    bucket_counts = list(compatibility_sweep.get("primary_bucket_counts", [])) or [2, 4]
    scoring_prefix_rows = len(scoring_rows)
    manifest = {
        "schema_name": SCHEMA_NAME,
        "status": "PLAN_COMPLETE_PENDING_REVIEW_AND_GPU_SCORING",
        "paper_claim_allowed": False,
        "protocol_id": protocol_id,
        "tokenizer": tokenizer_name,
        "generated_outputs": str(generated_outputs_path),
        "generated_rows": len(generated_rows),
        "skipped_short_responses": skipped_short_responses,
        "total_response_tokens": total_response_tokens,
        "scoring_prefix_rows": scoring_prefix_rows,
        "mean_scoring_prefixes_per_generated_response": scoring_prefix_rows / max(1, len(generated_rows)),
        "selector": {
            **selector,
            "min_response_prefix_tokens": min_response_prefix_tokens,
            "candidate_offsets": "all_actual_response_offsets_ranked_by_key_then_spacing_filtered",
        },
        "planned_gpu_scoring": {
            "needed": True,
            "reason": "reference-model top-k candidate scoring at actual generated prefixes",
            "candidate_top_k_values": candidate_top_k_values,
            "bucket_counts": bucket_counts,
            "estimated_topk_candidate_rows": {
                str(top_k): scoring_prefix_rows * int(top_k) for top_k in candidate_top_k_values
            },
            "recommended_first_job": {
                "tokenizer_key": "qwen",
                "candidate_top_k": min(int(value) for value in candidate_top_k_values),
                "bucket_count": 4 if 4 in [int(v) for v in bucket_counts] else int(bucket_counts[0]),
                "scope": "diagnostic_actual_prefix_scoring_only",
                "training_allowed": False,
            },
        },
        "compatibility_aware_selector_repair": {
            "construction_order": [
                "freeze generated transcripts",
                "enumerate actual prefixes with keyed selector",
                "score reference-model top-k at actual prefixes",
                "apply token surface filters",
                "compatibility score candidate replacements at actual suffix windows",
                "bucketize after compatibility filtering",
                "audit held-out density and decode capacity",
            ],
            "static_bucket_salvage_result": "insufficient_do_not_reuse_static_bucket_sets_as_main",
        },
        "result_claim": "actual_prefix_scoring_plan_not_model_scoring_not_payload_recovery",
        "outputs": {
            "scoring_input_jsonl": str(output_dir / "actual_prefix_scoring_input.jsonl"),
            "manifest_json": str(output_dir / "actual_prefix_scoring_manifest.json"),
            "by_unit_csv": str(output_dir / "actual_prefix_scoring_by_unit.csv"),
            "examples_jsonl": str(output_dir / "actual_prefix_scoring_examples.jsonl"),
        },
    }
    write_jsonl(output_dir / "actual_prefix_scoring_input.jsonl", scoring_rows)
    write_json(output_dir / "actual_prefix_scoring_manifest.json", manifest)
    write_csv(output_dir / "actual_prefix_scoring_by_unit.csv", by_unit, list(by_unit[0].keys()) if by_unit else [])
    write_jsonl(output_dir / "actual_prefix_scoring_examples.jsonl", examples)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    manifest = run_plan(
        config_path=resolve_repo_path(args.config, root),
        generated_outputs_path=resolve_repo_path(args.generated_outputs, root),
        tokenizer_name=str(args.tokenizer_name),
        output_dir=resolve_repo_path(args.output_dir, root),
        max_generated_rows=int(args.max_generated_rows),
        min_response_prefix_tokens=int(args.min_response_prefix_tokens),
        max_examples=int(args.max_examples),
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
