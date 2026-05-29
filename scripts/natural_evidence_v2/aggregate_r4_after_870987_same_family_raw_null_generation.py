from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.aggregate_r4_after_869348_locked_scale_generation import (  # noqa: E402
    discover_shards,
    int_value,
    read_csv,
    read_json,
    read_jsonl,
    summarize_first_token,
    summarize_full_phrase,
    write_csv,
    write_json,
)
from scripts.natural_evidence_v2.classify_r4_forbidden_surface_context_v2 import classify_text  # noqa: E402


MODELS = (
    "qwen2_5_3b_instruct_raw",
    "qwen2_5_7b_instruct_raw",
    "qwen2_5_14b_instruct_raw",
)
ARMS = ("protected", "raw", "task_only", "wrong_key", "wrong_payload")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate R4 after-870987 same-family raw-null generation artifacts. "
            "This is artifact-only: it reads generated/decode/trace artifacts and writes "
            "review tables. It does not generate, train, submit Slurm, run Llama, run FAR, "
            "or make paper-facing claims."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-roots",
        type=Path,
        nargs="+",
        required=True,
        help="Model root directories, each containing shard_XX subdirectories under shards/.",
    )
    parser.add_argument("--expected-shards-per-model", type=int, default=64)
    parser.add_argument("--expected-generated-rows-per-shard", type=int, default=1024)
    parser.add_argument("--raw-accepts-max", type=int, default=0)
    parser.add_argument("--global-duplicate-extra-max", type=int, default=0)
    parser.add_argument("--within-block-duplicate-max", type=int, default=0)
    parser.add_argument("--technical-forbidden-max", type=int, default=0)
    parser.add_argument("--ambiguous-forbidden-max", type=int, default=0)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def model_slug_from_root(path: Path) -> str:
    return path.name


def iter_generated_rows(shard_dirs: Mapping[int, Path]) -> Iterable[tuple[int, dict[str, Any]]]:
    for shard_index, shard_dir in sorted(shard_dirs.items()):
        for row in read_jsonl(shard_dir / "r4_generated_outputs.jsonl"):
            yield shard_index, row


def summarize_generated_and_contextual(
    model_slug: str,
    shard_dirs: Mapping[int, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    response_hash_counts: Counter[str] = Counter()
    generation_id_counts: Counter[str] = Counter()
    rows_by_condition: Counter[str] = Counter()
    row_count = 0
    wrong_key_replay_accept_rows = 0
    wrong_payload_replay_accept_rows = 0
    contextual_rows: list[dict[str, Any]] = []
    contextual_totals = Counter()

    for shard_index, row in iter_generated_rows(shard_dirs):
        row_count += 1
        response_hash = str(row.get("output_text_sha256") or row.get("response_text_sha256") or "")
        generation_id = str(row.get("generation_id", ""))
        condition = str(row.get("generation_condition", row.get("arm", "")))
        if response_hash:
            response_hash_counts[response_hash] += 1
        if generation_id:
            generation_id_counts[generation_id] += 1
        if condition:
            rows_by_condition[condition] += 1
        wrong_key_replay_accept_rows += int(bool(row.get("wrong_key_replay_accept")))
        wrong_payload_replay_accept_rows += int(bool(row.get("wrong_payload_replay_accept")))

        classification = classify_text(str(row.get("response_text") or row.get("output_text") or "")).to_dict()
        contextual_totals["technical_forbidden_public_surface_count"] += int(
            classification["technical_forbidden_public_surface_count"]
        )
        contextual_totals["ordinary_domain_literal_count"] += int(classification["ordinary_domain_literal_count"])
        contextual_totals["ambiguous_forbidden_surface_count"] += int(
            classification["ambiguous_forbidden_surface_count"]
        )
        if (
            classification["technical_forbidden_public_surface_count"]
            or classification["ordinary_domain_literal_count"]
            or classification["ambiguous_forbidden_surface_count"]
        ):
            contextual_rows.append(
                {
                    "model_slug": model_slug,
                    "shard_index": shard_index,
                    "generation_id": row.get("generation_id", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "arm": row.get("arm", ""),
                    "generation_condition": condition,
                    "technical_hits": ";".join(classification["technical_hits"]),
                    "ordinary_domain_literals": ";".join(classification["ordinary_domain_literals"]),
                    "ambiguous_hits": ";".join(classification["ambiguous_hits"]),
                    "technical_forbidden_public_surface_count": classification[
                        "technical_forbidden_public_surface_count"
                    ],
                    "ordinary_domain_literal_count": classification["ordinary_domain_literal_count"],
                    "ambiguous_forbidden_surface_count": classification["ambiguous_forbidden_surface_count"],
                    "output_text_sha256": response_hash,
                    "response_excerpt": str(row.get("response_text") or row.get("output_text") or "")[:240],
                }
            )

    duplicate_response_groups = {key: count for key, count in response_hash_counts.items() if count > 1}
    duplicate_generation_id_groups = {key: count for key, count in generation_id_counts.items() if count > 1}
    return (
        {
            "generated_rows": row_count,
            "rows_by_condition": dict(sorted(rows_by_condition.items())),
            "unique_response_hashes": len(response_hash_counts),
            "global_duplicate_response_hash_extra_rows": sum(count - 1 for count in duplicate_response_groups.values()),
            "global_duplicate_response_hash_group_count": len(duplicate_response_groups),
            "max_response_hash_group_size": max(duplicate_response_groups.values(), default=1),
            "duplicate_generation_id_extra_rows": sum(count - 1 for count in duplicate_generation_id_groups.values()),
            "duplicate_generation_id_group_count": len(duplicate_generation_id_groups),
            "max_generation_id_group_size": max(duplicate_generation_id_groups.values(), default=1),
            "wrong_key_replay_accept_rows": wrong_key_replay_accept_rows,
            "wrong_payload_replay_accept_rows": wrong_payload_replay_accept_rows,
            "contextual_forbidden_surface_summary": dict(contextual_totals),
        },
        contextual_rows,
    )


def first_token_rows_for_shards(model_slug: str, shard_dirs: Mapping[int, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard_index, shard_dir in sorted(shard_dirs.items()):
        for row in read_csv(shard_dir / "first_token_event_decode/first_token_event_per_block.csv"):
            rows.append(dict(row) | {"model_slug": model_slug, "shard_index": shard_index, "source_shard_dir": str(shard_dir)})
    return rows


def full_phrase_rows_for_shards(model_slug: str, shard_dirs: Mapping[int, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard_index, shard_dir in sorted(shard_dirs.items()):
        for mode_dir in ("decode_all", "decode_none"):
            for row in read_csv(shard_dir / f"{mode_dir}/per_block_decode.csv"):
                rows.append(dict(row) | {"model_slug": model_slug, "shard_index": shard_index, "source_shard_dir": str(shard_dir)})
    return rows


def trace_rows_for_shards(model_slug: str, shard_dirs: Mapping[int, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard_index, shard_dir in sorted(shard_dirs.items()):
        trace = read_json(shard_dir / "trace_binding_validation.json")
        rows.append(
            {
                "model_slug": model_slug,
                "shard_index": shard_index,
                "shard_id": f"shard_{shard_index:02d}",
                "status": trace.get("status", ""),
                "checked_rows": int_value(trace.get("checked_rows")),
                "invalid_rows": int_value(trace.get("invalid_rows")),
                "source_shard_dir": str(shard_dir),
            }
        )
    return rows


def write_review_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    shards_per_model = int(summary["gate_targets"]["shards_per_model"])
    scope_line = (
        "- This aggregate gate covers the full 64-shard same-family raw-null package, "
        "but it is still not full FAR or a paper-facing positive claim."
        if shards_per_model >= 64
        else "- This is a capacity-limited same-family raw-null confirmation, not a full 64-shard package."
    )
    lines = [
        "# R4 Same-Family Raw-Null Generation Review",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This review is artifact-only. It does not unlock Llama, FAR aggregation, sanitizer, "
        "payload diversity, training, or paper-facing claims.",
        "",
        "## Model Summary",
        "",
        "| Model | Shards | Raw accepts | Raw accepts ignoring quality | Duplicate extra rows | Technical forbidden | Ambiguous forbidden | Ordinary literals | Trace invalid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_slug, model in sorted(summary["models"].items()):
        raw = model["first_token_event_summary_by_arm"]["raw"]
        dup = model["generation_duplicate_summary"]
        ctx = dup["contextual_forbidden_surface_summary"]
        trace = model["trace_binding"]
        lines.append(
            "| "
            + " | ".join(
                [
                    model_slug,
                    str(model["complete_shard_count"]),
                    str(raw["accepts"]),
                    str(raw["accepts_ignoring_quality"]),
                    str(dup["global_duplicate_response_hash_extra_rows"]),
                    str(ctx.get("technical_forbidden_public_surface_count", 0)),
                    str(ctx.get("ambiguous_forbidden_surface_count", 0)),
                    str(ctx.get("ordinary_domain_literal_count", 0)),
                    str(trace["invalid_rows"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Claim Control",
            "",
            (
                "- This aggregate gate uses "
                f"{shards_per_model}/{shards_per_model} complete shards per model "
                "with 0 raw accepts."
            ),
            scope_line,
            "- Ordinary domain literals are reported separately from technical forbidden surfaces.",
            "- Full-phrase decoder rows remain report-only and are not a text-only success claim.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    model_roots = [resolve(path) for path in args.model_roots]
    for root in model_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"model root missing: {root}")

    model_summaries: dict[str, Any] = {}
    all_first_token_rows: list[dict[str, Any]] = []
    all_full_phrase_rows: list[dict[str, Any]] = []
    all_trace_rows: list[dict[str, Any]] = []
    all_contextual_rows: list[dict[str, Any]] = []
    shard_matrix: list[dict[str, Any]] = []

    for root in model_roots:
        model_slug = model_slug_from_root(root)
        complete, partial, missing, incomplete = discover_shards(
            [root / "shards"],
            expected_shards=int(args.expected_shards_per_model),
            require_generated_jsonl=True,
        )
        first_token_rows = first_token_rows_for_shards(model_slug, complete)
        full_phrase_rows = full_phrase_rows_for_shards(model_slug, complete)
        trace_rows = trace_rows_for_shards(model_slug, complete)
        generation_summary, contextual_rows = summarize_generated_and_contextual(model_slug, complete)
        first_token_summary = summarize_first_token(first_token_rows)
        full_phrase_summary = summarize_full_phrase(full_phrase_rows)
        trace_checked_rows = sum(int(row["checked_rows"]) for row in trace_rows)
        trace_invalid_rows = sum(int(row["invalid_rows"]) for row in trace_rows)
        trace_status_pass = bool(trace_rows) and all(row["status"] == "PASS_R4_FIRST_TOKEN_EVENT_TRACE_BINDING" for row in trace_rows)
        all_complete = len(complete) == int(args.expected_shards_per_model)

        model_summaries[model_slug] = {
            "model_root": str(root),
            "all_shards_complete": all_complete,
            "complete_shard_count": len(complete),
            "expected_shards": int(args.expected_shards_per_model),
            "missing_shards": missing,
            "incomplete_shards": incomplete,
            "partial_shards": {str(k): [str(p) for p in v] for k, v in partial.items() if v},
            "first_token_event_summary_by_arm": first_token_summary,
            "full_phrase_decoder_report_only_summary": full_phrase_summary,
            "trace_binding": {
                "checked_rows": trace_checked_rows,
                "expected_checked_rows": int(args.expected_shards_per_model) * int(args.expected_generated_rows_per_shard),
                "invalid_rows": trace_invalid_rows,
                "all_shard_trace_status_pass": trace_status_pass,
            },
            "generation_duplicate_summary": generation_summary,
        }
        for index in range(int(args.expected_shards_per_model)):
            shard_matrix.append(
                {
                    "model_slug": model_slug,
                    "shard_index": index,
                    "shard_id": f"shard_{index:02d}",
                    "status": "complete" if index in complete else ("partial" if partial.get(index) else "missing"),
                    "selected_shard_dir": str(complete.get(index, "")),
                }
            )
        all_first_token_rows.extend(first_token_rows)
        all_full_phrase_rows.extend(full_phrase_rows)
        all_trace_rows.extend(trace_rows)
        all_contextual_rows.extend(contextual_rows)

    expected_model_set = set(MODELS)
    observed_model_set = set(model_summaries)
    per_model_pass: dict[str, bool] = {}
    for model_slug, model in model_summaries.items():
        raw = model["first_token_event_summary_by_arm"]["raw"]
        duplicate = model["generation_duplicate_summary"]
        contextual = duplicate["contextual_forbidden_surface_summary"]
        trace = model["trace_binding"]
        per_model_pass[model_slug] = (
            model["all_shards_complete"]
            and raw["blocks"] == int(args.expected_shards_per_model)
            and raw["accepts"] <= int(args.raw_accepts_max)
            and raw["accepts_ignoring_quality"] <= int(args.raw_accepts_max)
            and raw["duplicate_response_hash_count"] <= int(args.within_block_duplicate_max)
            and duplicate["generated_rows"] == int(args.expected_shards_per_model) * int(args.expected_generated_rows_per_shard)
            and duplicate["rows_by_condition"] == {"raw": duplicate["generated_rows"]}
            and duplicate["global_duplicate_response_hash_extra_rows"] <= int(args.global_duplicate_extra_max)
            and duplicate["duplicate_generation_id_extra_rows"] <= 0
            and duplicate["wrong_key_replay_accept_rows"] == 0
            and duplicate["wrong_payload_replay_accept_rows"] == 0
            and contextual.get("technical_forbidden_public_surface_count", 0) <= int(args.technical_forbidden_max)
            and contextual.get("ambiguous_forbidden_surface_count", 0) <= int(args.ambiguous_forbidden_max)
            and trace["invalid_rows"] == 0
            and trace["checked_rows"] == trace["expected_checked_rows"]
            and trace["all_shard_trace_status_pass"]
        )

    all_models_complete = observed_model_set == expected_model_set and all(
        model["all_shards_complete"] for model in model_summaries.values()
    )
    gate_pass = all_models_complete and all(per_model_pass.values())
    status = (
        "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE"
        if gate_pass
        else (
            "INCOMPLETE_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_NO_GATE"
            if not all_models_complete
            else "FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE"
        )
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_870987_same_family_raw_null_generation_aggregate_v1",
        "status": status,
        "same_family_raw_null_gate_pass": bool(gate_pass),
        "expected_models": list(MODELS),
        "observed_models": sorted(observed_model_set),
        "all_models_complete": all_models_complete,
        "per_model_pass": per_model_pass,
        "models": model_summaries,
        "aggregate": {
            "generated_rows": sum(m["generation_duplicate_summary"]["generated_rows"] for m in model_summaries.values()),
            "trace_checked_rows": sum(m["trace_binding"]["checked_rows"] for m in model_summaries.values()),
            "trace_invalid_rows": sum(m["trace_binding"]["invalid_rows"] for m in model_summaries.values()),
            "ordinary_domain_literal_count": sum(
                m["generation_duplicate_summary"]["contextual_forbidden_surface_summary"].get(
                    "ordinary_domain_literal_count", 0
                )
                for m in model_summaries.values()
            ),
            "technical_forbidden_public_surface_count": sum(
                m["generation_duplicate_summary"]["contextual_forbidden_surface_summary"].get(
                    "technical_forbidden_public_surface_count", 0
                )
                for m in model_summaries.values()
            ),
            "ambiguous_forbidden_surface_count": sum(
                m["generation_duplicate_summary"]["contextual_forbidden_surface_summary"].get(
                    "ambiguous_forbidden_surface_count", 0
                )
                for m in model_summaries.values()
            ),
        },
        "gate_targets": {
            "shards_per_model": int(args.expected_shards_per_model),
            "raw_accepts_max_per_model": int(args.raw_accepts_max),
            "raw_accepts_ignoring_quality_max_per_model": int(args.raw_accepts_max),
            "global_duplicate_response_hash_extra_rows_max_per_model": int(args.global_duplicate_extra_max),
            "within_block_duplicate_response_hash_count_max_per_model": int(args.within_block_duplicate_max),
            "technical_forbidden_public_surface_count_max_per_model": int(args.technical_forbidden_max),
            "ambiguous_forbidden_surface_count_max_per_model": int(args.ambiguous_forbidden_max),
            "ordinary_domain_literals_policy": "report_only_not_fatal",
            "trace_binding_validity_required": 1.0,
        },
        "claim_control": {
            "same_family_raw_null_rejection_allowed": bool(gate_pass),
            "paper_claim_allowed": False,
            "training_allowed": False,
            "llama_allowed": False,
            "sanitizer_allowed": False,
            "far_aggregation_allowed": False,
            "payload_diversity_tested": False,
            "text_only_phrase_decoder_success_claim": False,
            "full_far_claim_allowed": False,
        },
    }

    write_csv(
        output_dir / "same_family_raw_null_shard_matrix.csv",
        shard_matrix,
        ["model_slug", "shard_index", "shard_id", "status", "selected_shard_dir"],
    )
    write_csv(
        output_dir / "same_family_raw_null_first_token_blocks.csv",
        all_first_token_rows,
        [
            "model_slug",
            "shard_index",
            "source_shard_dir",
            "block_id",
            "arm",
            "source_condition",
            "accept",
            "accept_ignoring_quality",
            "complete_pairs",
            "required_pairs",
            "decoded_bits",
            "expected_bits",
            "bits_match_condition",
            "checksum_valid",
            "forbidden_public_surface_count",
            "duplicate_response_hash_count",
        ],
    )
    write_csv(
        output_dir / "same_family_raw_null_full_phrase_blocks.csv",
        all_full_phrase_rows,
        [
            "model_slug",
            "shard_index",
            "source_shard_dir",
            "block_id",
            "arm",
            "accept",
            "complete_pairs",
            "required_pairs",
            "selected_coordinates_observed",
            "selected_coordinates_total",
            "min_pair_support",
            "matched_surface_count",
            "selected_surface_count",
            "checksum_valid",
            "bits_match_condition",
            "forbidden_public_surface_count",
            "format_scrub_mode",
        ],
    )
    write_csv(
        output_dir / "same_family_raw_null_trace_binding_by_shard.csv",
        all_trace_rows,
        ["model_slug", "shard_index", "shard_id", "status", "checked_rows", "invalid_rows", "source_shard_dir"],
    )
    write_csv(
        output_dir / "same_family_raw_null_contextual_forbidden_rows.csv",
        all_contextual_rows,
        [
            "model_slug",
            "shard_index",
            "generation_id",
            "prompt_id",
            "arm",
            "generation_condition",
            "technical_hits",
            "ordinary_domain_literals",
            "ambiguous_hits",
            "technical_forbidden_public_surface_count",
            "ordinary_domain_literal_count",
            "ambiguous_forbidden_surface_count",
            "output_text_sha256",
            "response_excerpt",
        ],
    )
    write_json(output_dir / "same_family_raw_null_summary.json", summary)
    write_review_markdown(output_dir / "same_family_raw_null_review.md", summary)
    print(json.dumps({"status": status, "output_dir": str(output_dir)}, sort_keys=True))
    if not all_models_complete and not args.allow_incomplete:
        return 2
    if all_models_complete and not gate_pass:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
