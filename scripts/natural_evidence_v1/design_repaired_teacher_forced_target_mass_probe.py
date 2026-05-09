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

from scripts.natural_evidence_v1.common import stable_hash_hex, write_csv, write_json, write_jsonl


SCHEMA_NAME = "natural_evidence_v1_repaired_teacher_forced_target_mass_probe_design_v1"
PLAN_SCHEMA_NAME = "natural_evidence_v1_repaired_teacher_forced_target_mass_score_plan_v1"
PRIMARY_TIERS = {
    "PRIMARY_COMPATIBLE_NON_TARGET_LOW_DELTA",
    "PRIMARY_COMPATIBLE_NON_TARGET_PROXY_PASS",
    "PRIMARY_OUT_OF_CANDIDATE_SET_LOW_DELTA",
}
DEFAULT_CHECKPOINT_ROOT = (
    "/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/"
    "natural_evidence_v1/qwen_natural_e2e_pilot/training"
)
MODEL_CONDITIONS = ("base", "protected_trained", "task_only_lora")
SLICE_FIELDS = [
    "group_kind",
    "group_value",
    "candidate_rows",
    "scoring_plan_rows",
    "base_score_rows",
    "protected_score_rows",
    "task_only_score_rows",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-only repaired teacher-forced target-mass probe design "
            "over primary branch-aware candidates. This writes a concrete score "
            "spec and plan; it does not score a model, generate, train, rerun "
            "E2E, estimate FAR, or make paper-facing claims."
        )
    )
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--branch-summary-json", required=True)
    parser.add_argument("--balanced-examples-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--training-job-id", default="846585")
    parser.add_argument("--payload-ids", default="P0421,P1729")
    parser.add_argument("--seeds", default="17,23")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            yield payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _bucket_token_ids(bucket_tokens: object) -> list[int]:
    output: list[int] = []
    if not isinstance(bucket_tokens, list):
        return output
    for item in bucket_tokens:
        if not isinstance(item, Mapping):
            continue
        try:
            output.append(int(item.get("token_id")))
        except (TypeError, ValueError):
            continue
    return output


def _candidate_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("prompt_id", "")),
        str(row.get("model_condition", "")),
        str(row.get("expected_payload_id", "")),
        str(row.get("seed", "")),
        str(row.get("query_index", "")),
        str(row.get("frame_index", "")),
        str(row.get("frame_digit_index", "")),
        str(row.get("observed_token_text", "")),
        str(row.get("target_bucket", "")),
        str(row.get("match_policy", "")),
    )


def _example_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, ...], Mapping[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_candidate_key(row)].append(row)
    duplicates = {key: values for key, values in grouped.items() if len(values) > 1}
    if duplicates:
        first_key = next(iter(duplicates))
        raise ValueError(f"Balanced example join is not unique for key: {first_key}")
    return {key: values[0] for key, values in grouped.items()}


def _is_primary_candidate(row: Mapping[str, Any]) -> bool:
    return (
        _as_bool(row.get("primary_probe_candidate"))
        and str(row.get("candidate_tier", "")) in PRIMARY_TIERS
        and _as_bool(row.get("branch_aware_proxy_pass"))
        and not _as_bool(row.get("training_started"))
        and not _as_bool(row.get("generation_started"))
        and not _as_bool(row.get("e2e_eval_started"))
    )


def _checkpoint_path(checkpoint_root: str, arm: str, payload_id: str, seed: str, training_job_id: str) -> str:
    prefix = "qwen_protected" if arm == "protected_trained" else "qwen_task_only_lora"
    return (
        f"{checkpoint_root}/{prefix}_{payload_id}_seed{seed}_{training_job_id}/"
        "checkpoints/natural_bucket_lora_last"
    )


def _score_units_for_candidate(
    *,
    row: Mapping[str, Any],
    seeds: Sequence[str],
    checkpoint_root: str,
    training_job_id: str,
) -> list[dict[str, str]]:
    payload_id = str(row.get("expected_payload_id", "")) or str(row.get("payload_id", ""))
    row_seed = str(row.get("seed", ""))
    lora_seeds = [row_seed] if row_seed else list(seeds)
    units = [
        {
            "scoring_model_condition": "base",
            "scoring_payload_id": payload_id,
            "scoring_seed": row_seed,
            "adapter_dir": "",
        }
    ]
    for seed in lora_seeds:
        units.append(
            {
                "scoring_model_condition": "protected_trained",
                "scoring_payload_id": payload_id,
                "scoring_seed": seed,
                "adapter_dir": _checkpoint_path(checkpoint_root, "protected_trained", payload_id, seed, training_job_id),
            }
        )
        units.append(
            {
                "scoring_model_condition": "task_only_lora",
                "scoring_payload_id": payload_id,
                "scoring_seed": seed,
                "adapter_dir": _checkpoint_path(checkpoint_root, "task_only_lora", payload_id, seed, training_job_id),
            }
        )
    return units


def _build_score_plan(
    *,
    candidates: Sequence[Mapping[str, Any]],
    examples_by_key: Mapping[tuple[str, ...], Mapping[str, Any]],
    seeds: Sequence[str],
    model_name: str,
    tokenizer_name: str,
    checkpoint_root: str,
    training_job_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    join_failures: list[str] = []
    for candidate in candidates:
        example = examples_by_key.get(_candidate_key(candidate))
        if example is None:
            join_failures.append(str(candidate.get("candidate_id", "")))
            continue
        bucket_token_texts = example.get("candidate_bucket_token_texts", {})
        if not isinstance(bucket_token_texts, Mapping):
            bucket_token_texts = {}
        bucket_to_token_ids = {
            str(bucket_id): _bucket_token_ids(tokens)
            for bucket_id, tokens in bucket_token_texts.items()
        }
        target_bucket = str(candidate.get("target_bucket", ""))
        target_ids = _bucket_token_ids(candidate.get("target_bucket_tokens", []))
        if not target_ids:
            target_ids = [int(value) for value in example.get("target_bucket_token_ids", [])]
        non_target_compatible_bucket_ids = [
            str(bucket_id)
            for bucket_id in candidate.get("compatible_bucket_ids", [])
            if str(bucket_id) != target_bucket
        ]
        non_target_compatible_token_ids = [
            token_id
            for bucket_id in non_target_compatible_bucket_ids
            for token_id in bucket_to_token_ids.get(str(bucket_id), [])
        ]
        for unit in _score_units_for_candidate(
            row=candidate,
            seeds=seeds,
            checkpoint_root=checkpoint_root,
            training_job_id=training_job_id,
        ):
            plan_id = "rtfm_" + stable_hash_hex(
                [
                    candidate.get("candidate_id", ""),
                    unit["scoring_model_condition"],
                    unit["scoring_payload_id"],
                    unit["scoring_seed"],
                ]
            )[:16]
            rows.append(
                {
                    "schema_name": PLAN_SCHEMA_NAME,
                    "plan_row_id": plan_id,
                    "candidate_id": candidate.get("candidate_id", ""),
                    "source_row_id": candidate.get("source_row_id", candidate.get("row_id", "")),
                    "source_model_condition": candidate.get("model_condition", ""),
                    "scoring_model_condition": unit["scoring_model_condition"],
                    "expected_payload_id": candidate.get("expected_payload_id", ""),
                    "scoring_payload_id": unit["scoring_payload_id"],
                    "source_seed": candidate.get("seed", ""),
                    "scoring_seed": unit["scoring_seed"],
                    "model_name": model_name,
                    "tokenizer_name": tokenizer_name,
                    "adapter_dir": unit["adapter_dir"],
                    "prompt_id": candidate.get("prompt_id", ""),
                    "prompt_slot": candidate.get("prompt_slot", ""),
                    "match_policy": candidate.get("match_policy", ""),
                    "drift_reason": candidate.get("drift_reason", ""),
                    "observed_token_class": candidate.get("observed_token_class", ""),
                    "prompt": candidate.get("prompt", ""),
                    "prefix_before_observed": candidate.get("prefix_before_observed", ""),
                    "repaired_prefix_text": str(candidate.get("prompt", "")) + str(candidate.get("prefix_before_observed", "")),
                    "target_bucket": target_bucket,
                    "target_bucket_token_ids": target_ids,
                    "bucket_to_token_ids": bucket_to_token_ids,
                    "compatible_bucket_ids": [str(value) for value in candidate.get("compatible_bucket_ids", [])],
                    "non_target_compatible_bucket_ids": non_target_compatible_bucket_ids,
                    "non_target_compatible_token_ids": non_target_compatible_token_ids,
                    "target_token_text": candidate.get("target_token_text", ""),
                    "observed_token_text": candidate.get("observed_token_text", ""),
                    "frame_index": candidate.get("frame_index", ""),
                    "frame_digit_index": candidate.get("frame_digit_index", ""),
                    "token_index": candidate.get("token_index", ""),
                    "paper_claim_allowed": False,
                    "training_started": False,
                    "generation_started": False,
                    "e2e_eval_started": False,
                    "model_scoring_started": False,
                    "result_claim": "repaired_teacher_forced_target_mass_probe_design_not_scored_not_far",
                }
            )
    return rows, join_failures


def _group_value(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    return "|".join(str(row.get(field, "")) if str(row.get(field, "")) else "<empty>" for field in fields)


def _slice_rows(candidates: Sequence[Mapping[str, Any]], score_plan: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    group_specs = [
        ("payload_id", ("expected_payload_id",)),
        ("seed", ("seed",)),
        ("condition", ("model_condition",)),
        ("drift_reason", ("drift_reason",)),
        ("token_class", ("observed_token_class",)),
        ("prompt_id", ("prompt_id",)),
        ("payload_seed", ("expected_payload_id", "seed")),
        ("condition_drift_reason", ("model_condition", "drift_reason")),
        ("payload_seed_condition", ("expected_payload_id", "seed", "model_condition")),
        ("prompt_id_condition", ("prompt_id", "model_condition")),
    ]
    output: list[dict[str, Any]] = []
    for group_kind, fields in group_specs:
        candidate_counts = Counter(_group_value(row, fields) for row in candidates)
        score_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in score_plan:
            source_projection = {
                "expected_payload_id": row.get("expected_payload_id", ""),
                "seed": row.get("source_seed", ""),
                "model_condition": row.get("source_model_condition", ""),
                "drift_reason": row.get("drift_reason", ""),
                "observed_token_class": row.get("observed_token_class", ""),
                "prompt_id": row.get("prompt_id", ""),
            }
            value = _group_value(source_projection, fields)
            score_counts[value][str(row.get("scoring_model_condition", ""))] += 1
        for value in sorted(candidate_counts):
            counts = score_counts[value]
            output.append(
                {
                    "group_kind": group_kind,
                    "group_value": value,
                    "candidate_rows": candidate_counts[value],
                    "scoring_plan_rows": sum(counts.values()),
                    "base_score_rows": counts["base"],
                    "protected_score_rows": counts["protected_trained"],
                    "task_only_score_rows": counts["task_only_lora"],
                }
            )
    return output


def _write_markdown(path: Path, summary: Mapping[str, Any], top_slices: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Repaired teacher-forced target-mass probe design",
        "",
        "This is an artifact-only design for scoring repaired branch-aware candidates. It did not load a model, score probabilities, generate text, train, rerun E2E, estimate FAR, or make a paper-facing claim.",
        "",
        "## Status",
        "",
        f"`{summary['status']}`",
        "",
        "## Inputs",
        "",
        f"- primary candidate JSONL: `{summary['inputs']['candidate_jsonl']}`",
        f"- balanced examples JSONL: `{summary['inputs']['balanced_examples_jsonl']}`",
        f"- branch interpretation summary: `{summary['inputs']['branch_summary_json']}`",
        "",
        "## Scoring Arms",
        "",
        "- base: Qwen/Qwen2.5-7B-Instruct with no adapter",
        "- protected_trained: payload/seed-matched `qwen_protected_*_846585` adapter",
        "- task_only_lora: payload/seed-matched `qwen_task_only_lora_*_846585` adapter",
        "",
        "Raw-source candidates have no source seed, so the design expands them across the configured seeds for the expected payload while keeping one base row.",
        "",
        "## Planned Rows",
        "",
        f"- candidate rows: `{summary['candidate_rows']}`",
        f"- score-plan rows: `{summary['score_plan_rows']}`",
        f"- bucket joins: `{summary['bucket_join']['matched_candidate_rows']}/{summary['candidate_rows']}`",
        "",
        "## Pass/Fail Thresholds",
        "",
        f"- aggregate protected-base target candidate mass lift must be at least `{summary['thresholds']['aggregate']['min_protected_minus_base_target_candidate_mass']}`",
        f"- aggregate protected-task-only target candidate mass lift must be at least `{summary['thresholds']['aggregate']['min_protected_minus_task_only_target_candidate_mass']}`",
        f"- target rank-1 lift over both controls must be at least `{summary['thresholds']['aggregate']['min_target_rank1_lift']}`",
        f"- all required slices with at least `{summary['thresholds']['slice_stability']['min_slice_candidate_rows']}` candidates must have positive protected lift",
        "",
        "A pass only permits decision review or repaired dataset preflight. It does not unlock training or E2E by itself.",
        "",
        "## Planned Slices",
        "",
        "| Slice | Candidate rows | Score rows | Base | Protected | Task-only |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in top_slices[:16]:
        lines.append(
            "| {group_kind}:{group_value} | {candidate_rows} | {scoring_plan_rows} | {base_score_rows} | {protected_score_rows} | {task_only_score_rows} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Next Allowed Action",
            "",
            "If model scoring is needed, submit exactly one allowlisted Slurm job that consumes this plan and writes repaired teacher-forced target-mass metrics. Do not train, generate, rerun E2E, run Llama, run same-family null, run sanitizer, aggregate FAR, or make paper-facing positive claims.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_design(
    *,
    candidate_jsonl: Path,
    branch_summary_json: Path,
    balanced_examples_jsonl: Path,
    output_dir: Path,
    model_name: str,
    tokenizer_name: str,
    checkpoint_root: str,
    training_job_id: str,
    payload_ids: Sequence[str],
    seeds: Sequence[str],
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "repaired_teacher_forced_target_mass_probe_design_summary.json",
        "score_plan": output_dir / "repaired_teacher_forced_target_mass_probe_scoring_plan.jsonl",
        "slice_plan": output_dir / "repaired_teacher_forced_target_mass_probe_slice_plan.csv",
        "markdown": output_dir / "repaired_teacher_forced_target_mass_probe_design.md",
    }
    for path in paths.values():
        if path.exists() and not force:
            raise FileExistsError(f"refusing to overwrite existing repaired probe design artifact: {path}")

    branch_summary = _read_json(branch_summary_json)
    candidate_rows_all = list(_iter_jsonl(candidate_jsonl))
    candidate_rows = [row for row in candidate_rows_all if _is_primary_candidate(row)]
    rejected_input_rows = len(candidate_rows_all) - len(candidate_rows)
    example_rows = list(_iter_jsonl(balanced_examples_jsonl))
    examples_by_key = _example_index(example_rows)
    score_plan_rows, join_failures = _build_score_plan(
        candidates=candidate_rows,
        examples_by_key=examples_by_key,
        seeds=seeds,
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        checkpoint_root=checkpoint_root,
        training_job_id=training_job_id,
    )
    if join_failures:
        raise ValueError(f"Missing balanced example join for candidate ids: {join_failures[:5]}")
    slice_rows = _slice_rows(candidate_rows, score_plan_rows)
    candidate_counts_by_condition = Counter(str(row.get("model_condition", "")) for row in candidate_rows)
    candidate_counts_by_drift = Counter(str(row.get("drift_reason", "")) for row in candidate_rows)
    candidate_counts_by_token = Counter(str(row.get("observed_token_class", "")) for row in candidate_rows)
    score_counts = Counter(str(row.get("scoring_model_condition", "")) for row in score_plan_rows)
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_REPAIRED_TEACHER_FORCED_TARGET_MASS_PROBE_DESIGN_NOT_SCORED",
        "protocol_id": "natural_evidence_v1",
        "phase": "POST_846699_BRANCH_AWARE_SCORE_INTERPRETATION_COMPLETE",
        "paper_claim_allowed": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "model_scoring_started": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "result_claim": "repaired_teacher_forced_target_mass_probe_design_not_scored_not_recovery_not_far",
        "inputs": {
            "candidate_jsonl": str(candidate_jsonl),
            "branch_summary_json": str(branch_summary_json),
            "branch_summary_status": branch_summary.get("status", ""),
            "balanced_examples_jsonl": str(balanced_examples_jsonl),
            "payload_ids": list(payload_ids),
            "seeds": list(seeds),
        },
        "candidate_filters": {
            "input_file_expected_role": "primary repaired target-mass probe candidates",
            "primary_probe_candidate": True,
            "candidate_tiers": sorted(PRIMARY_TIERS),
            "branch_aware_proxy_pass": True,
            "training_started": False,
            "generation_started": False,
            "e2e_eval_started": False,
            "filtered_out_input_rows": rejected_input_rows,
        },
        "candidate_rows": len(candidate_rows),
        "candidate_counts_by_condition": dict(sorted(candidate_counts_by_condition.items())),
        "candidate_counts_by_drift_reason": dict(sorted(candidate_counts_by_drift.items())),
        "candidate_counts_by_token_class": dict(sorted(candidate_counts_by_token.items())),
        "score_plan_rows": len(score_plan_rows),
        "score_plan_rows_by_model_condition": dict(sorted(score_counts.items())),
        "scoring_arms": {
            "base": {
                "model_name": model_name,
                "tokenizer_name": tokenizer_name,
                "adapter_dir": "",
            },
            "protected_trained": {
                "model_name": model_name,
                "tokenizer_name": tokenizer_name,
                "adapter_template": (
                    f"{checkpoint_root}/qwen_protected_{{payload_id}}_seed{{seed}}_"
                    f"{training_job_id}/checkpoints/natural_bucket_lora_last"
                ),
            },
            "task_only_lora": {
                "model_name": model_name,
                "tokenizer_name": tokenizer_name,
                "adapter_template": (
                    f"{checkpoint_root}/qwen_task_only_lora_{{payload_id}}_seed{{seed}}_"
                    f"{training_job_id}/checkpoints/natural_bucket_lora_last"
                ),
            },
        },
        "repaired_prefix_contract": {
            "context": "prompt + prefix_before_observed",
            "target_token_excluded_from_context": True,
            "target_bucket_source": "target_bucket and target_bucket_tokens from primary candidate row",
            "full_bucket_source": "candidate_bucket_token_texts joined one-to-one from balanced branch-aware examples",
            "non_target_compatible_mass": "sum compatible bucket masses excluding target_bucket",
        },
        "slice_outputs_required": [
            "payload_id",
            "seed",
            "source_model_condition",
            "drift_reason",
            "observed_token_class",
            "prompt_id",
            "payload_id|seed",
            "source_model_condition|drift_reason",
            "payload_id|seed|source_model_condition",
            "prompt_id|source_model_condition",
        ],
        "metrics_required": [
            "P(target bucket | repaired prefix)",
            "P(non-target compatible buckets | repaired prefix)",
            "target rank",
            "target-vs-best-other margin",
            "protected-base target candidate mass lift",
            "protected-task-only target candidate mass lift",
            "target rank-1 lift over controls",
        ],
        "thresholds": {
            "aggregate": {
                "min_protected_minus_base_target_candidate_mass": 0.05,
                "min_protected_minus_task_only_target_candidate_mass": 0.05,
                "min_target_rank1_lift": 0.05,
            },
            "slice_stability": {
                "min_slice_candidate_rows": 4,
                "required_min_lift_for_slices": 0.0,
                "high_n_slice_candidate_rows": 8,
                "min_high_n_slice_lift": 0.02,
                "max_negative_lift_allowed": -0.02,
            },
            "decision_rule": "passing thresholds permits review/preflight only; it does not authorize training or E2E",
        },
        "bucket_join": {
            "matched_candidate_rows": len(candidate_rows),
            "join_key": [
                "prompt_id",
                "model_condition",
                "expected_payload_id",
                "seed",
                "query_index",
                "frame_index",
                "frame_digit_index",
                "observed_token_text",
                "target_bucket",
                "match_policy",
            ],
        },
        "outputs": {
            "score_plan_jsonl": paths["score_plan"].name,
            "slice_plan_csv": paths["slice_plan"].name,
            "design_md": paths["markdown"].name,
        },
        "next_allowed_action": "If needed, submit exactly one Slurm-scored repaired teacher-forced target-mass probe; do not train, generate, rerun E2E, or make claims.",
    }
    write_json(paths["summary"], summary)
    write_jsonl(paths["score_plan"], score_plan_rows)
    write_csv(paths["slice_plan"], slice_rows, SLICE_FIELDS)
    top_slices = sorted(slice_rows, key=lambda row: (-int(row["candidate_rows"]), str(row["group_kind"]), str(row["group_value"])))
    _write_markdown(paths["markdown"], summary, top_slices)
    print(json.dumps(summary, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_design(
        candidate_jsonl=_resolve(args.candidate_jsonl),
        branch_summary_json=_resolve(args.branch_summary_json),
        balanced_examples_jsonl=_resolve(args.balanced_examples_jsonl),
        output_dir=_resolve(args.output_dir),
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        checkpoint_root=args.checkpoint_root.rstrip("/"),
        training_job_id=str(args.training_job_id),
        payload_ids=_parse_csv_list(args.payload_ids),
        seeds=_parse_csv_list(args.seeds),
        force=bool(args.force),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
