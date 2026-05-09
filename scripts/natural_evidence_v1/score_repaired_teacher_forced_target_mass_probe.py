from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.natural_evidence_v1.common import read_jsonl, write_csv, write_json, write_jsonl
from scripts.natural_evidence_v1.probe_qwen_teacher_forced_bucket_mass import (
    _classify_token_ids,
    _decode_token,
    _format_float,
    _load_model,
    _read_json,
    _release_model,
    _resolve,
    _unique_ints,
    bucket_probe_from_token_logits,
)
from scripts.natural_evidence_v1.replay_qwen_frame_completion import _hash_file, _rate


SCHEMA_NAME = "natural_evidence_v1_repaired_teacher_forced_target_mass_probe_scored_v1"
ROW_SCHEMA_NAME = "natural_evidence_v1_repaired_teacher_forced_target_mass_score_row_v1"
GROUP_FIELDS = [
    "group_kind",
    "group_value",
    "scored_rows",
    "mean_target_candidate_mass",
    "mean_non_target_compatible_candidate_mass",
    "mean_best_other_candidate_mass",
    "mean_target_margin",
    "mean_full_vocab_target_mass",
    "mean_target_rank",
    "target_rank1_rate",
    "target_rank_le2_rate",
    "positive_margin_rate",
]
ROW_FIELDS = [
    "plan_row_id",
    "candidate_id",
    "scoring_model_condition",
    "scoring_payload_id",
    "scoring_seed",
    "adapter_dir",
    "source_model_condition",
    "source_seed",
    "expected_payload_id",
    "prompt_id",
    "prompt_slot",
    "token_index",
    "frame_index",
    "frame_digit_index",
    "drift_reason",
    "observed_token_class",
    "observed_token_text",
    "target_token_text",
    "target_bucket",
    "target_bucket_token_class",
    "target_candidate_mass",
    "non_target_compatible_candidate_mass",
    "best_other_candidate_mass",
    "target_margin",
    "target_rank",
    "full_vocab_target_mass",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Slurm/GPU scorer for repaired teacher-forced target-mass probe "
            "plans. This scores existing plan prefixes only; it does not train, "
            "generate, rerun E2E, decode payloads, estimate FAR, or make claims."
        )
    )
    parser.add_argument("--score-plan-jsonl", required=True)
    parser.add_argument("--design-summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _token_ids(values: object) -> list[int]:
    if not isinstance(values, list):
        return []
    output: list[int] = []
    for value in values:
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            continue
    return output


def _bucket_to_token_ids(row: Mapping[str, Any]) -> dict[str, list[int]]:
    value = row.get("bucket_to_token_ids", {})
    if not isinstance(value, dict):
        return {}
    return {str(bucket_id): _token_ids(token_ids) for bucket_id, token_ids in value.items()}


def _read_plan(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows = [dict(row) for row in read_jsonl(path)]
    if max_rows > 0:
        rows = rows[:max_rows]
    if not rows:
        raise ValueError(f"no score-plan rows found: {path}")
    bad_rows = [
        str(row.get("plan_row_id", f"index_{index}"))
        for index, row in enumerate(rows)
        if row.get("schema_name") != "natural_evidence_v1_repaired_teacher_forced_target_mass_score_plan_v1"
        or row.get("model_scoring_started") is not False
        or row.get("training_started") is not False
        or row.get("generation_started") is not False
        or row.get("e2e_eval_started") is not False
        or row.get("paper_claim_allowed") is not False
    ]
    if bad_rows:
        raise ValueError(f"score plan rows failed claim/control checks: {bad_rows[:8]}")
    return rows


def _candidate_token_ids(rows: Sequence[Mapping[str, Any]]) -> set[int]:
    token_ids: set[int] = set()
    for row in rows:
        for bucket_tokens in _bucket_to_token_ids(row).values():
            token_ids.update(int(token_id) for token_id in bucket_tokens)
        token_ids.update(_token_ids(row.get("target_bucket_token_ids", [])))
    return token_ids


def _batch_contexts(
    *,
    torch_module: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    max_length: int,
    device: Any,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    pad_token_id = int(tokenizer.pad_token_id)
    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    specs: list[dict[str, Any]] = []
    max_width = 0
    for row in rows:
        try:
            context_ids = [
                int(token_id)
                for token_id in tokenizer.encode(
                    str(row.get("repaired_prefix_text", "")),
                    add_special_tokens=False,
                )
            ]
        except TypeError:
            context_ids = [int(token_id) for token_id in tokenizer.encode(str(row.get("repaired_prefix_text", "")))]
        if len(context_ids) > max_length:
            context_ids = context_ids[-max_length:]
        if not context_ids:
            continue
        bucket_to_token_ids = _bucket_to_token_ids(row)
        target_bucket = str(row.get("target_bucket", ""))
        candidate_ids = _unique_ints(
            [token_id for token_ids in bucket_to_token_ids.values() for token_id in token_ids]
        )
        if target_bucket not in bucket_to_token_ids or not candidate_ids:
            continue
        batch_row_index = len(input_rows)
        specs.append(
            {
                "batch_row_index": batch_row_index,
                "prediction_index": len(context_ids) - 1,
                "bucket_to_token_ids": bucket_to_token_ids,
                "candidate_token_ids": candidate_ids,
                "target_bucket": target_bucket,
                "target_bucket_token_ids": _token_ids(row.get("target_bucket_token_ids", [])),
                "non_target_compatible_bucket_ids": [
                    str(bucket_id) for bucket_id in row.get("non_target_compatible_bucket_ids", [])
                ],
                "row": row,
            }
        )
        input_rows.append(context_ids)
        attention_rows.append([1] * len(context_ids))
        max_width = max(max_width, len(context_ids))
    for input_ids, attention in zip(input_rows, attention_rows, strict=True):
        pad_width = max_width - len(input_ids)
        if pad_width > 0:
            input_ids.extend([pad_token_id] * pad_width)
            attention.extend([0] * pad_width)
    if not input_rows:
        raise ValueError("no score-plan rows had valid repaired prefixes and buckets")
    return (
        torch_module.tensor(input_rows, dtype=torch_module.long, device=device),
        torch_module.tensor(attention_rows, dtype=torch_module.long, device=device),
        specs,
    )


def _score_plan_rows(
    *,
    torch_module: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with torch_module.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            input_ids, attention_mask, specs = _batch_contexts(
                torch_module=torch_module,
                tokenizer=tokenizer,
                rows=batch,
                max_length=max_length,
                device=device,
            )
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            for spec in specs:
                source = spec["row"]
                row_logits = logits[int(spec["batch_row_index"]), int(spec["prediction_index"]), :].float()
                token_logits = {
                    int(token_id): float(row_logits[int(token_id)].detach().cpu().item())
                    for token_id in spec["candidate_token_ids"]
                }
                probe = bucket_probe_from_token_logits(
                    token_logits=token_logits,
                    bucket_to_token_ids=spec["bucket_to_token_ids"],
                    target_bucket=str(spec["target_bucket"]),
                )
                target_token_ids = _unique_ints(spec["target_bucket_token_ids"])
                if target_token_ids:
                    target_logits = row_logits[target_token_ids]
                    log_denom = torch_module.logsumexp(row_logits, dim=0)
                    full_vocab_target_mass = float(
                        torch_module.exp(torch_module.logsumexp(target_logits, dim=0) - log_denom)
                        .detach()
                        .cpu()
                        .item()
                    )
                else:
                    full_vocab_target_mass = 0.0
                bucket_masses = dict(probe["bucket_masses"])
                non_target_compatible_mass = sum(
                    float(bucket_masses.get(bucket_id, 0.0))
                    for bucket_id in spec["non_target_compatible_bucket_ids"]
                )
                output.append(
                    {
                        "schema_name": ROW_SCHEMA_NAME,
                        "plan_row_id": str(source.get("plan_row_id", "")),
                        "candidate_id": str(source.get("candidate_id", "")),
                        "scoring_model_condition": str(source.get("scoring_model_condition", "")),
                        "scoring_payload_id": str(source.get("scoring_payload_id", "")),
                        "scoring_seed": str(source.get("scoring_seed", "")),
                        "adapter_dir": str(source.get("adapter_dir", "")),
                        "source_model_condition": str(source.get("source_model_condition", "")),
                        "source_seed": str(source.get("source_seed", "")),
                        "expected_payload_id": str(source.get("expected_payload_id", "")),
                        "prompt_id": str(source.get("prompt_id", "")),
                        "prompt_slot": str(source.get("prompt_slot", "")),
                        "token_index": str(source.get("token_index", "")),
                        "frame_index": str(source.get("frame_index", "")),
                        "frame_digit_index": str(source.get("frame_digit_index", "")),
                        "drift_reason": str(source.get("drift_reason", "")),
                        "observed_token_class": str(source.get("observed_token_class", "")),
                        "observed_token_text": str(source.get("observed_token_text", "")),
                        "target_token_text": str(source.get("target_token_text", "")),
                        "target_bucket": str(source.get("target_bucket", "")),
                        "target_bucket_token_ids": target_token_ids,
                        "target_bucket_token_texts": [
                            _decode_token(tokenizer, token_id) for token_id in target_token_ids
                        ],
                        "target_bucket_token_class": _classify_token_ids(tokenizer, target_token_ids),
                        "bucket_masses": bucket_masses,
                        "target_candidate_mass": float(probe["target_candidate_mass"]),
                        "non_target_compatible_candidate_mass": float(non_target_compatible_mass),
                        "best_other_candidate_mass": float(probe["best_other_candidate_mass"]),
                        "target_margin": float(probe["target_margin"]),
                        "target_rank": int(probe["target_rank"]),
                        "full_vocab_target_mass": full_vocab_target_mass,
                        "paper_claim_allowed": False,
                        "training_started": False,
                        "generation_started": False,
                        "e2e_eval_started": False,
                        "not_payload_recovery": True,
                        "not_full_far": True,
                        "result_claim": "repaired_teacher_forced_target_mass_probe_scored_not_recovery_not_far",
                    }
                )
    return output


def _stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    rank1 = sum(1 for row in rows if int(row.get("target_rank", 0)) == 1)
    rank_le2 = sum(1 for row in rows if 0 < int(row.get("target_rank", 0)) <= 2)
    positive = sum(1 for row in rows if float(row.get("target_margin", 0.0)) > 0.0)
    return {
        "scored_rows": total,
        "mean_target_candidate_mass": _rate(
            sum(float(row.get("target_candidate_mass", 0.0)) for row in rows), total
        ),
        "mean_non_target_compatible_candidate_mass": _rate(
            sum(float(row.get("non_target_compatible_candidate_mass", 0.0)) for row in rows), total
        ),
        "mean_best_other_candidate_mass": _rate(
            sum(float(row.get("best_other_candidate_mass", 0.0)) for row in rows), total
        ),
        "mean_target_margin": _rate(sum(float(row.get("target_margin", 0.0)) for row in rows), total),
        "mean_full_vocab_target_mass": _rate(
            sum(float(row.get("full_vocab_target_mass", 0.0)) for row in rows), total
        ),
        "mean_target_rank": _rate(sum(float(row.get("target_rank", 0.0)) for row in rows), total),
        "target_rank1_rate": _rate(rank1, total),
        "target_rank_le2_rate": _rate(rank_le2, total),
        "positive_margin_rate": _rate(positive, total),
    }


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groupers = [
        ("all", lambda row: "all"),
        ("scoring_model_condition", lambda row: row.get("scoring_model_condition", "")),
        ("scoring_payload_id", lambda row: row.get("scoring_payload_id", "")),
        ("scoring_seed", lambda row: row.get("scoring_seed", "")),
        ("source_model_condition", lambda row: row.get("source_model_condition", "")),
        ("drift_reason", lambda row: row.get("drift_reason", "")),
        ("observed_token_class", lambda row: row.get("observed_token_class", "")),
        ("prompt_id", lambda row: row.get("prompt_id", "")),
        ("payload_seed", lambda row: f"{row.get('scoring_payload_id', '')}|{row.get('scoring_seed', '')}"),
        (
            "source_model_condition__drift_reason",
            lambda row: f"{row.get('source_model_condition', '')}|{row.get('drift_reason', '')}",
        ),
        (
            "payload_seed_source_model_condition",
            lambda row: (
                f"{row.get('scoring_payload_id', '')}|{row.get('scoring_seed', '')}|"
                f"{row.get('source_model_condition', '')}"
            ),
        ),
        (
            "prompt_id_source_model_condition",
            lambda row: f"{row.get('prompt_id', '')}|{row.get('source_model_condition', '')}",
        ),
    ]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for group_kind, getter in groupers:
            grouped[(group_kind, str(getter(row)))].append(row)
    output: list[dict[str, Any]] = []
    for (group_kind, group_value), group in sorted(grouped.items()):
        output.append({"group_kind": group_kind, "group_value": group_value, **_stats(group)})
    return output


def _format_group_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        formatted = dict(row)
        for field in GROUP_FIELDS:
            if field.startswith("mean_") or field.endswith("_rate"):
                formatted[field] = _format_float(float(row[field]))
        output.append(formatted)
    return output


def _format_position_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        formatted = {field: row.get(field, "") for field in ROW_FIELDS}
        for field in (
            "target_candidate_mass",
            "non_target_compatible_candidate_mass",
            "best_other_candidate_mass",
            "target_margin",
            "full_vocab_target_mass",
        ):
            formatted[field] = _format_float(float(row[field]))
        output.append(formatted)
    return output


def _condition_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("scoring_model_condition", ""))].append(row)
    for condition, group in sorted(by_condition.items()):
        output[condition] = _stats(group)
    return output


def _metric(stats: Mapping[str, Mapping[str, Any]], condition: str, field: str) -> float:
    return float(stats.get(condition, {}).get(field, 0.0))


def run_scoring(args: argparse.Namespace) -> dict[str, Any]:
    score_plan_jsonl = _resolve(args.score_plan_jsonl)
    design_summary_json = _resolve(args.design_summary_json)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "repaired_teacher_forced_target_mass_probe_score_summary.json"
    if summary_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing repaired score summary: {summary_path}")

    design_summary = _read_json(design_summary_json)
    plan_rows = _read_plan(score_plan_jsonl, int(args.max_rows))
    score_units: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in plan_rows:
        score_units[
            (
                str(row.get("scoring_model_condition", "")),
                str(row.get("scoring_payload_id", "")),
                str(row.get("scoring_seed", "")),
                str(row.get("model_name", "")),
                str(row.get("adapter_dir", "")),
            )
        ].append(row)

    scored_rows: list[dict[str, Any]] = []
    for (condition, payload_id, seed, model_name, adapter_dir), unit_rows in sorted(score_units.items()):
        adapter_path = _resolve(adapter_dir) if adapter_dir else None
        torch_module, tokenizer, model, device = _load_model(
            model_name=model_name,
            tokenizer_name=str(unit_rows[0].get("tokenizer_name", model_name)),
            adapter_dir=adapter_path,
            require_cuda=bool(args.require_cuda),
        )
        try:
            scored_rows.extend(
                _score_plan_rows(
                    torch_module=torch_module,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    rows=unit_rows,
                    batch_size=max(1, int(args.batch_size)),
                    max_length=max(1, int(args.max_length)),
                )
            )
        finally:
            _release_model(torch_module, model)

    group_rows = _group_rows(scored_rows)
    condition_stats = _condition_stats(scored_rows)
    protected_minus_base = (
        _metric(condition_stats, "protected_trained", "mean_target_candidate_mass")
        - _metric(condition_stats, "base", "mean_target_candidate_mass")
    )
    protected_minus_task = (
        _metric(condition_stats, "protected_trained", "mean_target_candidate_mass")
        - _metric(condition_stats, "task_only_lora", "mean_target_candidate_mass")
    )
    protected_rank1_minus_task = (
        _metric(condition_stats, "protected_trained", "target_rank1_rate")
        - _metric(condition_stats, "task_only_lora", "target_rank1_rate")
    )
    thresholds = design_summary.get("thresholds", {}) if isinstance(design_summary.get("thresholds", {}), dict) else {}
    aggregate_thresholds = thresholds.get("aggregate", {}) if isinstance(thresholds.get("aggregate", {}), dict) else {}
    threshold_pass = (
        protected_minus_base >= float(aggregate_thresholds.get("min_protected_minus_base_target_candidate_mass", 0.05))
        and protected_minus_task >= float(aggregate_thresholds.get("min_protected_minus_task_only_target_candidate_mass", 0.05))
        and protected_rank1_minus_task >= float(aggregate_thresholds.get("min_target_rank1_lift", 0.05))
    )
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_REPAIRED_TEACHER_FORCED_TARGET_MASS_PROBE_SCORED_NOT_RECOVERY_NOT_FAR",
        "score_plan_rows": len(plan_rows),
        "scored_rows": len(scored_rows),
        "score_units": len(score_units),
        "score_plan_rows_by_model_condition": dict(
            sorted(Counter(str(row.get("scoring_model_condition", "")) for row in plan_rows).items())
        ),
        "scored_rows_by_model_condition": dict(
            sorted(Counter(str(row.get("scoring_model_condition", "")) for row in scored_rows).items())
        ),
        "inputs": {
            "score_plan_jsonl": str(score_plan_jsonl),
            "score_plan_hash": _hash_file(score_plan_jsonl),
            "design_summary_json": str(design_summary_json),
            "design_summary_hash": _hash_file(design_summary_json),
            "design_status": design_summary.get("status", ""),
            "max_rows": int(args.max_rows),
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
            "candidate_token_count": len(_candidate_token_ids(plan_rows)),
        },
        "condition_stats": condition_stats,
        "aggregate_lifts": {
            "protected_minus_base_target_candidate_mass": protected_minus_base,
            "protected_minus_task_only_target_candidate_mass": protected_minus_task,
            "protected_minus_task_only_target_rank1_rate": protected_rank1_minus_task,
            "threshold_pass": threshold_pass,
            "thresholds": thresholds,
        },
        "outputs": {
            "summary_json": summary_path.name,
            "rows_jsonl": "repaired_teacher_forced_target_mass_probe_score_rows.jsonl",
            "rows_csv": "repaired_teacher_forced_target_mass_probe_score_rows.csv",
            "by_group_csv": "repaired_teacher_forced_target_mass_probe_score_by_group.csv",
        },
        "paper_claim_allowed": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "result_claim": "repaired_teacher_forced_target_mass_probe_scored_not_recovery_not_far",
        "limitations": [
            "Teacher-forced target-mass scoring only; no free generation is performed.",
            "Passing thresholds would permit review/preflight only, not training or E2E.",
            "This is not payload recovery, FAR, sanitizer evidence, or a paper-facing claim.",
        ],
    }
    write_json(summary_path, summary)
    write_jsonl(output_dir / "repaired_teacher_forced_target_mass_probe_score_rows.jsonl", scored_rows)
    write_csv(
        output_dir / "repaired_teacher_forced_target_mass_probe_score_rows.csv",
        _format_position_rows(scored_rows),
        ROW_FIELDS,
    )
    write_csv(
        output_dir / "repaired_teacher_forced_target_mass_probe_score_by_group.csv",
        _format_group_rows(group_rows),
        GROUP_FIELDS,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_scoring(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "scored_rows": summary["scored_rows"],
                "output_dir": str(_resolve(args.output_dir)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
