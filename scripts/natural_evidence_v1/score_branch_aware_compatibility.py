from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.natural_evidence_v1.common import write_csv, write_json, write_jsonl


SCHEMA_NAME = "natural_evidence_v1_branch_aware_compatibility_score_v1"
ROW_SCHEMA = "natural_evidence_v1_branch_aware_compatibility_score_row_v1"
ROW_FIELDS = [
    "row_id",
    "model_condition",
    "payload_id",
    "expected_payload_id",
    "seed",
    "match_policy",
    "drift_reason",
    "observed_token_class",
    "target_bucket",
    "observed_token_text",
    "target_token_text",
    "response_delta_nll_per_token",
    "suffix_delta_nll_per_token",
    "response_naturalness_proxy_pass",
    "suffix_preserving_proxy_pass",
    "branch_aware_proxy_pass",
    "original_response_nll_per_token",
    "repaired_response_nll_per_token",
    "observed_suffix_nll_per_token",
    "target_suffix_nll_per_token",
    "suffix_token_count",
    "paper_claim_allowed",
    "training_started",
    "generation_started",
    "e2e_eval_started",
]
GROUP_FIELDS = [
    "group_kind",
    "group_value",
    "rows",
    "response_naturalness_proxy_pass_rows",
    "response_naturalness_proxy_pass_rate",
    "suffix_preserving_proxy_pass_rows",
    "suffix_preserving_proxy_pass_rate",
    "branch_aware_proxy_pass_rows",
    "branch_aware_proxy_pass_rate",
    "mean_response_delta_nll_per_token",
    "mean_suffix_delta_nll_per_token",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Slurm-scored branch-aware compatibility proxy diagnostic. Reads the "
            "balanced branch-aware scoring plan plus local-suffix dry-run rows, "
            "then scores original/repaired responses and local suffix windows "
            "under a reference causal LM. This does not train, generate text, "
            "rerun E2E, decode payload recovery, or estimate FAR."
        )
    )
    parser.add_argument("--scoring-plan-jsonl", required=True)
    parser.add_argument("--dry-run-rows-jsonl", required=True)
    parser.add_argument("--prepared-summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-response-delta-per-token", type=float, default=0.5)
    parser.add_argument("--max-suffix-delta-per-token", type=float, default=1.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{float(value):.17g}"


def _mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return sum(clean) / len(clean) if clean else float("nan")


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _load_rows(plan_jsonl: Path, dry_run_rows_jsonl: Path, max_rows: int) -> list[dict[str, Any]]:
    plan_by_row_id = {str(row.get("row_id", "")): row for row in _iter_jsonl(plan_jsonl)}
    rows: list[dict[str, Any]] = []
    for dry_row in _iter_jsonl(dry_run_rows_jsonl):
        row_id = str(dry_row.get("row_id", ""))
        plan_row = plan_by_row_id.get(row_id, {})
        merged = {**plan_row, **dry_row}
        if str(merged.get("dry_run_status", "")) != "REPAIR_DRY_RUN_TEXT_SUBSTITUTION_READY_NOT_REGENERATED":
            continue
        rows.append(merged)
        if max_rows > 0 and len(rows) >= max_rows:
            break
    return rows


def scoring_pairs_from_row(row: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    prompt = str(row.get("prompt", ""))
    original_response = str(row.get("original_response_text", ""))
    repaired_response = str(row.get("repaired_response_text", ""))
    prefix_before = str(row.get("prefix_before_observed", ""))
    observed_match = str(row.get("observed_match_text", "")) or str(row.get("observed_token_text", ""))
    target_token = str(row.get("target_token_text", ""))
    suffix_window = str(row.get("local_suffix_window_after_observed", ""))
    return {
        "original_response": (prompt, original_response),
        "repaired_response": (prompt, repaired_response),
        "observed_suffix_window": (prompt + prefix_before + observed_match, suffix_window),
        "target_suffix_window": (prompt + prefix_before + target_token, suffix_window),
    }


def _score_pass_flags(
    *,
    response_delta: float,
    suffix_delta: float,
    suffix_token_count: int,
    max_response_delta_per_token: float,
    max_suffix_delta_per_token: float,
) -> dict[str, bool]:
    response_pass = not math.isnan(response_delta) and response_delta <= max_response_delta_per_token
    suffix_pass = (
        suffix_token_count > 0
        and not math.isnan(suffix_delta)
        and suffix_delta <= max_suffix_delta_per_token
    )
    return {
        "response_naturalness_proxy_pass": response_pass,
        "suffix_preserving_proxy_pass": suffix_pass,
        "branch_aware_proxy_pass": response_pass and suffix_pass,
    }


def _encode_no_special(tokenizer: Any, text: str) -> list[int]:
    try:
        return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
    except TypeError:
        return [int(token_id) for token_id in tokenizer.encode(text)]


def _score_continuations(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    contexts: Sequence[str],
    continuations: Sequence[str],
    batch_size: int,
    max_length: int,
) -> list[dict[str, float]]:
    import torch

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    encoded: list[tuple[list[int], list[int]]] = []
    for context, continuation in zip(contexts, continuations):
        context_ids = _encode_no_special(tokenizer, str(context))
        continuation_ids = _encode_no_special(tokenizer, str(continuation))
        if not continuation_ids:
            encoded.append((context_ids[-max_length:], []))
            continue
        if len(continuation_ids) >= max_length:
            continuation_ids = continuation_ids[: max_length - 1]
            context_ids = context_ids[-1:]
        overflow = len(context_ids) + len(continuation_ids) - max_length
        if overflow > 0:
            context_ids = context_ids[overflow:]
        encoded.append((context_ids, continuation_ids))

    outputs: list[dict[str, float]] = []
    for start in range(0, len(encoded), int(batch_size)):
        batch = encoded[start : start + int(batch_size)]
        max_seq = max((len(context_ids) + len(cont_ids) for context_ids, cont_ids in batch), default=0)
        if max_seq <= 1:
            outputs.extend(
                {"nll_sum": float("nan"), "token_count": 0.0, "nll_per_token": float("nan")}
                for _ in batch
            )
            continue
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for context_ids, cont_ids in batch:
            input_ids = context_ids + cont_ids
            labels = [-100] * len(context_ids) + cont_ids
            padding = max_seq - len(input_ids)
            input_rows.append(input_ids + [int(pad_token_id)] * padding)
            label_rows.append(labels + [-100] * padding)
        input_tensor = torch.tensor(input_rows, dtype=torch.long, device=device)
        labels_tensor = torch.tensor(label_rows, dtype=torch.long, device=device)
        attention_mask = (input_tensor != int(pad_token_id)).long()
        with torch.no_grad():
            logits = model(input_ids=input_tensor, attention_mask=attention_mask).logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = labels_tensor[:, 1:]
        mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~mask, 0)
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_nll = token_nll.masked_fill(~mask, 0.0)
        nll_sums = token_nll.sum(dim=1)
        token_counts = mask.sum(dim=1)
        for nll_sum, token_count in zip(nll_sums.detach().cpu(), token_counts.detach().cpu()):
            count = int(token_count.item())
            total = float(nll_sum.item())
            outputs.append(
                {
                    "nll_sum": total,
                    "token_count": float(count),
                    "nll_per_token": total / count if count else float("nan"),
                }
            )
    return outputs


def _group_rows(rows: Sequence[Mapping[str, Any]], group_kind: str, key_fn: Any) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(row)
    output: list[dict[str, Any]] = []
    for group_value, group_rows in sorted(grouped.items()):
        total = len(group_rows)
        response_pass = sum(1 for row in group_rows if row.get("response_naturalness_proxy_pass"))
        suffix_pass = sum(1 for row in group_rows if row.get("suffix_preserving_proxy_pass"))
        branch_pass = sum(1 for row in group_rows if row.get("branch_aware_proxy_pass"))
        response_deltas = [_as_float(row.get("response_delta_nll_per_token")) for row in group_rows]
        suffix_deltas = [_as_float(row.get("suffix_delta_nll_per_token")) for row in group_rows]
        output.append(
            {
                "group_kind": group_kind,
                "group_value": group_value,
                "rows": total,
                "response_naturalness_proxy_pass_rows": response_pass,
                "response_naturalness_proxy_pass_rate": _rate(response_pass, total),
                "suffix_preserving_proxy_pass_rows": suffix_pass,
                "suffix_preserving_proxy_pass_rate": _rate(suffix_pass, total),
                "branch_aware_proxy_pass_rows": branch_pass,
                "branch_aware_proxy_pass_rate": _rate(branch_pass, total),
                "mean_response_delta_nll_per_token": _mean(response_deltas),
                "mean_suffix_delta_nll_per_token": _mean(suffix_deltas),
            }
        )
    return output


def run_scoring(
    *,
    scoring_plan_jsonl: Path,
    dry_run_rows_jsonl: Path,
    prepared_summary_json: Path,
    output_dir: Path,
    model_name: str,
    tokenizer_name: str,
    batch_size: int,
    max_length: int,
    max_response_delta_per_token: float,
    max_suffix_delta_per_token: float,
    max_rows: int,
    require_cuda: bool,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "branch_aware_compatibility_score_summary.json"
    if summary_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing branch-aware score summary: {summary_path}")
    prepared_summary = _read_json(prepared_summary_json)
    rows = _load_rows(scoring_plan_jsonl, dry_run_rows_jsonl, max_rows)
    if not rows:
        raise ValueError("No text-substitution-ready rows found for branch-aware scoring")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    pair_names = [
        "original_response",
        "repaired_response",
        "observed_suffix_window",
        "target_suffix_window",
    ]
    all_contexts: list[str] = []
    all_continuations: list[str] = []
    row_pairs: list[dict[str, tuple[str, str]]] = []
    for row in rows:
        pairs = scoring_pairs_from_row(row)
        row_pairs.append(pairs)
        for pair_name in pair_names:
            context, continuation = pairs[pair_name]
            all_contexts.append(context)
            all_continuations.append(continuation)
    scores = _score_continuations(
        model=model,
        tokenizer=tokenizer,
        device=device,
        contexts=all_contexts,
        continuations=all_continuations,
        batch_size=batch_size,
        max_length=max_length,
    )

    scored_rows: list[dict[str, Any]] = []
    score_index = 0
    for row in rows:
        row_scores: dict[str, dict[str, float]] = {}
        for pair_name in pair_names:
            row_scores[pair_name] = scores[score_index]
            score_index += 1
        response_delta = (
            float(row_scores["repaired_response"]["nll_per_token"])
            - float(row_scores["original_response"]["nll_per_token"])
        )
        suffix_delta = (
            float(row_scores["target_suffix_window"]["nll_per_token"])
            - float(row_scores["observed_suffix_window"]["nll_per_token"])
        )
        flags = _score_pass_flags(
            response_delta=response_delta,
            suffix_delta=suffix_delta,
            suffix_token_count=int(row_scores["target_suffix_window"]["token_count"]),
            max_response_delta_per_token=max_response_delta_per_token,
            max_suffix_delta_per_token=max_suffix_delta_per_token,
        )
        scored = {
            "schema_name": ROW_SCHEMA,
            "row_id": row.get("row_id", ""),
            "model_condition": row.get("model_condition", ""),
            "payload_id": row.get("payload_id", ""),
            "expected_payload_id": row.get("expected_payload_id", ""),
            "seed": row.get("seed", ""),
            "prompt_id": row.get("prompt_id", ""),
            "prompt_slot": row.get("prompt_slot", ""),
            "match_policy": row.get("match_policy", ""),
            "drift_reason": row.get("drift_reason", ""),
            "observed_token_class": row.get("observed_token_class", ""),
            "target_bucket": row.get("target_bucket", ""),
            "observed_token_text": row.get("observed_token_text", ""),
            "target_token_text": row.get("target_token_text", ""),
            "original_response_nll_per_token": row_scores["original_response"]["nll_per_token"],
            "repaired_response_nll_per_token": row_scores["repaired_response"]["nll_per_token"],
            "response_delta_nll_per_token": response_delta,
            "observed_suffix_nll_per_token": row_scores["observed_suffix_window"]["nll_per_token"],
            "target_suffix_nll_per_token": row_scores["target_suffix_window"]["nll_per_token"],
            "suffix_delta_nll_per_token": suffix_delta,
            "suffix_token_count": int(row_scores["target_suffix_window"]["token_count"]),
            **flags,
            "thresholds": {
                "max_response_delta_per_token": max_response_delta_per_token,
                "max_suffix_delta_per_token": max_suffix_delta_per_token,
            },
            "claim_control": {
                "paper_claim_allowed": False,
                "training_started": False,
                "generation_started": False,
                "e2e_eval_started": False,
                "not_payload_recovery": True,
                "not_full_far": True,
            },
            "paper_claim_allowed": False,
            "training_started": False,
            "generation_started": False,
            "e2e_eval_started": False,
            "result_claim": "branch_aware_compatibility_proxy_score_not_training_not_far",
        }
        scored_rows.append(scored)

    by_rows: list[dict[str, Any]] = []
    by_rows.extend(_group_rows(scored_rows, "model_condition", lambda row: row.get("model_condition", "")))
    by_rows.extend(_group_rows(scored_rows, "drift_reason", lambda row: row.get("drift_reason", "")))
    by_rows.extend(_group_rows(scored_rows, "token_class", lambda row: row.get("observed_token_class", "")))
    by_rows.extend(
        _group_rows(
            scored_rows,
            "model_condition__drift_reason",
            lambda row: f"{row.get('model_condition', '')}|{row.get('drift_reason', '')}",
        )
    )
    condition_counts = Counter(str(row.get("model_condition", "")) for row in scored_rows)
    total = len(scored_rows)
    branch_pass = sum(1 for row in scored_rows if row["branch_aware_proxy_pass"])
    response_pass = sum(1 for row in scored_rows if row["response_naturalness_proxy_pass"])
    suffix_pass = sum(1 for row in scored_rows if row["suffix_preserving_proxy_pass"])
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_BRANCH_AWARE_COMPATIBILITY_MODEL_SCORED_PROXY_NOT_GENERATED",
        "claim_control": {
            "paper_claim_allowed": False,
            "training_started": False,
            "generation_started": False,
            "e2e_eval_started": False,
            "not_payload_recovery": True,
            "not_full_far": True,
            "result_claim": "branch_aware_compatibility_proxy_score_not_training_not_far",
        },
        "inputs": {
            "scoring_plan_jsonl": str(scoring_plan_jsonl),
            "dry_run_rows_jsonl": str(dry_run_rows_jsonl),
            "prepared_summary_json": str(prepared_summary_json),
            "prepared_status": prepared_summary.get("status", ""),
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "max_length": max_length,
            "batch_size": batch_size,
            "max_response_delta_per_token": max_response_delta_per_token,
            "max_suffix_delta_per_token": max_suffix_delta_per_token,
        },
        "scored_rows": total,
        "condition_counts": dict(sorted(condition_counts.items())),
        "response_naturalness_proxy_pass_rows": response_pass,
        "response_naturalness_proxy_pass_rate": _rate(response_pass, total),
        "suffix_preserving_proxy_pass_rows": suffix_pass,
        "suffix_preserving_proxy_pass_rate": _rate(suffix_pass, total),
        "branch_aware_proxy_pass_rows": branch_pass,
        "branch_aware_proxy_pass_rate": _rate(branch_pass, total),
        "mean_response_delta_nll_per_token": _mean(
            [_as_float(row["response_delta_nll_per_token"]) for row in scored_rows]
        ),
        "mean_suffix_delta_nll_per_token": _mean(
            [_as_float(row["suffix_delta_nll_per_token"]) for row in scored_rows]
        ),
        "limitations": [
            "This is a model-scored proxy diagnostic, not generated branch continuation.",
            "It does not train, run E2E, decode payload recovery, or estimate FAR.",
            "Pass thresholds are diagnostic thresholds, not paper-facing success criteria.",
        ],
        "next_allowed_action": (
            "Review branch-aware proxy pass/fail slices. If compatible, design "
            "repaired training-target preflight; do not train from this result alone."
        ),
    }
    csv_rows = []
    for row in scored_rows:
        csv_rows.append(
            {
                **row,
                "original_response_nll_per_token": _format_float(row["original_response_nll_per_token"]),
                "repaired_response_nll_per_token": _format_float(row["repaired_response_nll_per_token"]),
                "response_delta_nll_per_token": _format_float(row["response_delta_nll_per_token"]),
                "observed_suffix_nll_per_token": _format_float(row["observed_suffix_nll_per_token"]),
                "target_suffix_nll_per_token": _format_float(row["target_suffix_nll_per_token"]),
                "suffix_delta_nll_per_token": _format_float(row["suffix_delta_nll_per_token"]),
            }
        )
    write_json(summary_path, summary)
    write_jsonl(output_dir / "branch_aware_compatibility_score_rows.jsonl", scored_rows)
    write_csv(output_dir / "branch_aware_compatibility_score_rows.csv", csv_rows, ROW_FIELDS)
    write_csv(output_dir / "branch_aware_compatibility_score_by_group.csv", by_rows, GROUP_FIELDS)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_scoring(
        scoring_plan_jsonl=_resolve(args.scoring_plan_jsonl),
        dry_run_rows_jsonl=_resolve(args.dry_run_rows_jsonl),
        prepared_summary_json=_resolve(args.prepared_summary_json),
        output_dir=_resolve(args.output_dir),
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        max_response_delta_per_token=float(args.max_response_delta_per_token),
        max_suffix_delta_per_token=float(args.max_suffix_delta_per_token),
        max_rows=int(args.max_rows),
        require_cuda=bool(args.require_cuda),
        force=bool(args.force),
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(_resolve(args.output_dir))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
