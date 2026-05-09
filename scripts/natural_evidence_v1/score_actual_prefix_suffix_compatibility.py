from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.natural_evidence_v1.common import (
    bucket_mass_metrics,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    write_csv,
    write_json,
)


SCHEMA_NAME = "natural_evidence_actual_prefix_suffix_compatibility_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score suffix compatibility for actual-prefix bucketized candidates. "
            "This is reference-model scoring only: no training, E2E eval, FAR "
            "aggregation, or payload recovery claim."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--generated-outputs", required=True)
    parser.add_argument("--bucketized-candidates-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--by-entry-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--progress-json", default="")
    parser.add_argument("--suffix-window-tokens", type=int, default=16)
    parser.add_argument("--max-candidates-per-bucket", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--delta-nll-threshold", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _model_config(config: Mapping[str, Any], tokenizer_key: str) -> dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, dict) or tokenizer_key not in models:
        raise ValueError(f"Missing model config for tokenizer key {tokenizer_key!r}")
    model_cfg = models[tokenizer_key]
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Model config for {tokenizer_key!r} must be a mapping")
    return dict(model_cfg)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded.get("input_ids", [])
    if not isinstance(token_ids, list):
        return []
    return [int(token_id) for token_id in token_ids]


def _generated_by_index(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {index: row for index, row in enumerate(rows)}


def _select_candidates_by_bucket(
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_candidates_per_bucket: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        bucket_id = str(candidate.get("bucket_id", ""))
        if bucket_id == "":
            continue
        grouped[bucket_id].append(dict(candidate))
    selected: list[dict[str, Any]] = []
    for bucket_id in sorted(grouped, key=lambda value: int(value) if value.isdigit() else value):
        bucket_candidates = sorted(
            grouped[bucket_id],
            key=lambda row: (
                -float(row.get("probability", 0.0)),
                int(row.get("rank", 0)),
                int(row.get("token_id", -1)),
            ),
        )
        limit = max_candidates_per_bucket if max_candidates_per_bucket > 0 else len(bucket_candidates)
        selected.extend(bucket_candidates[:limit])
    return selected


def _suffix_losses(
    *,
    model: Any,
    torch_module: Any,
    sequences: Sequence[Sequence[int]],
    suffix_start: int,
    pad_token_id: int,
    device: Any,
) -> list[tuple[float, float, int]]:
    if not sequences:
        return []
    max_len = max(len(sequence) for sequence in sequences)
    input_ids = torch_module.full(
        (len(sequences), max_len),
        int(pad_token_id),
        dtype=torch_module.long,
        device=device,
    )
    attention_mask = torch_module.zeros((len(sequences), max_len), dtype=torch_module.long, device=device)
    for row_index, sequence in enumerate(sequences):
        ids = torch_module.tensor([int(token_id) for token_id in sequence], dtype=torch_module.long, device=device)
        input_ids[row_index, : len(sequence)] = ids
        attention_mask[row_index, : len(sequence)] = 1
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch_module.nn.functional.log_softmax(outputs.logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    label_positions = torch_module.arange(1, max_len, device=device).unsqueeze(0).expand_as(labels)
    mask = (attention_mask[:, 1:] == 1) & (label_positions >= int(suffix_start))
    safe_labels = labels.clamp_min(0)
    token_nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_nll = token_nll * mask
    raw_nll = token_nll.sum(dim=1)
    lengths = mask.sum(dim=1).clamp_min(1)
    per_token = raw_nll / lengths
    return [
        (float(raw_nll[index].item()), float(per_token[index].item()), int(lengths[index].item()))
        for index in range(len(sequences))
    ]


def _entry_summary_row(
    *,
    row: Mapping[str, Any],
    scored_rows: Sequence[Mapping[str, Any]],
    bucket_count: int,
    min_members_per_bucket: int,
    min_bucket_mass: float,
    max_bucket_mass_ratio: float,
    min_bucket_entropy_fraction: float,
) -> dict[str, Any]:
    expected_buckets = [str(bucket_id) for bucket_id in range(bucket_count)]
    scored_by_bucket = Counter(str(scored_row.get("bucket_id", "")) for scored_row in scored_rows)
    compatible_rows = [scored_row for scored_row in scored_rows if bool(scored_row.get("compatibility_pass", False))]
    compatible_by_bucket = Counter(str(scored_row.get("bucket_id", "")) for scored_row in compatible_rows)
    compatible_probability_by_bucket: dict[str, float] = {
        bucket_id: sum(
            float(scored_row.get("probability", 0.0))
            for scored_row in compatible_rows
            if str(scored_row.get("bucket_id", "")) == bucket_id
        )
        for bucket_id in expected_buckets
    }
    compatible_counts = {bucket_id: int(compatible_by_bucket.get(bucket_id, 0)) for bucket_id in expected_buckets}
    scored_counts = {bucket_id: int(scored_by_bucket.get(bucket_id, 0)) for bucket_id in expected_buckets}
    min_compatible = min(compatible_counts.values()) if compatible_counts else 0
    compatible_bucket_count = sum(1 for count in compatible_counts.values() if count > 0)
    would_accept_min1 = compatible_bucket_count == len(expected_buckets)
    would_accept_configured_min = would_accept_min1 and min_compatible >= int(min_members_per_bucket)
    mass_metrics = bucket_mass_metrics([compatible_probability_by_bucket[bucket_id] for bucket_id in expected_buckets])
    would_accept_probability_gates = (
        would_accept_configured_min
        and mass_metrics["min_bucket_mass"] >= float(min_bucket_mass)
        and mass_metrics["bucket_mass_ratio"] <= float(max_bucket_mass_ratio)
        and mass_metrics["bucket_entropy_fraction"] >= float(min_bucket_entropy_fraction)
    )
    if any(scored_counts[bucket_id] == 0 for bucket_id in expected_buckets):
        rejection_reason = "incomplete_bucket_scores"
    elif not would_accept_min1:
        rejection_reason = "missing_compatible_bucket"
    elif not would_accept_configured_min:
        rejection_reason = "below_configured_min_compatible_members"
    elif not would_accept_probability_gates:
        rejection_reason = "below_probability_balance_gates"
    else:
        rejection_reason = ""
    return {
        "bank_entry_id": row.get("bank_entry_id", ""),
        "prompt_id": row.get("prompt_id", ""),
        "prompt_split": row.get("prompt_split", ""),
        "model_condition": row.get("model_condition", ""),
        "payload_id": row.get("payload_id", ""),
        "seed": row.get("seed", ""),
        "query_index": row.get("query_index", 0),
        "generated_row_index": row.get("generated_row_index", 0),
        "position_index": row.get("position_index", 0),
        "bucket_count": bucket_count,
        "scored_candidate_count": len(scored_rows),
        "compatible_candidate_count": len(compatible_rows),
        "compatible_bucket_count": compatible_bucket_count,
        "min_compatible_members_per_bucket": min_compatible,
        "would_accept_min1": would_accept_min1,
        "would_accept_configured_min": would_accept_configured_min,
        "compatible_min_bucket_mass": mass_metrics["min_bucket_mass"],
        "compatible_bucket_mass_ratio": mass_metrics["bucket_mass_ratio"],
        "compatible_bucket_entropy_fraction": mass_metrics["bucket_entropy_fraction"],
        "would_accept_probability_gates": would_accept_probability_gates,
        "rejection_reason": rejection_reason,
        "scored_counts_by_bucket_json": json.dumps(scored_counts, sort_keys=True),
        "compatible_counts_by_bucket_json": json.dumps(compatible_counts, sort_keys=True),
        "compatible_probability_by_bucket_json": json.dumps(compatible_probability_by_bucket, sort_keys=True),
    }


def _write_progress(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is not None:
        write_json(path, payload)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    model_cfg = _model_config(config, args.tokenizer_key)
    model_name = str(model_cfg.get("model_name", ""))
    tokenizer_name = str(model_cfg.get("tokenizer_name", model_name))
    if not model_name or not tokenizer_name:
        raise ValueError(f"Missing model/tokenizer name for {args.tokenizer_key}")
    bucket_cfg = dict(config.get("bucket_bank", {}))
    quality_gates = dict(bucket_cfg.get("quality_gates", {}))
    bucket_count = int(
        dict(bucket_cfg.get("compatibility_adjusted_capacity", {}))
        .get("diagnostic_high_risk_gate", {})
        .get("bucket_count", 4)
    )
    min_members_per_bucket = int(bucket_cfg.get("min_members_per_bucket", 2))
    min_bucket_mass = float(quality_gates.get("min_bucket_mass", bucket_cfg.get("min_bucket_mass", 0.0)))
    max_bucket_mass_ratio = float(quality_gates.get("max_bucket_mass_ratio", float("inf")))
    min_bucket_entropy_fraction = float(quality_gates.get("min_bucket_entropy_fraction", 0.0))

    output_path = resolve_repo_path(args.output_jsonl, root)
    by_entry_path = resolve_repo_path(args.by_entry_csv, root)
    summary_path = resolve_repo_path(args.summary_json, root)
    progress_path = resolve_repo_path(args.progress_json, root) if args.progress_json else None
    existing_outputs = [str(path) for path in (output_path, by_entry_path, summary_path) if path.exists()]
    if existing_outputs:
        raise FileExistsError("Refusing to overwrite existing suffix compatibility outputs: " + ", ".join(existing_outputs))

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("actual-prefix suffix compatibility requires torch and transformers") from error

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is False")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()

    generated_rows = read_jsonl(resolve_repo_path(args.generated_outputs, root))
    generated_by_index = _generated_by_index(generated_rows)
    bucketized_rows = read_jsonl(resolve_repo_path(args.bucketized_candidates_jsonl, root))
    if args.max_records > 0:
        bucketized_rows = bucketized_rows[: args.max_records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_entry_rows: list[dict[str, Any]] = []
    invalid_counts: Counter[str] = Counter()
    scored_candidate_count = 0
    compatible_candidate_count = 0
    processed_records = 0
    total_records = len(bucketized_rows)

    _write_progress(
        progress_path,
        {
            "schema_name": "natural_evidence_actual_prefix_suffix_compatibility_progress_v1",
            "status": "running",
            "updated_time": _utc_now(),
            "processed_records": 0,
            "total_records": total_records,
            "scored_candidate_count": 0,
            "compatible_candidate_count": 0,
            "paper_claim_allowed": False,
            "training_started": False,
            "result_claim": "actual_prefix_suffix_compatibility_progress_not_payload_recovery",
        },
    )

    with output_path.open("w", encoding="utf-8") as output_handle, torch.no_grad():
        for row in bucketized_rows:
            processed_records += 1
            generated_index = int(row.get("generated_row_index", -1))
            generated = generated_by_index.get(generated_index)
            if generated is None:
                invalid_counts["missing_generated_row"] += 1
                continue
            prompt = str(generated.get("prompt", ""))
            response = str(generated.get("response_text", ""))
            response_ids = _token_ids(tokenizer, response)
            offset = int(row.get("prefix_response_token_count", 0))
            if offset < 0 or offset >= len(response_ids) - 1:
                invalid_counts["invalid_or_boundary_suffix_offset"] += 1
                continue
            suffix_ids = response_ids[offset + 1 : offset + 1 + max(1, int(args.suffix_window_tokens))]
            if not suffix_ids:
                invalid_counts["empty_suffix_window"] += 1
                continue
            observed_token_id = int(row.get("observed_token_id", response_ids[offset]))
            if observed_token_id != int(response_ids[offset]):
                invalid_counts["observed_token_mismatch"] += 1
            prefix_ids = [int(token_id) for token_id in row.get("prefix_token_ids", [])]
            if not prefix_ids:
                prompt_ids = _token_ids(tokenizer, prompt)
                prefix_ids = [*prompt_ids, *response_ids[:offset]]
            selected_candidates = _select_candidates_by_bucket(
                row.get("candidates", []),
                max_candidates_per_bucket=int(args.max_candidates_per_bucket),
            )
            if not selected_candidates:
                invalid_counts["no_bucketized_candidates"] += 1
                continue

            sequences = [[*prefix_ids, observed_token_id, *suffix_ids]]
            sequences.extend([[*prefix_ids, int(candidate["token_id"]), *suffix_ids] for candidate in selected_candidates])
            losses = _suffix_losses(
                model=model,
                torch_module=torch,
                sequences=sequences,
                suffix_start=len(prefix_ids) + 1,
                pad_token_id=pad_token_id,
                device=device,
            )
            baseline_raw, baseline_per_token, suffix_length = losses[0]
            scored_rows_for_entry: list[dict[str, Any]] = []
            for candidate, (candidate_raw, candidate_per_token, _) in zip(selected_candidates, losses[1:], strict=True):
                delta_per_token = candidate_per_token - baseline_per_token
                delta_raw = candidate_raw - baseline_raw
                compatibility_pass = delta_per_token <= float(args.delta_nll_threshold)
                scored_candidate_count += 1
                if compatibility_pass:
                    compatible_candidate_count += 1
                scored_row = {
                    "schema_name": SCHEMA_NAME,
                    "protocol_id": row.get("protocol_id", "natural_evidence_v1"),
                    "bank_id": row.get("bank_id", ""),
                    "bank_entry_id": row.get("bank_entry_id", ""),
                    "context_signature": row.get("context_signature", ""),
                    "tokenizer_key": args.tokenizer_key,
                    "tokenizer_name": tokenizer_name,
                    "model_name": model_name,
                    "model_condition": row.get("model_condition", ""),
                    "payload_id": row.get("payload_id", ""),
                    "seed": row.get("seed", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "prompt_split": row.get("prompt_split", ""),
                    "query_index": row.get("query_index", 0),
                    "generated_row_index": generated_index,
                    "position_index": row.get("position_index", 0),
                    "prefix_response_token_count": offset,
                    "bucket_id": str(candidate.get("bucket_id", "")),
                    "token_id": int(candidate.get("token_id", -1)),
                    "token_text": candidate.get("text", ""),
                    "rank": candidate.get("rank", ""),
                    "probability": float(candidate.get("probability", 0.0)),
                    "suffix_window_length": suffix_length,
                    "baseline_suffix_nll_raw": baseline_raw,
                    "candidate_suffix_nll_raw": candidate_raw,
                    "delta_suffix_nll_raw": delta_raw,
                    "baseline_suffix_nll_per_token": baseline_per_token,
                    "candidate_suffix_nll_per_token": candidate_per_token,
                    "delta_suffix_nll_per_token": delta_per_token,
                    "baseline_suffix_nll": baseline_per_token,
                    "candidate_suffix_nll": candidate_per_token,
                    "delta_suffix_nll": delta_per_token,
                    "compatibility_pass": compatibility_pass,
                    "observed_token_id": observed_token_id,
                    "observed_token_text": row.get("observed_token_text", ""),
                    "observed_token_bucket_id": row.get("observed_token_bucket_id", ""),
                    "paper_claim_allowed": False,
                    "training_started": False,
                    "result_claim": "actual_prefix_suffix_compatibility_not_payload_recovery",
                    "fingerprint_claim": False,
                }
                scored_rows_for_entry.append(scored_row)
                output_handle.write(json.dumps(scored_row, sort_keys=True) + "\n")
            by_entry_rows.append(
                _entry_summary_row(
                    row=row,
                    scored_rows=scored_rows_for_entry,
                    bucket_count=bucket_count,
                    min_members_per_bucket=min_members_per_bucket,
                    min_bucket_mass=min_bucket_mass,
                    max_bucket_mass_ratio=max_bucket_mass_ratio,
                    min_bucket_entropy_fraction=min_bucket_entropy_fraction,
                )
            )
            if progress_path and processed_records % max(1, int(args.progress_every)) == 0:
                _write_progress(
                    progress_path,
                    {
                        "schema_name": "natural_evidence_actual_prefix_suffix_compatibility_progress_v1",
                        "status": "running",
                        "updated_time": _utc_now(),
                        "processed_records": processed_records,
                        "total_records": total_records,
                        "scored_candidate_count": scored_candidate_count,
                        "compatible_candidate_count": compatible_candidate_count,
                        "paper_claim_allowed": False,
                        "training_started": False,
                        "result_claim": "actual_prefix_suffix_compatibility_progress_not_payload_recovery",
                    },
                )

    min1_rows = [row for row in by_entry_rows if bool(row["would_accept_min1"])]
    configured_rows = [row for row in by_entry_rows if bool(row["would_accept_configured_min"])]
    probability_rows = [row for row in by_entry_rows if bool(row["would_accept_probability_gates"])]
    rejection_counts = Counter(str(row["rejection_reason"]) for row in by_entry_rows if row["rejection_reason"])
    write_csv(by_entry_path, by_entry_rows, list(by_entry_rows[0].keys()) if by_entry_rows else [])
    summary = {
        "schema_name": "natural_evidence_actual_prefix_suffix_compatibility_summary_v1",
        "status": "COMPLETE_PENDING_REVIEW",
        "protocol_id": str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1")),
        "tokenizer_key": args.tokenizer_key,
        "tokenizer_name": tokenizer_name,
        "model_name": model_name,
        "generated_outputs": str(resolve_repo_path(args.generated_outputs, root)),
        "bucketized_candidates_jsonl": str(resolve_repo_path(args.bucketized_candidates_jsonl, root)),
        "input_records": total_records,
        "processed_records": processed_records,
        "invalid_counts": dict(sorted(invalid_counts.items())),
        "scored_candidate_count": scored_candidate_count,
        "compatible_candidate_count": compatible_candidate_count,
        "compatibility_pass_rate": compatible_candidate_count / scored_candidate_count if scored_candidate_count else 0.0,
        "min1_compatible_entries": len(min1_rows),
        "configured_min_compatible_entries": len(configured_rows),
        "probability_gated_compatible_entries": len(probability_rows),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "delta_nll_threshold": float(args.delta_nll_threshold),
        "suffix_window_tokens": int(args.suffix_window_tokens),
        "max_candidates_per_bucket": int(args.max_candidates_per_bucket),
        "outputs": {
            "compatibility_jsonl": str(output_path),
            "by_entry_csv": str(by_entry_path),
            "summary_json": str(summary_path),
            "progress_json": str(progress_path) if progress_path else "",
        },
        "next_minimal_action": (
            "Review actual-prefix compatibility counts and decide whether a diagnostic E2E "
            "repair has enough compatible observations; do not train or rerun E2E from "
            "compatibility scoring alone."
        ),
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "result_claim": "actual_prefix_suffix_compatibility_summary_not_payload_recovery",
        "fingerprint_claim": False,
    }
    write_json(summary_path, summary)
    _write_progress(
        progress_path,
        {
            "schema_name": "natural_evidence_actual_prefix_suffix_compatibility_progress_v1",
            "status": "complete",
            "updated_time": _utc_now(),
            "processed_records": processed_records,
            "total_records": total_records,
            "scored_candidate_count": scored_candidate_count,
            "compatible_candidate_count": compatible_candidate_count,
            "summary_json": str(summary_path),
            "paper_claim_allowed": False,
            "training_started": False,
            "result_claim": "actual_prefix_suffix_compatibility_progress_not_payload_recovery",
        },
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
