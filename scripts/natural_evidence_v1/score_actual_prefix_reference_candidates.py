from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.natural_evidence_v1.common import (
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    token_surface_allowed,
    write_json,
    write_jsonl,
)


SCHEMA_NAME = "natural_evidence_actual_prefix_reference_topk_candidates_v1"
SUMMARY_SCHEMA_NAME = "natural_evidence_actual_prefix_reference_scoring_summary_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score reference-model top-k next-token candidates at frozen actual "
            "generated prefixes. This is scoring only: it does not train, decode a "
            "payload, bucketize, or claim recovery."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--candidate-top-k", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--progress-json", default="")
    parser.add_argument("--progress-every", type=int, default=256)
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


def _validate_prefix_token_ids(row: Mapping[str, Any]) -> list[int]:
    token_ids = row.get("prefix_token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("actual-prefix scoring row must include non-empty prefix_token_ids")
    return [int(token_id) for token_id in token_ids]


def _sharded_rows(
    rows: list[dict[str, Any]], *, shard_index: int, shard_count: int
) -> list[tuple[int, dict[str, Any]]]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    return [(index, row) for index, row in enumerate(rows) if index % shard_count == shard_index]


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))


def _candidate_record(
    *,
    row_index: int,
    row: Mapping[str, Any],
    protocol_id: str,
    tokenizer_key: str,
    tokenizer_name: str,
    model_name: str,
    candidate_top_k: int,
    top_probabilities: list[float],
    top_token_ids: list[int],
    tokenizer: Any,
) -> dict[str, Any]:
    observed_token_id = int(row.get("observed_token_id", -1))
    observed_rank: int | None = None
    observed_probability: float | None = None
    candidates: list[dict[str, Any]] = []
    surface_allowed_count = 0
    for rank, (probability, token_id) in enumerate(zip(top_probabilities, top_token_ids), start=1):
        token_text = _decode_token(tokenizer, int(token_id))
        surface_allowed = token_surface_allowed(token_text)
        if surface_allowed:
            surface_allowed_count += 1
        if int(token_id) == observed_token_id:
            observed_rank = rank
            observed_probability = float(probability)
        candidates.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "text": token_text,
                "probability": float(probability),
                "surface_allowed": surface_allowed,
            }
        )
    return {
        "schema_name": SCHEMA_NAME,
        "protocol_id": str(row.get("protocol_id", protocol_id)),
        "tokenizer_key": tokenizer_key,
        "tokenizer_name": tokenizer_name,
        "model_name": model_name,
        "candidate_top_k": int(candidate_top_k),
        "source_row_index": int(row_index),
        "prefix_signature": str(row.get("prefix_signature", "")),
        "selector_version": str(row.get("selector_version", "")),
        "model_family": str(row.get("model_family", "")),
        "model_condition": str(row.get("model_condition", "")),
        "payload_id": str(row.get("payload_id", "")),
        "seed": str(row.get("seed", "")),
        "prompt_id": str(row.get("prompt_id", f"row_{row_index:06d}")),
        "prompt_split": str(row.get("prompt_split", "")),
        "query_index": int(row.get("query_index", 0)),
        "generated_row_index": int(row.get("generated_row_index", row_index)),
        "position_index": int(row.get("position_index", 0)),
        "prefix_response_token_count": int(row.get("prefix_response_token_count", 0)),
        "response_token_count": int(row.get("response_token_count", 0)),
        "prefix_token_ids": _validate_prefix_token_ids(row),
        "observed_token_id": observed_token_id,
        "observed_token_text": str(row.get("observed_token_text", "")),
        "observed_token_in_topk": observed_rank is not None,
        "observed_token_rank": observed_rank,
        "observed_token_probability": observed_probability,
        "surface_allowed_candidate_count": surface_allowed_count,
        "candidates": candidates,
        "result_claim": "actual_prefix_reference_scoring_not_bucketization_not_payload_recovery",
    }


def _write_progress(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    write_json(path, dict(payload))


def _progress_payload(
    *,
    status: str,
    tokenizer_key: str,
    candidate_top_k: int,
    shard_index: int,
    shard_count: int,
    processed_records: int,
    total_records: int,
    output_rows: int,
    observed_token_in_topk_rows: int,
    output_jsonl: Path,
    last_prefix_signature: str,
) -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_actual_prefix_reference_scoring_progress_v1",
        "status": status,
        "updated_time": _utc_now(),
        "tokenizer_key": tokenizer_key,
        "candidate_top_k": int(candidate_top_k),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "processed_records": int(processed_records),
        "total_records": int(total_records),
        "output_rows": int(output_rows),
        "observed_token_in_topk_rows": int(observed_token_in_topk_rows),
        "output_jsonl": str(output_jsonl),
        "last_prefix_signature": last_prefix_signature,
        "paper_claim_allowed": False,
        "training_started": False,
        "result_claim": "progress_for_reference_scoring_only",
    }


def _batched(items: list[tuple[int, dict[str, Any]]], batch_size: int) -> list[list[tuple[int, dict[str, Any]]]]:
    size = max(1, int(batch_size))
    return [items[index : index + size] for index in range(0, len(items), size)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = resolve_repo_path(args.config, root)
    input_path = resolve_repo_path(args.input_jsonl, root)
    manifest_path = resolve_repo_path(args.manifest_json, root)
    output_path = resolve_repo_path(args.output_jsonl, root)
    summary_path = resolve_repo_path(args.summary_json, root)
    progress_path = resolve_repo_path(args.progress_json, root) if args.progress_json else None

    if output_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing scoring outputs under {output_path.parent}")

    config = read_yaml(config_path)
    protocol_id = str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1"))
    model_cfg = _model_config(config, args.tokenizer_key)
    model_name = str(model_cfg.get("model_name", ""))
    tokenizer_name = str(model_cfg.get("tokenizer_name", model_name))
    if not model_name or not tokenizer_name:
        raise ValueError(f"Missing model/tokenizer name for {args.tokenizer_key}")

    plan_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if plan_manifest.get("status") != "PLAN_COMPLETE_PENDING_REVIEW_AND_GPU_SCORING":
        raise ValueError("actual-prefix plan manifest is not ready for GPU scoring")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "score_actual_prefix_reference_candidates requires torch and transformers"
        ) from error

    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is False")
    device = torch.device("cuda" if cuda_available else "cpu")
    torch_dtype = torch.bfloat16 if cuda_available and torch.cuda.is_bf16_supported() else None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()

    input_rows = read_jsonl(input_path)
    if args.max_records > 0:
        input_rows = input_rows[: args.max_records]
    work_rows = _sharded_rows(input_rows, shard_index=args.shard_index, shard_count=args.shard_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_records = 0
    output_rows = 0
    observed_token_in_topk_rows = 0
    surface_allowed_candidate_total = 0
    last_prefix_signature = ""
    records: list[dict[str, Any]] = []
    output_handle = output_path.open("w", encoding="utf-8") if args.stream_output else None
    _write_progress(
        progress_path,
        _progress_payload(
            status="running",
            tokenizer_key=args.tokenizer_key,
            candidate_top_k=args.candidate_top_k,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            processed_records=0,
            total_records=len(work_rows),
            output_rows=0,
            observed_token_in_topk_rows=0,
            output_jsonl=output_path,
            last_prefix_signature="",
        ),
    )
    try:
        with torch.no_grad():
            for batch in _batched(work_rows, args.batch_size):
                prefix_lists = [_validate_prefix_token_ids(row) for _, row in batch]
                lengths = [len(prefix_ids) for prefix_ids in prefix_lists]
                max_length = max(lengths)
                input_ids = torch.full(
                    (len(batch), max_length),
                    pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long, device=device)
                for row_offset, prefix_ids in enumerate(prefix_lists):
                    length = lengths[row_offset]
                    input_ids[row_offset, :length] = torch.tensor(prefix_ids, dtype=torch.long, device=device)
                    attention_mask[row_offset, :length] = 1
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                next_logits = outputs.logits[torch.arange(len(batch), device=device), torch.tensor(lengths, device=device) - 1]
                probabilities = torch.softmax(next_logits, dim=-1)
                top_probabilities, top_token_ids = torch.topk(probabilities, k=int(args.candidate_top_k), dim=-1)
                for batch_offset, (row_index, row) in enumerate(batch):
                    processed_records += 1
                    record = _candidate_record(
                        row_index=row_index,
                        row=row,
                        protocol_id=protocol_id,
                        tokenizer_key=args.tokenizer_key,
                        tokenizer_name=tokenizer_name,
                        model_name=model_name,
                        candidate_top_k=int(args.candidate_top_k),
                        top_probabilities=[float(value) for value in top_probabilities[batch_offset].tolist()],
                        top_token_ids=[int(value) for value in top_token_ids[batch_offset].tolist()],
                        tokenizer=tokenizer,
                    )
                    output_rows += 1
                    if record["observed_token_in_topk"]:
                        observed_token_in_topk_rows += 1
                    surface_allowed_candidate_total += int(record["surface_allowed_candidate_count"])
                    last_prefix_signature = str(record["prefix_signature"])
                    if output_handle is not None:
                        output_handle.write(json.dumps(record, sort_keys=True) + "\n")
                        if output_rows % max(1, args.progress_every) == 0:
                            output_handle.flush()
                    else:
                        records.append(record)
                    if output_rows % max(1, args.progress_every) == 0:
                        _write_progress(
                            progress_path,
                            _progress_payload(
                                status="running",
                                tokenizer_key=args.tokenizer_key,
                                candidate_top_k=args.candidate_top_k,
                                shard_index=args.shard_index,
                                shard_count=args.shard_count,
                                processed_records=processed_records,
                                total_records=len(work_rows),
                                output_rows=output_rows,
                                observed_token_in_topk_rows=observed_token_in_topk_rows,
                                output_jsonl=output_path,
                                last_prefix_signature=last_prefix_signature,
                            ),
                        )
    finally:
        if output_handle is not None:
            output_handle.close()
    if output_handle is None:
        write_jsonl(output_path, records)

    summary = {
        "schema_name": SUMMARY_SCHEMA_NAME,
        "status": "SCORING_COMPLETE_PENDING_COMPATIBILITY_FILTERING_AND_BUCKETIZATION",
        "paper_claim_allowed": False,
        "training_started": False,
        "protocol_id": protocol_id,
        "tokenizer_key": args.tokenizer_key,
        "tokenizer_name": tokenizer_name,
        "model_name": model_name,
        "candidate_top_k": int(args.candidate_top_k),
        "batch_size": int(args.batch_size),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "input_jsonl": str(input_path),
        "plan_manifest_json": str(manifest_path),
        "output_jsonl": str(output_path),
        "summary_json": str(summary_path),
        "progress_json": str(progress_path) if progress_path is not None else "",
        "input_rows_loaded": len(input_rows),
        "total_records_for_shard": len(work_rows),
        "processed_records": processed_records,
        "output_rows": output_rows,
        "topk_candidate_rows": output_rows * int(args.candidate_top_k),
        "observed_token_in_topk_rows": observed_token_in_topk_rows,
        "observed_token_in_topk_rate": observed_token_in_topk_rows / max(1, output_rows),
        "surface_allowed_candidate_total": surface_allowed_candidate_total,
        "surface_allowed_candidates_per_prefix": surface_allowed_candidate_total / max(1, output_rows),
        "next_minimal_action": (
            "run actual-prefix compatibility scoring and bucketization audit; "
            "do not run diagnostic E2E eval or training from this artifact alone"
        ),
        "result_claim": "reference_model_topk_scoring_only_not_payload_recovery",
    }
    write_json(summary_path, summary)
    _write_progress(
        progress_path,
        _progress_payload(
            status="complete",
            tokenizer_key=args.tokenizer_key,
            candidate_top_k=args.candidate_top_k,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            processed_records=processed_records,
            total_records=len(work_rows),
            output_rows=output_rows,
            observed_token_in_topk_rows=observed_token_in_topk_rows,
            output_jsonl=output_path,
            last_prefix_signature=last_prefix_signature,
        ),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
