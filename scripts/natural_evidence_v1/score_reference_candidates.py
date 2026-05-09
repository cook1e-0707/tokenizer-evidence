from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from scripts.natural_evidence_v1.common import read_jsonl, read_yaml, resolve_repo_path, write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score reference top-k next-token candidates for natural_evidence_v1 "
            "bucket-bank construction."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--candidate-top-k", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-prefixes-per-response", type=int, default=16)
    parser.add_argument("--prefix-stride", type=int, default=8)
    parser.add_argument("--min-response-prefix-tokens", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--progress-json", default="")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


def _model_config(config: dict[str, Any], tokenizer_key: str) -> dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, dict) or tokenizer_key not in models:
        raise ValueError(f"Missing model config for tokenizer key {tokenizer_key!r}")
    model_cfg = models[tokenizer_key]
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Model config for {tokenizer_key!r} must be a mapping")
    return model_cfg


def _record_texts(row: dict[str, Any]) -> tuple[str, str]:
    prompt = str(row.get("prompt", row.get("user_probe", "")))
    response = str(row.get("response_text", row.get("output_text", "")))
    if not prompt.strip():
        raise ValueError("reference input row must include prompt or user_probe")
    if not response.strip():
        raise ValueError("reference input row must include response_text or output_text")
    return prompt, response


def _candidate_prefix_offsets(
    *,
    response_token_count: int,
    min_response_prefix_tokens: int,
    prefix_stride: int,
    max_prefixes_per_response: int,
) -> list[int]:
    if response_token_count <= 0:
        return []
    start = max(1, min_response_prefix_tokens)
    stride = max(1, prefix_stride)
    offsets = list(range(start, response_token_count + 1, stride))
    if response_token_count not in offsets:
        offsets.append(response_token_count)
    return offsets[: max(1, max_prefixes_per_response)]


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))


def _sharded_rows(rows: list[dict[str, Any]], *, shard_index: int, shard_count: int) -> list[tuple[int, dict[str, Any]]]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    return [(index, row) for index, row in enumerate(rows) if index % shard_count == shard_index]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _maybe_write_progress(
    *,
    path: Path | None,
    status: str,
    shard_index: int,
    shard_count: int,
    processed_records: int,
    total_records: int,
    output_rows: int,
    output_jsonl: Path,
    last_prompt_id: str,
    last_prefix_response_token_count: int,
) -> None:
    if path is None:
        return
    _write_progress(
        path,
        {
            "schema_name": "natural_evidence_reference_candidate_scoring_progress_v1",
            "status": status,
            "updated_time": _utc_now(),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "processed_records": processed_records,
            "total_records": total_records,
            "output_rows": output_rows,
            "output_jsonl": str(output_jsonl),
            "last_prompt_id": last_prompt_id,
            "last_prefix_response_token_count": last_prefix_response_token_count,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    model_cfg = _model_config(config, args.tokenizer_key)
    bucket_cfg = dict(config.get("bucket_bank", {}))
    protocol_id = str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1"))
    candidate_top_k = args.candidate_top_k or int(bucket_cfg.get("candidate_top_k", 64))
    model_name = str(model_cfg.get("model_name", ""))
    tokenizer_name = str(model_cfg.get("tokenizer_name", model_name))
    if not model_name or not tokenizer_name:
        raise ValueError(f"Missing model/tokenizer name for {args.tokenizer_key}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "score_reference_candidates requires optional hf dependencies: torch and transformers"
        ) from error

    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is False")
    device = torch.device("cuda" if cuda_available else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    input_rows = read_jsonl(resolve_repo_path(args.input_jsonl, root))
    if args.max_records > 0:
        input_rows = input_rows[: args.max_records]

    sharded_rows = _sharded_rows(
        input_rows,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    if args.output_jsonl:
        output_path = resolve_repo_path(args.output_jsonl, root)
    else:
        output_path = resolve_repo_path(
            dict(bucket_cfg.get("reference_candidates", {}))[args.tokenizer_key],
            root,
        )
    progress_path = resolve_repo_path(args.progress_json, root) if args.progress_json else None
    progress_every = max(1, args.progress_every)
    output_handle = None
    output_rows: list[dict[str, Any]] = []
    output_count = 0
    processed_records = 0
    last_prompt_id = ""
    last_prefix_response_token_count = 0
    if args.stream_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")
    _maybe_write_progress(
        path=progress_path,
        status="running",
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        processed_records=0,
        total_records=len(sharded_rows),
        output_rows=0,
        output_jsonl=output_path,
        last_prompt_id="",
        last_prefix_response_token_count=0,
    )
    try:
        with torch.no_grad():
            for row_index, row in sharded_rows:
                processed_records += 1
                prompt, response = _record_texts(row)
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
                if not isinstance(prompt_ids, list) or not isinstance(response_ids, list):
                    raise ValueError("Tokenizer returned unexpected non-list input ids")
                offsets = _candidate_prefix_offsets(
                    response_token_count=len(response_ids),
                    min_response_prefix_tokens=args.min_response_prefix_tokens,
                    prefix_stride=args.prefix_stride,
                    max_prefixes_per_response=args.max_prefixes_per_response,
                )
                for offset in offsets:
                    prefix_token_ids = list(prompt_ids) + list(response_ids[:offset])
                    if not prefix_token_ids:
                        continue
                    input_ids = torch.tensor([prefix_token_ids], dtype=torch.long, device=device)
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[0, -1, :]
                    probabilities = torch.softmax(logits, dim=-1)
                    top_probabilities, top_token_ids = torch.topk(probabilities, k=candidate_top_k)
                    candidates = []
                    for rank, (probability, token_id) in enumerate(
                        zip(top_probabilities.tolist(), top_token_ids.tolist()),
                        start=1,
                    ):
                        candidates.append(
                            {
                                "rank": rank,
                                "token_id": int(token_id),
                                "text": _decode_token(tokenizer, int(token_id)),
                                "probability": float(probability),
                            }
                        )
                    output_row = {
                        "schema_name": "natural_evidence_reference_topk_candidates_v1",
                        "protocol_id": protocol_id,
                        "tokenizer_key": args.tokenizer_key,
                        "tokenizer_name": tokenizer_name,
                        "model_name": model_name,
                        "prompt_id": str(row.get("prompt_id", f"row_{row_index:06d}")),
                        "split": str(row.get("split", row.get("prompt_split", ""))),
                        "prompt_split": str(row.get("prompt_split", row.get("split", ""))),
                        "freeze_id": str(row.get("freeze_id", "")),
                        "response_id": str(row.get("response_id", row.get("output_id", ""))),
                        "prefix_response_token_count": int(offset),
                        "prefix_token_ids": prefix_token_ids,
                        "candidates": candidates,
                    }
                    output_count += 1
                    last_prompt_id = str(output_row["prompt_id"])
                    last_prefix_response_token_count = int(offset)
                    if output_handle is not None:
                        output_handle.write(json.dumps(output_row, sort_keys=True) + "\n")
                        if output_count % progress_every == 0:
                            output_handle.flush()
                    else:
                        output_rows.append(output_row)
                    if output_count % progress_every == 0:
                        _maybe_write_progress(
                            path=progress_path,
                            status="running",
                            shard_index=args.shard_index,
                            shard_count=args.shard_count,
                            processed_records=processed_records,
                            total_records=len(sharded_rows),
                            output_rows=output_count,
                            output_jsonl=output_path,
                            last_prompt_id=last_prompt_id,
                            last_prefix_response_token_count=last_prefix_response_token_count,
                        )
                _maybe_write_progress(
                    path=progress_path,
                    status="running",
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                    processed_records=processed_records,
                    total_records=len(sharded_rows),
                    output_rows=output_count,
                    output_jsonl=output_path,
                    last_prompt_id=last_prompt_id,
                    last_prefix_response_token_count=last_prefix_response_token_count,
                )
    finally:
        if output_handle is not None:
            output_handle.close()
    if output_handle is None:
        write_jsonl(output_path, output_rows)
    _maybe_write_progress(
        path=progress_path,
        status="complete",
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        processed_records=processed_records,
        total_records=len(sharded_rows),
        output_rows=output_count,
        output_jsonl=output_path,
        last_prompt_id=last_prompt_id,
        last_prefix_response_token_count=last_prefix_response_token_count,
    )
    print(
        json.dumps(
            {
                "rows": output_count,
                "output_jsonl": str(output_path),
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
