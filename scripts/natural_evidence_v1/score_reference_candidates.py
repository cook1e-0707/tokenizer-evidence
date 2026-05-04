from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from typing import Any

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

    output_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for row_index, row in enumerate(input_rows):
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
                output_rows.append(
                    {
                        "schema_name": "natural_evidence_reference_topk_candidates_v1",
                        "protocol_id": protocol_id,
                        "tokenizer_key": args.tokenizer_key,
                        "tokenizer_name": tokenizer_name,
                        "model_name": model_name,
                        "prompt_id": str(row.get("prompt_id", f"row_{row_index:06d}")),
                        "response_id": str(row.get("response_id", row.get("output_id", ""))),
                        "prefix_response_token_count": int(offset),
                        "prefix_token_ids": prefix_token_ids,
                        "candidates": candidates,
                    }
                )

    if args.output_jsonl:
        output_path = resolve_repo_path(args.output_jsonl, root)
    else:
        output_path = resolve_repo_path(
            dict(bucket_cfg.get("reference_candidates", {}))[args.tokenizer_key],
            root,
        )
    write_jsonl(output_path, output_rows)
    print(json.dumps({"rows": len(output_rows), "output_jsonl": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

