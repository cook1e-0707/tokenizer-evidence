from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score fingerprint response probabilities.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--fingerprints-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--use-chat-template", action="store_true")
    return parser.parse_args()


def _load_items(path: Path, limit: int) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for item in payload[:limit]:
        if isinstance(item, dict) and "key" in item and "response" in item:
            rows.append({"key": str(item["key"]), "response": str(item["response"])})
    return rows


def _prompt_ids(tokenizer: Any, key: str, use_chat_template: bool) -> torch.Tensor:
    if use_chat_template:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": key}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        return encoded[0]
    return tokenizer(key, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def _score_one(model: Any, tokenizer: Any, key: str, response: str, use_chat_template: bool) -> dict[str, Any]:
    prompt_ids = _prompt_ids(tokenizer, key, use_chat_template)
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if response_ids.numel() == 0:
        return {"token_count": 0, "sequence_logprob": None, "first_token_probability": None}

    input_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0).to(model.device)
    with torch.inference_mode():
        logits = model(input_ids).logits[0]
    logprobs = torch.log_softmax(logits, dim=-1)

    start = int(prompt_ids.numel())
    token_logprobs = []
    for offset, token_id in enumerate(response_ids.tolist()):
        pred_pos = start + offset - 1
        token_logprobs.append(float(logprobs[pred_pos, token_id].detach().cpu()))
    sequence_logprob = float(sum(token_logprobs))
    return {
        "token_count": int(response_ids.numel()),
        "sequence_logprob": sequence_logprob,
        "first_token_probability": float(math.exp(token_logprobs[0])),
    }


def main() -> int:
    args = parse_args()
    items = _load_items(Path(args.fingerprints_file), args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    rows = [
        {
            "index": idx,
            "key": item["key"],
            "response": item["response"],
            **_score_one(model, tokenizer, item["key"], item["response"], args.use_chat_template),
        }
        for idx, item in enumerate(items)
    ]
    valid = [row for row in rows if row["sequence_logprob"] is not None]
    summary = {
        "model_path": args.model_path,
        "fingerprints_file": args.fingerprints_file,
        "use_chat_template": bool(args.use_chat_template),
        "count": len(rows),
        "valid_count": len(valid),
        "mean_sequence_logprob": sum(float(row["sequence_logprob"]) for row in valid) / len(valid)
        if valid
        else None,
        "mean_first_token_probability": sum(float(row["first_token_probability"]) for row in valid) / len(valid)
        if valid
        else None,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

