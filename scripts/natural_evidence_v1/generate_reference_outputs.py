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


PROMPT_TEMPLATES = (
    "Give a concise plan for preparing a safe weekend {topic}.",
    "Write a short practical checklist for {topic}.",
    "Explain the main tradeoffs when planning {topic} in plain language.",
    "Summarize three steps someone should take before starting {topic}.",
    "Draft a brief advisory note about {topic} for a careful beginner.",
    "Compare two sensible approaches to {topic} and recommend one.",
    "List the most important risks in {topic} and how to reduce them.",
    "Give a calm, direct answer to a friend asking for help with {topic}.",
)

TOPICS = (
    "a weekend hike",
    "a train trip",
    "a small home repair",
    "a family budget review",
    "a study schedule",
    "a team project kickoff",
    "a garden planting plan",
    "a healthy meal plan",
    "a conference packing list",
    "a software release checklist",
    "a museum day visit",
    "a neighborhood cleanup",
    "a personal data backup",
    "a first apartment move",
    "a public talk outline",
    "a volunteer event",
    "an office relocation",
    "a fitness routine",
    "a morning commute change",
    "a basic emergency kit",
    "a customer support reply",
    "a reading plan",
    "a small workshop agenda",
    "a weather-aware travel plan",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate natural reference outputs for natural_evidence_v1 Phase A."
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--prompt-bank-jsonl", default="")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--prompt-count", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args(argv)


def _model_config(config: dict[str, Any], tokenizer_key: str) -> dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, dict) or tokenizer_key not in models:
        raise ValueError(f"Missing model config for tokenizer key {tokenizer_key!r}")
    model_cfg = models[tokenizer_key]
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Model config for {tokenizer_key!r} must be a mapping")
    return model_cfg


def _built_in_prompts(count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(max(0, count)):
        template = PROMPT_TEMPLATES[index % len(PROMPT_TEMPLATES)]
        topic = TOPICS[(index // len(PROMPT_TEMPLATES)) % len(TOPICS)]
        variant = index // (len(PROMPT_TEMPLATES) * len(TOPICS))
        user_probe = template.format(topic=topic)
        if variant:
            user_probe = f"{user_probe} Keep the answer concrete and avoid unnecessary detail."
        rows.append(
            {
                "prompt_id": f"nat_prompt_{index:06d}",
                "user_probe": user_probe,
                "prompt_family": f"PF{(index % 4) + 1}",
                "topic": topic,
            }
        )
    return rows


def _load_prompt_rows(path: Path, max_records: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if max_records > 0:
        rows = rows[:max_records]
    return rows


def _render_prompt(tokenizer: Any, user_probe: str) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            rendered = apply_chat_template(
                [{"role": "user", "content": user_probe}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass
    return f"User: {user_probe}\nAssistant:"


def _decode_completion(tokenizer: Any, generated_ids: Any, prompt_length: int) -> str:
    completion_ids = generated_ids[prompt_length:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return str(text).strip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    protocol_id = str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1"))
    model_cfg = _model_config(config, args.tokenizer_key)
    model_name = str(model_cfg.get("model_name", ""))
    tokenizer_name = str(model_cfg.get("tokenizer_name", model_name))
    if not model_name or not tokenizer_name:
        raise ValueError(f"Missing model/tokenizer name for {args.tokenizer_key}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "generate_reference_outputs requires optional hf dependencies: torch and transformers"
        ) from error

    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is False")
    device = torch.device("cuda" if cuda_available else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if cuda_available and torch.cuda.is_bf16_supported() else None
    model_kwargs: dict[str, Any] = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    if args.prompt_bank_jsonl:
        prompt_rows = _load_prompt_rows(resolve_repo_path(args.prompt_bank_jsonl, root), args.max_records)
    else:
        prompt_rows = _built_in_prompts(args.prompt_count)
    if args.max_records > 0:
        prompt_rows = prompt_rows[: args.max_records]

    output_rows: list[dict[str, Any]] = []
    batch_size = max(1, args.batch_size)
    with torch.no_grad():
        for start in range(0, len(prompt_rows), batch_size):
            batch = prompt_rows[start : start + batch_size]
            rendered_prompts = [
                _render_prompt(tokenizer, str(row.get("user_probe", row.get("prompt", ""))))
                for row in batch
            ]
            tokenized = tokenizer(
                rendered_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            prompt_lengths = [int(mask.sum().item()) for mask in tokenized["attention_mask"]]
            do_sample = args.temperature > 0.0
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": max(1, args.max_new_tokens),
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = float(args.temperature)
            generated = model.generate(**tokenized, **generate_kwargs)
            for row, rendered_prompt, generated_row, prompt_length in zip(
                batch,
                rendered_prompts,
                generated,
                prompt_lengths,
                strict=True,
            ):
                response_text = _decode_completion(tokenizer, generated_row, prompt_length)
                output_rows.append(
                    {
                        "schema_name": "natural_evidence_reference_output_v1",
                        "protocol_id": protocol_id,
                        "tokenizer_key": args.tokenizer_key,
                        "tokenizer_name": tokenizer_name,
                        "model_name": model_name,
                        "prompt_id": str(row.get("prompt_id", f"nat_prompt_{len(output_rows):06d}")),
                        "prompt_family": str(row.get("prompt_family", "")),
                        "topic": str(row.get("topic", "")),
                        "user_probe": str(row.get("user_probe", row.get("prompt", ""))),
                        "prompt": rendered_prompt,
                        "response_text": response_text,
                    }
                )

    output_path = resolve_repo_path(args.output_jsonl, root)
    write_jsonl(output_path, output_rows)
    print(json.dumps({"rows": len(output_rows), "output_jsonl": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

