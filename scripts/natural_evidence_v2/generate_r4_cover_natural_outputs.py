from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.generate_wp6_e2e_outputs import (  # noqa: E402
    batched,
    build_chat_text,
    load_model_condition,
)
from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    read_jsonl,
    resolve,
    sha256_file,
    sha256_text,
    technical_literal_hits,
    write_json_new,
    write_jsonl_new,
)


DEFAULT_PROMPTS = ROOT / "results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/dev_prompts.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate R4 cover-natural dev diagnostic outputs for Qwen protected, "
            "raw, and task-only conditions. Wrong-key and wrong-payload are "
            "decoder controls over protected transcripts. This script does not "
            "train, run Llama, aggregate FAR, or make paper claims."
        )
    )
    parser.add_argument("--prompts-jsonl", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--protected-adapter", type=Path, required=True)
    parser.add_argument("--task-only-adapter", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-prompts", type=int, default=512)
    parser.add_argument("--prompt-file-row-start", type=int, required=True)
    parser.add_argument("--prompt-file-row-end", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--replicate-group-id", default="local")
    parser.add_argument("--validate-plan-only", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def read_jsonl_with_file_index(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = read_jsonl(path)
    return [(index, dict(row)) for index, row in enumerate(rows)]


def select_prompts(
    rows: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    split: str,
    max_prompts: int,
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    if end - start + 1 != max_prompts:
        raise ValueError("prompt file-row window must match max-prompts")
    selected = [(idx, dict(row)) for idx, row in rows if start <= idx <= end and str(row.get("split", "")) == split]
    if len(selected) != max_prompts:
        raise ValueError(f"selected {len(selected)} prompts for rows {start}..{end}; expected {max_prompts}")
    output: list[dict[str, Any]] = []
    for selected_index, (file_row_index, row) in enumerate(selected):
        text = str(row.get("prompt_text", ""))
        if "Step " in text or "slot" in text.lower() or "exactly 16" in text:
            raise ValueError(f"R4 prompt contains forbidden structural instruction: {row.get('prompt_id')}")
        hits = technical_literal_hits(text)
        if hits:
            raise ValueError(f"R4 prompt contains technical literal {hits}: {row.get('prompt_id')}")
        row["selected_prompt_file_row_index"] = int(file_row_index)
        row["selected_prompt_index"] = int(selected_index)
        row["prompt_index"] = int(selected_index)
        output.append(row)
    if len({str(row["prompt_id"]) for row in output}) != len(output):
        raise ValueError("duplicate selected prompt_id")
    return output


def write_plan_summary(
    *,
    args: argparse.Namespace,
    prompts_path: Path,
    output_dir: Path,
    prompts: Sequence[Mapping[str, Any]],
    generation_started: bool,
) -> None:
    write_json_new(
        output_dir / "r4_generation_plan_summary.json",
        {
            "schema_name": "natural_evidence_v2_r4_generation_plan_v1",
            "artifact_role": "r4_cover_natural_dev_diagnostic_generation_plan_or_summary",
            "prompts_path": str(prompts_path),
            "prompts_sha256": sha256_file(prompts_path),
            "selected_prompt_file_row_start": int(prompts[0]["selected_prompt_file_row_index"]),
            "selected_prompt_file_row_end_inclusive": int(prompts[-1]["selected_prompt_file_row_index"]),
            "selected_prompt_jsonl_sha256": hashlib.sha256(
                "\n".join(str(row["prompt_id"]) for row in prompts).encode("utf-8")
            ).hexdigest(),
            "prompt_count": len(prompts),
            "split": str(args.split),
            "generation_conditions": ["protected", "raw", "task_only"],
            "decode_conditions": ["protected", "raw", "task_only", "wrong_key", "wrong_payload"],
            "model_name": str(args.model_name),
            "tokenizer_name": str(args.tokenizer_name),
            "max_new_tokens": int(args.max_new_tokens),
            "replicate_group_id": str(args.replicate_group_id),
            "generation_started": bool(generation_started),
            "training_started": False,
            "llama_started": False,
            "far_aggregation_started": False,
            "paper_claim_allowed": False,
        },
    )


def generate_condition(
    *,
    args: argparse.Namespace,
    prompts: Sequence[Mapping[str, Any]],
    condition: str,
    adapter_path: Path | None,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model, device = load_model_condition(
        model_name=args.model_name,
        adapter_path=adapter_path,
        require_cuda=bool(args.require_cuda),
    )
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(batched(prompts, args.batch_size)):
            chat_texts = [build_chat_text(tokenizer, str(row["prompt_text"])) for row in batch]
            encoded = tokenizer(chat_texts, return_tensors="pt", padding=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max(1, int(args.max_new_tokens)),
                pad_token_id=tokenizer.pad_token_id,
            )
            input_width = int(encoded["input_ids"].shape[1])
            for row_index, (prompt_row, output_ids) in enumerate(zip(batch, generated, strict=True)):
                response_text = tokenizer.decode(output_ids[input_width:], skip_special_tokens=True).strip()
                generation_id = "qwen_v2_r4_gen_" + sha256_text(
                    json.dumps(
                        {
                            "batch_index": batch_index,
                            "condition": condition,
                            "prompt_id": str(prompt_row["prompt_id"]),
                            "row_index": row_index,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )[:20]
                outputs.append(
                    {
                        "adapter_path": str(adapter_path) if adapter_path is not None else "",
                        "artifact_role": "r4_cover_natural_dev_diagnostic_transcript",
                        "contract_id": "a55e",
                        "generation_condition": condition,
                        "generation_id": generation_id,
                        "generation_mode": "deterministic_greedy",
                        "max_new_tokens": int(args.max_new_tokens),
                        "model_name": str(args.model_name),
                        "paper_claim_allowed": False,
                        "prompt_id": str(prompt_row["prompt_id"]),
                        "prompt_index": int(prompt_row["prompt_index"]),
                        "prompt_file_row_index": int(prompt_row["selected_prompt_file_row_index"]),
                        "prompt_text": str(prompt_row["prompt_text"]),
                        "prompt_text_sha256": str(prompt_row.get("prompt_text_sha256", "")),
                        "replicate_group_id": str(args.replicate_group_id),
                        "response_text": response_text,
                        "response_text_sha256": sha256_text(response_text),
                        "schema_name": "natural_evidence_v2_r4_generated_output_v1",
                        "split": str(prompt_row.get("split", "")),
                        "tokenizer_name": str(args.tokenizer_name),
                        "training_started": False,
                    }
                )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def main() -> int:
    args = parse_args()
    prompts_path = resolve(args.prompts_jsonl)
    output_dir = resolve(args.output_dir)
    protected_adapter = resolve(args.protected_adapter)
    task_only_adapter = resolve(args.task_only_adapter)
    prompts = select_prompts(
        read_jsonl_with_file_index(prompts_path),
        split=str(args.split),
        max_prompts=int(args.max_prompts),
        start=int(args.prompt_file_row_start),
        end=int(args.prompt_file_row_end),
    )
    if args.validate_plan_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_plan_summary(args=args, prompts_path=prompts_path, output_dir=output_dir, prompts=prompts, generation_started=False)
        print(json.dumps({"status": "PLAN_ONLY_PASS", "output_dir": str(output_dir)}, sort_keys=True))
        return 0
    if not (protected_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"protected adapter missing: {protected_adapter}")
    if not (task_only_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"task-only adapter missing: {task_only_adapter}")
    if (output_dir / "r4_generated_outputs.jsonl").exists():
        raise FileExistsError(f"refusing to overwrite R4 generated outputs in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for condition, adapter in (("protected", protected_adapter), ("raw", None), ("task_only", task_only_adapter)):
        rows.extend(generate_condition(args=args, prompts=prompts, condition=condition, adapter_path=adapter))
    write_jsonl_new(output_dir / "r4_generated_outputs.jsonl", rows)
    write_plan_summary(args=args, prompts_path=prompts_path, output_dir=output_dir, prompts=prompts, generation_started=True)
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
