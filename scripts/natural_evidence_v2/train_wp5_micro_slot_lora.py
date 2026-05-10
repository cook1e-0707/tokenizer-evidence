from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_PRIMARY_BANK = ROOT / "results/natural_evidence_v2/buckets/qwen_v2_primary_2way_bank.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a natural_evidence_v2 WP5 micro-slot LoRA arm. This script "
            "implements slot exact-token CE masking plus protected-arm bucket "
            "margin loss. It does not run E2E, decode payloads, aggregate FAR, "
            "or make paper-facing positive claims."
        )
    )
    parser.add_argument("--train-rows", type=Path, required=True)
    parser.add_argument("--primary-bank", type=Path, default=DEFAULT_PRIMARY_BANK)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=["protected", "task_only"], required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--task-ce-weight", type=float, default=1.0)
    parser.add_argument("--margin-lambda", type=float, default=5.0)
    parser.add_argument("--margin-tau", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_new(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def load_bank(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"expected one primary bank row: {path}")
    bank = rows[0]
    if bank.get("schema_name") != "natural_evidence_v2_primary_2way_micro_slot_bank_v1":
        raise ValueError("primary bank schema mismatch")
    return {
        "bank_id": bank.get("bank_id"),
        "bucket_0_token_ids": [int(item) for item in bank.get("bucket_0_token_ids", [])],
        "bucket_1_token_ids": [int(item) for item in bank.get("bucket_1_token_ids", [])],
        "bucket_0_surfaces": [str(item) for item in bank.get("bucket_0_surfaces", [])],
        "bucket_1_surfaces": [str(item) for item in bank.get("bucket_1_surfaces", [])],
    }


def chat_full_text(tokenizer: Any, prompt_text: str, response_text: str) -> str:
    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": response_text},
    ]
    if getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return f"User: {prompt_text}\nAssistant: {response_text}"


def find_slot_surface_spans(response_text: str, slot_targets: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for slot in slot_targets:
        line = str(slot.get("target_line", ""))
        surface = str(slot.get("target_surface", ""))
        if not line or not surface:
            continue
        line_start = response_text.find(line, cursor)
        if line_start < 0:
            line_start = response_text.find(line)
        if line_start < 0:
            raise ValueError(f"target line not found in response: {line!r}")
        surface_start_in_line = line.find(surface)
        if surface_start_in_line < 0:
            raise ValueError(f"target surface not found in target line: {surface!r}")
        start = line_start + surface_start_in_line
        end = start + len(surface)
        spans.append((start, end))
        cursor = line_start + len(line)
    return spans


def overlaps(span_a: tuple[int, int], span_b: tuple[int, int]) -> bool:
    return max(span_a[0], span_b[0]) < min(span_a[1], span_b[1])


def prepare_example(tokenizer: Any, row: Mapping[str, Any], bank: Mapping[str, Any], *, max_length: int) -> dict[str, Any]:
    prompt_text = str(row.get("prompt_text", ""))
    response_text = str(row.get("target_response_text", ""))
    full_text = chat_full_text(tokenizer, prompt_text, response_text)
    response_start = full_text.rfind(response_text)
    if response_start < 0:
        raise ValueError("target response text not found inside chat-formatted training text")

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids = [int(item) for item in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    labels = list(input_ids)
    slot_spans_full: list[tuple[int, int]] = []
    slot_targets = row.get("slot_targets", []) if isinstance(row.get("slot_targets", []), list) else []
    for start, end in find_slot_surface_spans(response_text, slot_targets):
        slot_spans_full.append((response_start + start, response_start + end))

    margin_positions: list[dict[str, Any]] = []
    for token_index, offset in enumerate(offsets):
        if offset[1] <= response_start:
            labels[token_index] = -100
            continue
        for slot_span in slot_spans_full:
            if overlaps(offset, slot_span):
                labels[token_index] = -100
                break

    for slot in slot_targets:
        target_surface = str(slot.get("target_surface", ""))
        target_line = str(slot.get("target_line", ""))
        surface_start_in_line = target_line.find(target_surface)
        line_start = response_text.find(target_line)
        if line_start < 0 or surface_start_in_line < 0:
            continue
        surface_start = response_start + line_start + surface_start_in_line
        surface_token_index = None
        for token_index, offset in enumerate(offsets):
            if offset[0] <= surface_start < offset[1]:
                surface_token_index = token_index
                break
        if surface_token_index is None or surface_token_index == 0:
            continue
        target_bucket_id = str(slot.get("target_bucket_id"))
        other_bucket_id = "1" if target_bucket_id == "0" else "0"
        margin_positions.append(
            {
                "logit_position": surface_token_index - 1,
                "target_ids": list(bank[f"bucket_{target_bucket_id}_token_ids"]),
                "other_ids": list(bank[f"bucket_{other_bucket_id}_token_ids"]),
            }
        )

    if len(input_ids) >= max_length:
        raise ValueError("training example was truncated; increase --max-length")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "margin_positions": margin_positions,
    }


def collate(batch: Sequence[Mapping[str, Any]], *, pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention_mask: list[list[int]] = []
    margin_positions: list[list[Mapping[str, Any]]] = []
    for item in batch:
        length = len(item["input_ids"])
        pad = max_len - length
        input_ids.append(list(item["input_ids"]) + [pad_token_id] * pad)
        labels.append(list(item["labels"]) + [-100] * pad)
        attention_mask.append([1] * length + [0] * pad)
        margin_positions.append(list(item.get("margin_positions", [])))
    return {
        "attention_mask": attention_mask,
        "input_ids": input_ids,
        "labels": labels,
        "margin_positions": margin_positions,
    }


def batched_cycle(items: Sequence[Mapping[str, Any]], batch_size: int, steps: int) -> Iterable[list[Mapping[str, Any]]]:
    if not items:
        raise ValueError("empty dataset")
    index = 0
    for _ in range(steps):
        batch = []
        for _ in range(batch_size):
            batch.append(items[index % len(items)])
            index += 1
        yield batch


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output dir: {output_dir}")
    output_dir.mkdir(parents=True)

    rows = read_jsonl(args.train_rows)[: args.max_rows]
    bank = load_bank(args.primary_bank)
    if not rows:
        raise ValueError("no training rows selected")

    dry_summary = {
        "arm": args.arm,
        "artifact_role": "wp5_micro_slot_lora_training",
        "bank_id": bank.get("bank_id"),
        "e2e_eval_started": False,
        "input_row_count": len(rows),
        "max_steps": args.max_steps,
        "model_name": args.model_name,
        "not_full_far": True,
        "not_payload_recovery": True,
        "paper_claim_allowed": False,
        "schema_name": "natural_evidence_v2_wp5_micro_slot_lora_train_summary_v1",
        "training_started": not args.dry_run,
    }
    if args.dry_run:
        write_json(output_dir / "wp5_micro_slot_lora_train_summary.json", {**dry_summary, "status": "DRY_RUN_VALIDATED_INPUTS"})
        print(json.dumps({**dry_summary, "status": "DRY_RUN_VALIDATED_INPUTS"}, indent=2, sort_keys=True))
        return 0

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but CUDA is not available")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.train()
    examples = [prepare_example(tokenizer, row, bank, max_length=args.max_length) for row in rows]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    metrics: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    for step_index, batch_examples in enumerate(batched_cycle(examples, args.batch_size, args.max_steps), start=1):
        batch = collate(batch_examples, pad_token_id=int(tokenizer.pad_token_id))
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
        labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        ce_loss = outputs.loss
        margin_loss = torch.zeros((), dtype=ce_loss.dtype, device=device)
        margin_count = 0
        if args.arm == "protected":
            for batch_index, slot_items in enumerate(batch["margin_positions"]):
                for slot in slot_items:
                    position = int(slot["logit_position"])
                    if position < 0 or position >= outputs.logits.shape[1]:
                        continue
                    probs = torch.softmax(outputs.logits[batch_index, position, :], dim=-1)
                    target_ids = torch.tensor(slot["target_ids"], dtype=torch.long, device=device)
                    other_ids = torch.tensor(slot["other_ids"], dtype=torch.long, device=device)
                    target_mass = probs.index_select(0, target_ids).sum()
                    other_mass = probs.index_select(0, other_ids).sum()
                    margin_loss = margin_loss + F.relu(args.margin_tau + other_mass - target_mass)
                    margin_count += 1
            if margin_count:
                margin_loss = margin_loss / margin_count
        loss = args.task_ce_weight * ce_loss + (args.margin_lambda * margin_loss if args.arm == "protected" else 0.0)
        (loss / args.gradient_accumulation_steps).backward()
        if step_index % args.gradient_accumulation_steps == 0 or step_index == args.max_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        metrics.append(
            {
                "ce_loss": float(ce_loss.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "margin_count": float(margin_count),
                "margin_loss": float(margin_loss.detach().cpu()),
                "step": float(step_index),
            }
        )

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    summary = {
        **dry_summary,
        "adapter_dir": str(adapter_dir),
        "final_ce_loss": metrics[-1]["ce_loss"] if metrics else math.nan,
        "final_loss": metrics[-1]["loss"] if metrics else math.nan,
        "final_margin_loss": metrics[-1]["margin_loss"] if metrics else math.nan,
        "learning_rate": args.learning_rate,
        "lora_alpha": args.lora_alpha,
        "lora_r": args.lora_r,
        "metrics_tail": metrics[-10:],
        "status": "TRAINED_ADAPTER_NOT_E2E_NOT_FAR",
    }
    write_json(output_dir / "wp5_micro_slot_lora_train_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
