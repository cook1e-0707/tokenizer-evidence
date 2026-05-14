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
    parser.add_argument("--row-mode", choices=["wp5_micro_slot", "r4_prefix_native_surface"], default="wp5_micro_slot")
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
    parser.add_argument("--target-mass-floor", type=float, default=0.0)
    parser.add_argument("--target-mass-floor-lambda", type=float, default=0.0)
    parser.add_argument("--target-mass-ceiling", type=float, default=0.0)
    parser.add_argument("--target-mass-ceiling-lambda", type=float, default=0.0)
    parser.add_argument("--stratum-weighting-mode", choices=["none", "r4_candidate_v3_failed_surface"], default="none")
    parser.add_argument("--stratum-weight-max", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def validate_objective_args(args: argparse.Namespace) -> None:
    allowed_weighting_modes = {"none", "r4_candidate_v3_failed_surface"}
    if str(args.stratum_weighting_mode) not in allowed_weighting_modes:
        raise ValueError(
            "stratum-weighting-mode must be one of "
            f"{sorted(allowed_weighting_modes)}"
        )
    nonnegative_fields = (
        "task_ce_weight",
        "margin_lambda",
        "margin_tau",
        "target_mass_floor",
        "target_mass_floor_lambda",
        "target_mass_ceiling",
        "target_mass_ceiling_lambda",
    )
    for field in nonnegative_fields:
        value = float(getattr(args, field))
        if not math.isfinite(value):
            raise ValueError(f"{field.replace('_', '-')} must be finite")
        if value < 0:
            raise ValueError(f"{field.replace('_', '-')} must be non-negative")
    if float(args.target_mass_floor) > 1.0:
        raise ValueError("target-mass-floor must be <= 1.0")
    if float(args.target_mass_ceiling) > 1.0:
        raise ValueError("target-mass-ceiling must be <= 1.0")
    if float(args.target_mass_floor) > 0 and float(args.target_mass_ceiling) > 0:
        if float(args.target_mass_floor) > float(args.target_mass_ceiling):
            raise ValueError("target-mass-floor must be <= target-mass-ceiling when both are enabled")
    if not math.isfinite(float(args.stratum_weight_max)):
        raise ValueError("stratum-weight-max must be finite")
    if float(args.stratum_weight_max) <= 0:
        raise ValueError("stratum-weight-max must be positive")


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


def extract_training_stratum(row: Mapping[str, Any], slot: Mapping[str, Any] | None = None) -> str:
    slot = slot or {}
    for key in ("assistant_prefix_model_text", "prefix_family", "target_surface", "surface_label"):
        value = slot.get(key, row.get(key))
        if value:
            return f"{key}:{value}"
    family_id = str(row.get("family_id", "unknown_family"))
    bit_role = str(slot.get("bit_role", row.get("bit_role", "unknown_bit_role")))
    return f"family_id:{family_id}|bit_role:{bit_role}"


def weight_for_stratum(stratum: str, mode: str, max_weight: float) -> float:
    if max_weight <= 0:
        raise ValueError("stratum weight max must be positive")
    if mode == "none":
        return 1.0
    if mode != "r4_candidate_v3_failed_surface":
        raise ValueError(f"unsupported stratum weighting mode: {mode}")
    reviewed_weights = {
        "assistant_prefix_model_text:A useful action is:": 3.0,
        "target_surface:Prepare a note": 3.0,
        "target_surface:Prepare questions": 2.0,
        "target_surface:Create a simple timeline": 1.5,
    }
    if "|" in stratum:
        return min(float(max_weight), max(reviewed_weights.get(part, 1.0) for part in stratum.split("|")))
    return min(float(max_weight), reviewed_weights.get(stratum, 1.0))


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> dict[str, float]:
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if not values:
        return {"mean": 0.0, "raw_count": 0.0, "effective_weighted_count": 0.0}
    total_weight = sum(float(weight) for weight in weights)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")
    weighted_total = sum(float(value) * float(weight) for value, weight in zip(values, weights))
    return {
        "mean": weighted_total / total_weight,
        "raw_count": float(len(values)),
        "effective_weighted_count": float(total_weight),
    }


def target_mass_floor_loss(target_mass: float, floor: float) -> float:
    return max(0.0, float(floor) - float(target_mass))


def target_mass_ceiling_loss(target_mass: float, ceiling: float) -> float:
    if float(ceiling) <= 0:
        return 0.0
    return max(0.0, float(target_mass) - float(ceiling))


def chat_full_text(tokenizer: Any, prompt_text: str, response_text: str) -> str:
    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": response_text},
    ]
    if getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return f"User: {prompt_text}\nAssistant: {response_text}"


def chat_prefix_text(tokenizer: Any, prompt_text: str, assistant_prefix: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    if getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)) + assistant_prefix
    return f"User: {prompt_text}\nAssistant: {assistant_prefix}"


def r4_surface_boundary(row: Mapping[str, Any]) -> dict[str, Any]:
    assistant_prefix = str(row.get("assistant_prefix_before_surface", ""))
    assistant_prefix_model_text = assistant_prefix.rstrip()
    surface_prefix_text = assistant_prefix[len(assistant_prefix_model_text) :]
    target_bit = int(row.get("target_bit"))
    other_bit = 1 - target_bit
    return {
        "assistant_prefix_model_text": assistant_prefix_model_text,
        "surface_prefix_text": surface_prefix_text,
        "target_surface": str(row.get("target_surface", "")),
        "target_surfaces": [str(item) for item in row.get(f"bucket_{target_bit}_surfaces", [])],
        "other_surfaces": [str(item) for item in row.get(f"bucket_{other_bit}_surfaces", [])],
    }


def first_token_id_after_prefix(tokenizer: Any, prefix: str, surface: str) -> int:
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    combined_ids = tokenizer(prefix + surface, add_special_tokens=False).input_ids
    if len(combined_ids) <= len(prefix_ids):
        raise ValueError(f"surface produced no next token: {surface!r}")
    if combined_ids[: len(prefix_ids)] == prefix_ids:
        return int(combined_ids[len(prefix_ids)])
    surface_ids = tokenizer(surface, add_special_tokens=False).input_ids
    if not surface_ids:
        raise ValueError(f"surface tokenized to empty ids: {surface!r}")
    return int(surface_ids[0])


def first_token_ids_for_surfaces(tokenizer: Any, prefix: str, surface_prefix_text: str, surfaces: Sequence[str]) -> list[int]:
    ids: list[int] = []
    for surface in surfaces:
        ids.append(first_token_id_after_prefix(tokenizer, prefix, surface_prefix_text + str(surface)))
    return ids


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
                "stratum": extract_training_stratum(row, slot),
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


def prepare_r4_prefix_native_example(tokenizer: Any, row: Mapping[str, Any], *, max_length: int) -> dict[str, Any]:
    prompt_text = str(row.get("prompt_text", ""))
    response_text = str(row.get("target_response_text", ""))
    boundary = r4_surface_boundary(row)
    assistant_prefix_model_text = str(boundary["assistant_prefix_model_text"])
    surface_prefix_text = str(boundary["surface_prefix_text"])
    target_surface = str(boundary["target_surface"])
    assistant_prefix_before_surface = assistant_prefix_model_text + surface_prefix_text
    if not response_text.startswith(assistant_prefix_before_surface):
        raise ValueError("R4 target response does not start with assistant_prefix_before_surface")
    if response_text[len(assistant_prefix_before_surface) :].find(target_surface) != 0:
        raise ValueError("R4 target response does not place target_surface immediately after prefix")

    prefix_text = chat_prefix_text(tokenizer, prompt_text, assistant_prefix_model_text)
    continuation_text = response_text[len(assistant_prefix_model_text) :]
    full_text = prefix_text + continuation_text
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

    surface_span_start = len(prefix_text)
    surface_span_end = surface_span_start + len(surface_prefix_text) + len(target_surface)
    first_surface_token_index = None
    for token_index, offset in enumerate(offsets):
        if offset[1] <= len(prefix_text):
            labels[token_index] = -100
        if max(offset[0], surface_span_start) < min(offset[1], surface_span_end):
            labels[token_index] = -100
            if first_surface_token_index is None:
                first_surface_token_index = token_index

    if first_surface_token_index is None or first_surface_token_index == 0:
        raise ValueError("could not locate R4 prefix-native surface token position")

    target_ids = first_token_ids_for_surfaces(
        tokenizer,
        prefix_text,
        surface_prefix_text,
        list(boundary["target_surfaces"]),
    )
    other_ids = first_token_ids_for_surfaces(
        tokenizer,
        prefix_text,
        surface_prefix_text,
        list(boundary["other_surfaces"]),
    )
    overlap = set(target_ids) & set(other_ids)
    if overlap:
        raise ValueError(f"R4 target/other token id overlap: {sorted(overlap)}")

    if len(input_ids) >= max_length:
        raise ValueError("training example was truncated; increase --max-length")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "margin_positions": [
            {
                "logit_position": first_surface_token_index - 1,
                "stratum": (
                    f"assistant_prefix_model_text:{assistant_prefix_model_text}"
                    f"|target_surface:{target_surface}"
                ),
                "target_ids": target_ids,
                "other_ids": other_ids,
            }
        ],
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
    validate_objective_args(args)
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output dir: {output_dir}")
    output_dir.mkdir(parents=True)

    rows = read_jsonl(args.train_rows)[: args.max_rows]
    bank = None if args.row_mode == "r4_prefix_native_surface" else load_bank(args.primary_bank)
    if not rows:
        raise ValueError("no training rows selected")

    dry_summary = {
        "arm": args.arm,
        "artifact_role": "wp5_micro_slot_lora_training",
        "bank_id": bank.get("bank_id") if bank is not None else None,
        "e2e_eval_started": False,
        "input_row_count": len(rows),
        "max_steps": args.max_steps,
        "model_name": args.model_name,
        "not_full_far": True,
        "not_payload_recovery": True,
        "paper_claim_allowed": False,
        "schema_name": "natural_evidence_v2_wp5_micro_slot_lora_train_summary_v1",
        "row_mode": args.row_mode,
        "training_started": not args.dry_run,
        "ce_masking_active": True,
        "margin_lambda": args.margin_lambda,
        "margin_tau": args.margin_tau,
        "raw_margin_count": 0.0,
        "effective_weighted_margin_count": 0.0,
        "stratum_weight_max": args.stratum_weight_max,
        "stratum_weighting_mode": args.stratum_weighting_mode,
        "target_mass_floor": args.target_mass_floor,
        "target_mass_ceiling": args.target_mass_ceiling,
        "target_mass_ceiling_count": 0.0,
        "target_mass_ceiling_lambda": args.target_mass_ceiling_lambda,
        "target_mass_floor_count": 0.0,
        "target_mass_floor_lambda": args.target_mass_floor_lambda,
        "task_only_bucket_margin_active": False,
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
    if args.row_mode == "r4_prefix_native_surface":
        examples = [prepare_r4_prefix_native_example(tokenizer, row, max_length=args.max_length) for row in rows]
    else:
        if bank is None:
            raise ValueError("primary bank is required for wp5_micro_slot row mode")
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
        floor_loss = torch.zeros((), dtype=ce_loss.dtype, device=device)
        ceiling_loss = torch.zeros((), dtype=ce_loss.dtype, device=device)
        margin_count = 0
        weighted_margin_count = 0.0
        floor_count = 0
        ceiling_count = 0
        if args.arm == "protected":
            weighted_margin_sum = torch.zeros((), dtype=ce_loss.dtype, device=device)
            weighted_floor_sum = torch.zeros((), dtype=ce_loss.dtype, device=device)
            weighted_ceiling_sum = torch.zeros((), dtype=ce_loss.dtype, device=device)
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
                    weight = weight_for_stratum(
                        str(slot.get("stratum", "")),
                        args.stratum_weighting_mode,
                        args.stratum_weight_max,
                    )
                    weighted_margin_sum = weighted_margin_sum + (float(weight) * F.relu(args.margin_tau + other_mass - target_mass))
                    margin_count += 1
                    weighted_margin_count += float(weight)
                    if args.target_mass_floor > 0:
                        weighted_floor_sum = weighted_floor_sum + (float(weight) * F.relu(args.target_mass_floor - target_mass))
                        floor_count += 1
                    if args.target_mass_ceiling > 0:
                        weighted_ceiling_sum = weighted_ceiling_sum + (float(weight) * F.relu(target_mass - args.target_mass_ceiling))
                        ceiling_count += 1
            if margin_count:
                margin_loss = weighted_margin_sum / weighted_margin_count
            if floor_count:
                floor_loss = weighted_floor_sum / weighted_margin_count
            if ceiling_count:
                ceiling_loss = weighted_ceiling_sum / weighted_margin_count
        loss = args.task_ce_weight * ce_loss
        if args.arm == "protected":
            loss = (
                loss
                + args.margin_lambda * margin_loss
                + args.target_mass_floor_lambda * floor_loss
                + args.target_mass_ceiling_lambda * ceiling_loss
            )
        (loss / args.gradient_accumulation_steps).backward()
        if step_index % args.gradient_accumulation_steps == 0 or step_index == args.max_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        metrics.append(
            {
                "ce_loss": float(ce_loss.detach().cpu()),
                "ceiling_loss": float(ceiling_loss.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "floor_loss": float(floor_loss.detach().cpu()),
                "effective_weighted_margin_count": float(weighted_margin_count),
                "margin_count": float(margin_count),
                "margin_loss": float(margin_loss.detach().cpu()),
                "target_mass_floor_count": float(floor_count),
                "target_mass_ceiling_count": float(ceiling_count),
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
        "final_ceiling_loss": metrics[-1]["ceiling_loss"] if metrics else math.nan,
        "final_floor_loss": metrics[-1]["floor_loss"] if metrics else math.nan,
        "final_loss": metrics[-1]["loss"] if metrics else math.nan,
        "final_margin_loss": metrics[-1]["margin_loss"] if metrics else math.nan,
        "learning_rate": args.learning_rate,
        "lora_alpha": args.lora_alpha,
        "lora_r": args.lora_r,
        "metrics_tail": metrics[-10:],
        "raw_margin_count": metrics[-1]["margin_count"] if metrics else 0.0,
        "effective_weighted_margin_count": metrics[-1]["effective_weighted_margin_count"] if metrics else 0.0,
        "target_mass_floor_count": metrics[-1]["target_mass_floor_count"] if metrics else 0.0,
        "target_mass_ceiling_count": metrics[-1]["target_mass_ceiling_count"] if metrics else 0.0,
        "status": "TRAINED_ADAPTER_NOT_E2E_NOT_FAR",
    }
    write_json(output_dir / "wp5_micro_slot_lora_train_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
