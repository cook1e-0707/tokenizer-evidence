from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.core.scaffolded_completion import FieldwiseGenerationPlan
from src.training.dataset import TrainingExample


class HFCausalLMTrainingError(RuntimeError):
    """Raised when the minimal HF causal-LM training path cannot run."""


@dataclass(frozen=True)
class HFCausalLMTrainingResult:
    status: str
    steps: int
    examples_seen: int
    final_loss: float
    checkpoint_dir: str
    generated_text: str
    generation_diagnostics: dict[str, object]


def _training_text(example: TrainingExample) -> str:
    completion = str(example.metadata.get("completion", "")).strip()
    if not completion:
        completion = " ".join(example.target_symbols).strip()
    if completion:
        return f"{example.prompt}{completion}".strip()
    return example.prompt.strip() or "Ownership verification training sample."


def _tokenize_text(tokenizer: object, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return [int(token_id) for token_id in encode(text, add_special_tokens=False)]
        except TypeError:
            return [int(token_id) for token_id in encode(text)]
    tokenized = tokenizer(text, add_special_tokens=False)
    input_ids = tokenized.get("input_ids", [])
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        return [int(token_id) for token_id in input_ids[0]]
    if hasattr(input_ids, "tolist"):
        rows = input_ids.tolist()
        if rows and isinstance(rows[0], list):
            return [int(token_id) for token_id in rows[0]]
        return [int(token_id) for token_id in rows]
    payload = getattr(input_ids, "payload", None)
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        return [int(token_id) for token_id in payload[0]]
    return [int(token_id) for token_id in input_ids]


def _tensor_rows(tensor: object) -> list[list[int]]:
    if hasattr(tensor, "tolist"):
        return [[int(token_id) for token_id in row] for row in tensor.tolist()]
    payload = getattr(tensor, "payload", tensor)
    return [[int(token_id) for token_id in row] for row in payload]


def _build_labels_tensor(torch_module: object, labels_payload: list[list[int]], device: object) -> object:
    tensor_ctor = getattr(torch_module, "tensor", None)
    if callable(tensor_ctor):
        return tensor_ctor(labels_payload).to(device)
    return labels_payload


def _resolve_contextual_single_token_ids(
    tokenizer: object,
    *,
    prompt: str,
    allowed_values: Sequence[str],
) -> tuple[dict[str, int], dict[int, str]]:
    prompt_ids = _tokenize_text(tokenizer, prompt)
    value_to_token_id: dict[str, int] = {}
    token_id_to_value: dict[int, str] = {}
    for value in allowed_values:
        full_ids = _tokenize_text(tokenizer, f"{prompt}{value}")
        continuation_ids = full_ids[len(prompt_ids) :]
        if len(continuation_ids) != 1:
            raise HFCausalLMTrainingError(
                "Field-wise constrained decoding requires every allowed carrier to map to exactly "
                f"one continuation token in context; value={value!r} produced {continuation_ids!r}"
            )
        token_id = int(continuation_ids[0])
        if token_id in token_id_to_value and token_id_to_value[token_id] != value:
            raise HFCausalLMTrainingError(
                "Constrained decoding found ambiguous token ids for distinct allowed carriers: "
                f"token_id={token_id}, carriers=({token_id_to_value[token_id]!r}, {value!r})"
            )
        value_to_token_id[value] = token_id
        token_id_to_value[token_id] = value
    return value_to_token_id, token_id_to_value


def _build_generation_kwargs(
    *,
    tokenizer: object,
    max_new_tokens: int,
    generation_do_sample: bool,
    allowed_token_ids: Sequence[int] | None = None,
) -> dict[str, object]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    generation_kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": generation_do_sample,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
    }
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if allowed_token_ids is not None:
        generation_kwargs["prefix_allowed_tokens_fn"] = (
            lambda _batch_id, _input_ids, allowed=tuple(int(token_id) for token_id in allowed_token_ids): list(allowed)
        )
    return generation_kwargs


def _compute_fieldwise_generation_diagnostics(
    *,
    plan: FieldwiseGenerationPlan,
    generated_values: Sequence[str],
    slot_results: Sequence[dict[str, object]],
) -> dict[str, object]:
    total_slots = len(plan.slot_targets)
    exact_matches = sum(
        1
        for observed, expected in zip(generated_values, plan.expected_slot_values, strict=True)
        if observed == expected
    )
    parse_success_count = sum(1 for slot_result in slot_results if slot_result["field_valid"])
    valid_blocks = 0
    for start in range(0, total_slots, plan.fields_per_block):
        block_results = slot_results[start : start + plan.fields_per_block]
        if len(block_results) != plan.fields_per_block:
            break
        if all(bool(slot_result["field_valid"]) for slot_result in block_results):
            valid_blocks += 1
    per_field_totals: dict[str, int] = {}
    per_field_exact: dict[str, int] = {}
    for slot_result, expected_value in zip(slot_results, plan.expected_slot_values, strict=True):
        field_name = str(slot_result["slot_type"])
        per_field_totals[field_name] = per_field_totals.get(field_name, 0) + 1
        if slot_result["token_text"] == expected_value:
            per_field_exact[field_name] = per_field_exact.get(field_name, 0) + 1
    per_field_accuracy = {
        field_name: per_field_exact.get(field_name, 0) / total
        for field_name, total in per_field_totals.items()
        if total
    }
    decode_success_rate = (
        1.0 if all(bool(slot_result["bucket_correct"]) for slot_result in slot_results) else 0.0
    )
    return {
        "slot_results": list(slot_results),
        "per_field_accuracy": per_field_accuracy,
        "per_slot_exact_rate": exact_matches / total_slots if total_slots else 0.0,
        "valid_canonical_block_count": valid_blocks,
        "parse_success_rate": parse_success_count / total_slots if total_slots else 0.0,
        "decode_success_rate": decode_success_rate,
    }


def run_minimal_hf_causal_lm_training(
    *,
    model_name_or_path: str,
    max_length: int,
    dataset: Sequence[TrainingExample],
    batch_size: int,
    epochs: int,
    learning_rate: float,
    run_dir: Path,
    require_cuda: bool = False,
    generation_prompt: str = "",
    generation_do_sample: bool = False,
    generation_max_new_tokens: int = 16,
    generation_stop_strings: Sequence[str] = (),
    generation_bad_words: Sequence[str] = (),
    generation_suppress_tokens: Sequence[int] = (),
    generation_sequence_bias: Mapping[str, float] | None = None,
    adapter_mode: str = "full",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    lora_target_modules: Sequence[str] = (),
    fieldwise_generation_plan: FieldwiseGenerationPlan | None = None,
) -> HFCausalLMTrainingResult:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise HFCausalLMTrainingError(
            "Minimal HF causal-LM training requires both 'torch' and 'transformers' to be installed."
        ) from error

    if not model_name_or_path.strip():
        raise HFCausalLMTrainingError("model_name_or_path must be non-empty for HF causal-LM training")
    if not dataset:
        raise HFCausalLMTrainingError("HF causal-LM training requires at least one training example")
    if generation_max_new_tokens <= 0:
        raise HFCausalLMTrainingError("generation_max_new_tokens must be positive")
    normalized_adapter_mode = adapter_mode.strip().lower() or "full"
    if normalized_adapter_mode not in {"full", "lora"}:
        raise HFCausalLMTrainingError(
            f"Unsupported adapter_mode={adapter_mode!r}; expected 'full' or 'lora'"
        )
    if lora_r <= 0:
        raise HFCausalLMTrainingError("lora_r must be positive")
    if lora_alpha <= 0:
        raise HFCausalLMTrainingError("lora_alpha must be positive")
    if lora_dropout < 0:
        raise HFCausalLMTrainingError("lora_dropout must be non-negative")

    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise HFCausalLMTrainingError(
            "GPU training was requested but torch.cuda.is_available() is False inside the job. "
            "This usually means the runtime environment or CUDA driver on the allocated node is "
            "not compatible with the installed PyTorch build."
        )

    device = torch.device("cuda" if cuda_available else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if normalized_adapter_mode == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as error:
            raise HFCausalLMTrainingError(
                "adapter_mode='lora' requires the optional 'peft' package to be installed."
            ) from error
        target_modules = tuple(lora_target_modules) or _default_lora_target_modules(model_name_or_path)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    model.to(device)
    model.train()

    texts = [_training_text(example) for example in dataset]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    effective_batch_size = max(1, batch_size)
    total_steps = 0
    examples_seen = 0
    final_loss = 0.0
    for _epoch in range(max(1, epochs)):
        for start in range(0, len(texts), effective_batch_size):
            batch_texts = texts[start : start + effective_batch_size]
            batch_examples = dataset[start : start + effective_batch_size]
            tokenized = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            input_id_rows = _tensor_rows(input_ids)
            attention_rows = _tensor_rows(attention_mask)
            prompt_token_lengths = [
                len(_tokenize_text(tokenizer, example.prompt))
                for example in batch_examples
            ]
            labels_payload: list[list[int]] = []
            for row_index, input_row in enumerate(input_id_rows):
                label_row = list(input_row)
                prompt_length = prompt_token_lengths[row_index]
                for token_index, mask_value in enumerate(attention_rows[row_index]):
                    if not mask_value or token_index < prompt_length:
                        label_row[token_index] = -100
                labels_payload.append(label_row)
            labels = _build_labels_tensor(torch, labels_payload, device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_steps += 1
            examples_seen += len(batch_texts)
            final_loss = float(loss.detach().cpu().item())

    checkpoint_dir = run_dir / "checkpoints" / "hf_last"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    model.eval()
    generation_diagnostics: dict[str, object] = {}
    with torch.no_grad():
        if fieldwise_generation_plan is not None:
            generated_values: list[str] = []
            slot_results: list[dict[str, object]] = []
            for slot_target in fieldwise_generation_plan.slot_targets:
                value_to_token_id, token_id_to_value = _resolve_contextual_single_token_ids(
                    tokenizer,
                    prompt=slot_target.prompt,
                    allowed_values=slot_target.allowed_values,
                )
                generation_inputs = tokenizer(
                    slot_target.prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                generation_inputs = {key: value.to(device) for key, value in generation_inputs.items()}
                prompt_rows = _tensor_rows(generation_inputs["input_ids"])
                prompt_length = len(prompt_rows[0]) if prompt_rows else 0
                generation_kwargs = _build_generation_kwargs(
                    tokenizer=tokenizer,
                    max_new_tokens=1,
                    generation_do_sample=False,
                    allowed_token_ids=tuple(value_to_token_id.values()),
                )
                generated_tokens = model.generate(
                    **generation_inputs,
                    **generation_kwargs,
                )
                generated_rows = _tensor_rows(generated_tokens)
                chosen_token_id = (
                    int(generated_rows[0][prompt_length])
                    if generated_rows and len(generated_rows[0]) > prompt_length
                    else None
                )
                chosen_text = (
                    token_id_to_value[chosen_token_id]
                    if chosen_token_id in token_id_to_value
                    else ""
                )
                field_valid = chosen_text in slot_target.allowed_value_bucket_ids
                chosen_bucket_id = (
                    slot_target.allowed_value_bucket_ids.get(chosen_text)
                    if field_valid
                    else None
                )
                bucket_correct = field_valid and chosen_text == slot_target.expected_value
                generated_values.append(chosen_text)
                slot_results.append(
                    {
                        "slot_index": slot_target.slot_index,
                        "slot_type": slot_target.field_name,
                        "allowed_token_count": len(value_to_token_id),
                        "chosen_token_id": chosen_token_id,
                        "token_text": chosen_text,
                        "field_valid": field_valid,
                        "bucket_correct": bucket_correct,
                        "expected_bucket_id": slot_target.expected_bucket_id,
                        "chosen_bucket_id": chosen_bucket_id,
                    }
                )
            generated_text = "\n".join(generated_values).strip()
            generation_diagnostics = _compute_fieldwise_generation_diagnostics(
                plan=fieldwise_generation_plan,
                generated_values=tuple(generated_values),
                slot_results=tuple(slot_results),
            )
        else:
            prompt = generation_prompt if generation_prompt else dataset[0].prompt.strip()
            if not prompt:
                prompt = "Ownership verification prompt."
            generation_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            generation_inputs = {key: value.to(device) for key, value in generation_inputs.items()}
            generation_kwargs = _build_generation_kwargs(
                tokenizer=tokenizer,
                max_new_tokens=generation_max_new_tokens,
                generation_do_sample=generation_do_sample,
            )
            if generation_bad_words:
                bad_words_ids: list[list[int]] = []
                for bad_word in generation_bad_words:
                    if not bad_word.strip():
                        continue
                    token_ids = _tokenize_text(tokenizer, bad_word)
                    if token_ids:
                        bad_words_ids.append(list(token_ids))
                if bad_words_ids:
                    generation_kwargs["bad_words_ids"] = bad_words_ids
            if generation_suppress_tokens:
                generation_kwargs["suppress_tokens"] = [int(token_id) for token_id in generation_suppress_tokens]
            if generation_sequence_bias:
                sequence_bias: dict[tuple[int, ...], float] = {}
                for text, bias in generation_sequence_bias.items():
                    if not str(text).strip():
                        continue
                    token_ids = _tokenize_text(tokenizer, str(text))
                    if token_ids:
                        sequence_bias[tuple(int(token_id) for token_id in token_ids)] = float(bias)
                if sequence_bias:
                    generation_kwargs["sequence_bias"] = sequence_bias
            generated_tokens = model.generate(
                **generation_inputs,
                **generation_kwargs,
            )
            generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

            if prompt and generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt) :].lstrip()

            for stop_string in generation_stop_strings:
                if not stop_string:
                    continue
                stop_index = generated_text.find(stop_string)
                if stop_index >= 0:
                    generated_text = generated_text[:stop_index]
                    break

            generated_text = generated_text.strip()

    return HFCausalLMTrainingResult(
        status="completed",
        steps=total_steps,
        examples_seen=examples_seen,
        final_loss=round(final_loss, 6),
        checkpoint_dir=str(checkpoint_dir),
        generated_text=generated_text,
        generation_diagnostics=generation_diagnostics,
    )


def _default_lora_target_modules(model_name_or_path: str) -> tuple[str, ...]:
    normalized_name = model_name_or_path.strip().lower()
    if "qwen" in normalized_name or "llama" in normalized_name or "mistral" in normalized_name:
        return ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    if "gpt2" in normalized_name:
        return ("c_attn", "c_proj", "c_fc")
    return ("q_proj", "v_proj")
