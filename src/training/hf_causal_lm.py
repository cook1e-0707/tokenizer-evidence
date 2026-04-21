from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.core.contextual_alignment import (
    ContextualCarrierAuditResult,
    ContextualSlotTarget,
    audit_contextual_slot_targets,
)
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
    health_diagnostics: dict[str, object]


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
        dtype = getattr(torch_module, "long", None)
        if dtype is not None:
            try:
                return tensor_ctor(labels_payload, dtype=dtype).to(device)
            except TypeError:
                pass
        return tensor_ctor(labels_payload).to(device)
    return labels_payload


def _build_input_tensor(torch_module: object, payload: list[list[int]], device: object) -> object:
    tensor_ctor = getattr(torch_module, "tensor", None)
    if callable(tensor_ctor):
        dtype = getattr(torch_module, "long", None)
        if dtype is not None:
            try:
                return tensor_ctor(payload, dtype=dtype).to(device)
            except TypeError:
                pass
        return tensor_ctor(payload).to(device)
    return payload


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
        "top_k": 0,
        "renormalize_logits": True,
        "remove_invalid_values": True,
    }
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if allowed_token_ids is not None:
        generation_kwargs["prefix_allowed_tokens_fn"] = (
            lambda _batch_id, _input_ids, allowed=tuple(int(token_id) for token_id in allowed_token_ids): list(allowed)
        )
    return generation_kwargs


def _resolve_fieldwise_contextual_token_map(
    *,
    tokenizer: object,
    plan: FieldwiseGenerationPlan,
) -> tuple[
    ContextualCarrierAuditResult,
    dict[tuple[str, str], tuple[dict[str, int], dict[int, str]]],
]:
    audit_result = audit_contextual_slot_targets(
        slot_targets=[
            ContextualSlotTarget(
                field_name=target.field_name,
                exact_slot_prefix=target.exact_slot_prefix,
                allowed_values=target.allowed_values,
            )
            for target in plan.slot_targets
        ],
        tokenizer=tokenizer,
        prompt_contract_name=plan.prompt_contract_name,
    )
    if not audit_result.is_context_safe:
        first_failure = next(item for item in audit_result.diagnostics if not item.is_valid_next_token)
        raise HFCausalLMTrainingError(
            "Field-wise constrained decoding requires every allowed carrier to map to exactly one "
            "next token under the exact slot prefix; "
            f"field={first_failure.field_name!r}, prefix={first_failure.exact_slot_prefix!r}, "
            f"value={first_failure.carrier!r}, reasons={list(first_failure.reasons)!r}, "
            f"matching_token_ids={list(first_failure.matching_token_ids)!r}"
        )
    slot_maps: dict[tuple[str, str], tuple[dict[str, int], dict[int, str]]] = {}
    for field_name, prefix_map in audit_result.valid_token_map.items():
        for exact_slot_prefix, value_to_token_id in prefix_map.items():
            token_id_to_value = {
                int(token_id): value
                for value, token_id in value_to_token_id.items()
            }
            if len(token_id_to_value) != len(value_to_token_id):
                raise HFCausalLMTrainingError(
                    "Constrained decoding found ambiguous token ids for distinct allowed carriers "
                    f"in field={field_name!r}, prefix={exact_slot_prefix!r}"
                )
            slot_maps[(field_name, exact_slot_prefix)] = (
                {value: int(token_id) for value, token_id in value_to_token_id.items()},
                token_id_to_value,
            )
    return audit_result, slot_maps


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


def _build_fieldwise_training_batch(
    *,
    torch_module: object,
    tokenizer: object,
    batch_examples: Sequence[TrainingExample],
    slot_token_maps: Mapping[tuple[str, str], tuple[dict[str, int], dict[int, str]]],
    max_length: int,
    device: object,
) -> tuple[object, object, object]:
    if tokenizer.pad_token_id is None:
        raise HFCausalLMTrainingError("Field-wise training requires tokenizer.pad_token_id to be defined")

    input_payload: list[list[int]] = []
    attention_payload: list[list[int]] = []
    labels_payload: list[list[int]] = []
    max_width = 0
    for example in batch_examples:
        completion = str(example.metadata.get("completion", "")).strip()
        field_name = str(example.metadata.get("slot_type", "")).strip()
        if not completion:
            raise HFCausalLMTrainingError(
                f"Field-wise training example is missing a completion value for prompt={example.prompt!r}"
            )
        if not field_name:
            raise HFCausalLMTrainingError(
                f"Field-wise training example is missing slot_type metadata for prompt={example.prompt!r}"
            )
        slot_key = (field_name, example.prompt)
        if slot_key not in slot_token_maps:
            raise HFCausalLMTrainingError(
                "Field-wise training example prompt was not audited as a valid exact slot prefix; "
                f"field={field_name!r}, prompt={example.prompt!r}"
            )
        value_to_token_id, _token_id_to_value = slot_token_maps[slot_key]
        if completion not in value_to_token_id:
            raise HFCausalLMTrainingError(
                "Field-wise training example completion is not a contextual single-token carrier "
                f"under its exact slot prefix; field={field_name!r}, value={completion!r}"
            )
        prompt_ids = _tokenize_text(tokenizer, example.prompt)
        target_token_id = int(value_to_token_id[completion])
        row = [*prompt_ids, target_token_id]
        if len(row) > max_length:
            raise HFCausalLMTrainingError(
                "Field-wise training example exceeds model max_length after appending the "
                f"single-token carrier; length={len(row)}, max_length={max_length}, "
                f"field={field_name!r}, value={completion!r}"
            )
        labels = [-100] * len(prompt_ids) + [target_token_id]
        input_payload.append(row)
        attention_payload.append([1] * len(row))
        labels_payload.append(labels)
        max_width = max(max_width, len(row))

    for row, attention_row, label_row in zip(input_payload, attention_payload, labels_payload, strict=True):
        pad_width = max_width - len(row)
        if pad_width <= 0:
            continue
        row.extend([int(tokenizer.pad_token_id)] * pad_width)
        attention_row.extend([0] * pad_width)
        label_row.extend([-100] * pad_width)

    return (
        _build_input_tensor(torch_module, input_payload, device),
        _build_input_tensor(torch_module, attention_payload, device),
        _build_labels_tensor(torch_module, labels_payload, device),
    )


def _build_compiled_training_batch(
    *,
    torch_module: object,
    tokenizer: object,
    batch_examples: Sequence[TrainingExample],
    max_length: int,
    device: object,
) -> tuple[object, object]:
    if tokenizer.pad_token_id is None:
        raise HFCausalLMTrainingError("Compiled bucket training requires tokenizer.pad_token_id to be defined")

    input_payload: list[list[int]] = []
    attention_payload: list[list[int]] = []
    max_width = 0
    for example in batch_examples:
        prompt_token_ids = tuple(int(token_id) for token_id in example.metadata.get("compiled_prompt_token_ids", ()))
        if not prompt_token_ids:
            prompt_token_ids = tuple(_tokenize_text(tokenizer, example.prompt))
        if not prompt_token_ids:
            raise HFCausalLMTrainingError(
                f"Compiled bucket training example has no prompt tokens: prompt={example.prompt!r}"
            )
        if len(prompt_token_ids) > max_length:
            raise HFCausalLMTrainingError(
                f"Compiled bucket training prompt exceeds model max_length; "
                f"length={len(prompt_token_ids)}, max_length={max_length}, prompt={example.prompt!r}"
            )
        row = list(prompt_token_ids)
        input_payload.append(row)
        attention_payload.append([1] * len(row))
        max_width = max(max_width, len(row))

    for row, attention_row in zip(input_payload, attention_payload, strict=True):
        pad_width = max_width - len(row)
        if pad_width <= 0:
            continue
        row.extend([int(tokenizer.pad_token_id)] * pad_width)
        attention_row.extend([0] * pad_width)

    return (
        _build_input_tensor(torch_module, input_payload, device),
        _build_input_tensor(torch_module, attention_payload, device),
    )


def _compute_grad_norm(model: object) -> float:
    total = 0.0
    for parameter in model.parameters():
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        try:
            grad_norm = float(gradient.detach().data.norm(2).cpu().item())
        except AttributeError:
            try:
                grad_norm = float(gradient.detach().norm(2).cpu().item())
            except AttributeError:
                continue
        total += grad_norm * grad_norm
    return total ** 0.5


def _compute_compiled_bucket_loss(
    *,
    torch_module: object,
    logits: object,
    attention_mask: object,
    batch_examples: Sequence[TrainingExample],
    objective_mode: str,
) -> tuple[object, float]:
    normalized_objective_mode = objective_mode.strip().lower()
    if normalized_objective_mode not in {"bucket_mass", "fixed_representative", "uniform_bucket"}:
        raise HFCausalLMTrainingError(
            f"Unsupported compiled objective mode {objective_mode!r}; "
            "expected one of {'bucket_mass', 'fixed_representative', 'uniform_bucket'}"
        )
    attention_rows = _tensor_rows(attention_mask)
    sample_losses: list[object] = []
    max_logit = 0.0
    for row_index, example in enumerate(batch_examples):
        active_width = sum(int(mask_value) for mask_value in attention_rows[row_index])
        if active_width <= 0:
            raise HFCausalLMTrainingError(
                f"Compiled bucket training batch row has zero active tokens at row_index={row_index}"
            )
        row_logits = logits[row_index, active_width - 1, :]
        allowed_token_ids = [int(token_id) for token_id in example.metadata.get("compiled_allowed_token_ids", ())]
        if not allowed_token_ids:
            raise HFCausalLMTrainingError(
                f"Compiled bucket training example has no allowed_token_ids: prompt={example.prompt!r}"
            )
        bucket_to_token_ids_payload = dict(example.metadata.get("compiled_bucket_to_token_ids", {}))
        if not bucket_to_token_ids_payload:
            raise HFCausalLMTrainingError(
                f"Compiled bucket training example has no bucket_to_token_ids: prompt={example.prompt!r}"
            )
        bucket_to_token_ids_raw: dict[int, tuple[int, ...]] = {
            int(bucket_id): tuple(int(token_id) for token_id in token_ids_raw)
            for bucket_id, token_ids_raw in bucket_to_token_ids_payload.items()
        }
        allowed_logits = row_logits[allowed_token_ids]
        try:
            max_logit = max(max_logit, float(allowed_logits.detach().abs().max().cpu().item()))
        except AttributeError:
            pass
        allowed_log_probs = torch_module.log_softmax(allowed_logits, dim=0)
        token_id_to_position = {
            token_id: position
            for position, token_id in enumerate(allowed_token_ids)
        }
        bucket_log_probs: dict[int, object] = {}
        for bucket_id, token_ids_raw in bucket_to_token_ids_raw.items():
            positions = [
                token_id_to_position[int(token_id)]
                for token_id in token_ids_raw
                if int(token_id) in token_id_to_position
            ]
            if not positions:
                raise HFCausalLMTrainingError(
                    f"Compiled bucket objective found an empty token set for bucket={bucket_id} "
                    f"and prompt={example.prompt!r}"
                )
            bucket_log_probs[bucket_id] = torch_module.logsumexp(allowed_log_probs[positions], dim=0)
        target_bucket_id = int(example.metadata.get("compiled_target_bucket_id"))
        if target_bucket_id not in bucket_log_probs:
            raise HFCausalLMTrainingError(
                f"Compiled bucket target bucket is missing from masked logits: "
                f"bucket={target_bucket_id}, prompt={example.prompt!r}"
            )
        if normalized_objective_mode == "bucket_mass":
            sample_losses.append(-bucket_log_probs[target_bucket_id])
            continue

        target_bucket_token_ids = tuple(
            int(token_id)
            for token_id in bucket_to_token_ids_raw.get(target_bucket_id, ())
            if int(token_id) in token_id_to_position
        )
        if not target_bucket_token_ids:
            raise HFCausalLMTrainingError(
                f"Compiled objective found no valid token ids for target bucket={target_bucket_id} "
                f"under prompt={example.prompt!r}"
            )
        target_bucket_positions = [token_id_to_position[token_id] for token_id in target_bucket_token_ids]
        if normalized_objective_mode == "uniform_bucket":
            sample_losses.append(-allowed_log_probs[target_bucket_positions].mean())
            continue

        target_token_id = int(example.metadata.get("compiled_target_token_id"))
        if target_token_id not in token_id_to_position:
            raise HFCausalLMTrainingError(
                f"Compiled fixed representative target token is missing from masked logits: "
                f"token_id={target_token_id}, prompt={example.prompt!r}"
            )
        sample_losses.append(-allowed_log_probs[token_id_to_position[target_token_id]])

    return torch_module.stack(sample_losses).mean(), max_logit


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
    use_compiled_bucket_objective: bool = False,
    compiled_objective_mode: str = "bucket_mass",
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
    normalized_compiled_objective_mode = compiled_objective_mode.strip().lower() or "bucket_mass"
    if use_compiled_bucket_objective and normalized_compiled_objective_mode not in {
        "bucket_mass",
        "fixed_representative",
        "uniform_bucket",
    }:
        raise HFCausalLMTrainingError(
            f"Unsupported compiled_objective_mode={compiled_objective_mode!r}; "
            "expected one of {'bucket_mass', 'fixed_representative', 'uniform_bucket'}"
        )

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
    fieldwise_contextual_audit_result: ContextualCarrierAuditResult | None = None
    fieldwise_slot_token_maps: dict[tuple[str, str], tuple[dict[str, int], dict[int, str]]] | None = None
    if fieldwise_generation_plan is not None:
        fieldwise_contextual_audit_result, fieldwise_slot_token_maps = _resolve_fieldwise_contextual_token_map(
            tokenizer=tokenizer,
            plan=fieldwise_generation_plan,
        )

    effective_batch_size = max(1, batch_size)
    total_steps = 0
    examples_seen = 0
    final_loss = 0.0
    health_diagnostics: dict[str, object] = {
        "first_nan_step": None,
        "first_nonfinite_step": None,
        "max_logit": 0.0,
        "last_grad_norm": 0.0,
        "max_grad_norm": 0.0,
        "objective_mode": (
            f"compiled_{normalized_compiled_objective_mode}"
            if use_compiled_bucket_objective
            else "token_supervision"
        ),
    }
    for _epoch in range(max(1, epochs)):
        for start in range(0, len(texts), effective_batch_size):
            batch_examples = dataset[start : start + effective_batch_size]
            if use_compiled_bucket_objective:
                input_ids, attention_mask = _build_compiled_training_batch(
                    torch_module=torch,
                    tokenizer=tokenizer,
                    batch_examples=batch_examples,
                    max_length=max_length,
                    device=device,
                )
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = getattr(outputs, "logits", None)
                if logits is None:
                    raise HFCausalLMTrainingError(
                        "Compiled bucket objective requires model outputs to expose logits"
                    )
                loss, batch_max_logit = _compute_compiled_bucket_loss(
                    torch_module=torch,
                    logits=logits,
                    attention_mask=attention_mask,
                    batch_examples=batch_examples,
                    objective_mode=normalized_compiled_objective_mode,
                )
                if not math.isfinite(float(batch_max_logit)):
                    health_diagnostics["first_nan_step"] = total_steps + 1
                    health_diagnostics["first_nonfinite_step"] = total_steps + 1
                    raise HFCausalLMTrainingError(
                        "Non-finite masked logit encountered during compiled bucket training; "
                        f"step={total_steps + 1}, epoch={_epoch + 1}, batch_start={start}"
                    )
                health_diagnostics["max_logit"] = max(
                    float(health_diagnostics["max_logit"]),
                    float(batch_max_logit),
                )
            elif fieldwise_slot_token_maps is not None:
                input_ids, attention_mask, labels = _build_fieldwise_training_batch(
                    torch_module=torch,
                    tokenizer=tokenizer,
                    batch_examples=batch_examples,
                    slot_token_maps=fieldwise_slot_token_maps,
                    max_length=max_length,
                    device=device,
                )
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            else:
                batch_texts = texts[start : start + effective_batch_size]
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
                supervised_token_count = sum(
                    1
                    for label_row in labels_payload
                    for label_token_id in label_row
                    if label_token_id != -100
                )
                if supervised_token_count <= 0:
                    raise HFCausalLMTrainingError(
                        "Sequence-level training batch contains no supervised tokens after label masking. "
                        "This usually means prompt-only tokenization consumed the intended completion under "
                        "the active tokenizer boundary rules."
                    )
                labels = _build_labels_tensor(torch, labels_payload, device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                health_diagnostics["first_nan_step"] = total_steps + 1
                health_diagnostics["first_nonfinite_step"] = total_steps + 1
                raise HFCausalLMTrainingError(
                    "Non-finite training loss encountered during HF causal-LM training; "
                    f"step={total_steps + 1}, epoch={_epoch + 1}, batch_start={start}, "
                    f"examples_in_batch={len(batch_examples)}, target_mode="
                    f"{fieldwise_generation_plan.prompt_contract_name if fieldwise_generation_plan else 'sequence'}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = _compute_grad_norm(model)
            if not math.isfinite(grad_norm):
                health_diagnostics["first_nan_step"] = total_steps + 1
                health_diagnostics["first_nonfinite_step"] = total_steps + 1
                raise HFCausalLMTrainingError(
                    "Non-finite gradient norm encountered during HF causal-LM training; "
                    f"step={total_steps + 1}, epoch={_epoch + 1}, batch_start={start}"
                )
            health_diagnostics["last_grad_norm"] = grad_norm
            health_diagnostics["max_grad_norm"] = max(
                float(health_diagnostics["max_grad_norm"]),
                float(grad_norm),
            )
            optimizer.step()

            total_steps += 1
            examples_seen += len(batch_examples)
            final_loss = loss_value

    checkpoint_dir = run_dir / "checkpoints" / "hf_last"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    model.eval()
    generation_diagnostics: dict[str, object] = {}
    with torch.no_grad():
        if fieldwise_generation_plan is not None:
            if fieldwise_contextual_audit_result is None or fieldwise_slot_token_maps is None:
                raise HFCausalLMTrainingError("Missing contextual audit data for field-wise generation")
            generated_values: list[str] = []
            slot_results: list[dict[str, object]] = []
            for slot_target in fieldwise_generation_plan.slot_targets:
                value_to_token_id, token_id_to_value = fieldwise_slot_token_maps[
                    (slot_target.field_name, slot_target.exact_slot_prefix)
                ]
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
                        "exact_slot_prefix": slot_target.exact_slot_prefix,
                        "allowed_values": list(slot_target.allowed_values),
                        "allowed_token_count": len(value_to_token_id),
                        "chosen_token_id": chosen_token_id,
                        "chosen_token_text": chosen_text,
                        "token_text": chosen_text,
                        "is_field_valid": field_valid,
                        "field_valid": field_valid,
                        "is_bucket_correct": bucket_correct,
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
            generation_diagnostics["contextual_carrier_audit"] = fieldwise_contextual_audit_result.to_dict()
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
        health_diagnostics=health_diagnostics,
    )


def _default_lora_target_modules(model_name_or_path: str) -> tuple[str, ...]:
    normalized_name = model_name_or_path.strip().lower()
    if "qwen" in normalized_name or "llama" in normalized_name or "mistral" in normalized_name:
        return ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    if "gpt2" in normalized_name:
        return ("c_attn", "c_proj", "c_fc")
    return ("q_proj", "v_proj")
