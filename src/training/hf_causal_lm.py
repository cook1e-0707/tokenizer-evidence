from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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


def _training_text(example: TrainingExample) -> str:
    completion = str(example.metadata.get("completion", "")).strip()
    if not completion:
        completion = " ".join(example.target_symbols).strip()
    if completion:
        return f"{example.prompt}\n{completion}".strip()
    return example.prompt.strip() or "Ownership verification training sample."


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
            tokenized = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
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
    with torch.no_grad():
        prompt = generation_prompt if generation_prompt else dataset[0].prompt.strip()
        if not prompt:
            prompt = "Ownership verification prompt."
        generation_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        generation_inputs = {key: value.to(device) for key, value in generation_inputs.items()}
        generation_kwargs = {
            "max_new_tokens": generation_max_new_tokens,
            "do_sample": generation_do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if generation_bad_words:
            bad_words_ids: list[list[int]] = []
            for bad_word in generation_bad_words:
                if not bad_word.strip():
                    continue
                try:
                    token_ids = tokenizer.encode(bad_word, add_special_tokens=False)
                except TypeError:
                    token_ids = tokenizer.encode(bad_word)
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
                try:
                    token_ids = tokenizer.encode(str(text), add_special_tokens=False)
                except TypeError:
                    token_ids = tokenizer.encode(str(text))
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
    )


def _default_lora_target_modules(model_name_or_path: str) -> tuple[str, ...]:
    normalized_name = model_name_or_path.strip().lower()
    if "qwen" in normalized_name or "llama" in normalized_name or "mistral" in normalized_name:
        return ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    if "gpt2" in normalized_name:
        return ("c_attn", "c_proj", "c_fc")
    return ("q_proj", "v_proj")
