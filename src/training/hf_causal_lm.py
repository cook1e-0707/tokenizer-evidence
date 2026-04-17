from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
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
        prompt = dataset[0].prompt.strip() or "Ownership verification prompt."
        generation_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        generation_inputs = {key: value.to(device) for key, value in generation_inputs.items()}
        generated_tokens = model.generate(
            **generation_inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return HFCausalLMTrainingResult(
        status="completed",
        steps=total_steps,
        examples_seen=examples_seen,
        final_loss=round(final_loss, 6),
        checkpoint_dir=str(checkpoint_dir),
        generated_text=generated_text,
    )
