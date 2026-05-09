from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUCKET_BANK = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "wp3_detector_bank_scaffold_repaired_20260508_2308/two_way_bucket_bank_scaffold.json"
)

DEFAULT_CONTEXTS: dict[str, tuple[str, ...]] = {
    "sentence_opener_sequence_v0": (
        "Write a practical answer. ",
        "Give the next sentence. ",
        "Begin the explanation. ",
    ),
    "step_opener_action_v0": (
        "Step 1: ",
        "For the next action, ",
        "In the checklist, ",
    ),
    "discourse_marker_additive_v0": (
        "The plan is simple. ",
        "This habit is useful. ",
        "The group can start small. ",
    ),
    "optional_hedge_frequency_v0": (
        "This approach ",
        "A simple checklist ",
        "A shared routine ",
    ),
    "transition_word_plain_v0": (
        "The first option is too slow. ",
        "The plan needs one more step. ",
        "The group has enough information. ",
    ),
    "function_word_conjunction_v0": (
        "Bring water ",
        "Review notes ",
        "Choose a simple plan ",
    ),
    "function_word_preposition_v0": (
        "Prepare the plan ",
        "Share the checklist ",
        "Review the notes ",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score natural_evidence_v2 WP3 two-way bucket next-token masses under "
            "a fixed reference model. This does not train, generate text, run "
            "E2E, estimate FAR, or make claims."
        )
    )
    parser.add_argument("--bucket-bank", type=Path, default=DEFAULT_BUCKET_BANK)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top level must be a mapping: {path}")
    return payload


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_new(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def encode_no_special(tokenizer: Any, text: str) -> list[int]:
    try:
        return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
    except TypeError:
        return [int(token_id) for token_id in tokenizer.encode(text)]


def load_model(*, model_name: str, tokenizer_name: str, require_cuda: bool) -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("WP3 bucket-mass scoring requires torch and transformers") from error

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True, "trust_remote_code": True}
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def bucket_token_ids(tokenizer: Any, bank: Mapping[str, Any]) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    for bucket_id, members in bank.get("buckets", {}).items():
        ids: list[int] = []
        for member in members:
            token_ids = encode_no_special(tokenizer, str(member))
            if len(token_ids) != 1:
                raise ValueError(
                    f"{bank.get('candidate_bank_id')}: member is not one token under scoring tokenizer: "
                    f"{member!r} -> {token_ids}"
                )
            ids.append(int(token_ids[0]))
        output[str(bucket_id)] = ids
    return output


def score_contexts(
    *,
    torch_module: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    bank: Mapping[str, Any],
    contexts: Sequence[str],
    max_length: int,
) -> list[dict[str, Any]]:
    token_ids_by_bucket = bucket_token_ids(tokenizer, bank)
    candidate_token_ids = sorted({token_id for ids in token_ids_by_bucket.values() for token_id in ids})
    rows: list[dict[str, Any]] = []
    with torch_module.no_grad():
        for context_index, context in enumerate(contexts):
            input_ids = encode_no_special(tokenizer, context)
            if not input_ids:
                raise ValueError(f"empty context after tokenization: {context!r}")
            input_ids = input_ids[-max_length:]
            tensor = torch_module.tensor([input_ids], dtype=torch_module.long, device=device)
            logits = model(input_ids=tensor).logits[0, -1, :].float()
            log_denom = torch_module.logsumexp(logits, dim=0)
            candidate_logits = logits[candidate_token_ids]
            candidate_denom = torch_module.logsumexp(candidate_logits, dim=0)
            full_masses: dict[str, float] = {}
            candidate_normalized_masses: dict[str, float] = {}
            for bucket_id, bucket_token_ids_value in sorted(token_ids_by_bucket.items()):
                bucket_tensor = torch_module.tensor(bucket_token_ids_value, dtype=torch_module.long, device=device)
                bucket_logits = logits[bucket_tensor]
                full_masses[bucket_id] = float(
                    torch_module.exp(torch_module.logsumexp(bucket_logits, dim=0) - log_denom)
                    .detach()
                    .cpu()
                    .item()
                )
                candidate_normalized_masses[bucket_id] = float(
                    torch_module.exp(torch_module.logsumexp(bucket_logits, dim=0) - candidate_denom)
                    .detach()
                    .cpu()
                    .item()
                )
            rows.append(
                {
                    "schema_name": "natural_evidence_v2_wp3_bucket_mass_context_score_v1",
                    "candidate_bank_id": str(bank["candidate_bank_id"]),
                    "slot_type": str(bank.get("slot_type", "")),
                    "context_index": context_index,
                    "context_text": context,
                    "context_token_count": len(input_ids),
                    "bucket_token_ids": token_ids_by_bucket,
                    "full_vocab_bucket_masses": full_masses,
                    "candidate_normalized_bucket_masses": candidate_normalized_masses,
                    "model_scoring_started": True,
                    "training_started": False,
                    "generation_started": False,
                    "e2e_eval_started": False,
                    "paper_claim_allowed": False,
                }
            )
    return rows


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ratio(masses: Mapping[str, float]) -> float | None:
    positives = [float(value) for value in masses.values() if float(value) > 0.0]
    if not positives:
        return None
    return max(positives) / min(positives)


def aggregate_mass_rows(context_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_bank: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in context_rows:
        by_bank[str(row["candidate_bank_id"])].append(row)
    output: list[dict[str, Any]] = []
    for bank_id, rows in sorted(by_bank.items()):
        bucket_ids = sorted(str(bucket_id) for bucket_id in rows[0]["full_vocab_bucket_masses"])
        full_masses = {
            bucket_id: mean([float(row["full_vocab_bucket_masses"][bucket_id]) for row in rows])
            for bucket_id in bucket_ids
        }
        candidate_masses = {
            bucket_id: mean([float(row["candidate_normalized_bucket_masses"][bucket_id]) for row in rows])
            for bucket_id in bucket_ids
        }
        output.append(
            {
                "schema_name": "natural_evidence_v2_wp3_bucket_mass_row_v1",
                "candidate_bank_id": bank_id,
                "bucket_masses": full_masses,
                "candidate_normalized_bucket_masses": candidate_masses,
                "full_vocab_min_bucket_mass": min(full_masses.values()) if full_masses else 0.0,
                "full_vocab_mass_ratio": ratio(full_masses),
                "candidate_normalized_mass_ratio": ratio(candidate_masses),
                "context_count": len(rows),
                "mass": full_masses,
                "model_scoring_started": True,
                "training_started": False,
                "generation_started": False,
                "e2e_eval_started": False,
                "paper_claim_allowed": False,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    bucket_bank_path = resolve_path(args.bucket_bank)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    bucket_bank = load_json(bucket_bank_path)
    torch_module, tokenizer, model, device = load_model(
        model_name=str(args.model_name),
        tokenizer_name=str(args.tokenizer_name),
        require_cuda=bool(args.require_cuda),
    )
    try:
        context_rows: list[dict[str, Any]] = []
        for bank in bucket_bank.get("candidate_banks", []):
            bank_id = str(bank["candidate_bank_id"])
            contexts = DEFAULT_CONTEXTS.get(bank_id)
            if not contexts:
                raise ValueError(f"no fixed scoring contexts for bank: {bank_id}")
            context_rows.extend(
                score_contexts(
                    torch_module=torch_module,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    bank=bank,
                    contexts=contexts,
                    max_length=max(1, int(args.max_length)),
                )
            )
    finally:
        del model
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()

    mass_rows = aggregate_mass_rows(context_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    mass_json = output_dir / "qwen_v2_wp3_bucket_mass_artifact.json"
    summary = {
        "schema_name": "natural_evidence_v2_wp3_bucket_mass_score_summary_v1",
        "status": "WP3_BUCKET_MASS_SCORED_NOT_TRAINING_NOT_GENERATION",
        "bucket_bank": str(bucket_bank_path),
        "model_name": str(args.model_name),
        "tokenizer_name": str(args.tokenizer_name),
        "context_score_rows": len(context_rows),
        "mass_rows": len(mass_rows),
        "mass_json": str(mass_json),
        "model_scoring_started": True,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "limitations": [
            "Fixed-prefix next-token mass scoring only.",
            "No text is generated and no model weights are changed.",
            "Mass gate outcomes are diagnostics, not payload recovery or FAR.",
        ],
    }
    write_jsonl(output_dir / "qwen_v2_wp3_bucket_mass_context_scores.jsonl", context_rows)
    write_json(mass_json, {"schema_name": "natural_evidence_v2_wp3_bucket_mass_artifact_v1", "mass_rows": mass_rows})
    write_json(output_dir / "qwen_v2_wp3_bucket_mass_score_summary.json", summary)
    print(json.dumps({"status": summary["status"], "mass_rows": len(mass_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
