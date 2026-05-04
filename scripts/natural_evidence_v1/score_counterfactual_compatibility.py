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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score local suffix compatibility for natural opportunity-bank candidates. "
            "This is a quality gate, not a fingerprint result."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--reference-outputs", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--suffix-window-tokens", type=int, default=16)
    parser.add_argument("--max-candidates-per-record", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--delta-nll-threshold", type=float, default=0.5)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


def _model_config(config: dict[str, Any], tokenizer_key: str) -> dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, dict) or tokenizer_key not in models:
        raise ValueError(f"Missing model config for tokenizer key {tokenizer_key!r}")
    model_cfg = models[tokenizer_key]
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Model config for {tokenizer_key!r} must be a mapping")
    return model_cfg


def _nll_for_suffix(model: Any, torch: Any, input_ids: list[int], suffix_start: int, device: Any) -> float:
    if suffix_start >= len(input_ids):
        return 0.0
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels = torch.full_like(tensor, -100)
    labels[0, suffix_start:] = tensor[0, suffix_start:]
    outputs = model(input_ids=tensor, labels=labels)
    return float(outputs.loss.item())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
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
            "score_counterfactual_compatibility requires optional hf dependencies: torch and transformers"
        ) from error

    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is False")
    device = torch.device("cuda" if cuda_available else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    references = {
        str(row.get("prompt_id", "")): row
        for row in read_jsonl(resolve_repo_path(args.reference_outputs, root))
        if row.get("prompt_id")
    }
    candidate_rows = read_jsonl(resolve_repo_path(args.candidate_jsonl, root))
    if args.max_records > 0:
        candidate_rows = candidate_rows[: args.max_records]

    output_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in candidate_rows:
            prompt_id = str(row.get("prompt_id", ""))
            reference = references.get(prompt_id)
            if reference is None:
                continue
            offset = int(row.get("prefix_response_token_count", 0))
            prompt = str(reference.get("prompt", reference.get("user_probe", "")))
            response = str(reference.get("response_text", reference.get("output_text", "")))
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
            if not isinstance(prompt_ids, list) or not isinstance(response_ids, list):
                continue
            if offset < 0 or offset >= len(response_ids) - 1:
                continue
            prefix_ids = list(prompt_ids) + list(response_ids[:offset])
            original_next_id = int(response_ids[offset])
            suffix_ids = list(response_ids[offset + 1 : offset + 1 + max(1, args.suffix_window_tokens)])
            if not suffix_ids:
                continue
            baseline_ids = prefix_ids + [original_next_id] + suffix_ids
            baseline_nll = _nll_for_suffix(
                model,
                torch,
                baseline_ids,
                suffix_start=len(prefix_ids) + 1,
                device=device,
            )
            for candidate in row.get("candidates", [])[: max(1, args.max_candidates_per_record)]:
                if not isinstance(candidate, dict):
                    continue
                candidate_id = int(candidate.get("token_id", -1))
                if candidate_id < 0:
                    continue
                candidate_ids = prefix_ids + [candidate_id] + suffix_ids
                candidate_nll = _nll_for_suffix(
                    model,
                    torch,
                    candidate_ids,
                    suffix_start=len(prefix_ids) + 1,
                    device=device,
                )
                delta_nll = candidate_nll - baseline_nll
                output_rows.append(
                    {
                        "schema_name": "natural_evidence_counterfactual_compatibility_v1",
                        "protocol_id": row.get("protocol_id", "natural_evidence_v1"),
                        "tokenizer_key": args.tokenizer_key,
                        "tokenizer_name": tokenizer_name,
                        "model_name": model_name,
                        "prompt_id": prompt_id,
                        "prefix_response_token_count": offset,
                        "token_id": candidate_id,
                        "token_text": candidate.get("text", ""),
                        "rank": candidate.get("rank", ""),
                        "probability": candidate.get("probability", ""),
                        "baseline_suffix_nll": baseline_nll,
                        "candidate_suffix_nll": candidate_nll,
                        "delta_suffix_nll": delta_nll,
                        "compatibility_pass": delta_nll <= float(args.delta_nll_threshold),
                        "fingerprint_claim": False,
                    }
                )

    output_path = resolve_repo_path(args.output_jsonl, root)
    write_jsonl(output_path, output_rows)
    print(json.dumps({"rows": len(output_rows), "output_jsonl": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
