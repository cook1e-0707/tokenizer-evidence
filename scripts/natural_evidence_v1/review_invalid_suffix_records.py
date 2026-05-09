from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.common import read_jsonl, read_yaml, resolve_repo_path, write_csv, write_json, write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review counterfactual candidate records skipped by suffix compatibility "
            "because they do not leave a valid suffix window. This is a CPU diagnostic."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--reference-outputs", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix-window-tokens", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=25)
    return parser.parse_args(argv)


def _model_config(config: dict[str, Any], tokenizer_key: str) -> dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, dict) or tokenizer_key not in models:
        raise ValueError(f"Missing model config for tokenizer key {tokenizer_key!r}")
    model_cfg = models[tokenizer_key]
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Model config for {tokenizer_key!r} must be a mapping")
    return model_cfg


def _reference_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("prompt_id", "")): row for row in rows if row.get("prompt_id")}


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded.get("input_ids", [])
    if not isinstance(token_ids, list):
        return []
    return [int(token_id) for token_id in token_ids]


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    try:
        return str(tokenizer.decode(token_ids, skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode(token_ids))


def _invalid_reason(offset: int | None, response_token_count: int, suffix_window_length: int) -> str:
    if offset is None:
        return "invalid_offset_non_integer"
    if response_token_count <= 0:
        return "empty_response_tokens"
    if offset < 0:
        return "negative_offset"
    if offset >= response_token_count:
        return "offset_beyond_response_tokens"
    if offset >= response_token_count - 1:
        return "offset_at_final_token_no_suffix"
    if suffix_window_length <= 0:
        return "empty_suffix_window"
    return ""


def _review_candidate_rows(
    *,
    tokenizer: Any,
    candidate_rows: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    suffix_window_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    invalid_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for row_index, row in enumerate(candidate_rows):
        prompt_id = str(row.get("prompt_id", ""))
        reference = references.get(prompt_id)
        if reference is None:
            reason = "missing_reference"
            response_ids: list[int] = []
            response = ""
            offset = None
            suffix_ids: list[int] = []
        else:
            response = str(reference.get("response_text", reference.get("output_text", "")))
            response_ids = _token_ids(tokenizer, response)
            try:
                offset = int(row.get("prefix_response_token_count", 0))
            except (TypeError, ValueError):
                offset = None
            if offset is None or offset < 0 or offset >= len(response_ids):
                suffix_ids = []
            else:
                suffix_ids = list(response_ids[offset + 1 : offset + 1 + max(1, suffix_window_tokens)])
            reason = _invalid_reason(offset, len(response_ids), len(suffix_ids))
        base = {
            "row_index": row_index,
            "bank_entry_id": str(row.get("bank_entry_id", "")),
            "prompt_id": prompt_id,
            "prefix_response_token_count": "" if offset is None else offset,
            "response_token_count": len(response_ids),
            "suffix_window_length": len(suffix_ids),
            "candidate_count": len(row.get("candidates", [])) if isinstance(row.get("candidates", []), list) else 0,
        }
        if reason:
            reason_counts[reason] += 1
            invalid_rows.append(
                {
                    **base,
                    "invalid_suffix_reason": reason,
                    "response_tail_surface": _decode_tokens(tokenizer, response_ids[max(0, len(response_ids) - 12) :]),
                    "offset_surface_context": (
                        _decode_tokens(tokenizer, response_ids[max(0, int(offset or 0) - 4) : min(len(response_ids), int(offset or 0) + 5)])
                        if offset is not None
                        else ""
                    ),
                    "reference_response_chars": len(response),
                }
            )
        else:
            valid_rows.append({**base, "invalid_suffix_reason": ""})
    return valid_rows, invalid_rows, reason_counts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    model_cfg = _model_config(config, args.tokenizer_key)
    tokenizer_name = str(model_cfg.get("tokenizer_name", model_cfg.get("model_name", "")))
    if not tokenizer_name:
        raise ValueError(f"Missing tokenizer name for {args.tokenizer_key}")

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("review_invalid_suffix_records requires transformers tokenizer support") from error

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    references = _reference_index(read_jsonl(resolve_repo_path(args.reference_outputs, root)))
    candidate_rows = read_jsonl(resolve_repo_path(args.candidate_jsonl, root))
    valid_rows, invalid_rows, reason_counts = _review_candidate_rows(
        tokenizer=tokenizer,
        candidate_rows=candidate_rows,
        references=references,
        suffix_window_tokens=args.suffix_window_tokens,
    )

    output_dir = resolve_repo_path(args.output_dir, root)
    invalid_csv = output_dir / "invalid_suffix_records.csv"
    examples_jsonl = output_dir / "invalid_suffix_examples.jsonl"
    summary_json = output_dir / "invalid_suffix_review_summary.json"
    fieldnames = [
        "row_index",
        "bank_entry_id",
        "prompt_id",
        "invalid_suffix_reason",
        "prefix_response_token_count",
        "response_token_count",
        "suffix_window_length",
        "candidate_count",
        "response_tail_surface",
        "offset_surface_context",
        "reference_response_chars",
    ]
    write_csv(invalid_csv, invalid_rows, fieldnames)
    write_jsonl(examples_jsonl, invalid_rows[: max(0, args.max_examples)])
    boundary_reasons = {"offset_at_final_token_no_suffix", "offset_beyond_response_tokens"}
    systemic_offset_bug_suspected = any(reason not in boundary_reasons for reason in reason_counts)
    summary = {
        "schema_name": "natural_evidence_invalid_suffix_review_summary_v1",
        "protocol_id": str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1")),
        "tokenizer_key": args.tokenizer_key,
        "tokenizer_name": tokenizer_name,
        "reference_outputs": str(resolve_repo_path(args.reference_outputs, root)),
        "candidate_jsonl": str(resolve_repo_path(args.candidate_jsonl, root)),
        "candidate_records": len(candidate_rows),
        "valid_suffix_records": len(valid_rows),
        "invalid_suffix_records": len(invalid_rows),
        "invalid_suffix_rate": len(invalid_rows) / len(candidate_rows) if candidate_rows else 0.0,
        "invalid_suffix_reason_counts": dict(sorted(reason_counts.items())),
        "systemic_offset_bug_suspected": systemic_offset_bug_suspected,
        "diagnostic_pilot_policy": "invalid rows may be excluded only if reasons are documented and not systemic tokenizer offset bugs",
        "paper_ready_policy": "fix_or_document_before_paper_ready_e2e",
        "result_claim": "invalid_suffix_review_not_payload_recovery",
        "outputs": {
            "invalid_suffix_records_csv": str(invalid_csv),
            "invalid_suffix_examples_jsonl": str(examples_jsonl),
            "invalid_suffix_review_summary_json": str(summary_json),
        },
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
