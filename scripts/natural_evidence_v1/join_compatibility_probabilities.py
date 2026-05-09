from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.common import read_jsonl, resolve_repo_path, write_json, write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join counterfactual compatibility rows back to original top-k candidate "
            "probabilities. This is a repair diagnostic, not a training run."
        )
    )
    parser.add_argument("--compatibility-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args(argv)


def _context_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("prompt_id", "")), int(row.get("prefix_response_token_count", 0))


def _candidate_token_id(candidate: dict[str, Any]) -> int:
    return int(candidate.get("token_id", -1))


def _candidate_probability(candidate: dict[str, Any]) -> float:
    return float(candidate.get("probability", 0.0))


def _candidate_rank(candidate: dict[str, Any]) -> int:
    return int(candidate.get("rank", 0))


def _stream_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            yield payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    compatibility_path = resolve_repo_path(args.compatibility_jsonl, root)
    candidate_path = resolve_repo_path(args.candidate_jsonl, root)
    output_path = resolve_repo_path(args.output_jsonl, root)
    summary_path = resolve_repo_path(args.summary_json, root)

    compatibility_rows = read_jsonl(compatibility_path)
    needed_tokens_by_context: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in compatibility_rows:
        token_id = int(row.get("token_id", -1))
        if token_id < 0:
            continue
        needed_tokens_by_context[_context_key(row)].add(token_id)

    candidate_lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    relevant_candidate_records = 0
    duplicate_matches = 0
    for record in _stream_jsonl(candidate_path):
        context = _context_key(record)
        needed = needed_tokens_by_context.get(context)
        if not needed:
            continue
        relevant_candidate_records += 1
        candidates = record.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            token_id = _candidate_token_id(candidate)
            if token_id not in needed:
                continue
            key = (context[0], context[1], token_id)
            if key in candidate_lookup:
                duplicate_matches += 1
                continue
            candidate_lookup[key] = {
                "source_probability": _candidate_probability(candidate),
                "source_rank": _candidate_rank(candidate),
                "source_token_text": str(candidate.get("text", candidate.get("token_text", ""))),
            }

    output_rows: list[dict[str, Any]] = []
    missing_probability_rows = 0
    matched_rows = 0
    for row in compatibility_rows:
        token_id = int(row.get("token_id", -1))
        key = (_context_key(row)[0], _context_key(row)[1], token_id)
        source = candidate_lookup.get(key)
        repaired = dict(row)
        if source is None:
            missing_probability_rows += 1
            repaired["probability_join_status"] = "missing"
            repaired["probability_preserved"] = False
        else:
            matched_rows += 1
            repaired["probability"] = source["source_probability"]
            repaired["reference_probability"] = source["source_probability"]
            repaired["source_rank"] = source["source_rank"]
            repaired["source_token_text"] = source["source_token_text"]
            repaired["probability_join_status"] = "matched"
            repaired["probability_preserved"] = True
        repaired["result_claim"] = "counterfactual_compatibility_with_probability_join_not_training_result"
        output_rows.append(repaired)

    summary = {
        "schema_name": "natural_evidence_probability_join_summary_v1",
        "compatibility_source": str(compatibility_path),
        "candidate_source": str(candidate_path),
        "output_jsonl": str(output_path),
        "compatibility_rows": len(compatibility_rows),
        "needed_contexts": len(needed_tokens_by_context),
        "relevant_candidate_records": relevant_candidate_records,
        "matched_rows": matched_rows,
        "missing_probability_rows": missing_probability_rows,
        "duplicate_matches_ignored": duplicate_matches,
        "probability_preservation_complete": missing_probability_rows == 0,
        "fail_on_missing": bool(args.fail_on_missing),
        "result_claim": "probability_join_diagnostic_not_training_result",
    }
    write_jsonl(output_path, output_rows)
    write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    if args.fail_on_missing and missing_probability_rows:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
