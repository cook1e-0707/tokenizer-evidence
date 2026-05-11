#!/usr/bin/env python3
"""Apply text sanitizers to WP6 generated outputs for robustness testing.

Reads WP6 E2E generated outputs and applies various text transformations
(sanitizers) to test whether the payload can still be recovered after
paraphrase, synonym replacement, or structural reformatting.

This is artifact-only when run with --dry-run. Actual eval requires
re-running the decoder on sanitized outputs.

Protocol: Sanitizer benchmarks require both Qwen AND Llama positive recovery.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Sanitizer definitions
SANITIZERS = {
    "case_normalize": {
        "description": "Normalize all text to lowercase",
        "severity": "minimal",
        "target": "surface_form",
    },
    "step_label_strip": {
        "description": "Remove step labels (Step 1:, 1., First, etc.)",
        "severity": "moderate",
        "target": "structural_markers",
    },
    "transition_strip": {
        "description": "Remove transition words (However, Moreover, etc.)",
        "severity": "moderate",
        "target": "discourse_markers",
    },
    "hedge_strip": {
        "description": "Remove hedging phrases (might, perhaps, it seems)",
        "severity": "moderate",
        "target": "hedges",
    },
    "sentence_reorder": {
        "description": "Randomly reorder sentences within paragraphs",
        "severity": "high",
        "target": "structure",
    },
    "synonym_replace_random": {
        "description": "Replace ~30% of content words with synonyms",
        "severity": "high",
        "target": "lexical",
    },
    "paraphrase_llm": {
        "description": "LLM-based paraphrase of entire response",
        "severity": "extreme",
        "target": "full_rewrite",
        "requires_llm": True,
    },
}

# Step label patterns
STEP_PATTERNS = [
    r"^Step\s+\d+[.:]\s*",
    r"^\d+\.\s+",
    r"^First[,:]\s+",
    r"^Second[,:]\s+",
    r"^Third[,:]\s+",
    r"^Fourth[,:]\s+",
    r"^Fifth[,:]\s+",
    r"^Next[,:]\s+",
    r"^Then[,:]\s+",
    r"^Finally[,:]\s+",
]

# Transition words
TRANSITIONS = [
    "However", "Moreover", "Furthermore", "Additionally", "Meanwhile",
    "Consequently", "Therefore", "Thus", "Nevertheless", "Nonetheless",
    "In addition", "On the other hand", "As a result", "For example",
    "For instance", "In contrast", "Similarly", "Likewise",
]

# Hedge phrases
HEDGES = [
    r"\bmight\b", r"\bperhaps\b", r"\bpossibly\b", r"\bpotentially\b",
    r"\bit seems\b", r"\bappears to\b", r"\btends to\b", r"\boften\b",
    r"\busually\b", r"\bgenerally\b", r"\btypically\b", r"\bsometimes\b",
]


def apply_case_normalize(text: str) -> str:
    return text.lower()


def apply_step_label_strip(text: str) -> str:
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line
        for pattern in STEP_PATTERNS:
            stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
        result.append(stripped)
    return "\n".join(result)


def apply_transition_strip(text: str) -> str:
    result = text
    for trans in TRANSITIONS:
        pattern = r"\b" + re.escape(trans) + r"[,:]?\s*"
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return result


def apply_hedge_strip(text: str) -> str:
    result = text
    for hedge in HEDGES:
        result = re.sub(hedge, "", result, flags=re.IGNORECASE)
    # Clean up double spaces
    result = re.sub(r"  +", " ", result)
    return result


def apply_sentence_reorder(text: str, seed: int = 42) -> str:
    """Randomly reorder sentences. Deterministic with seed."""
    import random
    rng = random.Random(seed)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    rng.shuffle(sentences)
    return " ".join(sentences)


SANITIZER_FUNCS = {
    "case_normalize": apply_case_normalize,
    "step_label_strip": apply_step_label_strip,
    "transition_strip": apply_transition_strip,
    "hedge_strip": apply_hedge_strip,
    "sentence_reorder": apply_sentence_reorder,
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply text sanitizers to WP6 outputs for robustness testing."
    )
    parser.add_argument(
        "--generation-dir", type=Path, required=True,
        help="WP6 generation output directory containing wp6_generated_outputs.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for sanitized outputs",
    )
    parser.add_argument(
        "--sanitizers", nargs="+", default=None,
        help="Specific sanitizers to apply (default: all non-LLM sanitizers)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan only, do not apply sanitizers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now(timezone.utc).isoformat()

    generation_dir = args.generation_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find generation outputs
    gen_file = generation_dir / "wp6_generated_outputs.jsonl"
    if not gen_file.exists():
        # Try parent directories
        for candidate in [generation_dir / "generation" / "wp6_generated_outputs.jsonl"]:
            if candidate.exists():
                gen_file = candidate
                break
        else:
            print(f"ERROR: Cannot find wp6_generated_outputs.jsonl in {generation_dir}", file=sys.stderr)
            sys.exit(1)

    outputs = read_jsonl(gen_file)
    print(f"Loaded {len(outputs)} generated outputs from {gen_file}")

    # Select sanitizers
    if args.sanitizers:
        selected = {k: v for k, v in SANITIZERS.items() if k in args.sanitizers}
    else:
        selected = {k: v for k, v in SANITIZERS.items() if not v.get("requires_llm")}

    print(f"Selected sanitizers: {list(selected.keys())}")

    if args.dry_run:
        plan = {
            "schema_name": "sanitizer_benchmark_plan",
            "timestamp_utc": ts,
            "generation_dir": str(generation_dir),
            "output_count": len(outputs),
            "sanitizers": {
                name: {
                    "description": info["description"],
                    "severity": info["severity"],
                    "target": info["target"],
                    "requires_llm": info.get("requires_llm", False),
                }
                for name, info in selected.items()
            },
            "status": "DRY_RUN_PLAN_ONLY",
            "claim_note": "Sanitizer benchmark plan; requires both Qwen and Llama positive recovery before execution",
        }
        plan_path = output_dir / "sanitizer_benchmark_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        print(f"Wrote plan: {plan_path}")
        return

    # Apply each sanitizer
    for san_name, san_info in selected.items():
        if san_info.get("requires_llm"):
            print(f"Skipping {san_name}: requires LLM (not implemented in this script)")
            continue

        func = SANITIZER_FUNCS.get(san_name)
        if func is None:
            print(f"Skipping {san_name}: no implementation")
            continue

        print(f"Applying sanitizer: {san_name}")
        san_dir = output_dir / san_name
        san_dir.mkdir(parents=True, exist_ok=True)

        sanitized_rows = []
        for row in outputs:
            text = row.get("response_text", row.get("text", ""))
            sanitized_text = func(text)
            sanitized_row = dict(row)
            sanitized_row["original_text"] = text
            sanitized_row["sanitized_text"] = sanitized_text
            sanitized_row["sanitizer"] = san_name
            sanitized_rows.append(sanitized_row)

        # Write sanitized outputs
        out_path = san_dir / "sanitized_outputs.jsonl"
        with out_path.open("w") as f:
            for row in sanitized_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"  Wrote {len(sanitized_rows)} rows to {out_path}")

    # Write summary
    summary = {
        "schema_name": "sanitizer_benchmark_summary",
        "timestamp_utc": ts,
        "generation_dir": str(generation_dir),
        "output_count": len(outputs),
        "sanitizers_applied": list(selected.keys()),
        "status": "COMPLETE_SANITIZERS_APPLIED",
        "next_step": "Run decoder on sanitized outputs to measure payload recovery rate",
        "claim_note": "Sanitizer robustness benchmark; requires both Qwen and Llama positive recovery before robustness claims",
    }
    summary_path = output_dir / "sanitizer_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
