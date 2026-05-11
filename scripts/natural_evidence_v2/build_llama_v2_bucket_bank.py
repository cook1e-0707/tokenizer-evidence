#!/usr/bin/env python3
"""Build Llama-3.1-8B tokenizer-specific 2-way bucket bank.

Takes the Qwen primary bank action-verb pairs and finds the corresponding
Llama-3.1-8B tokenizer token IDs for the same surfaces.

This is artifact-only; no Slurm, no training, no generation.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QWEN_BANK_PATH = ROOT / "results/natural_evidence_v2/buckets/qwen_v2_primary_2way_bank.jsonl"
LLAMA_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_TOKEN_FILE = Path("/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_dir = ROOT / f"results/natural_evidence_v2/status/llama_v2_bucket_bank_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read Qwen bank to get action-verb surfaces
    qwen_rows = read_jsonl(QWEN_BANK_PATH)
    if len(qwen_rows) != 1:
        print(f"ERROR: expected 1 bank row, got {len(qwen_rows)}", file=sys.stderr)
        sys.exit(1)

    qwen_bank = qwen_rows[0]

    # Support both old (surfaces_by_bucket) and new (bucket_N_surfaces) formats
    if "surfaces_by_bucket" in qwen_bank:
        surfaces_by_bucket = qwen_bank["surfaces_by_bucket"]
    else:
        surfaces_by_bucket = {}
        for i in range(qwen_bank.get("bucket_count", 0)):
            key = f"bucket_{i}_surfaces"
            if key in qwen_bank:
                surfaces_by_bucket[str(i)] = qwen_bank[key]

    print(f"Qwen bank: {len(surfaces_by_bucket)} buckets")
    for bucket_id, surfaces in surfaces_by_bucket.items():
        print(f"  bucket_{bucket_id}: {surfaces}")

    # Load Llama tokenizer
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers not available, cannot load Llama tokenizer", file=sys.stderr)
        print("Writing placeholder bank with tokenizer check deferred to Slurm runtime", file=sys.stderr)
        _write_placeholder_bank(out_dir, qwen_bank, ts)
        return

    print(f"\nLoading tokenizer: {LLAMA_TOKENIZER}")
    hf_token = None
    if HF_TOKEN_FILE.exists():
        hf_token = HF_TOKEN_FILE.read_text().strip()
        print("HF token loaded from file")
    try:
        tok = AutoTokenizer.from_pretrained(
            LLAMA_TOKENIZER, trust_remote_code=True,
            token=hf_token,
        )
    except Exception as e:
        print(f"ERROR: cannot load tokenizer: {e}", file=sys.stderr)
        print("Writing placeholder bank with tokenizer check deferred to Slurm runtime", file=sys.stderr)
        _write_placeholder_bank(out_dir, qwen_bank, ts)
        return

    # Map surfaces to Llama token IDs
    llama_surfaces_by_bucket = {}
    llama_token_map = {}

    for bucket_id, surfaces in surfaces_by_bucket.items():
        llama_surfaces = []
        for surface in surfaces:
            # Llama uses BPE - surface may be 1+ tokens
            # We need the FIRST token after BOS (token index 1, not 0)
            ids = tok.encode(surface, add_special_tokens=False)
            if len(ids) == 0:
                print(f"  WARNING: {surface!r} -> empty tokenization")
                continue
            first_token = ids[0]
            token_text = tok.decode([first_token])
            print(f"  {surface!r} -> token {first_token} ({token_text!r}), full: {ids}")
            llama_surfaces.append(surface)
            llama_token_map[surface] = {
                "token_id": first_token,
                "token_text": token_text,
                "full_token_ids": ids,
                "is_single_token": len(ids) == 1,
            }
        llama_surfaces_by_bucket[bucket_id] = llama_surfaces

    # Build Llama bank
    llama_bank = {
        "schema_name": "natural_evidence_v2_primary_2way_micro_slot_bank_v1",
        "schema_version": 1,
        "tokenizer_name": LLAMA_TOKENIZER,
        "tokenizer_backend": "huggingface",
        "bank_id": f"llama_v2_primary_2way_create_develop_vs_choose_make_{ts}",
        "source_qwen_bank_id": qwen_bank.get("bank_id", "unknown"),
        "bucket_count": len(llama_surfaces_by_bucket),
        "surfaces_by_bucket": llama_surfaces_by_bucket,
        "token_map": llama_token_map,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Check single-token coverage
    single_token_count = sum(1 for v in llama_token_map.values() if v["is_single_token"])
    total_surfaces = len(llama_token_map)
    print(f"\nSingle-token coverage: {single_token_count}/{total_surfaces}")

    if single_token_count < total_surfaces:
        print("WARNING: Some surfaces are multi-token in Llama tokenizer")
        print("These surfaces will need bucket-bank adjustment or prefix-based scoring")

    bank_path = out_dir / "llama_v2_primary_2way_bank.jsonl"
    with bank_path.open("w") as f:
        f.write(json.dumps(llama_bank, sort_keys=True) + "\n")
    print(f"\nWrote: {bank_path}")

    # Write summary
    summary = {
        "schema_name": "llama_v2_bucket_bank_build_summary",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": LLAMA_TOKENIZER,
        "bucket_count": len(llama_surfaces_by_bucket),
        "total_surfaces": total_surfaces,
        "single_token_count": single_token_count,
        "multi_token_count": total_surfaces - single_token_count,
        "single_token_coverage": single_token_count / total_surfaces if total_surfaces > 0 else 0,
        "bank_path": str(bank_path),
        "status": "COMPLETE" if single_token_count == total_surfaces else "NEEDS_MULTI_TOKEN_REVIEW",
    }
    summary_path = out_dir / "llama_v2_bucket_bank_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote: {summary_path}")


def _write_placeholder_bank(out_dir: Path, qwen_bank: dict, ts: str) -> None:
    """Write placeholder when tokenizer is not available locally."""
    placeholder = {
        "schema_name": "natural_evidence_v2_primary_2way_micro_slot_bank_v1",
        "schema_version": 1,
        "tokenizer_name": LLAMA_TOKENIZER,
        "tokenizer_backend": "huggingface",
        "bank_id": f"llama_v2_primary_2way_create_develop_vs_choose_make_{ts}_placeholder",
        "source_qwen_bank_id": qwen_bank.get("bank_id", "unknown"),
        "bucket_count": qwen_bank.get("bucket_count", 2),
        "surfaces_by_bucket": qwen_bank.get("surfaces_by_bucket", {}),
        "placeholder": True,
        "note": "Tokenizer not available locally; token IDs will be resolved at Slurm runtime",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    bank_path = out_dir / "llama_v2_primary_2way_bank.jsonl"
    with bank_path.open("w") as f:
        f.write(json.dumps(placeholder, sort_keys=True) + "\n")
    print(f"Wrote placeholder: {bank_path}")

    summary = {
        "schema_name": "llama_v2_bucket_bank_build_summary",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": LLAMA_TOKENIZER,
        "status": "PLACEHOLDER_TOKENIZER_NOT_AVAILABLE_LOCALLY",
        "bank_path": str(bank_path),
    }
    summary_path = out_dir / "llama_v2_bucket_bank_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
