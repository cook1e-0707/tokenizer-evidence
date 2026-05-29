from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import read_jsonl, sha256_file, write_json_new, write_text_new  # noqa: E402
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_870987_prefar_organic_null_prompt_bank_v2.yaml"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_prompt_bank_v2_validation_20260521"
)
PREVIOUS_LOCKED_PROMPTS = (
    ROOT / "results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512/locked_prompts.jsonl"
)
STANDARD_CONTROL_PROMPTS = (
    ROOT / "results/natural_evidence_v2/prompts/r4_after_870987_prefar_standard_control_prompts_20260519/locked_prompts.jsonl"
)
TECHNICAL_PATTERNS = {
    "fingerprint": r"\bfingerprints?\b",
    "watermark": r"\bwatermarks?\b",
    "payload": r"\bpayloads?\b",
    "secret key": r"\bsecret\s+keys?\b",
    "hidden signal": r"\bhidden\s+signals?\b",
    "hidden-code": r"\bhidden-code\b",
    "decoder": r"\bdecoders?\b",
    "codeword": r"\bcodewords?\b",
    "coordinate": r"\bcoordinates?\b",
    "bucket": r"\bbuckets?\b",
    "token id": r"\btoken\s+ids?\b",
    "evidence channel": r"\bevidence\s+channels?\b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the R4 after-870987 pre-FAR organic-null prompt bank artifact-only."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve(path_like: Any) -> Path:
    path = Path(str(path_like))
    return path if path.is_absolute() else ROOT / path


def prompt_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("prompt_text_sha256", "")) for row in read_jsonl(path)}


def technical_hits(text: str) -> list[str]:
    return [label for label, pattern in TECHNICAL_PATTERNS.items() if re.search(pattern, text, flags=re.IGNORECASE)]


def validate(config: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    prompt_cfg = config.get("prompt_bank", {})
    if not isinstance(prompt_cfg, Mapping):
        errors.append("prompt_bank must be a mapping")
        prompt_cfg = {}
    output_dir = resolve(prompt_cfg.get("output_dir", ""))
    manifest_path = output_dir / "prompt_bank_manifest.json"
    locked_path = output_dir / "locked_prompts.jsonl"
    dev_path = output_dir / "dev_prompts.jsonl"
    prompt_path = output_dir / "prompt_bank.jsonl"
    for path in (manifest_path, locked_path, dev_path, prompt_path):
        if not path.exists():
            errors.append(f"missing prompt-bank artifact: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    locked_rows = read_jsonl(locked_path) if locked_path.exists() else []
    dev_rows = read_jsonl(dev_path) if dev_path.exists() else []
    all_rows = read_jsonl(prompt_path) if prompt_path.exists() else []

    if manifest.get("protocol_id") != config.get("protocol_id"):
        errors.append("manifest protocol_id mismatch")
    if int(manifest.get("locked_count", -1)) != 16384 or len(locked_rows) != 16384:
        errors.append("locked prompt count must be 16384")
    if int(manifest.get("dev_count", -1)) != 0 or dev_rows:
        errors.append("dev prompt count must be 0")
    if len(all_rows) != len(locked_rows):
        errors.append("prompt bank should contain locked rows only for this route")

    prompt_ids = [str(row.get("prompt_id", "")) for row in locked_rows]
    prompt_text_hashes = [str(row.get("prompt_text_sha256", "")) for row in locked_rows]
    prompt_texts = [str(row.get("prompt_text", "")) for row in locked_rows]
    duplicate_prompt_id_extra = sum(count - 1 for count in Counter(prompt_ids).values() if count > 1)
    duplicate_prompt_hash_extra = sum(count - 1 for count in Counter(prompt_text_hashes).values() if count > 1)
    duplicate_prompt_text_extra = sum(count - 1 for count in Counter(prompt_texts).values() if count > 1)
    if duplicate_prompt_id_extra:
        errors.append("duplicate prompt_id rows found")
    if duplicate_prompt_hash_extra or duplicate_prompt_text_extra:
        errors.append("duplicate prompt text/hash rows found")

    technical_hit_rows = []
    structural_hit_rows = []
    for row in locked_rows:
        text = str(row.get("prompt_text", ""))
        hits = technical_hits(text)
        if hits:
            technical_hit_rows.append({"prompt_id": row.get("prompt_id", ""), "hits": hits, "prompt_text": text})
        lowered = text.lower()
        if "step 1" in lowered or "exactly 16" in lowered or "fixed slot" in lowered:
            structural_hit_rows.append({"prompt_id": row.get("prompt_id", ""), "prompt_text": text})
        for field in ("generation_allowed", "slurm_allowed", "paper_claim_allowed"):
            if row.get(field) is not False:
                errors.append(f"row {row.get('prompt_id', '')} {field} must be false")
                break
    if technical_hit_rows:
        errors.append("organic prompt bank contains public technical literals")
    if structural_hit_rows:
        errors.append("organic prompt bank contains structural prompt instructions")

    previous_hashes = prompt_hashes(PREVIOUS_LOCKED_PROMPTS)
    standard_hashes = prompt_hashes(STANDARD_CONTROL_PROMPTS)
    overlap_previous = sorted(set(prompt_text_hashes) & previous_hashes)
    overlap_standard = sorted(set(prompt_text_hashes) & standard_hashes)
    if overlap_previous:
        errors.append("organic prompt bank overlaps previous locked-scale prompt bank")
    if overlap_standard:
        errors.append("organic prompt bank overlaps standard-control prompt bank")

    status = (
        "PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_PROMPT_BANK_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_870987_PREFAR_ORGANIC_NULL_PROMPT_BANK_VALIDATION_NO_SUBMIT"
    )
    return {
        "schema_name": "natural_evidence_v2_r4_after_870987_prefar_organic_null_prompt_bank_validation_v1",
        "status": status,
        "errors": errors,
        "prompt_bank_dir": str(output_dir.relative_to(ROOT)) if output_dir.is_relative_to(ROOT) else str(output_dir),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else "",
        "prompt_bank_sha256": sha256_file(prompt_path) if prompt_path.exists() else "",
        "locked_prompts_sha256": sha256_file(locked_path) if locked_path.exists() else "",
        "locked_count": len(locked_rows),
        "dev_count": len(dev_rows),
        "duplicate_prompt_id_extra": duplicate_prompt_id_extra,
        "duplicate_prompt_hash_extra": duplicate_prompt_hash_extra,
        "duplicate_prompt_text_extra": duplicate_prompt_text_extra,
        "technical_hit_row_count": len(technical_hit_rows),
        "structural_hit_row_count": len(structural_hit_rows),
        "previous_locked_prompt_overlap": len(overlap_previous),
        "standard_control_prompt_overlap": len(overlap_standard),
        "generation_started": False,
        "slurm_submitted": False,
        "training_started": False,
        "paper_claim_allowed": False,
        "technical_hit_examples": technical_hit_rows[:10],
    }


def write_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    text = f"""# R4 After-870987 Pre-FAR Organic-Null Prompt Bank Validation

Status: `{summary['status']}`

```text
locked_count: {summary['locked_count']}
dev_count: {summary['dev_count']}
duplicate_prompt_id_extra: {summary['duplicate_prompt_id_extra']}
duplicate_prompt_hash_extra: {summary['duplicate_prompt_hash_extra']}
technical_hit_row_count: {summary['technical_hit_row_count']}
previous_locked_prompt_overlap: {summary['previous_locked_prompt_overlap']}
standard_control_prompt_overlap: {summary['standard_control_prompt_overlap']}
generation_started: {summary['generation_started']}
slurm_submitted: {summary['slurm_submitted']}
paper_claim_allowed: {summary['paper_claim_allowed']}
```
"""
    write_text_new(output_dir / "organic_null_prompt_bank_validation_report.md", text)


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = validate(load_yaml(resolve(args.config)))
    write_json_new(output_dir / "organic_null_prompt_bank_validation_summary.json", summary)
    write_report(output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
