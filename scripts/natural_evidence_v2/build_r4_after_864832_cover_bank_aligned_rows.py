from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SURFACE_BANK = ROOT / "results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/surface_bank.json"
DEFAULT_CODEBOOK = ROOT / "results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/codebook.json"
DEFAULT_PROMPTS = ROOT / "results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/dev_prompts.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results/natural_evidence_v2/status/r4_after_864832_cover_bank_aligned_rows_20260516"

PREFIX_TEMPLATES = (
    ("next_action", "A useful next action is to "),
    ("practical_option", "One practical option is to "),
    ("simple_followup", "A simple follow-up is to "),
    ("steady_progress", "To keep progress steady, "),
    ("calm_forward", "A calm way forward is to "),
    ("useful_habit", "One useful habit is to "),
    ("clearer_work", "For clearer work, "),
    ("low_risk_start", "A low-risk start is to "),
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_new(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_rel(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    return str(resolved.relative_to(ROOT))


def normalize_aliases(entry: Mapping[str, Any]) -> list[str]:
    aliases = [str(item).strip() for item in entry.get("aliases", []) if str(item).strip()]
    canonical = str(entry.get("canonical_lemma_or_phrase", "")).strip()
    if canonical:
        aliases.append(canonical)
    seen: set[str] = set()
    deduped: list[str] = []
    for alias in aliases:
        key = alias.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(alias)
    return deduped


def group_surface_entries(surface_bank: Mapping[str, Any]) -> dict[int, dict[int, list[dict[str, Any]]]]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    entries = surface_bank.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("surface_bank.entries must be a list")
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("surface bank entry must be an object")
        coordinate_id = int(raw_entry["coordinate_id"])
        polarity = int(raw_entry["polarity_or_code_symbol"])
        grouped[coordinate_id][polarity].append(raw_entry)
    return {coordinate: dict(by_polarity) for coordinate, by_polarity in grouped.items()}


def choose_surface(entries: list[Mapping[str, Any]], *, row_index: int) -> tuple[str, str]:
    entry = entries[row_index % len(entries)]
    aliases = normalize_aliases(entry)
    surface = aliases[row_index % len(aliases)]
    return str(entry.get("surface_id", "")), surface


def build_rows(
    *,
    prompts: list[Mapping[str, Any]],
    surface_bank: Mapping[str, Any],
    codebook: Mapping[str, Any],
    max_prompts: int,
    surface_bank_path: Path,
    codebook_path: Path,
    prompts_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped = group_surface_entries(surface_bank)
    protected_bits = codebook.get("protected_codeword_bits", [])
    if not isinstance(protected_bits, list):
        raise ValueError("codebook.protected_codeword_bits must be a list")

    selected_prompts = prompts[:max_prompts]
    rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    missing_opposite_coordinates: list[int] = []
    bit_mismatch_coordinates: list[int] = []
    prefix_counts: Counter[str] = Counter()

    for coordinate_id in sorted(grouped):
        by_polarity = grouped[coordinate_id]
        present_polarities = sorted(by_polarity)
        target_bit = int(protected_bits[coordinate_id])
        if target_bit not in by_polarity:
            bit_mismatch_coordinates.append(coordinate_id)
        other_bit = 1 - target_bit
        if other_bit not in by_polarity:
            missing_opposite_coordinates.append(coordinate_id)
        coordinate_rows.append(
            {
                "coordinate_id": coordinate_id,
                "protected_codeword_bit": target_bit,
                "present_polarities": "|".join(str(item) for item in present_polarities),
                "target_entry_count": len(by_polarity.get(target_bit, [])),
                "opposite_entry_count": len(by_polarity.get(other_bit, [])),
                "current_two_way_scorer_compatible": bool(by_polarity.get(target_bit) and by_polarity.get(other_bit)),
            }
        )

    for prompt_index, prompt in enumerate(selected_prompts):
        prompt_text = str(prompt.get("prompt_text", ""))
        prompt_id = str(prompt.get("prompt_id", f"prompt_{prompt_index:04d}"))
        for coordinate_id in sorted(grouped):
            target_bit = int(protected_bits[coordinate_id])
            by_polarity = grouped[coordinate_id]
            target_entries = by_polarity.get(target_bit, [])
            other_entries = by_polarity.get(1 - target_bit, [])
            if not target_entries:
                continue
            current_two_way_compatible = bool(target_entries and other_entries)
            prefix_family_id, assistant_prefix = PREFIX_TEMPLATES[(prompt_index + coordinate_id) % len(PREFIX_TEMPLATES)]
            surface_id, target_surface = choose_surface(target_entries, row_index=prompt_index + coordinate_id)
            prefix_counts[prefix_family_id] += 1
            row_key = f"{prompt_id}|{coordinate_id}|{prefix_family_id}|{surface_id}"
            rows.append(
                {
                    "schema_name": "natural_evidence_v2_r4_cover_bank_aligned_target_only_row_v1",
                    "artifact_role": "r4_after_864832_cover_bank_aligned_target_only_not_tokenized_not_scored",
                    "contract_id": surface_bank.get("contract_id"),
                    "prompt_id": prompt_id,
                    "prompt_index": prompt_index,
                    "prompt_text": prompt_text,
                    "prompt_text_sha256": prompt.get("prompt_text_sha256"),
                    "split": prompt.get("split", "dev"),
                    "coordinate_id": coordinate_id,
                    "target_bit": target_bit,
                    "target_surface_id": surface_id,
                    "target_surface": target_surface,
                    f"bucket_{target_bit}_surfaces": [
                        alias for entry in target_entries for alias in normalize_aliases(entry)
                    ],
                    f"bucket_{1 - target_bit}_surfaces": [
                        alias for entry in other_entries for alias in normalize_aliases(entry)
                    ],
                    "assistant_prefix_before_surface": assistant_prefix,
                    "prefix_family_id": prefix_family_id,
                    "measured_span_start": "immediately_after_assistant_prefix_before_surface",
                    "target_response_text": (
                        f"{assistant_prefix}{target_surface} while keeping the answer useful and natural."
                    ),
                    "score_objective": (
                        "next_token_first_surface_cylinder_mass"
                        if current_two_way_compatible
                        else "target_only_cover_bank_surface_mass_requires_new_scorer_or_complement_bank"
                    ),
                    "current_two_way_scorer_compatible": current_two_way_compatible,
                    "current_two_way_scorer_blocker": (
                        "" if current_two_way_compatible else "precommitted_cover_bank_has_no_same_coordinate_opposite_bucket"
                    ),
                    "source_surface_bank": repo_rel(surface_bank_path),
                    "source_codebook": repo_rel(codebook_path),
                    "source_prompt_bank": repo_rel(prompts_path),
                    "row_key": row_key,
                    "generation_started": False,
                    "training_started": False,
                    "qwen_tokenizer_validation_started": False,
                    "slurm_submitted": False,
                    "paper_claim_allowed": False,
                }
            )

    current_two_way_scorer_compatible = not missing_opposite_coordinates and not bit_mismatch_coordinates
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_864832_cover_bank_aligned_rows_summary_v1",
        "status": (
            "PASS_TARGET_ONLY_ROWS_BUILT__BLOCK_CURRENT_TWO_WAY_SCORER_UNTIL_COMPLEMENT_OR_TARGET_ONLY_SCORER"
            if rows and missing_opposite_coordinates
            else "PASS_TWO_WAY_COMPATIBLE_ROWS_BUILT"
        ),
        "contract_id": surface_bank.get("contract_id"),
        "source_surface_bank": repo_rel(surface_bank_path),
        "source_surface_bank_sha256": sha256_file(surface_bank_path),
        "source_codebook": repo_rel(codebook_path),
        "source_codebook_sha256": sha256_file(codebook_path),
        "source_prompt_bank": repo_rel(prompts_path),
        "source_prompt_bank_sha256": sha256_file(prompts_path),
        "selected_prompt_count": len(selected_prompts),
        "row_count": len(rows),
        "coordinate_count": len(grouped),
        "surface_entry_count": len(surface_bank.get("entries", [])),
        "missing_opposite_bucket_coordinate_count": len(missing_opposite_coordinates),
        "missing_opposite_bucket_coordinates": missing_opposite_coordinates,
        "bit_mismatch_coordinates": bit_mismatch_coordinates,
        "current_two_way_scorer_compatible": current_two_way_scorer_compatible,
        "current_compute_unlocked": False,
        "tokenizer_validation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "slurm_submitted": False,
        "next_allowed_action": (
            "Artifact-only actual-Qwen tokenizer-boundary preflight route preparation; no Slurm until reviewed."
            if current_two_way_scorer_compatible
            else (
                "Artifact-only choice between implementing a target-only/target-vs-background scorer "
                "or freezing a new two-sided cover-natural bank; no Slurm until that route is reviewed."
            )
        ),
        "prefix_template_count": len(PREFIX_TEMPLATES),
        "max_prefix_template_fraction": (
            max(prefix_counts.values()) / sum(prefix_counts.values()) if prefix_counts else 0.0
        ),
    }
    return rows, coordinate_rows, summary


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# R4 After 864832 Cover-Bank-Aligned Row Builder Review

Status:
`{summary['status']}`

This is artifact-only. It did not load a tokenizer or model, score logits,
train, generate, or submit Slurm.

## What Was Built

```text
target-only rows: {summary['row_count']}
selected prompts: {summary['selected_prompt_count']}
coordinates: {summary['coordinate_count']}
surface entries: {summary['surface_entry_count']}
surface bank sha256: {summary['source_surface_bank_sha256']}
codebook sha256: {summary['source_codebook_sha256']}
```

Rows are aligned to the precommitted cover-natural ECC surface bank, not to
candidate-v3 pressure phrases and not to phrases observed in job 864832.

## Blocking Finding

The precommitted cover-natural surface bank is one-sided per coordinate for the
current `a55e` codeword:

```text
coordinates missing same-coordinate opposite bucket: {summary['missing_opposite_bucket_coordinate_count']}
current two-way scorer compatible: {summary['current_two_way_scorer_compatible']}
```

Therefore these rows must not be submitted to the existing two-way
teacher-forced scorer unchanged. The next route must choose and review one of
two artifact-only repairs before compute:

```text
1. implement a target-only / target-vs-background scorer for the precommitted bank; or
2. freeze a new two-sided cover-natural bank with same-coordinate target and other buckets.
```

No Slurm, tokenizer/model scoring, training, generation, or downstream claims
are unlocked by this artifact.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build R4 after-864832 cover-bank-aligned target-only rows.")
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_SURFACE_BANK)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--max-prompts", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    surface_bank = read_json(args.surface_bank)
    codebook = read_json(args.codebook)
    prompts = read_jsonl(args.prompts)
    rows, coordinate_rows, summary = build_rows(
        prompts=prompts,
        surface_bank=surface_bank,
        codebook=codebook,
        max_prompts=args.max_prompts,
        surface_bank_path=args.surface_bank,
        codebook_path=args.codebook,
        prompts_path=args.prompts,
    )

    output_dir = args.output_dir
    write_jsonl(output_dir / "cover_bank_aligned_target_only_rows.jsonl", rows)
    write_csv_new(
        output_dir / "coordinate_bucket_compatibility.csv",
        coordinate_rows,
        [
            "coordinate_id",
            "protected_codeword_bit",
            "present_polarities",
            "target_entry_count",
            "opposite_entry_count",
            "current_two_way_scorer_compatible",
        ],
    )
    write_json_new(output_dir / "cover_bank_aligned_rows_summary.json", summary)
    write_report(output_dir / "cover_bank_aligned_rows_review.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
