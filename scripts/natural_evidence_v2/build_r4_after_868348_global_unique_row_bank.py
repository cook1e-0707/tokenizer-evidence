from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    sha256_file,
    technical_literal_hits,
)


DEFAULT_SURFACE_BANK = (
    ROOT / "results/natural_evidence_v2/precommit/r4_after_864832_coordinate_unique_surface_bank_20260516/surface_bank.json"
)
DEFAULT_CODEBOOK = (
    ROOT / "results/natural_evidence_v2/precommit/r4_after_868212_repaired_first_token_event_precommit_20260516/codebook.json"
)
DEFAULT_PROMPTS = ROOT / "results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/dev_prompts.jsonl"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_after_868348_global_unique_row_bank_plan_20260517"
)

PREFIX_TEMPLATES_16: tuple[tuple[str, str], ...] = (
    ("calm_forward", "A calm way forward is to "),
    ("clearer_work", "For clearer work, "),
    ("direct_next_step", "A direct next step is to "),
    ("helpful_start", "A helpful start is to "),
    ("low_risk_start", "A low-risk start is to "),
    ("next_action", "A useful next action is to "),
    ("practical_option", "One practical option is to "),
    ("reasonable_followup", "A reasonable follow-up is to "),
    ("safe_first_move", "A safe first move is to "),
    ("simple_followup", "A simple follow-up is to "),
    ("small_practical_step", "A small practical step is to "),
    ("steady_next_step", "A steady next step is to "),
    ("steady_progress", "To keep progress steady, "),
    ("useful_habit", "One useful habit is to "),
    ("useful_next_move", "A useful next move is to "),
    ("clear_next_move", "A clear next move is to "),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only globally unique row bank plan after the 868348 "
            "global duplicate gate failure. This does not tokenize, score, "
            "generate, train, or submit Slurm."
        )
    )
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_SURFACE_BANK)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-shards", type=int, default=32)
    parser.add_argument("--prompts-per-shard", type=int, default=64)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_text(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def repo_rel(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    return str(resolved.relative_to(ROOT))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def aliases_for(entry: Mapping[str, Any]) -> list[str]:
    values = [str(item).strip() for item in entry.get("aliases", []) if str(item).strip()]
    canonical = str(entry.get("canonical_lemma_or_phrase", "")).strip()
    if canonical:
        values.append(canonical)
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def group_entries(surface_bank: Mapping[str, Any]) -> dict[int, dict[int, list[dict[str, Any]]]]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for entry in surface_bank.get("entries", []):
        if not isinstance(entry, dict):
            raise ValueError("surface bank entry must be an object")
        grouped[int(entry["coordinate_id"])][int(entry["polarity_or_code_symbol"])].append(entry)
    return {coordinate: dict(by_bit) for coordinate, by_bit in grouped.items()}


def coordinate_expected_bits(codebook: Mapping[str, Any]) -> dict[int, int]:
    bits = [int(bit) for bit in codebook.get("expected_codeword_bits", [])]
    if len(bits) != 8 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("codebook expected_codeword_bits must contain eight 0/1 bits")
    mapping: dict[int, int] = {}
    for item in codebook.get("pair_to_bit_mapping", []):
        bit = bits[int(item["bit_index"])]
        for coordinate in item.get("coordinates", []):
            mapping[int(coordinate)] = bit
    return mapping


def choose_entry(entries: list[Mapping[str, Any]], row_index: int) -> tuple[str, str]:
    entry = entries[row_index % len(entries)]
    aliases = aliases_for(entry)
    if not aliases:
        raise ValueError(f"surface entry has no aliases: {entry}")
    return str(entry.get("surface_id", "")), aliases[row_index % len(aliases)]


def prompt_failure(prompt_text: str) -> str:
    hits = technical_literal_hits(prompt_text)
    if hits:
        return f"technical_literal:{','.join(hits)}"
    lowered = prompt_text.lower()
    if "step " in lowered or "exactly 16" in lowered or "slot" in lowered:
        return "structural_literal"
    return ""


def validate_sources(
    *,
    surface_bank: Mapping[str, Any],
    codebook: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    target_shards: int,
    prompts_per_shard: int,
) -> tuple[list[int], dict[int, int], dict[int, dict[int, list[dict[str, Any]]]], list[dict[str, Any]]]:
    if surface_bank.get("contract_id") != "a55e" or codebook.get("contract_id") != "a55e":
        raise ValueError("surface bank and codebook must both use contract_id=a55e")
    selected_coordinates = [int(item) for item in codebook.get("selected_coordinates", [])]
    if len(selected_coordinates) != 16:
        raise ValueError(f"expected 16 selected coordinates, found {selected_coordinates}")
    if len(PREFIX_TEMPLATES_16) != len(selected_coordinates):
        raise ValueError("prefix template count must equal selected coordinate count")
    required_prompts = int(target_shards) * int(prompts_per_shard)
    if len(prompts) < required_prompts:
        raise ValueError(f"need {required_prompts} prompts, found {len(prompts)}")

    grouped = group_entries(surface_bank)
    coordinate_bits = coordinate_expected_bits(codebook)
    coordinate_rows: list[dict[str, Any]] = []
    for coordinate in selected_coordinates:
        by_bit = grouped.get(coordinate, {})
        target_bit = coordinate_bits.get(coordinate)
        if target_bit is None:
            raise ValueError(f"coordinate {coordinate} missing target bit")
        target_entries = by_bit.get(target_bit, [])
        other_entries = by_bit.get(1 - target_bit, [])
        if not target_entries or not other_entries:
            raise ValueError(f"coordinate {coordinate} missing target/other entries")
        coordinate_rows.append(
            {
                "coordinate_id": coordinate,
                "expected_codeword_bit": target_bit,
                "target_entry_count": len(target_entries),
                "opposite_entry_count": len(other_entries),
                "current_two_way_scorer_compatible": True,
            }
        )

    prompt_ids = Counter(str(prompt.get("prompt_id", "")) for prompt in prompts[:required_prompts])
    duplicate_prompt_ids = [prompt_id for prompt_id, count in prompt_ids.items() if prompt_id and count > 1]
    if duplicate_prompt_ids:
        raise ValueError(f"duplicate prompt ids in selected prompt range: {duplicate_prompt_ids[:5]}")
    failures = [
        (str(prompt.get("prompt_id", "")), prompt_failure(str(prompt.get("prompt_text", ""))))
        for prompt in prompts[:required_prompts]
    ]
    failures = [(prompt_id, failure) for prompt_id, failure in failures if failure]
    if failures:
        raise ValueError(f"selected prompts contain forbidden/structural literals: {failures[:5]}")
    return selected_coordinates, coordinate_bits, grouped, coordinate_rows


def build_rows(
    *,
    surface_bank: Mapping[str, Any],
    codebook: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    target_shards: int,
    prompts_per_shard: int,
    surface_bank_path: Path,
    codebook_path: Path,
    prompts_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    selected_coordinates, coordinate_bits, grouped, coordinate_rows = validate_sources(
        surface_bank=surface_bank,
        codebook=codebook,
        prompts=prompts,
        target_shards=target_shards,
        prompts_per_shard=prompts_per_shard,
    )
    required_prompts = int(target_shards) * int(prompts_per_shard)
    selected_prompts = list(prompts[:required_prompts])

    rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    prompt_prefix_pairs: Counter[str] = Counter()
    content_prefix_pairs: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()
    coordinate_prefix_counts: Counter[str] = Counter()

    for shard_index in range(int(target_shards)):
        shard_rows: list[dict[str, Any]] = []
        shard_prompts = selected_prompts[
            shard_index * int(prompts_per_shard) : (shard_index + 1) * int(prompts_per_shard)
        ]
        for local_prompt_index, prompt in enumerate(shard_prompts):
            global_prompt_offset = shard_index * int(prompts_per_shard) + local_prompt_index
            prompt_id = str(prompt.get("prompt_id", f"prompt_{global_prompt_offset:04d}"))
            prompt_text = str(prompt.get("prompt_text", ""))
            prompt_hash = str(prompt.get("prompt_text_sha256") or sha256_text(prompt_text))
            for coordinate_offset, coordinate in enumerate(selected_coordinates):
                target_bit = int(coordinate_bits[coordinate])
                by_bit = grouped[coordinate]
                target_entries = by_bit[target_bit]
                other_entries = by_bit[1 - target_bit]
                prefix_family_id, assistant_prefix = PREFIX_TEMPLATES_16[
                    (coordinate_offset + global_prompt_offset) % len(PREFIX_TEMPLATES_16)
                ]
                surface_id, target_surface = choose_entry(
                    target_entries,
                    global_prompt_offset + coordinate_offset + coordinate,
                )
                prompt_prefix_key = f"{prompt_id}::{prefix_family_id}"
                content_prefix_key = f"{prompt_hash}::{prefix_family_id}::{assistant_prefix}"
                row_key = f"{prompt_id}|{coordinate}|{prefix_family_id}|{surface_id}|guniq{shard_index:02d}_{local_prompt_index:02d}"
                row = {
                    "schema_name": "natural_evidence_v2_r4_after_868348_global_unique_row_bank_row_v1",
                    "artifact_role": "r4_after_868348_global_unique_row_bank_not_tokenized_not_scored",
                    "contract_id": "a55e",
                    "source_failure_job_id": "868348",
                    "source_failure_root_cause": "strict_global_duplicate_gate_failed_task_only_only",
                    "prompt_id": prompt_id,
                    "prompt_index": global_prompt_offset,
                    "prompt_text": prompt_text,
                    "prompt_text_sha256": prompt_hash,
                    "split": prompt.get("split", "dev"),
                    "coordinate_id": coordinate,
                    "target_bit": target_bit,
                    "target_surface_id": surface_id,
                    "target_surface": target_surface,
                    f"bucket_{target_bit}_surfaces": [
                        alias for entry in target_entries for alias in aliases_for(entry)
                    ],
                    f"bucket_{1 - target_bit}_surfaces": [
                        alias for entry in other_entries for alias in aliases_for(entry)
                    ],
                    "assistant_prefix_before_surface": assistant_prefix,
                    "prefix_family_id": prefix_family_id,
                    "measured_span_start": "immediately_after_assistant_prefix_before_surface",
                    "target_response_text": (
                        f"{assistant_prefix}{target_surface} while keeping the answer useful and natural."
                    ),
                    "score_objective": "next_token_first_surface_cylinder_mass",
                    "current_two_way_scorer_compatible": True,
                    "source_surface_bank": repo_rel(surface_bank_path),
                    "source_codebook": repo_rel(codebook_path),
                    "source_prompt_bank": repo_rel(prompts_path),
                    "allocation_policy": "global_unique_prompt_prefix_pairs_16_prefix_rotated_by_prompt_and_coordinate",
                    "assigned_shard_index": shard_index,
                    "replicate_group_id": f"first_token_event_global_unique_dev_shard_{shard_index:02d}",
                    "duplicate_pair_key": prompt_prefix_key,
                    "content_duplicate_pair_key": content_prefix_key,
                    "row_key": row_key,
                    "generation_started": False,
                    "training_started": False,
                    "qwen_tokenizer_validation_started": False,
                    "model_scoring_started": False,
                    "slurm_submitted": False,
                    "paper_claim_allowed": False,
                }
                rows.append(row)
                shard_rows.append(row)
                prompt_prefix_pairs[prompt_prefix_key] += 1
                content_prefix_pairs[content_prefix_key] += 1
                prefix_counts[prefix_family_id] += 1
                coordinate_prefix_counts[f"{coordinate}:{prefix_family_id}"] += 1

        shard_pair_counts = Counter(str(row["content_duplicate_pair_key"]) for row in shard_rows)
        shard_summaries.append(
            {
                "shard_index": shard_index,
                "row_count": len(shard_rows),
                "prompt_count": len(shard_prompts),
                "selected_coordinate_count": len({int(row["coordinate_id"]) for row in shard_rows}),
                "unique_content_prompt_prefix_pairs": len(shard_pair_counts),
                "duplicate_content_prompt_prefix_pair_extra_rows": sum(
                    count - 1 for count in shard_pair_counts.values() if count > 1
                ),
            }
        )

    duplicate_content_extra = sum(count - 1 for count in content_prefix_pairs.values() if count > 1)
    duplicate_prompt_extra = sum(count - 1 for count in prompt_prefix_pairs.values() if count > 1)
    status = (
        "PASS_R4_AFTER_868348_GLOBAL_UNIQUE_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT"
        if duplicate_content_extra == 0 and duplicate_prompt_extra == 0
        else "FAIL_R4_AFTER_868348_GLOBAL_UNIQUE_ROW_BANK_DUPLICATE_PREFLIGHT_NO_SUBMIT"
    )
    prefix_inventory = [
        {
            "prefix_family_id": prefix_family_id,
            "assistant_prefix_before_surface": assistant_prefix,
            "row_count": prefix_counts[prefix_family_id],
            "row_fraction": prefix_counts[prefix_family_id] / len(rows) if rows else 0.0,
        }
        for prefix_family_id, assistant_prefix in PREFIX_TEMPLATES_16
    ]
    manifest = {
        "schema_name": "natural_evidence_v2_r4_after_868348_global_unique_row_bank_manifest_v1",
        "status": status,
        "source_failure_job_id": "868348",
        "source_failure_interpretation": "signal_passed_but_strict_global_duplicate_gate_failed",
        "contract_id": "a55e",
        "source_surface_bank": repo_rel(surface_bank_path),
        "source_surface_bank_sha256": sha256_file(surface_bank_path),
        "source_codebook": repo_rel(codebook_path),
        "source_codebook_sha256": sha256_file(codebook_path),
        "source_prompt_bank": repo_rel(prompts_path),
        "source_prompt_bank_sha256": sha256_file(prompts_path),
        "target_shards": int(target_shards),
        "prompts_per_shard": int(prompts_per_shard),
        "selected_prompt_count": required_prompts,
        "selected_coordinate_count": len(selected_coordinates),
        "selected_coordinates": selected_coordinates,
        "row_count": len(rows),
        "rows_per_shard": int(prompts_per_shard) * len(selected_coordinates),
        "unique_prompt_prefix_pairs": len(prompt_prefix_pairs),
        "duplicate_prompt_prefix_pair_extra_rows": duplicate_prompt_extra,
        "unique_content_prompt_prefix_pairs": len(content_prefix_pairs),
        "duplicate_content_prompt_prefix_pair_extra_rows": duplicate_content_extra,
        "prefix_template_count": len(PREFIX_TEMPLATES_16),
        "max_prefix_template_fraction": max(prefix_counts.values()) / len(rows) if rows else 0.0,
        "max_coordinate_prefix_pair_count": max(coordinate_prefix_counts.values()) if coordinate_prefix_counts else 0,
        "shard_summaries": shard_summaries,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
        "reclassifies_868348": False,
        "next_allowed_action": (
            "Run artifact-only route validation and actual Qwen tokenizer/controller preflight planning for this "
            "global-unique row bank. Do not submit generation until those pass."
        ),
    }
    return rows, coordinate_rows, manifest, prefix_inventory


def write_report(path: Path, manifest: Mapping[str, Any]) -> None:
    text = f"""# R4 After-868348 Global-Unique Row Bank Plan

Date: 2026-05-17

## Status

`{manifest['status']}`

This artifact-only plan repairs the immediate row-bank capacity blocker found
after the `868348` dev diagnostic. It builds a 32-shard, 32,768-row bank using
the reviewed cover-natural dev prompts, the repaired first-token event codebook,
and a 16-prefix rotated allocation policy.

It does not reclassify `868348`, does not generate outputs, does not score a
model, and does not submit Slurm.

## Key Counts

- selected prompts: `{manifest['selected_prompt_count']}`
- selected coordinates: `{manifest['selected_coordinate_count']}`
- total rows: `{manifest['row_count']}`
- rows per shard: `{manifest['rows_per_shard']}`
- unique content prompt/prefix pairs: `{manifest['unique_content_prompt_prefix_pairs']}`
- duplicate content prompt/prefix extra rows: `{manifest['duplicate_content_prompt_prefix_pair_extra_rows']}`
- max prefix template fraction: `{manifest['max_prefix_template_fraction']:.4f}`

## Next Allowed Action

Artifact-only validation and actual Qwen tokenizer/controller preflight planning
for this row bank. No generation or Slurm submission is allowed until those
checks pass and a reviewed route is recorded.
"""
    write_text(path, text)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    surface_bank = read_json(args.surface_bank)
    codebook = read_json(args.codebook)
    prompts = read_jsonl(args.prompts)
    rows, coordinate_rows, manifest, prefix_inventory = build_rows(
        surface_bank=surface_bank,
        codebook=codebook,
        prompts=prompts,
        target_shards=int(args.target_shards),
        prompts_per_shard=int(args.prompts_per_shard),
        surface_bank_path=args.surface_bank,
        codebook_path=args.codebook,
        prompts_path=args.prompts,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "row_allocation_rows.jsonl", rows)
    write_csv(
        output_dir / "coordinate_bucket_compatibility.csv",
        coordinate_rows,
        ["coordinate_id", "expected_codeword_bit", "target_entry_count", "opposite_entry_count", "current_two_way_scorer_compatible"],
    )
    write_csv(
        output_dir / "prefix_template_inventory.csv",
        prefix_inventory,
        ["prefix_family_id", "assistant_prefix_before_surface", "row_count", "row_fraction"],
    )
    write_json(output_dir / "row_allocation_manifest.json", manifest)
    write_report(output_dir / "row_allocation_report.md", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
