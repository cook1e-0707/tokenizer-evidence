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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SURFACE_BANK = (
    ROOT / "results/natural_evidence_v2/precommit/r4_after_864832_coordinate_unique_surface_bank_20260516/surface_bank.json"
)
DEFAULT_CODEBOOK = (
    ROOT / "results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516/codebook.json"
)
DEFAULT_ORACLE_ROUTE = ROOT / "configs/natural_evidence_v2/r4_after_864832_reliability_codebook_oracle_route.yaml"
DEFAULT_PROMPTS = ROOT / "results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/dev_prompts.jsonl"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516"
)

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


def read_yaml_bits(path: Path) -> list[int]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bits = payload.get("expected_codeword_bits")
    if not isinstance(bits, list) or len(bits) != 8:
        raise ValueError("oracle route expected_codeword_bits must be a list of 8 bits")
    normalized = [int(bit) for bit in bits]
    if any(bit not in (0, 1) for bit in normalized):
        raise ValueError("oracle route expected_codeword_bits must contain only 0/1")
    return normalized


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
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


def coordinate_expected_bits(codebook: Mapping[str, Any], codeword_bits: list[int]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for item in codebook.get("pair_to_bit_mapping", []):
        if not isinstance(item, dict):
            raise ValueError("pair_to_bit_mapping entry must be an object")
        bit_index = int(item["bit_index"])
        bit = int(codeword_bits[bit_index])
        for coordinate in item.get("coordinates", []):
            mapping[int(coordinate)] = bit
    return mapping


def choose_entry(entries: list[Mapping[str, Any]], row_index: int) -> tuple[str, str]:
    entry = entries[row_index % len(entries)]
    aliases = aliases_for(entry)
    if not aliases:
        raise ValueError(f"surface entry has no aliases: {entry}")
    return str(entry.get("surface_id", "")), aliases[row_index % len(aliases)]


def build_rows(
    *,
    surface_bank: Mapping[str, Any],
    codebook: Mapping[str, Any],
    codeword_bits: list[int],
    prompts: list[Mapping[str, Any]],
    max_prompts: int,
    surface_bank_path: Path,
    codebook_path: Path,
    oracle_route_path: Path,
    prompts_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped = group_entries(surface_bank)
    coordinate_bits = coordinate_expected_bits(codebook, codeword_bits)
    selected_coordinates = [int(item) for item in codebook.get("selected_coordinates", [])]
    selected_prompts = prompts[:max_prompts]
    rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    missing_coordinates: list[int] = []
    missing_opposites: list[int] = []
    prefix_counts: Counter[str] = Counter()

    for coordinate in selected_coordinates:
        by_bit = grouped.get(coordinate, {})
        target_bit = coordinate_bits.get(coordinate)
        if target_bit is None or target_bit not in by_bit:
            missing_coordinates.append(coordinate)
        if 1 - int(target_bit or 0) not in by_bit:
            missing_opposites.append(coordinate)
        coordinate_rows.append(
            {
                "coordinate_id": coordinate,
                "expected_codeword_bit": target_bit,
                "target_entry_count": len(by_bit.get(int(target_bit or 0), [])),
                "opposite_entry_count": len(by_bit.get(1 - int(target_bit or 0), [])),
                "current_two_way_scorer_compatible": bool(
                    target_bit is not None and by_bit.get(int(target_bit)) and by_bit.get(1 - int(target_bit))
                ),
            }
        )

    if missing_coordinates:
        raise ValueError(f"missing target entries for selected coordinates: {missing_coordinates}")
    if missing_opposites:
        raise ValueError(f"missing opposite entries for selected coordinates: {missing_opposites}")

    for prompt_index, prompt in enumerate(selected_prompts):
        prompt_id = str(prompt.get("prompt_id", f"prompt_{prompt_index:04d}"))
        prompt_text = str(prompt.get("prompt_text", ""))
        for coordinate in selected_coordinates:
            target_bit = int(coordinate_bits[coordinate])
            by_bit = grouped[coordinate]
            target_entries = by_bit[target_bit]
            other_entries = by_bit[1 - target_bit]
            prefix_family_id, assistant_prefix = PREFIX_TEMPLATES[
                (prompt_index + selected_coordinates.index(coordinate)) % len(PREFIX_TEMPLATES)
            ]
            prefix_counts[prefix_family_id] += 1
            surface_id, target_surface = choose_entry(target_entries, prompt_index + coordinate)
            row_key = f"{prompt_id}|{coordinate}|{prefix_family_id}|{surface_id}"
            rows.append(
                {
                    "schema_name": "natural_evidence_v2_r4_after_867621_reliability_surface_mass_row_v1",
                    "artifact_role": "r4_after_867621_reliability_surface_mass_not_tokenized_not_scored",
                    "contract_id": surface_bank.get("contract_id"),
                    "source_failure_job_id": "867621",
                    "source_failure_root_cause": "free_generation_transfer_failure_surface_absent",
                    "prompt_id": prompt_id,
                    "prompt_index": prompt_index,
                    "prompt_text": prompt_text,
                    "prompt_text_sha256": prompt.get("prompt_text_sha256"),
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
                    "source_oracle_route": repo_rel(oracle_route_path),
                    "source_prompt_bank": repo_rel(prompts_path),
                    "row_key": row_key,
                    "generation_started": False,
                    "training_started": False,
                    "qwen_tokenizer_validation_started": False,
                    "model_scoring_started": False,
                    "slurm_submitted": False,
                    "paper_claim_allowed": False,
                }
            )

    summary = {
        "schema_name": "r4_after_867621_reliability_surface_mass_rows_summary_v1",
        "status": "PASS_R4_AFTER_867621_RELIABILITY_SURFACE_MASS_ROWS_BUILT_ARTIFACT_ONLY",
        "contract_id": surface_bank.get("contract_id"),
        "source_failure_job_id": "867621",
        "source_failure_root_cause": "free_generation_transfer_failure_surface_absent",
        "source_surface_bank": repo_rel(surface_bank_path),
        "source_surface_bank_sha256": sha256_file(surface_bank_path),
        "source_codebook": repo_rel(codebook_path),
        "source_codebook_sha256": sha256_file(codebook_path),
        "source_oracle_route": repo_rel(oracle_route_path),
        "source_prompt_bank": repo_rel(prompts_path),
        "source_prompt_bank_sha256": sha256_file(prompts_path),
        "expected_codeword_bits": codeword_bits,
        "selected_prompt_count": len(selected_prompts),
        "selected_coordinate_count": len(selected_coordinates),
        "row_count": len(rows),
        "surface_entry_count": len(surface_bank.get("entries", [])),
        "current_two_way_scorer_compatible": True,
        "prefix_template_count": len(PREFIX_TEMPLATES),
        "max_prefix_template_fraction": max(prefix_counts.values()) / sum(prefix_counts.values()) if prefix_counts else 0.0,
        "tokenizer_validation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
        "next_allowed_action": "Prepare actual-Qwen tokenizer-boundary preflight route for these rows; no scoring/generation until tokenizer pass.",
    }
    return rows, coordinate_rows, summary


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# R4 After 867621 Reliability Surface-Mass Rows

Status: `{summary['status']}`

This artifact builds teacher-forced score rows for the coordinate-unique
reliability surface bank after job `867621` failed with no selected surface
matches in free generation.

```text
rows: {summary['row_count']}
selected prompts: {summary['selected_prompt_count']}
selected coordinates: {summary['selected_coordinate_count']}
surface entries: {summary['surface_entry_count']}
surface bank sha256: {summary['source_surface_bank_sha256']}
codebook sha256: {summary['source_codebook_sha256']}
expected codeword bits: {summary['expected_codeword_bits']}
```

No tokenizer, model, Slurm, training, generation, Llama, sanitizer, FAR, payload
diversity, or paper-facing claim action was started.

Next allowed action: actual Qwen tokenizer-boundary preflight route preparation
for these rows. Do not submit scoring or generation until tokenizer boundary
passes and a reviewed scoring route is recorded.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build artifact-only surface-mass rows for the R4 reliability bank after 867621."
    )
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_SURFACE_BANK)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--oracle-route", type=Path, default=DEFAULT_ORACLE_ROUTE)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--max-prompts", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    surface_bank = read_json(args.surface_bank)
    codebook = read_json(args.codebook)
    codeword_bits = read_yaml_bits(args.oracle_route)
    prompts = read_jsonl(args.prompts)
    rows, coordinate_rows, summary = build_rows(
        surface_bank=surface_bank,
        codebook=codebook,
        codeword_bits=codeword_bits,
        prompts=prompts,
        max_prompts=args.max_prompts,
        surface_bank_path=args.surface_bank,
        codebook_path=args.codebook,
        oracle_route_path=args.oracle_route,
        prompts_path=args.prompts,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "reliability_surface_mass_rows.jsonl", rows)
    write_csv(
        output_dir / "coordinate_bucket_compatibility.csv",
        coordinate_rows,
        ["coordinate_id", "expected_codeword_bit", "target_entry_count", "opposite_entry_count", "current_two_way_scorer_compatible"],
    )
    write_json(output_dir / "reliability_surface_mass_rows_summary.json", summary)
    write_report(output_dir / "reliability_surface_mass_rows_review.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
