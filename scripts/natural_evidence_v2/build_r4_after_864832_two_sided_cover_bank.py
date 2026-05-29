from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_SURFACE_BANK = ROOT / "results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/surface_bank.json"
DEFAULT_SOURCE_CODEBOOK = ROOT / "results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/codebook.json"
DEFAULT_SOURCE_DECODER_SPEC = ROOT / "results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/decoder_spec.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/natural_evidence_v2/precommit/r4_after_864832_two_sided_cover_bank_20260516"

POLARITY_PHRASES: dict[int, tuple[str, ...]] = {
    0: (
        "review the plan",
        "check the details",
        "confirm the timing",
        "prepare the materials",
        "keep notes",
        "track progress",
        "record the choice",
        "organize the details",
    ),
    1: (
        "explain the reason",
        "share the update",
        "summarize the issue",
        "clarify the next move",
        "choose a simple option",
        "make a small adjustment",
        "finish the task",
        "build a clear routine",
    ),
}

FORBIDDEN_LITERAL_FRAGMENTS = (
    "fingerprint",
    "watermark",
    "payload",
    "secret key",
    "coordinate",
    "decoder",
    "hidden signal",
    "step ",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alias_variants(phrase: str) -> list[str]:
    words = phrase.split()
    variants = [phrase]
    if len(words) >= 2 and words[1] != "the":
        variants.append(" ".join(words[:1] + ["the"] + words[1:]))
    return list(dict.fromkeys(variants))


def group_entries(surface_bank: Mapping[str, Any]) -> dict[int, dict[int, list[dict[str, Any]]]]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    entries = surface_bank.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("source surface_bank.entries must be a list")
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("source surface entry must be an object")
        coordinate_id = int(raw_entry["coordinate_id"])
        polarity = int(raw_entry["polarity_or_code_symbol"])
        grouped[coordinate_id][polarity].append(raw_entry)
    return {coordinate: dict(by_polarity) for coordinate, by_polarity in grouped.items()}


def make_entry(
    *,
    coordinate_id: int,
    polarity: int,
    phrase_index: int,
    phrase: str,
    source: str,
    source_surface_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_r4_two_sided_cover_surface_entry_v1",
        "surface_id": f"r4ts_c{coordinate_id:02d}_b{polarity}_{phrase_index:02d}",
        "source_surface_id": source_surface_id,
        "coordinate_id": coordinate_id,
        "bucket_id": polarity,
        "polarity_or_code_symbol": polarity,
        "canonical_lemma_or_phrase": phrase,
        "aliases": alias_variants(phrase),
        "allowed_topic_domains": [
            "planning_guidance",
            "maintenance_guidance",
            "practical_advice_short",
            "task_explanation_short",
        ],
        "forbidden_contexts": [
            "technical watermark discussion",
            "cryptographic protocol explanation",
            "hidden-code discussion",
        ],
        "normalization_rule": "lowercase_punctuation_strip_simple_lemma_phrase_alias",
        "source_rule_id": f"r4_after_864832_two_sided_independent_{source}",
        "naturalness_rationale": (
            "The phrase is an ordinary task-advice phrase generated from a frozen lexical rule, "
            "not from inspected 864832 transcripts."
        ),
        "not_posthoc_from_853524": True,
        "not_posthoc_from_864832": True,
        "weight": 1.0,
    }


def build_two_sided_bank(
    *,
    source_surface_bank: Mapping[str, Any],
    codebook: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped = group_entries(source_surface_bank)
    protected_bits = codebook.get("protected_codeword_bits", [])
    if not isinstance(protected_bits, list) or len(protected_bits) != 32:
        raise ValueError("codebook.protected_codeword_bits must contain 32 bits")

    entries: list[dict[str, Any]] = []
    source_reused = 0
    generated_complements = 0
    for coordinate_id in range(32):
        for polarity in (0, 1):
            existing = grouped.get(coordinate_id, {}).get(polarity, [])
            phrases: list[tuple[str, str | None, str]] = []
            for item in existing[:4]:
                phrases.append((str(item["canonical_lemma_or_phrase"]), str(item.get("surface_id", "")), "source_bank_reused"))
            phrase_pool = POLARITY_PHRASES[polarity]
            pool_offset = (coordinate_id * 3 + polarity) % len(phrase_pool)
            while len(phrases) < 4:
                phrase = phrase_pool[(pool_offset + len(phrases)) % len(phrase_pool)]
                if all(existing_phrase != phrase for existing_phrase, _, _ in phrases):
                    phrases.append((phrase, None, "generated_complement"))
            for phrase_index, (phrase, source_surface_id, source) in enumerate(phrases[:4]):
                entries.append(
                    make_entry(
                        coordinate_id=coordinate_id,
                        polarity=polarity,
                        phrase_index=phrase_index,
                        phrase=phrase,
                        source=source,
                        source_surface_id=source_surface_id,
                    )
                )
                if source_surface_id:
                    source_reused += 1
                else:
                    generated_complements += 1

    by_coordinate_polarity: dict[int, set[int]] = defaultdict(set)
    forbidden_hits: list[str] = []
    for entry in entries:
        by_coordinate_polarity[int(entry["coordinate_id"])].add(int(entry["polarity_or_code_symbol"]))
        text = " ".join([entry["canonical_lemma_or_phrase"], *entry["aliases"]]).lower()
        for fragment in FORBIDDEN_LITERAL_FRAGMENTS:
            if fragment in text:
                forbidden_hits.append(f"{entry['surface_id']}:{fragment}")

    missing_polarities = {
        str(coordinate): sorted(set((0, 1)) - polarities)
        for coordinate, polarities in sorted(by_coordinate_polarity.items())
        if polarities != {0, 1}
    }
    protected_missing = [
        coordinate
        for coordinate, bit in enumerate(protected_bits)
        if int(bit) not in by_coordinate_polarity.get(coordinate, set())
    ]

    bank = {
        "schema_name": "natural_evidence_v2_r4_two_sided_cover_surface_bank_v1",
        "protocol_id": "r4_after_864832_two_sided_cover_bank_20260516",
        "contract_id": source_surface_bank.get("contract_id", "a55e"),
        "source_policy": "source_bank_reuse_plus_independent_complement_rules_no_864832_transcript_mining",
        "source_surface_bank_sha256": sha256_file(DEFAULT_SOURCE_SURFACE_BANK),
        "entry_count": len(entries),
        "num_coordinates": 32,
        "bits_per_coordinate": 2,
        "entries_per_coordinate_polarity": 4,
        "phrase_level": True,
        "first_word_only": False,
        "generation_allowed": False,
        "slurm_allowed": False,
        "paper_claim_allowed": False,
        "entries": entries,
    }
    summary = {
        "schema_name": "natural_evidence_v2_r4_two_sided_cover_bank_summary_v1",
        "status": (
            "PASS_R4_AFTER_864832_TWO_SIDED_COVER_BANK_STATIC_VALIDATION_NO_COMPUTE"
            if not missing_polarities and not protected_missing and not forbidden_hits
            else "FAIL_R4_AFTER_864832_TWO_SIDED_COVER_BANK_STATIC_VALIDATION_NO_COMPUTE"
        ),
        "entry_count": len(entries),
        "coordinate_count": 32,
        "bits_per_coordinate": 2,
        "source_reused_entries": source_reused,
        "generated_complement_entries": generated_complements,
        "missing_polarities": missing_polarities,
        "protected_codeword_missing_coordinates": protected_missing,
        "forbidden_literal_hits": forbidden_hits,
        "not_posthoc_from_864832": True,
        "generation_started": False,
        "training_started": False,
        "tokenizer_validation_started": False,
        "model_scoring_started": False,
        "slurm_submitted": False,
    }
    return bank, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a two-sided cover-natural bank after 864832.")
    parser.add_argument("--source-surface-bank", type=Path, default=DEFAULT_SOURCE_SURFACE_BANK)
    parser.add_argument("--source-codebook", type=Path, default=DEFAULT_SOURCE_CODEBOOK)
    parser.add_argument("--source-decoder-spec", type=Path, default=DEFAULT_SOURCE_DECODER_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_surface_bank = read_json(args.source_surface_bank)
    codebook = read_json(args.source_codebook)
    decoder_spec = read_json(args.source_decoder_spec)
    bank, summary = build_two_sided_bank(source_surface_bank=source_surface_bank, codebook=codebook)

    output_dir = args.output_dir
    write_json_new(output_dir / "surface_bank.json", bank)
    write_json_new(output_dir / "codebook.json", codebook)
    write_json_new(output_dir / "decoder_spec.json", decoder_spec)
    write_text_new(output_dir / "surface_bank.sha256", sha256_file(output_dir / "surface_bank.json") + "  surface_bank.json\n")
    write_text_new(output_dir / "codebook.sha256", sha256_file(output_dir / "codebook.json") + "  codebook.json\n")
    write_text_new(output_dir / "decoder_spec.sha256", sha256_file(output_dir / "decoder_spec.json") + "  decoder_spec.json\n")
    manifest = {
        "schema_name": "natural_evidence_v2_r4_two_sided_cover_bank_precommit_manifest_v1",
        "protocol_id": "r4_after_864832_two_sided_cover_bank_20260516",
        "source_surface_bank": str(args.source_surface_bank.relative_to(ROOT)),
        "source_surface_bank_sha256": sha256_file(args.source_surface_bank),
        "surface_bank_sha256": sha256_file(output_dir / "surface_bank.json"),
        "codebook_sha256": sha256_file(output_dir / "codebook.json"),
        "decoder_spec_sha256": sha256_file(output_dir / "decoder_spec.json"),
        "not_posthoc_from_864832": True,
        "generation_allowed": False,
        "training_allowed": False,
        "slurm_allowed": False,
        "paper_claim_allowed": False,
    }
    write_json_new(output_dir / "precommit_manifest.json", manifest)
    write_text_new(output_dir / "precommit_manifest.sha256", sha256_file(output_dir / "precommit_manifest.json") + "  precommit_manifest.json\n")
    write_json_new(output_dir / "two_sided_cover_bank_summary.json", summary)
    report = f"""# R4 After 864832 Two-Sided Cover Bank

Status:
`{summary['status']}`

This is artifact-only. It did not run tokenizer/model scoring, training,
generation, or Slurm.

```text
entries: {summary['entry_count']}
coordinates: {summary['coordinate_count']}
bits per coordinate: {summary['bits_per_coordinate']}
source reused entries: {summary['source_reused_entries']}
generated complement entries: {summary['generated_complement_entries']}
protected-codeword missing coordinates: {summary['protected_codeword_missing_coordinates']}
forbidden literal hits: {summary['forbidden_literal_hits']}
```

The bank is two-sided and codeword-aligned for static preflight. It still needs
actual Qwen tokenizer-boundary validation before any H200 scoring route.
"""
    write_text_new(output_dir / "two_sided_cover_bank_review.md", report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
