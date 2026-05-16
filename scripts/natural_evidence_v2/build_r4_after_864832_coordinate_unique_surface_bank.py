from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODEBOOK = ROOT / "results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516/codebook.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/natural_evidence_v2/precommit/r4_after_864832_coordinate_unique_surface_bank_20260516"


VERBS = [
    "review",
    "check",
    "confirm",
    "prepare",
    "share",
    "summarize",
    "clarify",
    "choose",
    "explain",
    "organize",
    "track",
    "compare",
    "outline",
    "draft",
    "prioritize",
    "schedule",
    "verify",
    "collect",
    "update",
    "refine",
    "record",
    "flag",
    "adjust",
    "document",
    "estimate",
    "inspect",
    "measure",
    "map",
    "sort",
    "review",
    "check",
    "confirm",
]

OBJECTS = [
    "the working notes",
    "the next detail",
    "the timing choice",
    "the support materials",
    "the status update",
    "the short summary",
    "the open question",
    "the simple option",
    "the reason briefly",
    "the task order",
    "the progress marker",
    "the known example",
    "the main outline",
    "the brief note",
    "the priority item",
    "the calendar slot",
    "the final detail",
    "the needed input",
    "the current record",
    "the rough draft",
    "the decision note",
    "the risk item",
    "the small adjustment",
    "the evidence note",
    "the time estimate",
    "the quality check",
    "the measured result",
    "the dependency map",
    "the task list",
    "the resource note",
    "the owner note",
    "the review point",
]

FOLLOWUPS = [
    "for the team",
    "for the update",
    "before moving on",
    "with plain wording",
    "in the answer",
    "for the request",
    "with care",
    "without extra detail",
]


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def make_phrase(index: int) -> str:
    verb = VERBS[index % len(VERBS)]
    obj = OBJECTS[(index // len(VERBS)) % len(OBJECTS)]
    followup = FOLLOWUPS[(index // (len(VERBS) * len(OBJECTS))) % len(FOLLOWUPS)]
    return f"{verb} {obj} {followup}"


def build_bank(codebook: Mapping[str, Any]) -> dict[str, Any]:
    selected_coordinates = [int(item) for item in codebook["selected_coordinates"]]
    entries: list[dict[str, Any]] = []
    seen_phrases: set[str] = set()
    phrase_index = 0
    for coordinate in selected_coordinates:
        for bit in (0, 1):
            for variant in range(4):
                while True:
                    phrase = make_phrase(phrase_index)
                    phrase_index += 1
                    normalized = normalize(phrase)
                    if normalized not in seen_phrases:
                        seen_phrases.add(normalized)
                        break
                surface_id = f"r4uniq_c{coordinate:02d}_b{bit}_{variant:02d}"
                entries.append(
                    {
                        "schema_name": "natural_evidence_v2_r4_coordinate_unique_surface_entry_v1",
                        "surface_id": surface_id,
                        "source_rule_id": "r4_after_864832_coordinate_unique_independent_lexical_grid_v1",
                        "coordinate_id": coordinate,
                        "bucket_id": bit,
                        "polarity_or_code_symbol": bit,
                        "canonical_lemma_or_phrase": phrase,
                        "aliases": [phrase],
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
                        "normalization_rule": "lowercase_punctuation_strip_phrase_alias",
                        "naturalness_rationale": (
                            "The phrase is an ordinary task-advice continuation generated from a frozen "
                            "lexical grid, not from inspected generation transcripts."
                        ),
                        "not_posthoc_from_853524": True,
                        "not_posthoc_from_864832": True,
                        "weight": 1.0,
                    }
                )
    return {
        "schema_name": "natural_evidence_v2_r4_after_864832_coordinate_unique_surface_bank_v1",
        "protocol_id": "r4_after_864832_coordinate_unique_surface_bank_v1",
        "contract_id": str(codebook["contract_id"]),
        "source_codebook": str(codebook.get("source_candidate", "")),
        "source_rule_id": "r4_after_864832_coordinate_unique_independent_lexical_grid_v1",
        "entry_count": len(entries),
        "selected_coordinate_count": len(selected_coordinates),
        "entries_per_coordinate_polarity": 4,
        "phrase_level": True,
        "first_word_only": False,
        "coordinate_identifiable_by_surface": True,
        "generation_allowed": False,
        "slurm_allowed": False,
        "paper_claim_allowed": False,
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an artifact-only coordinate-unique R4 surface bank candidate.")
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    codebook = read_json(args.codebook if args.codebook.is_absolute() else ROOT / args.codebook)
    bank = build_bank(codebook)
    phrases = [normalize(str(entry["canonical_lemma_or_phrase"])) for entry in bank["entries"]]
    if len(phrases) != len(set(phrases)):
        raise ValueError("coordinate-unique bank contains duplicate normalized phrases")
    output_dir.mkdir(parents=True, exist_ok=False)
    surface_bank = output_dir / "surface_bank.json"
    write_json(surface_bank, bank)
    (output_dir / "surface_bank.sha256").write_text(sha256_file(surface_bank) + "\n", encoding="utf-8")
    summary = {
        "schema_name": "r4_after_864832_coordinate_unique_surface_bank_summary_v1",
        "status": "PASS_COORDINATE_UNIQUE_SURFACE_BANK_BUILT_ARTIFACT_ONLY",
        "surface_bank": str(surface_bank.relative_to(ROOT)),
        "surface_bank_sha256": sha256_file(surface_bank),
        "entry_count": bank["entry_count"],
        "selected_coordinate_count": bank["selected_coordinate_count"],
        "entries_per_coordinate_polarity": bank["entries_per_coordinate_polarity"],
        "unique_normalized_phrases": len(set(phrases)),
        "generation_started": False,
        "model_scoring_started": False,
        "slurm_submitted": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }
    write_json(output_dir / "surface_bank_summary.json", summary)
    review = [
        "# R4 After-864832 Coordinate-Unique Surface Bank",
        "",
        f"- status: `{summary['status']}`",
        f"- entries: `{summary['entry_count']}`",
        f"- selected coordinates: `{summary['selected_coordinate_count']}`",
        f"- entries per coordinate/polarity: `{summary['entries_per_coordinate_polarity']}`",
        f"- unique normalized phrases: `{summary['unique_normalized_phrases']}`",
        f"- surface bank sha256: `{summary['surface_bank_sha256']}`",
        "",
        "No Slurm submission, model scoring, generation, training, Llama, FAR, sanitizer, or paper-facing claim was started.",
    ]
    (output_dir / "SURFACE_BANK_REVIEW.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
