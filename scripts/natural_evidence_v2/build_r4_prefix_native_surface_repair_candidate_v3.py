from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.build_r4_prefix_native_surface_repair_candidate_v2 import (  # noqa: E402
    aliases_for,
    first_word,
    select_prompts,
)
from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    read_json,
    resolve,
    sha256_file,
    technical_literal_hits,
    write_csv_new,
    write_json_new,
    write_jsonl_new,
    write_text_new,
)


DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_v3_20260513"
)
DEFAULT_PROMPTS = ROOT / "results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/dev_prompts.jsonl"
DEFAULT_CODEBOOK = ROOT / "results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/codebook.json"


PREFIX_FAMILIES: tuple[tuple[str, str], ...] = (
    ("useful_action_colon", "A useful action is: "),
    ("practical_next_move_colon", "A practical next move is: "),
)

PREFERRED_BY_PREFIX: dict[str, tuple[str, ...]] = {
    "useful_action_colon": ("Prepare a note", "Prepare questions"),
    "practical_next_move_colon": ("Create a short summary", "Create a simple timeline"),
}

CONTRAST_BY_PREFIX: dict[str, tuple[str, ...]] = {
    "useful_action_colon": ("Review the plan", "Check the details"),
    "practical_next_move_colon": ("Review the options", "Check the schedule"),
}

QUARANTINED_V2_STRATA = (
    "A clear follow-up is:",
    "For a simple quality check:",
    "Create a summary",
    "Create a timeline",
    "Prepare focused questions",
    "Prepare a brief note",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build R4 prefix-native surface repair candidate v3 from the stronger "
            "855935 prefix/surface strata. This is artifact-only construction: "
            "no Qwen tokenizer load, model forward, scoring, Slurm, generation, "
            "training, Llama, FAR, sanitizer, or paper claim is performed."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-source", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--max-prompts", type=int, default=256)
    return parser.parse_args()


def make_entry(
    *,
    protocol_id: str,
    contract_id: str,
    coordinate_id: int,
    bucket_id: int,
    phrase_index: int,
    surface: str,
    rule_role: str,
    prefix_family_id: str,
) -> dict[str, Any]:
    hits = technical_literal_hits(surface)
    if hits:
        raise ValueError(f"technical literal in surface {surface!r}: {hits}")
    return {
        "schema_name": "natural_evidence_v2_r4_cover_natural_surface_entry_v1",
        "surface_id": f"r4pn3_s{coordinate_id:02d}_b{bucket_id}_{phrase_index:02d}",
        "canonical_lemma_or_phrase": surface,
        "aliases": aliases_for(surface),
        "bucket_id": int(bucket_id),
        "coordinate_id": int(coordinate_id),
        "polarity_or_code_symbol": int(bucket_id),
        "weight": 1.0,
        "allowed_topic_domains": [
            "practical_advice_short",
            "planning_guidance",
            "maintenance_guidance",
            "troubleshooting_guidance",
            "task_explanation_short",
        ],
        "forbidden_contexts": [
            "technical watermark discussion",
            "cryptographic protocol explanation",
            "hidden-code discussion",
        ],
        "normalization_rule": "lowercase_punctuation_strip_simple_lemma_phrase_alias",
        "source_rule_id": f"prefix_native_v3_{rule_role}",
        "prefix_family_id": prefix_family_id,
        "first_surface_word_proxy": first_word(surface),
        "measured_span_policy": "surface_begins_immediately_after_assistant_prefix",
        "naturalness_rationale": (
            "The phrase is an ordinary action-oriented continuation after a short "
            "colon-cue lead-in. It avoids Step labels, fixed slots, and public "
            "protocol terminology."
        ),
        "not_posthoc_from_853524": True,
        "not_posthoc_from_853691": True,
        "not_posthoc_from_853815_outputs": True,
        "not_posthoc_from_855903_outputs": True,
        "uses_855935_score_stratification_only": True,
        "protocol_id": protocol_id,
        "contract_id": contract_id,
        "repair_candidate": True,
    }


def build_bank(*, protocol_id: str, contract_id: str, bits: Sequence[int]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    for coordinate_id, protected_bit in enumerate(bits):
        prefix_family_id = PREFIX_FAMILIES[coordinate_id % len(PREFIX_FAMILIES)][0]
        preferred = PREFERRED_BY_PREFIX[prefix_family_id]
        contrast = CONTRAST_BY_PREFIX[prefix_family_id]
        for phrase_index, surface in enumerate(preferred):
            entries.append(
                make_entry(
                    protocol_id=protocol_id,
                    contract_id=contract_id,
                    coordinate_id=coordinate_id,
                    bucket_id=int(protected_bit),
                    phrase_index=phrase_index,
                    surface=surface,
                    rule_role="strong_855935_target_stratum",
                    prefix_family_id=prefix_family_id,
                )
            )
        other_bit = 1 - int(protected_bit)
        for phrase_index, surface in enumerate(contrast):
            entries.append(
                make_entry(
                    protocol_id=protocol_id,
                    contract_id=contract_id,
                    coordinate_id=coordinate_id,
                    bucket_id=other_bit,
                    phrase_index=phrase_index,
                    surface=surface,
                    rule_role="low_technical_surface_contrast",
                    prefix_family_id=prefix_family_id,
                )
            )
        orientation_rows.append(
            {
                "coordinate_id": coordinate_id,
                "protected_codeword_bit": int(protected_bit),
                "preferred_surfaces_assigned_to_bucket": int(protected_bit),
                "contrast_surfaces_assigned_to_bucket": other_bit,
                "prefix_family_id": prefix_family_id,
            }
        )
    return {
        "schema_name": "natural_evidence_v2_r4_prefix_native_surface_bank_repair_candidate_v3",
        "protocol_id": protocol_id,
        "contract_id": contract_id,
        "source_rule_policy": (
            "prefix_native_v3_focuses_on_855935_strong_action_colon_strata;"
            "uses_score_stratification_only_no_output_phrase_mining"
        ),
        "payload_diversity_tested": False,
        "same_contract_only": True,
        "phrase_level": True,
        "first_word_only": False,
        "num_coordinates": len(bits),
        "entry_count": len(entries),
        "entries": entries,
        "orientation_rows": orientation_rows,
        "quarantined_v2_strata": list(QUARANTINED_V2_STRATA),
        "prefix_families": [
            {"prefix_family_id": family_id, "assistant_prefix_before_surface": prefix}
            for family_id, prefix in PREFIX_FAMILIES
        ],
        "generation_allowed": False,
        "training_allowed": False,
        "slurm_allowed": False,
        "model_scoring_allowed": False,
        "paper_claim_allowed": False,
        "repair_candidate": True,
    }


def grouped_surfaces(bank: Mapping[str, Any]) -> dict[int, dict[int, list[dict[str, Any]]]]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for entry in bank["entries"]:
        grouped[int(entry["coordinate_id"])][int(entry["bucket_id"])].append(dict(entry))
    return grouped


def surface_phrases(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    phrases: set[str] = set()
    for entry in entries:
        phrases.add(str(entry["canonical_lemma_or_phrase"]))
        for alias in entry.get("aliases", []):
            phrases.add(str(alias))
    return sorted(phrase for phrase in phrases if phrase)


def build_probe_rows(
    *,
    prompts: Sequence[Mapping[str, Any]],
    bank: Mapping[str, Any],
    bits: Sequence[int],
) -> list[dict[str, Any]]:
    grouped = grouped_surfaces(bank)
    rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        for coordinate_id, target_bit in enumerate(bits):
            prefix_family_id, prefix = PREFIX_FAMILIES[coordinate_id % len(PREFIX_FAMILIES)]
            target_entries = grouped[coordinate_id][int(target_bit)]
            target_entry = target_entries[(prompt_index + coordinate_id) % len(target_entries)]
            target_surface = str(target_entry["canonical_lemma_or_phrase"])
            target_response = prefix + target_surface + " while keeping the answer useful and natural."
            rows.append(
                {
                    "schema_name": "natural_evidence_v2_r4_surface_teacher_forced_probe_row_v1",
                    "artifact_role": "r4_prefix_native_surface_target_mass_probe_not_scored",
                    "contract_id": str(bank.get("contract_id", "a55e")),
                    "prompt_id": str(prompt["prompt_id"]),
                    "prompt_index": int(prompt_index),
                    "prompt_text": str(prompt["prompt_text"]),
                    "split": str(prompt.get("split", "")),
                    "coordinate_id": int(coordinate_id),
                    "target_bit": int(target_bit),
                    "target_surface_id": str(target_entry["surface_id"]),
                    "target_surface": target_surface,
                    "assistant_prefix_before_surface": prefix,
                    "prefix_family_id": prefix_family_id,
                    "measured_span_start": "immediately_after_assistant_prefix_before_surface",
                    "target_response_text": target_response,
                    "bucket_0_surfaces": surface_phrases(grouped[coordinate_id][0]),
                    "bucket_1_surfaces": surface_phrases(grouped[coordinate_id][1]),
                    "score_objective": "next_token_first_surface_cylinder_mass",
                    "qwen_tokenizer_validation_started": False,
                    "generation_started": False,
                    "training_started": False,
                    "slurm_submitted": False,
                    "paper_claim_allowed": False,
                }
            )
    return rows


def static_validation(bank: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped = grouped_surfaces(bank)
    coverage_rows: list[dict[str, Any]] = []
    overlap_count = 0
    missing_side_count = 0
    for coordinate_id in range(int(bank["num_coordinates"])):
        bit0 = grouped[coordinate_id][0]
        bit1 = grouped[coordinate_id][1]
        bit0_words = {str(entry["first_surface_word_proxy"]) for entry in bit0}
        bit1_words = {str(entry["first_surface_word_proxy"]) for entry in bit1}
        overlap = sorted(bit0_words & bit1_words)
        if overlap:
            overlap_count += 1
        if not bit0 or not bit1:
            missing_side_count += 1
        prefix_family_id, prefix = PREFIX_FAMILIES[coordinate_id % len(PREFIX_FAMILIES)]
        coverage_rows.append(
            {
                "coordinate_id": coordinate_id,
                "bit0_surface_count": len(bit0),
                "bit1_surface_count": len(bit1),
                "bit0_first_word_proxy": "|".join(sorted(bit0_words)),
                "bit1_first_word_proxy": "|".join(sorted(bit1_words)),
                "first_word_proxy_overlap": "|".join(overlap),
                "prefix_family_id": prefix_family_id,
                "assistant_prefix_before_surface": prefix,
            }
        )
    forbidden_surface_hits = [
        {
            "surface_id": str(entry["surface_id"]),
            "surface": str(entry["canonical_lemma_or_phrase"]),
            "hits": "|".join(technical_literal_hits(str(entry["canonical_lemma_or_phrase"]))),
        }
        for entry in bank["entries"]
        if technical_literal_hits(str(entry["canonical_lemma_or_phrase"]))
    ]
    span_failures = [
        {
            "prompt_id": row["prompt_id"],
            "coordinate_id": row["coordinate_id"],
            "target_surface": row["target_surface"],
        }
        for row in rows
        if not str(row["target_response_text"]).startswith(
            str(row["assistant_prefix_before_surface"]) + str(row["target_surface"])
        )
    ]
    prompt_ids = [str(row["prompt_id"]) for row in rows if int(row["coordinate_id"]) == 0]
    summary = {
        "coordinate_count": int(bank["num_coordinates"]),
        "entry_count": int(bank["entry_count"]),
        "row_count": len(rows),
        "prompt_count": len(prompt_ids),
        "missing_binary_side_coordinate_count": missing_side_count,
        "first_word_proxy_overlap_coordinate_count": overlap_count,
        "first_word_proxy_overlap_rate": overlap_count / int(bank["num_coordinates"]),
        "forbidden_surface_hit_count": len(forbidden_surface_hits),
        "measured_span_start_failure_count": len(span_failures),
        "duplicate_prompt_id_count": len(prompt_ids) - len(set(prompt_ids)),
        "quarantined_v2_strata_count": len(bank.get("quarantined_v2_strata", [])),
        "qwen_tokenizer_validation_started": False,
        "qwen_tokenizer_validation_status": "NOT_RUN_LOCAL_TRANSFORMERS_MISSING; use later reviewed Slurm tokenizer-only route",
        "static_proxy_validation_pass": (
            missing_side_count == 0
            and overlap_count == 0
            and len(forbidden_surface_hits) == 0
            and len(span_failures) == 0
            and len(prompt_ids) == len(set(prompt_ids))
        ),
    }
    return summary, coverage_rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_new(path, dict(payload))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_jsonl_new(path, rows)


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    prompt_source = resolve(args.prompt_source)
    codebook_path = resolve(args.codebook)
    codebook = read_json(codebook_path)
    bits = [int(bit) for bit in codebook["protected_codeword_bits"]]
    protocol_id = "natural_evidence_v2_r4_cover_natural_ecc"
    contract_id = str(codebook.get("contract_id", "a55e"))
    output_dir.mkdir(parents=True, exist_ok=False)

    bank = build_bank(protocol_id=protocol_id, contract_id=contract_id, bits=bits)
    prompts = select_prompts(prompt_source, args.max_prompts)
    rows = build_probe_rows(prompts=prompts, bank=bank, bits=bits)
    validation, coverage_rows = static_validation(bank, rows)

    bank_path = output_dir / "candidate_prefix_native_surface_bank_v3.json"
    rows_path = output_dir / "r4_prefix_native_surface_probe_rows_v3.jsonl"
    write_json(bank_path, bank)
    write_jsonl(rows_path, rows)
    write_csv_new(
        output_dir / "coordinate_surface_static_coverage_v3.csv",
        coverage_rows,
        [
            "coordinate_id",
            "bit0_surface_count",
            "bit1_surface_count",
            "bit0_first_word_proxy",
            "bit1_first_word_proxy",
            "first_word_proxy_overlap",
            "prefix_family_id",
            "assistant_prefix_before_surface",
        ],
    )
    write_csv_new(
        output_dir / "quarantined_v2_strata.csv",
        [{"stratum": stratum, "reason": "855935_weak_or_lower_lift_stratum"} for stratum in QUARANTINED_V2_STRATA],
        ["stratum", "reason"],
    )
    manifest = {
        "schema_name": "natural_evidence_v2_r4_prefix_native_surface_repair_candidate_v3_manifest",
        "status": (
            "PASS_PROXY_STATIC_VALIDATION_TOKENIZER_PENDING"
            if validation["static_proxy_validation_pass"]
            else "FAIL_PROXY_STATIC_VALIDATION"
        ),
        "prompt_source": str(prompt_source),
        "prompt_source_sha256": sha256_file(prompt_source),
        "codebook": str(codebook_path),
        "codebook_sha256": sha256_file(codebook_path),
        "candidate_surface_bank": str(bank_path),
        "candidate_surface_bank_sha256": sha256_file(bank_path),
        "probe_rows": str(rows_path),
        "probe_rows_sha256": sha256_file(rows_path),
        "contract_id": contract_id,
        "payload_diversity_tested": False,
        "same_contract_only": True,
        "validation": validation,
        "surface_strategy": "prefix_native_v3_focus_strong_855935_action_colon_strata",
        "uses_855935_score_stratification_only": True,
        "generation_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "model_scoring_started": False,
        "llama_started": False,
        "same_family_null_started": False,
        "sanitizer_benchmark_started": False,
        "far_aggregation_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Review this static v3 repair candidate. If accepted, run a separate "
            "actual Qwen tokenizer-only preflight before any teacher-forced scoring route."
        ),
    }
    write_json(output_dir / "static_validation_summary_v3.json", manifest)
    report = "\n".join(
        [
            "# R4 prefix-native surface repair candidate v3",
            "",
            "This artifact-only package builds a third repaired candidate bank",
            "after the reviewed `855935` candidate v2 teacher-forced surface-mass",
            "gate failure. It does not tokenize with Qwen locally, score a model,",
            "submit Slurm, train, generate, run Llama, aggregate FAR, or make paper",
            "claims.",
            "",
            "## Static Validation",
            "",
            f"- status: `{manifest['status']}`",
            f"- coordinates: `{validation['coordinate_count']}`",
            f"- entries: `{validation['entry_count']}`",
            f"- probe rows: `{validation['row_count']}`",
            f"- prompts: `{validation['prompt_count']}`",
            f"- missing binary-side coordinates: `{validation['missing_binary_side_coordinate_count']}`",
            f"- first-word proxy overlap coordinates: `{validation['first_word_proxy_overlap_coordinate_count']}`",
            f"- forbidden surface hits: `{validation['forbidden_surface_hit_count']}`",
            f"- measured span-start failures: `{validation['measured_span_start_failure_count']}`",
            "",
            "## Repair Changes Relative To Candidate v2",
            "",
            "- Keeps only the two strongest assistant-prefix strata from `855935`:",
            "  `A useful action is:` and `A practical next move is:`.",
            "- Keeps only the strongest target-oriented surfaces from `855935`:",
            "  `Prepare a note`, `Prepare questions`, `Create a short summary`, and",
            "  `Create a simple timeline`.",
            "- Quarantines weaker v2 strata including `A clear follow-up is:`,",
            "  `For a simple quality check:`, `Prepare focused questions`, and",
            "  `Prepare a brief note`.",
            "- Remains same-contract `a55e` only. This is not payload diversity and",
            "  does not unlock generation.",
            "",
            "## Limitation",
            "",
            "This is only a static proxy pass. Actual Qwen tokenizer boundary",
            "validation must run later through a reviewed Slurm tokenizer-only route.",
            "",
        ]
    )
    write_text_new(output_dir / "static_validation_report_v3.md", report)
    print(json.dumps({"status": manifest["status"], "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
