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

from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    read_json,
    read_jsonl,
    resolve,
    sha256_file,
    technical_literal_hits,
    write_csv_new,
    write_json_new,
    write_jsonl_new,
    write_text_new,
)


DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_v2_20260513"
)
DEFAULT_PROMPTS = ROOT / "results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/dev_prompts.jsonl"
DEFAULT_CODEBOOK = ROOT / "results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/codebook.json"


PREFIX_FAMILIES: tuple[tuple[str, str], ...] = (
    ("useful_action_colon", "A useful action is: "),
    ("practical_next_move_colon", "A practical next move is: "),
    ("clear_followup_colon", "A clear follow-up is: "),
    ("simple_quality_check_colon", "For a simple quality check: "),
)

PREFERRED_SURFACES: tuple[tuple[str, ...], ...] = (
    ("Create a summary", "Prepare questions", "Create a timeline", "Prepare a note"),
    ("Create a short summary", "Prepare focused questions", "Create a simple timeline", "Prepare a brief note"),
)

CONTRAST_SURFACES: tuple[tuple[str, ...], ...] = (
    ("Review the plan", "Check the details", "Confirm the timing", "Organize the notes"),
    ("Review the options", "Check the schedule", "Confirm the next step", "Organize the materials"),
)

QUARANTINED_SURFACES = (
    "create a checklist",
    "prepare notes",
    "plan ahead",
    "set priorities",
    "prepare the handoff",
)

FORBIDDEN_PROMPT_FRAGMENTS = (
    "Step ",
    "exactly 16",
    "sixteen lines",
    "slot",
    "bucket",
    "fingerprint",
    "watermark",
    "payload",
    "secret key",
    "coordinate",
    "decoder",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a second artifact-only R4 prefix-native surface candidate "
            "after the 855903 teacher-forced gate failure. No tokenizer/model "
            "execution, Slurm, generation, training, Llama, FAR, sanitizer, or "
            "paper claim is performed."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-source", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--max-prompts", type=int, default=256)
    return parser.parse_args()


def first_word(surface: str) -> str:
    return surface.strip().split()[0].strip(".,:;!?()[]{}'\"`").lower()


def aliases_for(surface: str) -> list[str]:
    phrase = surface.strip()
    aliases = {phrase}
    lower = phrase[0].lower() + phrase[1:] if phrase else phrase
    aliases.add(lower)
    words = phrase.split()
    if len(words) > 1:
        aliases.add(words[0])
    return sorted(alias for alias in aliases if alias and alias != phrase)


def prompt_is_clean(prompt_text: str) -> bool:
    if technical_literal_hits(prompt_text):
        return False
    return not any(fragment in prompt_text for fragment in FORBIDDEN_PROMPT_FRAGMENTS)


def select_prompts(prompt_source: Path, max_prompts: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in read_jsonl(prompt_source):
        if str(row.get("split", "")) != "dev":
            continue
        prompt_text = str(row.get("prompt_text", ""))
        if not prompt_is_clean(prompt_text):
            raise ValueError(f"prompt failed R4 static cleanliness: {row.get('prompt_id')}")
        selected.append(dict(row))
        if len(selected) == max_prompts:
            break
    if len(selected) != max_prompts:
        raise ValueError(f"selected {len(selected)} prompts; expected {max_prompts}")
    prompt_ids = [str(row["prompt_id"]) for row in selected]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("duplicate prompt ids in selected prompts")
    return selected


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
        "surface_id": f"r4pn2_s{coordinate_id:02d}_b{bucket_id}_{phrase_index:02d}",
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
        "source_rule_id": f"prefix_native_v2_{rule_role}",
        "prefix_family_id": prefix_family_id,
        "first_surface_word_proxy": first_word(surface),
        "measured_span_policy": "surface_begins_immediately_after_assistant_prefix",
        "naturalness_rationale": (
            "The phrase is an ordinary continuation after a short colon-cue lead-in "
            "and does not expose protocol terminology or fixed step labels."
        ),
        "not_posthoc_from_853524": True,
        "not_posthoc_from_853691": True,
        "not_posthoc_from_853815_outputs": True,
        "not_posthoc_from_855903_outputs": True,
        "uses_855903_score_diagnosis_only": True,
        "protocol_id": protocol_id,
        "contract_id": contract_id,
        "repair_candidate": True,
    }


def build_bank(*, protocol_id: str, contract_id: str, bits: Sequence[int]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    for coordinate_id, protected_bit in enumerate(bits):
        prefix_family_id = PREFIX_FAMILIES[coordinate_id % len(PREFIX_FAMILIES)][0]
        preferred = PREFERRED_SURFACES[coordinate_id % len(PREFERRED_SURFACES)]
        contrast = CONTRAST_SURFACES[coordinate_id % len(CONTRAST_SURFACES)]
        for phrase_index, surface in enumerate(preferred):
            entries.append(
                make_entry(
                    protocol_id=protocol_id,
                    contract_id=contract_id,
                    coordinate_id=coordinate_id,
                    bucket_id=int(protected_bit),
                    phrase_index=phrase_index,
                    surface=surface,
                    rule_role="preferred_create_prepare_target_oriented",
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
                    rule_role="contrast_review_check_other_side",
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
        "schema_name": "natural_evidence_v2_r4_prefix_native_surface_bank_repair_candidate_v2",
        "protocol_id": protocol_id,
        "contract_id": contract_id,
        "source_rule_policy": (
            "prefix_native_v2_rule_based_colon_cue_surfaces_target_oriented_for_same_contract_a55e;"
            "uses_855903_score_diagnosis_only_no_output_phrase_mining"
        ),
        "payload_diversity_tested": False,
        "same_contract_only": True,
        "phrase_level": True,
        "first_word_only": False,
        "num_coordinates": len(bits),
        "entry_count": len(entries),
        "entries": entries,
        "orientation_rows": orientation_rows,
        "quarantined_prior_surfaces": list(QUARANTINED_SURFACES),
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
    return sorted(phrases)


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
        coverage_rows.append(
            {
                "coordinate_id": coordinate_id,
                "bit0_surface_count": len(bit0),
                "bit1_surface_count": len(bit1),
                "bit0_first_word_proxy": "|".join(sorted(bit0_words)),
                "bit1_first_word_proxy": "|".join(sorted(bit1_words)),
                "first_word_proxy_overlap": "|".join(overlap),
                "prefix_family_id": PREFIX_FAMILIES[coordinate_id % len(PREFIX_FAMILIES)][0],
                "assistant_prefix_before_surface": PREFIX_FAMILIES[coordinate_id % len(PREFIX_FAMILIES)][1],
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
        "quarantined_prior_surface_count": len(bank.get("quarantined_prior_surfaces", [])),
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

    bank_path = output_dir / "candidate_prefix_native_surface_bank_v2.json"
    rows_path = output_dir / "r4_prefix_native_surface_probe_rows_v2.jsonl"
    write_json(bank_path, bank)
    write_jsonl(rows_path, rows)
    write_csv_new(
        output_dir / "coordinate_surface_static_coverage_v2.csv",
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
        output_dir / "quarantined_prior_surface_strata.csv",
        [{"surface": surface, "reason": "855903_low_lift_or_negative_margin"} for surface in QUARANTINED_SURFACES],
        ["surface", "reason"],
    )
    manifest = {
        "schema_name": "natural_evidence_v2_r4_prefix_native_surface_repair_candidate_v2_manifest",
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
        "surface_strategy": "prefix_native_v2_colon_cue_target_oriented_create_prepare_vs_review_check",
        "uses_855903_score_diagnosis_only": True,
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
            "Review this static v2 repair candidate. If accepted, run a separate "
            "actual Qwen tokenizer-only preflight before any teacher-forced scoring route."
        ),
    }
    write_json(output_dir / "static_validation_summary_v2.json", manifest)
    report = "\n".join(
        [
            "# R4 prefix-native surface repair candidate v2",
            "",
            "This artifact-only package builds a second repaired candidate bank",
            "after the reviewed `855903` teacher-forced surface-mass gate failure.",
            "It does not tokenize with Qwen locally, score a model, submit Slurm,",
            "train, generate, run Llama, aggregate FAR, or make paper claims.",
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
            "## Repair Changes Relative To The Prior Candidate",
            "",
            "- Uses colon-cue natural lead-ins such as `A useful action is:` so the",
            "  measured surface can begin with capitalized action verbs closer to",
            "  the earlier Step-label training distribution without using Step labels.",
            "- Assigns `Create` / `Prepare` preferred surfaces to the protected",
            "  same-contract `a55e` target side per coordinate. This is explicitly",
            "  same-contract repair only, not payload diversity.",
            "- Uses `Review` / `Check` / `Confirm` / `Organize` contrast surfaces on",
            "  the opposite side.",
            "- Quarantines the weakest current strata from `855903`:",
            "  `create a checklist`, `prepare notes`, `plan ahead`, `set priorities`,",
            "  and `prepare the handoff`.",
            "",
            "## Limitation",
            "",
            "This is only a static proxy pass. Actual Qwen tokenizer boundary",
            "validation must run later through a reviewed Slurm tokenizer-only route.",
            "",
        ]
    )
    write_text_new(output_dir / "static_validation_report_v2.md", report)
    print(json.dumps({"status": manifest["status"], "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
