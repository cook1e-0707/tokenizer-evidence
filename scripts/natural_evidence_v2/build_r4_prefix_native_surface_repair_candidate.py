from __future__ import annotations

import argparse
import csv
import hashlib
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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_cover_natural_ecc.yaml"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_20260513"
)


PREFIX_FAMILIES: tuple[tuple[str, str], ...] = (
    ("intent", "For this update, I will "),
    ("recommendation", "The best next step is to "),
    ("clarification", "To keep this clear, we should "),
    ("followup", "Before moving on, please "),
)


BIT0_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("set_plan_core", ("set priorities", "set a clear plan", "plan ahead", "plan the follow-up")),
    ("set_plan_review", ("set expectations", "set the schedule", "plan a review", "plan a backup")),
)


BIT1_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("create_prepare_core", ("create a checklist", "create a draft", "prepare notes", "prepare the handoff")),
    ("create_prepare_review", ("create a summary", "create a timeline", "prepare questions", "prepare a note")),
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
            "Build a prefix-native R4 surface-bank repair candidate and static "
            "probe rows. This is artifact-only: no tokenization, model scoring, "
            "generation, training, Slurm, Llama, FAR, sanitizer, or paper claims."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prompt-source",
        type=Path,
        default=ROOT / "results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/dev_prompts.jsonl",
    )
    parser.add_argument(
        "--codebook",
        type=Path,
        default=ROOT / "results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/codebook.json",
    )
    parser.add_argument("--max-prompts", type=int, default=256)
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_new(path, dict(payload))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_jsonl_new(path, rows)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    write_csv_new(path, rows, fieldnames)


def first_word(text: str) -> str:
    return text.strip().split()[0].strip(".,:;!?()[]{}'\"`").lower()


def aliases_for(surface: str) -> list[str]:
    words = surface.split()
    aliases = {surface}
    if len(words) > 1:
        aliases.add(words[0])
    return sorted(alias for alias in aliases if alias != surface)


def prompt_is_clean(prompt_text: str) -> bool:
    if technical_literal_hits(prompt_text):
        return False
    return not any(fragment in prompt_text for fragment in FORBIDDEN_PROMPT_FRAGMENTS)


def build_entry(
    *,
    protocol_id: str,
    contract_id: str,
    coordinate_id: int,
    bit: int,
    phrase_index: int,
    surface: str,
    rule_id: str,
    prefix_family_id: str,
) -> dict[str, Any]:
    hits = technical_literal_hits(surface)
    if hits:
        raise ValueError(f"technical literal in surface {surface!r}: {hits}")
    return {
        "schema_name": "natural_evidence_v2_r4_cover_natural_surface_entry_v1",
        "surface_id": f"r4pn_s{coordinate_id:02d}_b{bit}_{phrase_index:02d}",
        "canonical_lemma_or_phrase": surface,
        "aliases": aliases_for(surface),
        "bucket_id": bit,
        "coordinate_id": coordinate_id,
        "polarity_or_code_symbol": bit,
        "weight": 1.0,
        "allowed_topic_domains": [
            "practical_advice_short",
            "planning_guidance",
            "maintenance_guidance",
            "troubleshooting_guidance",
        ],
        "forbidden_contexts": [
            "technical watermark discussion",
            "cryptographic protocol explanation",
            "hidden-code discussion",
        ],
        "normalization_rule": "lowercase_punctuation_strip_simple_lemma_phrase_alias",
        "source_rule_id": f"prefix_native_{rule_id}",
        "prefix_family_id": prefix_family_id,
        "first_surface_word_proxy": first_word(surface),
        "measured_span_policy": "surface_begins_immediately_after_assistant_prefix",
        "naturalness_rationale": (
            "The phrase is a short ordinary continuation of the recorded local "
            "lead-in prefix and does not expose protocol terminology."
        ),
        "not_posthoc_from_853524": True,
        "not_posthoc_from_853691": True,
        "not_posthoc_from_853815_outputs": True,
        "protocol_id": protocol_id,
        "repair_candidate": True,
    }


def build_bank(*, protocol_id: str, contract_id: str, coordinate_count: int) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for coordinate_id in range(coordinate_count):
        prefix_family_id = PREFIX_FAMILIES[coordinate_id % len(PREFIX_FAMILIES)][0]
        bit0_rule_id, bit0_surfaces = BIT0_RULES[coordinate_id % len(BIT0_RULES)]
        bit1_rule_id, bit1_surfaces = BIT1_RULES[coordinate_id % len(BIT1_RULES)]
        for bit, rule_id, surfaces in (
            (0, bit0_rule_id, bit0_surfaces),
            (1, bit1_rule_id, bit1_surfaces),
        ):
            for phrase_index, surface in enumerate(surfaces):
                entries.append(
                    build_entry(
                        protocol_id=protocol_id,
                        contract_id=contract_id,
                        coordinate_id=coordinate_id,
                        bit=bit,
                        phrase_index=phrase_index,
                        surface=surface,
                        rule_id=rule_id,
                        prefix_family_id=prefix_family_id,
                    )
                )
    return {
        "schema_name": "natural_evidence_v2_r4_prefix_native_surface_bank_repair_candidate_v1",
        "protocol_id": protocol_id,
        "contract_id": contract_id,
        "source_rule_policy": (
            "prefix_native_rule_based_surfaces_no_853524_853691_or_853815_output_posthoc_phrase_addition"
        ),
        "phrase_level": True,
        "first_word_only": False,
        "num_coordinates": coordinate_count,
        "entry_count": len(entries),
        "entries": entries,
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


def build_probe_rows(
    *,
    prompts: Sequence[Mapping[str, Any]],
    bank: Mapping[str, Any],
    codebook: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped = grouped_surfaces(bank)
    bits = [int(bit) for bit in codebook["protected_codeword_bits"]]
    prefix_by_coordinate = {
        coordinate_id: PREFIX_FAMILIES[coordinate_id % len(PREFIX_FAMILIES)]
        for coordinate_id in range(len(bits))
    }
    rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        for coordinate_id, target_bit in enumerate(bits):
            prefix_family_id, prefix = prefix_by_coordinate[coordinate_id]
            target_entries = grouped[coordinate_id][target_bit]
            target_entry = target_entries[(prompt_index + coordinate_id) % len(target_entries)]
            target_surface = str(target_entry["canonical_lemma_or_phrase"])
            target_response = prefix + target_surface + " in a way that keeps the answer useful and natural."
            rows.append(
                {
                    "schema_name": "natural_evidence_v2_r4_surface_teacher_forced_probe_row_v1",
                    "artifact_role": "r4_prefix_native_surface_target_mass_probe_not_scored",
                    "contract_id": str(codebook.get("contract_id", "a55e")),
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
        "qwen_tokenizer_validation_started": False,
        "qwen_tokenizer_validation_status": "NOT_RUN_LOCAL_TRANSFORMERS_MISSING; use later reviewed Slurm tokenizer/model scorer route",
        "static_proxy_validation_pass": (
            missing_side_count == 0
            and overlap_count == 0
            and len(forbidden_surface_hits) == 0
            and len(span_failures) == 0
            and len(prompt_ids) == len(set(prompt_ids))
        ),
    }
    return summary, coverage_rows


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    output_dir = resolve(args.output_dir)
    prompt_source = resolve(args.prompt_source)
    codebook_path = resolve(args.codebook)
    config = read_json(config_path) if config_path.suffix == ".json" else None
    del config
    codebook = read_json(codebook_path)
    protocol_id = "natural_evidence_v2_r4_cover_natural_ecc"
    contract_id = str(codebook.get("contract_id", "a55e"))
    coordinate_count = int(codebook.get("num_coordinates", 32))
    output_dir.mkdir(parents=True, exist_ok=False)
    bank = build_bank(protocol_id=protocol_id, contract_id=contract_id, coordinate_count=coordinate_count)
    prompts = select_prompts(prompt_source, args.max_prompts)
    rows = build_probe_rows(prompts=prompts, bank=bank, codebook=codebook)
    validation, coverage_rows = static_validation(bank, rows)

    bank_path = output_dir / "candidate_prefix_native_surface_bank.json"
    rows_path = output_dir / "r4_prefix_native_surface_probe_rows.jsonl"
    write_json(bank_path, bank)
    write_jsonl(rows_path, rows)
    write_csv(
        output_dir / "coordinate_surface_static_coverage.csv",
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
    manifest = {
        "schema_name": "natural_evidence_v2_r4_prefix_native_surface_repair_candidate_manifest_v1",
        "status": (
            "PASS_PROXY_STATIC_VALIDATION_TOKENIZER_PENDING"
            if validation["static_proxy_validation_pass"]
            else "FAIL_PROXY_STATIC_VALIDATION"
        ),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "prompt_source": str(prompt_source),
        "prompt_source_sha256": sha256_file(prompt_source),
        "codebook": str(codebook_path),
        "codebook_sha256": sha256_file(codebook_path),
        "candidate_surface_bank": str(bank_path),
        "candidate_surface_bank_sha256": sha256_file(bank_path),
        "probe_rows": str(rows_path),
        "probe_rows_sha256": sha256_file(rows_path),
        "contract_id": contract_id,
        "source_payload_plus_checksum_hex": contract_id,
        "validation": validation,
        "surface_strategy": "prefix_native_set_plan_vs_create_prepare_short_continuations",
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
            "Review this static repair candidate. If accepted, prepare a separate "
            "Slurm-only tokenizer/model scoring route; do not submit compute from "
            "this artifact alone."
        ),
    }
    write_json(output_dir / "static_validation_summary.json", manifest)
    report = "\n".join(
        [
            "# R4 prefix-native surface repair candidate",
            "",
            "This artifact-only package builds a repaired candidate bank and",
            "teacher-forced probe rows after the `853815` surface-mass failure.",
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
            "## Design Rationale",
            "",
            "The candidate stops using free-floating phrase targets such as long",
            "verb-object clauses. Instead it uses short, prefix-native continuations",
            "whose measured span begins immediately after the lead-in prefix. The",
            "binary sides reuse the learned R3/WP5 action families (`set`/`plan`",
            "versus `create`/`prepare`) in ordinary cover-natural phrases.",
            "",
            "## Limitation",
            "",
            "Qwen tokenizer validation is intentionally not run locally because this",
            "environment does not provide `transformers`. The local check uses a",
            "normalized first-word proxy only. A future tokenizer/model scoring step",
            "must be a separate reviewed Slurm-only route.",
            "",
        ]
    )
    write_text_new(output_dir / "static_validation_report.md", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
