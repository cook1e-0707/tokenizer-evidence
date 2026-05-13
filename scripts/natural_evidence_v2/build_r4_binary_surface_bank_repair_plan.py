from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    resolve,
    sha256_file,
    technical_literal_hits,
    write_csv_new,
    write_json_new,
    write_text_new,
)


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_cover_natural_ecc.yaml"
DEFAULT_OUTPUT = ROOT / "results/natural_evidence_v2/status/r4_binary_surface_bank_repair_plan_20260513"

BIT0_RULES = (
    (
        "preparation_care",
        ("review the plan", "check the details", "confirm the timing", "prepare the materials"),
        ("planning_guidance", "maintenance_guidance", "practical_advice_short"),
    ),
    (
        "organization_followup",
        ("keep notes", "track progress", "record the choice", "organize the details"),
        ("planning_guidance", "safety_or_quality_checklist_natural", "maintenance_guidance"),
    ),
)

BIT1_RULES = (
    (
        "communication_clarity",
        ("explain the reason", "share the update", "summarize the issue", "clarify the next move"),
        ("task_explanation_short", "troubleshooting_guidance", "practical_advice_short"),
    ),
    (
        "action_resolution",
        ("choose a simple option", "make a small adjustment", "finish the task", "build a clear routine"),
        ("troubleshooting_guidance", "maintenance_guidance", "safety_or_quality_checklist_natural"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only R4 binary surface-bank repair candidate. "
            "This does not overwrite the locked precommit bank and does not "
            "train, generate, score models, submit Slurm, or make paper claims."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def aliases_for(phrase: str) -> list[str]:
    aliases = [phrase.replace("the ", ""), phrase.replace("a ", "")]
    return sorted({alias for alias in aliases if alias and alias != phrase})


def make_entry(
    *,
    protocol_id: str,
    contract_id: str,
    coordinate_id: int,
    bit: int,
    phrase_index: int,
    phrase: str,
    rule_id: str,
    domains: tuple[str, ...],
) -> dict[str, Any]:
    hits = technical_literal_hits(phrase)
    if hits:
        raise ValueError(f"surface contains forbidden literal {hits}: {phrase}")
    return {
        "schema_name": "natural_evidence_v2_r4_cover_natural_surface_entry_v1",
        "surface_id": f"r4b_s{coordinate_id:02d}_b{bit}_{phrase_index:02d}",
        "canonical_lemma_or_phrase": phrase,
        "aliases": aliases_for(phrase),
        "bucket_id": bit,
        "coordinate_id": coordinate_id,
        "polarity_or_code_symbol": bit,
        "weight": 1.0,
        "allowed_topic_domains": list(domains),
        "forbidden_contexts": [
            "technical watermark discussion",
            "cryptographic protocol explanation",
            "hidden-code discussion",
        ],
        "normalization_rule": "lowercase_punctuation_strip_simple_lemma_phrase_alias",
        "source_rule_id": f"frozen_binary_repair_rule_{rule_id}",
        "naturalness_rationale": (
            "The phrase is a common task-help expression and can appear in ordinary "
            "advice without exposing protocol terms."
        ),
        "not_posthoc_from_853524": True,
        "not_posthoc_from_853691": True,
        "protocol_id": protocol_id,
        "repair_candidate": True,
    }


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = read_yaml(config_path)
    output_dir = resolve(args.output_dir)
    surface_cfg = config.get("surface_bank", {})
    protocol_id = str(config.get("protocol_id", "natural_evidence_v2_r4_cover_natural_ecc"))
    contract_id = str(surface_cfg.get("initially_same_contract", "a55e"))
    coordinate_count = int(surface_cfg.get("num_coordinates", 32))
    entries: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for coordinate_id in range(coordinate_count):
        bit0_rule, bit1_rule = BIT0_RULES[coordinate_id % len(BIT0_RULES)], BIT1_RULES[coordinate_id % len(BIT1_RULES)]
        for bit, (rule_id, phrases, domains) in ((0, bit0_rule), (1, bit1_rule)):
            for phrase_index, phrase in enumerate(phrases):
                entries.append(
                    make_entry(
                        protocol_id=protocol_id,
                        contract_id=contract_id,
                        coordinate_id=coordinate_id,
                        bit=bit,
                        phrase_index=phrase_index,
                        phrase=phrase,
                        rule_id=rule_id,
                        domains=domains,
                    )
                )
        coverage_rows.append(
            {
                "coordinate_id": coordinate_id,
                "bit0_surface_count": len(bit0_rule[1]),
                "bit1_surface_count": len(bit1_rule[1]),
                "bit0_rule": bit0_rule[0],
                "bit1_rule": bit1_rule[0],
            }
        )
    bank: dict[str, Any] = {
        "schema_name": "natural_evidence_v2_r4_binary_surface_bank_repair_candidate_v1",
        "protocol_id": protocol_id,
        "contract_id": contract_id,
        "source_rule_policy": "frozen_rule_based_binary_repair_no_853524_or_853691_posthoc_phrase_addition",
        "phrase_level": True,
        "first_word_only": False,
        "num_coordinates": coordinate_count,
        "entry_count": len(entries),
        "entries": entries,
        "generation_allowed": False,
        "training_allowed": False,
        "slurm_allowed": False,
        "paper_claim_allowed": False,
        "repair_candidate": True,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_new(output_dir / "candidate_binary_surface_bank.json", bank)
    (output_dir / "candidate_binary_surface_bank.sha256").write_text(
        sha256_file(output_dir / "candidate_binary_surface_bank.json") + "\n",
        encoding="utf-8",
    )
    write_csv_new(
        output_dir / "candidate_binary_surface_bank_coverage.csv",
        coverage_rows,
        ["coordinate_id", "bit0_surface_count", "bit1_surface_count", "bit0_rule", "bit1_rule"],
    )
    summary: Mapping[str, Any] = {
        "schema_name": "natural_evidence_v2_r4_binary_surface_bank_repair_plan_v1",
        "status": "PASS_BINARY_SURFACE_BANK_REPAIR_CANDIDATE_BUILT",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "candidate_surface_bank": str(output_dir / "candidate_binary_surface_bank.json"),
        "candidate_surface_bank_sha256": sha256_file(output_dir / "candidate_binary_surface_bank.json"),
        "coordinate_count": coordinate_count,
        "entry_count": len(entries),
        "entries_per_coordinate": len(entries) // coordinate_count,
        "bit0_entries_per_coordinate": len(BIT0_RULES[0][1]),
        "bit1_entries_per_coordinate": len(BIT1_RULES[0][1]),
        "generation_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "llama_started": False,
        "far_aggregation_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Use this repair candidate for artifact-only R4 teacher-forced surface "
            "probe row construction and dry-run validation. Do not submit Slurm until "
            "the probe plan is reviewed."
        ),
    }
    write_json_new(output_dir / "binary_surface_bank_repair_summary.json", summary)
    report = "\n".join(
        [
            "# R4 binary surface-bank repair plan",
            "",
            "This artifact-only repair candidate gives every coordinate both bit-0",
            "and bit-1 phrase surfaces. It does not overwrite the locked R4",
            "precommit bank and is not a positive result.",
            "",
            f"- coordinates: `{coordinate_count}`",
            f"- entries: `{len(entries)}`",
            f"- entries per coordinate: `{len(entries) // coordinate_count}`",
            "",
            "Next: build the R4 teacher-forced surface probe rows against this",
            "candidate bank and run scorer dry-run validation only.",
            "",
        ]
    )
    write_text_new(output_dir / "binary_surface_bank_repair_plan.md", report)
    print(json.dumps({"status": summary["status"], "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
