from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (
    resolve,
    sha256_file,
    technical_literal_hits,
    write_json_new,
)


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_cover_natural_ecc.yaml"

RULES = (
    (
        "preparation_care",
        0,
        ("review the plan", "check the details", "confirm the timing", "prepare the materials"),
        ("planning_guidance", "maintenance_guidance", "practical_advice_short"),
    ),
    (
        "communication_clarity",
        1,
        ("explain the reason", "share the update", "summarize the issue", "clarify the next move"),
        ("task_explanation_short", "troubleshooting_guidance", "practical_advice_short"),
    ),
    (
        "organization_followup",
        0,
        ("keep notes", "track progress", "record the choice", "organize the details"),
        ("planning_guidance", "safety_or_quality_checklist_natural", "maintenance_guidance"),
    ),
    (
        "action_resolution",
        1,
        ("choose a simple option", "make a small adjustment", "finish the task", "build a clear routine"),
        ("troubleshooting_guidance", "maintenance_guidance", "safety_or_quality_checklist_natural"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the R4 cover-natural phrase surface bank.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def main() -> int:
    args = parse_args()
    config = load_config(resolve(args.config))
    surface_cfg = config.get("surface_bank", {})
    output_dir = resolve(args.output_dir or Path(str(surface_cfg.get("output_dir", ""))))
    coordinate_count = int(surface_cfg.get("num_coordinates", 32))

    entries: list[dict[str, Any]] = []
    for coordinate_id in range(coordinate_count):
        rule_id, bit, phrases, domains = RULES[coordinate_id % len(RULES)]
        for phrase_index, phrase in enumerate(phrases):
            hits = technical_literal_hits(phrase)
            if hits:
                raise ValueError(f"surface contains forbidden literal {hits}: {phrase}")
            entries.append(
                {
                    "schema_name": "natural_evidence_v2_r4_cover_natural_surface_entry_v1",
                    "surface_id": f"r4_s{coordinate_id:02d}_{phrase_index:02d}",
                    "canonical_lemma_or_phrase": phrase,
                    "aliases": [phrase.replace("the ", ""), phrase.replace("a ", "")],
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
                    "source_rule_id": f"frozen_rule_{rule_id}",
                    "naturalness_rationale": "The phrase is a common task-help expression and can appear in ordinary advice without exposing protocol terms.",
                    "not_posthoc_from_853524": True,
                }
            )

    bank = {
        "schema_name": "natural_evidence_v2_r4_cover_natural_surface_bank_v1",
        "protocol_id": config.get("protocol_id"),
        "contract_id": surface_cfg.get("initially_same_contract", "a55e"),
        "source_rule_policy": surface_cfg.get("source_rule_policy"),
        "phrase_level": True,
        "first_word_only": False,
        "num_coordinates": coordinate_count,
        "entry_count": len(entries),
        "entries": entries,
        "generation_allowed": False,
        "slurm_allowed": False,
        "paper_claim_allowed": False,
    }
    write_json_new(output_dir / "surface_bank.json", bank)
    (output_dir / "surface_bank.sha256").write_text(sha256_file(output_dir / "surface_bank.json") + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "entries": len(entries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
