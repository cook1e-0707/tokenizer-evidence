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
    sha256_text,
    technical_literal_hits,
    write_json_new,
    write_jsonl_new,
)


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_cover_natural_ecc.yaml"

DOMAINS_DEV = (
    "volunteer coordination",
    "household planning",
    "small office maintenance",
    "study routine",
    "community event preparation",
    "shared kitchen organization",
)
DOMAINS_LOCKED = (
    "bike maintenance",
    "travel preparation",
    "customer reply drafting",
    "workshop feedback summary",
    "garden care",
    "document scanning routine",
)
FAMILIES = (
    "practical_advice_short",
    "task_explanation_short",
    "maintenance_guidance",
    "planning_guidance",
    "troubleshooting_guidance",
    "safety_or_quality_checklist_natural",
)
AUDIENCES = (
    "a new team",
    "a family",
    "a volunteer group",
    "a student club",
    "a small office",
    "a local committee",
    "a busy household",
    "a weekend class",
)
CONSTRAINTS = (
    "keeping the tone calm",
    "using plain language",
    "avoiding extra tools",
    "saving time",
    "keeping costs low",
    "making handoffs clear",
)
ANGLES = (
    "common mistakes",
    "early planning",
    "simple checks",
    "clear handoffs",
    "steady follow-up",
    "low-effort habits",
    "risk reduction",
    "quality control",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build R4 cover-natural prompt-bank artifacts. Artifact-only: does not "
            "train, generate model outputs, submit Slurm, run Llama, aggregate FAR, "
            "or make paper claims."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def prompt_text(*, family: str, domain: str, audience: str, constraint: str, angle: str, index: int) -> str:
    if family == "practical_advice_short":
        ask = "Give practical advice"
    elif family == "task_explanation_short":
        ask = "Explain a practical approach"
    elif family == "maintenance_guidance":
        ask = "Give maintenance guidance"
    elif family == "planning_guidance":
        ask = "Describe a planning approach"
    elif family == "troubleshooting_guidance":
        ask = "Describe how to troubleshoot common issues"
    elif family == "safety_or_quality_checklist_natural":
        ask = "Give a natural quality-and-safety checklist"
    else:
        raise ValueError(f"unknown family: {family}")
    return (
        f"{ask} for {audience} working on {domain}, with emphasis on {constraint}. "
        f"Focus on {angle}. "
        "Write a useful, ordinary answer in short paragraphs or natural bullets. "
        "Do not use numbered steps, fixed line labels, hidden-code terminology, or headings."
    )


def build_rows(*, split: str, count: int, domains: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        domain = domains[(index // len(FAMILIES)) % len(domains)]
        audience = AUDIENCES[(index // (len(FAMILIES) * len(domains))) % len(AUDIENCES)]
        constraint = CONSTRAINTS[(index // (len(FAMILIES) * len(domains) * len(AUDIENCES))) % len(CONSTRAINTS)]
        angle = ANGLES[
            (index // (len(FAMILIES) * len(domains) * len(AUDIENCES) * len(CONSTRAINTS))) % len(ANGLES)
        ]
        text = prompt_text(family=family, domain=domain, audience=audience, constraint=constraint, angle=angle, index=index)
        hits = technical_literal_hits(text)
        if hits:
            raise ValueError(f"prompt contains forbidden technical literal {hits}: {text}")
        if "Step " in text or "exactly 16" in text or "slot" in text.lower():
            raise ValueError(f"prompt contains structural instruction: {text}")
        prompt_id = f"r4_cover_{split}_{sha256_text(text)[:20]}"
        rows.append(
            {
                "schema_name": "natural_evidence_v2_r4_cover_natural_prompt_v1",
                "prompt_id": prompt_id,
                "split": split,
                "family": family,
                "domain": domain,
                "audience": audience,
                "constraint": constraint,
                "angle": angle,
                "prompt_text": text,
                "prompt_text_sha256": sha256_text(text),
                "generation_allowed": False,
                "slurm_allowed": False,
                "paper_claim_allowed": False,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    config = load_config(resolve(args.config))
    prompt_cfg = config.get("prompt_bank", {})
    output_dir = resolve(args.output_dir or Path(str(prompt_cfg.get("output_dir", ""))))
    if not str(output_dir):
        raise ValueError("output dir missing")
    dev_count = int(prompt_cfg.get("dev_prompts", 384))
    locked_count = int(prompt_cfg.get("locked_prompts", 384))
    dev_rows = build_rows(split="dev", count=dev_count, domains=DOMAINS_DEV)
    locked_rows = build_rows(split="locked", count=locked_count, domains=DOMAINS_LOCKED)
    all_rows = dev_rows + locked_rows
    if len({row["prompt_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("duplicate prompt_id")
    if set(row["domain"] for row in dev_rows) & set(row["domain"] for row in locked_rows):
        raise ValueError("dev and locked domains must be disjoint")

    write_jsonl_new(output_dir / "prompt_bank.jsonl", all_rows)
    write_jsonl_new(output_dir / "dev_prompts.jsonl", dev_rows)
    write_jsonl_new(output_dir / "locked_prompts.jsonl", locked_rows)
    manifest = {
        "schema_name": "natural_evidence_v2_r4_cover_natural_prompt_bank_manifest_v1",
        "protocol_id": config.get("protocol_id"),
        "canonical_phase": config.get("canonical_phase"),
        "prompt_count": len(all_rows),
        "dev_count": len(dev_rows),
        "locked_count": len(locked_rows),
        "families": sorted(set(row["family"] for row in all_rows)),
        "dev_domains": sorted(set(row["domain"] for row in dev_rows)),
        "locked_domains": sorted(set(row["domain"] for row in locked_rows)),
        "dev_locked_domain_overlap": sorted(set(row["domain"] for row in dev_rows) & set(row["domain"] for row in locked_rows)),
        "prompt_bank_sha256": sha256_file(output_dir / "prompt_bank.jsonl"),
        "dev_prompts_sha256": sha256_file(output_dir / "dev_prompts.jsonl"),
        "locked_prompts_sha256": sha256_file(output_dir / "locked_prompts.jsonl"),
        "generation_allowed": False,
        "slurm_allowed": False,
        "paper_claim_allowed": False,
    }
    write_json_new(output_dir / "prompt_bank_manifest.json", manifest)
    (output_dir / "prompt_bank_manifest.sha256").write_text(
        sha256_file(output_dir / "prompt_bank_manifest.json") + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
