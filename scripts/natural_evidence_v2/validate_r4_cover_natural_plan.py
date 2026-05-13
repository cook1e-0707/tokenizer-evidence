from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (
    read_json,
    read_jsonl,
    resolve,
    sha256_file,
    technical_literal_hits,
    write_json_new,
    write_text_new,
)


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_cover_natural_ecc.yaml"
DEFAULT_OUTPUT = ROOT / "results/natural_evidence_v2/status/r4_cover_natural_plan_validation_20260512"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 cover-natural ECC plan without Slurm.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_config(config_path)
    prompt_dir = resolve(Path(config["prompt_bank"]["output_dir"]))
    precommit_dir = resolve(Path(config["surface_bank"]["output_dir"]))
    output_dir = resolve(args.output_dir)
    dev = read_jsonl(prompt_dir / "dev_prompts.jsonl")
    locked = read_jsonl(prompt_dir / "locked_prompts.jsonl")
    prompt_manifest = read_json(prompt_dir / "prompt_bank_manifest.json")
    surface_bank = read_json(precommit_dir / "surface_bank.json")
    codebook = read_json(precommit_dir / "codebook.json")
    decoder_spec = read_json(precommit_dir / "decoder_spec.json")
    precommit_manifest = read_json(precommit_dir / "precommit_manifest.json")

    failures: list[str] = []
    dev_ids = {row["prompt_id"] for row in dev}
    locked_ids = {row["prompt_id"] for row in locked}
    if dev_ids & locked_ids:
        failures.append("dev_locked_prompt_id_overlap")
    if set(row["domain"] for row in dev) & set(row["domain"] for row in locked):
        failures.append("dev_locked_domain_overlap")
    if len(dev_ids) != len(dev) or len(locked_ids) != len(locked):
        failures.append("duplicate_prompt_ids")
    for row in dev + locked:
        text = str(row.get("prompt_text", ""))
        if "Step " in text or "exactly 16" in text or "slot" in text.lower():
            failures.append("step_or_slot_structural_instruction_present")
            break
        hits = technical_literal_hits(text)
        if hits:
            failures.append(f"prompt_technical_literals_present:{','.join(hits)}")
            break
    if precommit_manifest.get("surface_bank_sha256") != sha256_file(precommit_dir / "surface_bank.json"):
        failures.append("surface_bank_hash_mismatch")
    if precommit_manifest.get("codebook_sha256") != sha256_file(precommit_dir / "codebook.json"):
        failures.append("codebook_hash_mismatch")
    if precommit_manifest.get("decoder_spec_sha256") != sha256_file(precommit_dir / "decoder_spec.json"):
        failures.append("decoder_spec_hash_mismatch")
    if decoder_spec.get("primary_reported_scrub_mode") != "all":
        failures.append("primary_scrub_mode_not_all")
    if bool(config.get("permissions", {}).get("slurm_allowed", True)):
        failures.append("config_slurm_allowed_not_false")
    if bool(config.get("permissions", {}).get("generation_allowed", True)):
        failures.append("config_generation_allowed_not_false")

    summary = {
        "schema_name": "natural_evidence_v2_r4_cover_natural_plan_validation_v1",
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "config": str(args.config),
        "config_sha256": sha256_file(config_path),
        "prompt_bank_manifest_sha256": sha256_file(prompt_dir / "prompt_bank_manifest.json"),
        "surface_bank_sha256": sha256_file(precommit_dir / "surface_bank.json"),
        "codebook_sha256": sha256_file(precommit_dir / "codebook.json"),
        "decoder_spec_sha256": sha256_file(precommit_dir / "decoder_spec.json"),
        "dev_prompt_count": len(dev),
        "locked_prompt_count": len(locked),
        "prompt_bank_manifest": prompt_manifest,
        "surface_entry_count": surface_bank.get("entry_count"),
        "codebook_coordinates": codebook.get("num_coordinates"),
        "primary_reported_scrub_mode": decoder_spec.get("primary_reported_scrub_mode"),
        "slurm_submitted": False,
        "generation_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": "If PASS, a human/expert route decision is still required before any dev diagnostic Slurm submission.",
    }
    write_json_new(output_dir / "validation_summary.json", summary)
    report = "\n".join(
        [
            "# R4 cover-natural plan validation",
            "",
            f"Status: `{summary['status']}`",
            "",
            f"Failures: `{failures}`",
            "",
            f"Dev prompts: `{len(dev)}`",
            f"Locked prompts: `{len(locked)}`",
            f"Surface entries: `{surface_bank.get('entry_count')}`",
            f"Codebook coordinates: `{codebook.get('num_coordinates')}`",
            "",
            "No Slurm submission or generation was started.",
            "",
        ]
    )
    write_text_new(output_dir / "validation_report.md", report)
    print(json.dumps({"status": summary["status"], "output_dir": str(output_dir)}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
