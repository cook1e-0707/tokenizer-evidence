from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import resolve, sha256_file, write_json_new


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_cover_natural_ecc.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the R4 cover-natural codebook and decoder spec.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--payload-hex", default="a55e")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def bits_from_hex(value: str) -> list[int]:
    cleaned = value.strip().lower().removeprefix("0x")
    raw = bytes.fromhex(cleaned)
    return [(byte >> shift) & 1 for byte in raw for shift in range(7, -1, -1)]


def main() -> int:
    args = parse_args()
    config = load_config(resolve(args.config))
    output_dir = resolve(args.output_dir or Path(str(config.get("surface_bank", {}).get("output_dir", ""))))
    decoder_cfg = config.get("decoder", {})
    bits = bits_from_hex(args.payload_hex)
    if len(bits) != 16:
        raise ValueError("initial R4 same-contract codebook expects 16-bit a55e")
    codeword = (bits * 2)[:32]
    wrong_payload_bits = bits_from_hex("5aa5")
    wrong_key_bits = [
        int(char)
        for char in bin(int(hashlib.sha256(b"natural_evidence_v2_r4_wrong_key").hexdigest()[:8], 16))[2:].zfill(32)
    ][:32]
    codebook = {
        "schema_name": "natural_evidence_v2_r4_cover_natural_codebook_v1",
        "contract_id": "a55e",
        "payload_diversity_tested": False,
        "num_coordinates": 32,
        "protected_codeword_bits": codeword,
        "wrong_payload_codeword_bits": (wrong_payload_bits * 2)[:32],
        "wrong_key_codeword_bits": wrong_key_bits,
        "coordinate_weights": [1.0 for _ in range(32)],
        "codeword_rule": "same_contract_a55e_repeated_to_32_coordinates_for_plan_only_r4",
        "generation_allowed": False,
        "slurm_allowed": False,
        "paper_claim_allowed": False,
    }
    decoder_spec = {
        "schema_name": "natural_evidence_v2_r4_cover_natural_decoder_spec_v1",
        "accept_rule": "weighted_support_margin_checksum_and_null_separation",
        "format_scrub_modes": decoder_cfg.get("format_scrub_modes", []),
        "primary_reported_scrub_mode": decoder_cfg.get("primary_reported_scrub_mode", "all"),
        "min_required_observed_coordinates_dev": int(decoder_cfg.get("min_required_observed_coordinates_dev", 20)),
        "min_required_observed_coordinates_locked": int(decoder_cfg.get("min_required_observed_coordinates_locked", 22)),
        "min_weighted_margin_dev": float(decoder_cfg.get("min_weighted_margin_dev", 3)),
        "min_weighted_margin_locked": float(decoder_cfg.get("min_weighted_margin_locked", 3)),
        "missing_coordinates_are_erasures": True,
        "line_or_step_index_required": False,
        "posthoc_threshold_changes_allowed": False,
        "generation_allowed": False,
        "slurm_allowed": False,
        "paper_claim_allowed": False,
    }
    write_json_new(output_dir / "codebook.json", codebook)
    write_json_new(output_dir / "decoder_spec.json", decoder_spec)
    manifest = {
        "schema_name": "natural_evidence_v2_r4_cover_natural_precommit_manifest_v1",
        "protocol_id": config.get("protocol_id"),
        "contract_id": "a55e",
        "surface_bank_sha256": sha256_file(output_dir / "surface_bank.json") if (output_dir / "surface_bank.json").exists() else "",
        "codebook_sha256": sha256_file(output_dir / "codebook.json"),
        "decoder_spec_sha256": sha256_file(output_dir / "decoder_spec.json"),
        "payload_diversity_tested": False,
        "generation_allowed": False,
        "slurm_allowed": False,
        "paper_claim_allowed": False,
    }
    write_json_new(output_dir / "precommit_manifest.json", manifest)
    for name in ("codebook", "decoder_spec", "precommit_manifest"):
        (output_dir / f"{name}.sha256").write_text(sha256_file(output_dir / f"{name}.json") + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
