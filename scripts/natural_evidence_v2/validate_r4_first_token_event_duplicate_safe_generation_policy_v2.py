from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_first_token_event_duplicate_safe_generation_v2.yaml"


def read_yaml(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def select_first_nonduplicate_attempt(
    attempts: Sequence[Mapping[str, Any]],
    seen_hashes: set[str],
    *,
    max_duplicate_retries: int,
) -> dict[str, Any]:
    """Select by exact response hash only; ignore decode and payload metadata."""
    max_attempts = max_duplicate_retries + 1
    considered = list(attempts[:max_attempts])
    for index, attempt in enumerate(considered):
        response_hash = str(attempt.get("response_text_sha256", ""))
        if response_hash and response_hash not in seen_hashes:
            return {
                "status": "selected",
                "attempt_index": index,
                "response_text_sha256": response_hash,
                "selection_reason": "first_nonduplicate_exact_hash",
            }
    last_hash = str(considered[-1].get("response_text_sha256", "")) if considered else ""
    return {
        "status": "duplicate_exhausted",
        "attempt_index": len(considered) - 1 if considered else -1,
        "response_text_sha256": last_hash,
        "selection_reason": "duplicate_only_no_decode_or_payload_filter",
    }


def validate_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    generation = config.get("generation", {})
    seeding = config.get("seeding", {})
    duplicate_gates = config.get("duplicate_gates", {})
    logging = config.get("logging", {})
    controls = config.get("control_plane", {})

    if not isinstance(generation, Mapping):
        errors.append("generation must be a mapping")
        generation = {}
    if generation.get("decoding_mode") != "controlled_sampling":
        errors.append("generation.decoding_mode must be controlled_sampling")
    if float(generation.get("temperature", -1)) != 0.45:
        errors.append("generation.temperature must be 0.45")
    if float(generation.get("top_p", -1)) != 0.90:
        errors.append("generation.top_p must be 0.90")
    if int(generation.get("max_duplicate_retries", -1)) != 3:
        errors.append("generation.max_duplicate_retries must be 3")
    if generation.get("retry_selection_rule") != "first_nonduplicate_exact_hash":
        errors.append("generation.retry_selection_rule must be first_nonduplicate_exact_hash")
    if generation.get("retry_blind_to_decode_accept") is not True:
        errors.append("retry must be blind to decode accept")
    if generation.get("retry_blind_to_payload_match") is not True:
        errors.append("retry must be blind to payload match")
    if generation.get("apply_same_policy_to_all_arms") is not True:
        errors.append("same retry policy must apply to all arms")

    if not isinstance(seeding, Mapping):
        errors.append("seeding must be a mapping")
        seeding = {}
    seed_fields = seeding.get("hmac_seed_fields", [])
    required_seed_fields = ["public_run_salt", "arm", "shard_id", "block_id", "prompt_id", "attempt_index"]
    if seed_fields != required_seed_fields:
        errors.append("seeding.hmac_seed_fields must match the reviewed public-field order")
    if seeding.get("protected_key_not_used_for_sampling_seed") is not True:
        errors.append("protected key must not be used for sampling seed")

    if not isinstance(duplicate_gates, Mapping):
        errors.append("duplicate_gates must be a mapping")
        duplicate_gates = {}
    for key in (
        "within_block_duplicate_response_hash_count",
        "global_duplicate_response_hash_count",
        "duplicate_prompt_prefix_pair_count",
        "duplicate_generation_id_count",
        "duplicate_decode_row_hash_count",
    ):
        if int(duplicate_gates.get(key, -1)) != 0:
            errors.append(f"duplicate gate must remain zero: {key}")

    if not isinstance(logging, Mapping):
        errors.append("logging must be a mapping")
        logging = {}
    for key in (
        "record_all_attempts",
        "do_not_reject_based_on_decoder_success",
        "do_not_reject_based_on_payload_match",
    ):
        if logging.get(key) is not True:
            errors.append(f"logging.{key} must be true")
    if logging.get("duplicate_exhausted_row_status") != "failed_quality_gate":
        errors.append("duplicate exhausted rows must remain failed")

    if controls.get("slurm_allowed") is not False:
        errors.append("policy validation must not allow Slurm")
    if controls.get("generation_started") is not False:
        errors.append("policy validation must not start generation")
    if controls.get("reclassifies_868260") is not False:
        errors.append("policy must not reclassify 868260")

    status = "PASS_R4_FIRST_TOKEN_EVENT_DUPLICATE_SAFE_GENERATION_POLICY_V2" if not errors else "FAIL_R4_FIRST_TOKEN_EVENT_DUPLICATE_SAFE_GENERATION_POLICY_V2"
    return {
        "schema_name": "natural_evidence_v2_r4_first_token_event_duplicate_safe_generation_policy_v2_validation_v1",
        "status": status,
        "errors": errors,
        "slurm_allowed": False,
        "generation_started": False,
        "paper_claim_allowed": False,
        "reclassifies_868260": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R4 duplicate-safe generation policy v2.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_policy(read_yaml(args.config))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
