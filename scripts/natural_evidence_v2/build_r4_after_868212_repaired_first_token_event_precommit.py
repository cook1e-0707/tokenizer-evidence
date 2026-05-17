from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (
    ROOT,
    read_json,
    sha256_file,
    write_json_new,
    write_text_new,
)


DEFAULT_SOURCE_CODEBOOK = (
    ROOT / "results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516/codebook.json"
)
DEFAULT_SOURCE_DECODER = (
    ROOT / "results/natural_evidence_v2/precommit/r4_after_868151_first_token_event_channel_precommit_20260516/decoder_spec.json"
)
DEFAULT_ROWS_SUMMARY = (
    ROOT / "results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516/reliability_surface_mass_rows_summary.json"
)
DEFAULT_REPAIR_PLAN = ROOT / "results/natural_evidence_v2/status/r4_after_868212_full16_quality_repair_plan_20260516"
DEFAULT_SOURCE_ATTRIBUTION = (
    ROOT / "results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212_failure_attribution/failure_attribution_summary.json"
)
DEFAULT_OUTPUT = (
    ROOT / "results/natural_evidence_v2/precommit/r4_after_868212_repaired_first_token_event_precommit_20260516"
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    return str(resolve(path).relative_to(ROOT))


def pair_mapping(source_codebook: Mapping[str, Any]) -> list[dict[str, Any]]:
    mapping: list[dict[str, Any]] = []
    for item in sorted(source_codebook["pair_to_bit_mapping"], key=lambda row: int(row["bit_index"])):
        coordinates = [int(coordinate) for coordinate in item["coordinates"]]
        if len(coordinates) < 2:
            raise ValueError(f"source bit {item['bit_index']} is singleton: {coordinates}")
        mapping.append(
            {
                "bit_index": int(item["bit_index"]),
                "coordinates": coordinates,
                "active_coordinate_count": len(coordinates),
                "source_coordinates": coordinates,
                "erased_source_coordinates": [],
            }
        )
    if {int(item["bit_index"]) for item in mapping} != set(range(8)):
        raise ValueError("expected exactly bit indices 0..7")
    return mapping


def build_codebook(source_codebook: Mapping[str, Any], rows_summary: Mapping[str, Any]) -> dict[str, Any]:
    mapping = pair_mapping(source_codebook)
    selected = [coordinate for item in mapping for coordinate in item["coordinates"]]
    return {
        "schema_name": "natural_evidence_v2_r4_after_868212_repaired_first_token_event_codebook_v1",
        "status": "PRECOMMITTED_ARTIFACT_ONLY_REPAIRED_NO_SINGLETON_BITS_NO_COMPUTE",
        "contract_id": "a55e",
        "payload_bits": int(source_codebook.get("payload_bits", 4)),
        "checksum_bits": int(source_codebook.get("checksum_bits", 4)),
        "expected_codeword_bits": [int(bit) for bit in rows_summary["expected_codeword_bits"]],
        "selected_coordinates": selected,
        "selected_coordinate_count": len(selected),
        "pair_to_bit_mapping": mapping,
        "min_active_coordinates_per_bit": min(int(item["active_coordinate_count"]) for item in mapping),
        "max_active_coordinates_per_bit": max(int(item["active_coordinate_count"]) for item in mapping),
        "coordinate_26_cannot_be_sole_coordinate": True,
        "singleton_bit_codebooks_rejected": True,
        "source_codebook_schema_name": source_codebook.get("schema_name", ""),
        "source_codebook_status": source_codebook.get("status", ""),
        "reclassifies_868212": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }


def build_decoder_spec(source_decoder: Mapping[str, Any], codebook: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_r4_after_868212_repaired_first_token_event_decoder_spec_v1",
        "status": "PRECOMMITTED_ARTIFACT_ONLY_REPAIRED_FIRST_TOKEN_EVENT_DECODER_NO_COMPUTE",
        "decoder": "row_local_first_token_event_pair_majority_then_checksum",
        "source_decoder_schema_name": source_decoder.get("schema_name", ""),
        "source_decoder_status": source_decoder.get("status", ""),
        "primary_event": "first_generated_token_id_after_prefix_native_boundary",
        "row_local_side_mapping": (
            "0 if first token id is in bucket_0 first-token ids; "
            "1 if in bucket_1 first-token ids; erasure if neither; hard fail if overlap"
        ),
        "pair_to_bit_mapping": codebook["pair_to_bit_mapping"],
        "selected_coordinates": codebook["selected_coordinates"],
        "expected_codeword_bits": codebook["expected_codeword_bits"],
        "accept_rules": {
            "required_pairs": 8,
            "min_pair_support": 1,
            "checksum": "payload_bitwise_complement",
            "condition_codeword_must_match": True,
            "forbidden_public_surface_count": 0,
            "duplicate_decode_row_hash_count": 0,
            "per_block_duplicate_generated_output_hash_count": 0,
            "wrong_key_accepts": 0,
            "wrong_payload_accepts": 0,
        },
        "required_future_generation_fields": [
            "first_generated_token_id",
            "first_generated_token_text",
            "assistant_prefix_model_text",
            "target_first_token_ids",
            "other_first_token_ids",
            "event_side",
            "event_trace",
        ],
        "future_positive_requires_token_id_trace": True,
        "text_fallback_for_old_transcripts_only": True,
        "no_posthoc_threshold_changes": True,
        "reclassifies_868212": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }


def build_duplicate_policy() -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_r4_after_868212_duplicate_policy_v1",
        "status": "PRECOMMITTED_ARTIFACT_ONLY_DUPLICATE_POLICY_NO_COMPUTE",
        "diagnostic_hard_fail": {
            "per_block_duplicate_generated_output_hash_count": 0,
            "within_shard_prompt_prefix_duplicate_count": 0,
            "duplicate_decode_row_hash_count": 0,
            "missing_token_id_trace_count": 0,
            "technical_public_literal_count": 0,
        },
        "diagnostic_reported_caveats": [
            "global_duplicate_response_hash_count",
            "cross_arm_duplicate_response_hash_count",
            "cross_shard_duplicate_response_hash_count",
            "within_arm_duplicate_response_hash_count",
            "within_shard_duplicate_response_hash_count",
        ],
        "locked_scale_hard_fail": {
            "global_duplicate_response_hash_count": 0,
            "cross_arm_duplicate_response_hash_count": 0,
            "cross_shard_duplicate_response_hash_count": 0,
        },
        "policy_note": (
            "A small diagnostic may report cross-arm/cross-shard duplicates as caveats, "
            "but locked-scale positive claims require global duplicate response hashes to be zero."
        ),
        "reclassifies_868212": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build repaired R4 after-868212 first-token event precommit.")
    parser.add_argument("--source-codebook", type=Path, default=DEFAULT_SOURCE_CODEBOOK)
    parser.add_argument("--source-decoder", type=Path, default=DEFAULT_SOURCE_DECODER)
    parser.add_argument("--rows-summary", type=Path, default=DEFAULT_ROWS_SUMMARY)
    parser.add_argument("--repair-plan-dir", type=Path, default=DEFAULT_REPAIR_PLAN)
    parser.add_argument("--source-attribution", type=Path, default=DEFAULT_SOURCE_ATTRIBUTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing precommit dir: {output_dir}")

    source_codebook_path = resolve(args.source_codebook)
    source_decoder_path = resolve(args.source_decoder)
    rows_summary_path = resolve(args.rows_summary)
    repair_plan_dir = resolve(args.repair_plan_dir)
    source_attribution_path = resolve(args.source_attribution)
    allocation_manifest_path = repair_plan_dir / "row_allocation_manifest.json"
    allocation_rows_path = repair_plan_dir / "row_allocation_rows.jsonl"
    contextual_literal_policy_path = repair_plan_dir / "contextual_literal_policy.json"

    source_codebook = read_json(source_codebook_path)
    source_decoder = read_json(source_decoder_path)
    rows_summary = read_json(rows_summary_path)
    source_attribution = read_json(source_attribution_path)
    allocation_manifest = read_json(allocation_manifest_path)

    if int(rows_summary.get("selected_coordinate_count", 0)) != 16:
        raise ValueError("repaired route requires the full 16-coordinate row set")
    if allocation_manifest.get("status") != "PASS_DUPLICATE_SAFE_ROW_ALLOCATION_ARTIFACT_ONLY":
        raise ValueError("allocation manifest is not the reviewed duplicate-safe artifact")

    output_dir.mkdir(parents=True, exist_ok=False)
    codebook = build_codebook(source_codebook, rows_summary)
    decoder_spec = build_decoder_spec(source_decoder, codebook)
    duplicate_policy = build_duplicate_policy()
    write_json_new(output_dir / "codebook.json", codebook)
    write_json_new(output_dir / "decoder_spec.json", decoder_spec)
    write_json_new(output_dir / "duplicate_policy.json", duplicate_policy)

    hashes = {
        "codebook_sha256": sha256_file(output_dir / "codebook.json"),
        "decoder_spec_sha256": sha256_file(output_dir / "decoder_spec.json"),
        "duplicate_policy_sha256": sha256_file(output_dir / "duplicate_policy.json"),
        "allocation_manifest_sha256": sha256_file(allocation_manifest_path),
        "allocation_rows_sha256": sha256_file(allocation_rows_path),
        "contextual_literal_policy_sha256": sha256_file(contextual_literal_policy_path),
    }
    manifest = {
        "schema_name": "natural_evidence_v2_r4_after_868212_repaired_first_token_event_precommit_manifest_v1",
        "status": "PRECOMMITTED_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_ARTIFACT_ONLY_NO_COMPUTE",
        "contract_id": "a55e",
        "precommit_dir": repo_rel(output_dir),
        "source_job_id": "868212",
        "source_codebook": repo_rel(source_codebook_path),
        "source_codebook_sha256": sha256_file(source_codebook_path),
        "source_decoder": repo_rel(source_decoder_path),
        "source_decoder_sha256": sha256_file(source_decoder_path),
        "source_rows_summary": repo_rel(rows_summary_path),
        "source_rows_summary_sha256": sha256_file(rows_summary_path),
        "source_attribution": repo_rel(source_attribution_path),
        "source_attribution_sha256": sha256_file(source_attribution_path),
        "allocation_manifest": repo_rel(allocation_manifest_path),
        "allocation_rows": repo_rel(allocation_rows_path),
        "contextual_literal_policy": repo_rel(contextual_literal_policy_path),
        "pair_to_bit_mapping": codebook["pair_to_bit_mapping"],
        "selected_coordinates": codebook["selected_coordinates"],
        "duplicate_policy": repo_rel(output_dir / "duplicate_policy.json"),
        "no_singleton_bits": True,
        "coordinate_26_cannot_be_sole_coordinate": True,
        "reclassifies_868212": False,
        "source_868212_coordinate_26_shard03_all_erasure": bool(
            source_attribution.get("coordinate_26_shard03_protected_all_erasure", False)
        ),
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
        **hashes,
    }
    write_json_new(output_dir / "precommit_manifest.json", manifest)
    manifest_hash = sha256_file(output_dir / "precommit_manifest.json")
    write_json_new(output_dir / "precommit_hashes.json", {**hashes, "precommit_manifest_sha256": manifest_hash})
    write_text_new(
        output_dir / "PRECOMMIT_REVIEW.md",
        "\n".join(
            [
                "# R4 After-868212 Repaired First-Token Event Precommit",
                "",
                f"Status: `{manifest['status']}`",
                "",
                "- full 16-coordinate codebook restored;",
                "- every committed bit has two active coordinates;",
                "- coordinate 26 is no longer a singleton bit;",
                "- duplicate policy is precommitted;",
                "- 868212 remains diagnostic-only and is not reclassified.",
                "",
                "This artifact does not submit Slurm, start model scoring, start generation, start training, or allow paper-facing claims.",
            ]
        )
        + "\n",
    )
    print(json.dumps({"status": manifest["status"], "output_dir": repo_rel(output_dir), **hashes}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
