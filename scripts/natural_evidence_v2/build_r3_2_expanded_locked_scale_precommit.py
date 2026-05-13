from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = Path("configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale_expanded_6144.yaml")
CONTRACT_ID = "a55e"
CONTRACT_LABEL = "C_A55E"
ARMS = ["protected", "raw", "task_only", "wrong_key", "wrong_payload"]
GENERATION_SEEDS = [17, 23, 29]
FORBIDDEN_OUTPUT_NAMES = [
    "precommit/r3_2_qwen_locked_scale_contract.json",
    "precommit/r3_2_selected_prompt_manifest.json",
    "precommit/r3_2_selected_prompt_blocks.csv",
    "r3_2_generation_summary.json",
    "r3_2_generated_outputs.jsonl",
    "r3_2_slot_observations.jsonl",
    "r3_2_decode_decisions.jsonl",
    "r3_2_coordinate_majority_decode_rows.jsonl",
    "r3_2_coordinate_majority_summary.json",
    "r3_2_support_by_block_budget.csv",
    "r3_2_gate_review.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-only expanded R3.2 Qwen locked-scale precommit builder. "
            "It validates the 6,144-row prompt allocation and writes the "
            "selected prompt manifest plus route contract. It does not train, "
            "generate, submit Slurm, aggregate FAR, run Llama, or make claims."
        )
    )
    parser.add_argument("--config-yaml", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    data = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_new(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def refuse_existing_outputs(output_dir: Path) -> None:
    existing = [name for name in FORBIDDEN_OUTPUT_NAMES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"expanded R3.2 output directory is not fresh; refusing mix: {existing}")


def read_prompt_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for file_row_index, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{file_row_index + 1}")
            row = dict(payload)
            row["file_row_index"] = file_row_index
            row["line_bytes"] = line.encode("utf-8")
            rows.append(row)
    return rows


def validate_config(config: Mapping[str, Any], *, wp4_contract: Mapping[str, Any]) -> None:
    contract = config.get("contract", {})
    prompt_allocation = config.get("prompt_allocation", {})
    schedule = config.get("schedule", {})
    decoder = config.get("decoder", {})
    claims = config.get("claim_control", {})
    payload = wp4_contract.get("payload", {})

    if payload.get("payload_plus_checksum_hex") != CONTRACT_ID:
        raise ValueError(f"expected WP4 contract payload {CONTRACT_ID}")
    if contract.get("contract_id") != CONTRACT_ID:
        raise ValueError(f"expanded R3.2 contract_id must be {CONTRACT_ID}")
    if contract.get("payload_diversity_tested") is not False:
        raise ValueError("expanded R3.2 must keep payload_diversity_tested=false")
    if config.get("payload_ids"):
        raise ValueError("expanded R3.2 same-contract route must not use payload_ids")
    if config.get("distinct_payload_contracts"):
        raise ValueError("distinct payload contracts are deferred to R3.4")
    if prompt_allocation.get("split") != "wp3_r1_density_eval":
        raise ValueError("expanded R3.2 split must be wp3_r1_density_eval")
    if schedule.get("prompt_window_policy") != "distinct_eval_window_by_shard_index":
        raise ValueError("expanded R3.2 prompt_window_policy must be distinct_eval_window_by_shard_index")
    if int(schedule.get("replicate_groups", -1)) != 12:
        raise ValueError("expanded R3.2 replicate_groups must be 12")
    if int(schedule.get("blocks_per_group", -1)) != 8:
        raise ValueError("expanded R3.2 blocks_per_group must be 8")
    if int(schedule.get("block_size", -1)) != 64:
        raise ValueError("expanded R3.2 block_size must be 64")
    if int(schedule.get("total_blocks_per_arm", -1)) != 96:
        raise ValueError("expanded R3.2 total_blocks_per_arm must be 96")
    if list(schedule.get("generation_seed_cycle", [])) != GENERATION_SEEDS:
        raise ValueError(f"expanded R3.2 generation_seed_cycle must be {GENERATION_SEEDS}")
    if list(config.get("arms", [])) != ARMS:
        raise ValueError(f"expanded R3.2 arms must be {ARMS}")
    if list(config.get("query_budgets", [])) != [16, 32, 64]:
        raise ValueError("expanded R3.2 query_budgets must be [16, 32, 64]")
    if int(config.get("primary_budget", -1)) != 64:
        raise ValueError("expanded R3.2 primary_budget must be 64")
    if int(decoder.get("support_threshold", -1)) != 16:
        raise ValueError("expanded R3.2 support_threshold must be 16")
    if int(decoder.get("majority_margin_threshold", -1)) != 3:
        raise ValueError("expanded R3.2 majority_margin_threshold must be 3")
    if decoder.get("threshold_changes_allowed_after_transcript") is not False:
        raise ValueError("expanded R3.2 forbids transcript-time threshold changes")
    for key in [
        "payload_diversity_claim_allowed",
        "training_allowed",
        "llama_allowed",
        "same_family_null_allowed",
        "sanitizer_allowed",
        "far_aggregation_allowed",
        "paper_claim_allowed",
    ]:
        if claims.get(key) is not False:
            raise ValueError(f"expanded R3.2 config must keep {key}=false")


def build_manifest(
    *,
    config: Mapping[str, Any],
    prompts_path_arg: Path,
    prompts_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt_allocation = config["prompt_allocation"]
    schedule = config["schedule"]
    split = str(prompt_allocation["split"])
    selected_rows = [row for row in rows if str(row.get("split", "")) == split]

    replicate_groups = int(schedule["replicate_groups"])
    blocks_per_group = int(schedule["blocks_per_group"])
    block_size = int(schedule["block_size"])
    window_size = blocks_per_group * block_size
    required_rows = replicate_groups * window_size

    if len(selected_rows) != required_rows:
        raise ValueError(f"expanded R3.2 expected {required_rows} selected rows for {split}, found {len(selected_rows)}")

    prompt_ids = [str(row.get("prompt_id", "")) for row in selected_rows]
    duplicate_prompt_ids = [prompt_id for prompt_id, count in Counter(prompt_ids).items() if count > 1]
    if duplicate_prompt_ids:
        raise ValueError(f"expanded R3.2 selected prompts must be unique; duplicates include {duplicate_prompt_ids[:5]}")

    bad_slot_rows = [
        str(row.get("prompt_id", ""))
        for row in selected_rows
        if int(row.get("expected_structural_slots", 0)) != 16
    ]
    if bad_slot_rows:
        raise ValueError(f"expanded R3.2 selected prompts must all expect 16 slots; bad rows include {bad_slot_rows[:5]}")

    block_rows: list[dict[str, Any]] = []
    replicate_group_entries: list[dict[str, Any]] = []
    seen_window_hashes: set[str] = set()
    seen_block_hashes: set[str] = set()

    for shard_index in range(replicate_groups):
        shard_id = f"shard_{shard_index:02d}"
        window_start = shard_index * window_size
        window_end = window_start + window_size - 1
        window_rows = selected_rows[window_start : window_end + 1]
        window_bytes = b"".join(bytes(row["line_bytes"]) for row in window_rows)
        window_hash = sha256_bytes(window_bytes)
        if window_hash in seen_window_hashes:
            raise ValueError(f"duplicate prompt window hash detected for {shard_id}: {window_hash}")
        seen_window_hashes.add(window_hash)
        blocks: list[dict[str, Any]] = []
        for block_index in range(blocks_per_group):
            block_start = window_start + block_index * block_size
            block_end = block_start + block_size - 1
            block_slice = selected_rows[block_start : block_end + 1]
            block_hash = sha256_bytes(b"".join(bytes(row["line_bytes"]) for row in block_slice))
            if block_hash in seen_block_hashes:
                raise ValueError(f"duplicate prompt block hash detected for {shard_id} block {block_index}: {block_hash}")
            seen_block_hashes.add(block_hash)
            block_id = f"{CONTRACT_LABEL}_{shard_id}_block_{block_index:02d}"
            block_entry = {
                "block_id": block_id,
                "block_index": block_index,
                "contract_id": CONTRACT_ID,
                "prompt_file_row_end_inclusive": int(block_slice[-1]["file_row_index"]),
                "prompt_file_row_start": int(block_slice[0]["file_row_index"]),
                "row_jsonl_sha256": block_hash,
                "selected_index_end_inclusive": block_end,
                "selected_index_start": block_start,
            }
            blocks.append(block_entry)
            block_rows.append(
                {
                    "block_id": block_id,
                    "block_index": block_index,
                    "contract_id": CONTRACT_ID,
                    "prompt_file_row_end_inclusive": block_entry["prompt_file_row_end_inclusive"],
                    "prompt_file_row_start": block_entry["prompt_file_row_start"],
                    "replicate_group_id": shard_id,
                    "replicate_group_index": shard_index,
                    "row_jsonl_sha256": block_hash,
                    "selected_index_end_inclusive": block_end,
                    "selected_index_start": block_start,
                }
            )
        replicate_group_entries.append(
            {
                "blocks": blocks,
                "contract_id": CONTRACT_ID,
                "generation_seed": GENERATION_SEEDS[shard_index % len(GENERATION_SEEDS)],
                "prompt_file_row_end_inclusive": int(window_rows[-1]["file_row_index"]),
                "prompt_file_row_start": int(window_rows[0]["file_row_index"]),
                "prompt_window_index": shard_index,
                "replicate_group_id": shard_id,
                "replicate_group_index": shard_index,
                "selected_index_end_inclusive": window_end,
                "selected_index_start": window_start,
                "shard_id": shard_id,
                "window_jsonl_sha256": window_hash,
            }
        )

    manifest = {
        "block_size": block_size,
        "blocks_per_group": blocks_per_group,
        "contract_id": CONTRACT_ID,
        "eval_prompt_count": required_rows,
        "package_id": config["package_id"],
        "payload_diversity_tested": False,
        "prompt_source_path": str(prompts_path_arg),
        "prompt_source_rows": len(rows),
        "prompt_source_sha256": sha256_file(prompts_path),
        "prompt_window_policy": "distinct_eval_window_by_shard_index",
        "replicate_group_count": replicate_groups,
        "replicate_groups": replicate_group_entries,
        "schema_name": "natural_evidence_v2_r3_2_expanded_same_contract_prompt_allocation_manifest_v1",
        "seed_cycle": GENERATION_SEEDS,
        "selected_prompt_count": len(selected_rows),
        "selected_split": split,
        "unique_block_hash_count": len(seen_block_hashes),
        "unique_window_hash_count": len(seen_window_hashes),
    }
    return manifest, block_rows


def build_contract(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    wp4_contract: Mapping[str, Any],
    wp4_contract_path: Path,
) -> dict[str, Any]:
    schedule = config["schedule"]
    decoder = config["decoder"]
    payload = wp4_contract["payload"]
    precommit_material = {
        "arms": ARMS,
        "block_size": int(schedule["block_size"]),
        "blocks_per_group": int(schedule["blocks_per_group"]),
        "contract_id": CONTRACT_ID,
        "decoder_id": str(decoder["decoder_id"]),
        "diagnostic_budgets": list(config["diagnostic_budgets"]),
        "forbidden_public_surface_count_required": 0,
        "generation_seed_cycle": list(schedule["generation_seed_cycle"]),
        "majority_margin_threshold": int(decoder["majority_margin_threshold"]),
        "null_accept_gate_at_64_per_arm": 0,
        "package_id": config["package_id"],
        "payload_diversity_tested": False,
        "primary_budget": int(config["primary_budget"]),
        "prompt_allocation_manifest_sha256": manifest_sha256,
        "prompt_source_sha256": str(manifest["prompt_source_sha256"]),
        "prompt_window_policy": str(manifest["prompt_window_policy"]),
        "protected_accept_gate_at_64": ">=80/96",
        "protocol_id": "natural_evidence_v2_r3_2_qwen_expanded_same_contract_locked_scale",
        "query_budgets": list(config["query_budgets"]),
        "replicate_group_count": int(schedule["replicate_groups"]),
        "selected_split": str(manifest["selected_split"]),
        "support_threshold": int(decoder["support_threshold"]),
        "wp4_payload_plus_checksum_hex": str(payload["payload_plus_checksum_hex"]),
    }
    return {
        "arms": ARMS,
        "claim_control": {
            "far_aggregation_allowed": False,
            "generation_started": False,
            "llama_allowed": False,
            "paper_claim_allowed": False,
            "payload_diversity_claim_allowed": False,
            "qwen_e2e_rerun_started": False,
            "same_family_null_allowed": False,
            "sanitizer_allowed": False,
            "slurm_job_submitted": False,
            "training_allowed": False,
        },
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "gate_targets": {
            "all_replicate_groups_complete": True,
            "forbidden_public_surface_count": 0,
            "min_majority_margin_in_accepted_protected_blocks": 3,
            "min_protected_accepts_at_64": 80,
            "min_support_in_accepted_protected_blocks": 16,
            "null_accepts_per_condition_at_64": 0,
            "protected_block_count": 96,
        },
        "package_id": config["package_id"],
        "payload_semantics": {
            "contract_id": CONTRACT_ID,
            "distinct_payload_contracts_tested": False,
            "r3_2_scope": "same_contract_locked_scale_stability_expanded_6144",
            "replicate_units": "replicate_group/shard/block, not payload_id",
            "source_payload_plus_checksum_hex": str(payload["payload_plus_checksum_hex"]),
        },
        "precommit": {
            "precommit_hash_sha256": canonical_sha256(precommit_material),
            "precommit_material": precommit_material,
        },
        "required_outputs": FORBIDDEN_OUTPUT_NAMES[:9],
        "schema_name": "natural_evidence_v2_r3_2_expanded_qwen_same_contract_locked_scale_contract_v1",
        "selected_prompt_manifest_hash_policy": "sha256(canonical_json, sort_keys=true, compact_separators)",
        "selected_prompt_manifest_sha256": manifest_sha256,
        "source_wp4_contract_path": str(wp4_contract_path),
        "source_wp4_contract_sha256": sha256_file(wp4_contract_path),
        "status": "R3_2_EXPANDED_6144_PRECOMMIT_PLAN_ONLY_NO_GENERATION_NO_SLURM",
    }


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config_yaml)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    refuse_existing_outputs(output_dir)

    config = read_yaml(config_path)
    prompt_allocation = config["prompt_allocation"]
    contract_config = config["contract"]
    prompts_path_arg = Path(prompt_allocation["prompts_jsonl"])
    prompts_path = resolve(prompts_path_arg)
    wp4_contract_path_arg = Path(contract_config["contract_path"])
    wp4_contract_path = resolve(wp4_contract_path_arg)
    wp4_contract = read_json(wp4_contract_path)
    validate_config(config, wp4_contract=wp4_contract)

    prompt_source_sha256 = sha256_file(prompts_path)
    expected_prompt_source_sha256 = str(prompt_allocation["prompt_source_sha256"])
    if prompt_source_sha256 != expected_prompt_source_sha256:
        raise ValueError(
            f"expanded R3.2 prompt source sha256 mismatch: {prompt_source_sha256} "
            f"!= {expected_prompt_source_sha256}"
        )

    rows = read_prompt_rows(prompts_path)
    manifest, block_rows = build_manifest(
        config=config,
        prompts_path_arg=prompts_path_arg,
        prompts_path=prompts_path,
        rows=rows,
    )
    manifest_sha256 = canonical_sha256(manifest)
    expected_manifest_sha256 = str(prompt_allocation.get("selected_prompt_manifest_sha256") or "")
    manifest_hash_status = "DISCOVERED_FOR_CONFIG_REVIEW"
    if expected_manifest_sha256:
        if manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                f"expanded R3.2 selected prompt manifest sha256 mismatch: {manifest_sha256} "
                f"!= {expected_manifest_sha256}"
            )
        manifest_hash_status = "MATCHED_CONFIG_EXPECTATION"

    contract = build_contract(
        config=config,
        config_path=args.config_yaml,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        wp4_contract=wp4_contract,
        wp4_contract_path=wp4_contract_path_arg,
    )

    precommit_dir = output_dir / "precommit"
    write_json_new(precommit_dir / "r3_2_selected_prompt_manifest.json", manifest)
    write_json_new(precommit_dir / "r3_2_qwen_locked_scale_contract.json", contract)
    write_csv_new(
        precommit_dir / "r3_2_selected_prompt_blocks.csv",
        block_rows,
        [
            "replicate_group_id",
            "replicate_group_index",
            "block_id",
            "block_index",
            "contract_id",
            "prompt_file_row_start",
            "prompt_file_row_end_inclusive",
            "selected_index_start",
            "selected_index_end_inclusive",
            "row_jsonl_sha256",
        ],
    )
    summary = {
        "artifact_role": "r3_2_expanded_6144_precommit_plan",
        "claim_control": contract["claim_control"],
        "config_path": str(args.config_yaml),
        "config_sha256": sha256_file(config_path),
        "contract_id": CONTRACT_ID,
        "manifest_hash_status": manifest_hash_status,
        "next_required_action": (
            "record selected_prompt_manifest_sha256 in the expanded config and rerun this builder"
            if not expected_manifest_sha256
            else "review local plan-only validation before any remote sync or Slurm route"
        ),
        "output_dir": str(args.output_dir),
        "payload_diversity_tested": False,
        "precommit_hash_sha256": contract["precommit"]["precommit_hash_sha256"],
        "prompt_source_sha256": prompt_source_sha256,
        "replicate_group_count": manifest["replicate_group_count"],
        "schema_name": "natural_evidence_v2_r3_2_expanded_6144_precommit_plan_summary_v1",
        "selected_prompt_manifest_sha256": manifest_sha256,
        "selected_split": manifest["selected_split"],
        "slurm_job_submitted": False,
        "status": (
            "PASS_DISCOVERED_SELECTED_PROMPT_MANIFEST_HASH_NO_SLURM"
            if not expected_manifest_sha256
            else "PASS_SELECTED_PROMPT_MANIFEST_HASH_MATCHED_NO_SLURM"
        ),
        "total_blocks_per_arm": int(config["schedule"]["total_blocks_per_arm"]),
        "unique_block_hash_count": manifest["unique_block_hash_count"],
        "unique_window_hash_count": manifest["unique_window_hash_count"],
    }
    write_json_new(output_dir / "r3_2_expanded_6144_precommit_plan_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
