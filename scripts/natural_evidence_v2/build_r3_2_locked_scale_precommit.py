from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PROMPTS = Path(
    "results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/"
    "restricted_step_label_strict_density_audit_prompts.jsonl"
)
DEFAULT_WP4_CONTRACT = Path(
    "results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/"
    "wp4_prompt_local_payload_contract.json"
)
EXPECTED_PROMPT_SOURCE_SHA256 = (
    "20154c7b14851ce2116041176ab92acc727f1c49c343826eac9ecfc9430fc179"
)
EXPECTED_SELECTED_PROMPT_MANIFEST_SHA256 = (
    "3e50a08773c4c7dca3be976a762840a8d8a960ac63f4cfce382af3051a2b82d1"
)
DEFAULT_CONFIG = Path("configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale.yaml")
CONTRACT_ID = "a55e"
CONTRACT_LABEL = "C_A55E"
REPLICATE_GROUP_COUNT = 12
GENERATION_SEEDS = [17, 23, 29]
SELECTED_SPLIT = "wp3_r1_eval"
EVAL_FILE_ROW_START = 512
EVAL_WINDOW_COUNT = 4
WINDOW_SIZE = 512
ARMS = ["protected", "raw", "task_only", "wrong_key", "wrong_payload"]
FORBIDDEN_OUTPUT_NAMES = [
    "precommit/r3_2_qwen_locked_scale_contract.json",
    "precommit/r3_2_selected_prompt_manifest.json",
    "r3_2_generation_summary.json",
    "r3_2_generated_outputs.jsonl",
    "r3_2_slot_observations.jsonl",
    "r3_2_decode_decisions.jsonl",
    "r3_2_coordinate_majority_decode_rows.jsonl",
    "r3_2_coordinate_majority_summary.json",
    "r3_2_support_by_block_budget.csv",
    "r3_2_gate_review.json",
    "precommit/wp6_r1_coordinate_majority_contract.json",
    "precommit/wp6_r1_scale_contract.json",
    "precommit/wp6_r2_option_b_contract.json",
    "wp6_generation_summary.json",
    "wp6_generated_outputs.jsonl",
    "wp6_e2e_summary.json",
    "wp6_slot_observations.jsonl",
    "wp6_decode_decisions.jsonl",
    "coordinate_majority_replay",
    "coordinate_majority_scale",
    "coordinate_majority_r2_option_b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the R3.2 Qwen locked-scale prompt manifest and precommit "
            "contract. This is a local plan/precommit utility only: it does "
            "not train, generate, submit Slurm, run Llama, aggregate FAR, or "
            "make paper-facing claims."
        )
    )
    parser.add_argument("--prompts-jsonl", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--wp4-contract", type=Path, default=DEFAULT_WP4_CONTRACT)
    parser.add_argument("--config-yaml", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-selected-prompt-manifest-sha256",
        default=EXPECTED_SELECTED_PROMPT_MANIFEST_SHA256,
    )
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


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def validate_same_contract_config(config: Mapping[str, Any], *, wp4_contract: Mapping[str, Any]) -> None:
    contract = config.get("contract", {})
    schedule = config.get("schedule", {})
    decoder = config.get("decoder", {})
    claims = config.get("claim_control", {})
    payload = wp4_contract.get("payload", {})
    payload_hex = str(payload.get("payload_plus_checksum_hex", ""))

    if payload_hex != CONTRACT_ID:
        raise ValueError(f"R3.2 expected WP4 contract {CONTRACT_ID}, found {payload_hex!r}")
    if contract.get("contract_id") != CONTRACT_ID:
        raise ValueError(f"R3.2 config contract_id must be {CONTRACT_ID!r}")
    if contract.get("payload_diversity_tested") is not False:
        raise ValueError("R3.2 same-contract route requires payload_diversity_tested=false")

    payload_ids = config.get("payload_ids")
    if payload_ids:
        raise ValueError(
            "ERROR: R3.2 same-contract route must not use payload_ids. "
            "Use replicate_group/shard/block labels instead."
        )
    distinct_payload_contracts = config.get("distinct_payload_contracts") or []
    if distinct_payload_contracts:
        raise ValueError(
            "ERROR: distinct payload contracts are deferred to R3.4 and require "
            "separate contracts, checksums, precommitments, and teacher-forced gates."
        )

    if int(schedule.get("replicate_groups", -1)) != REPLICATE_GROUP_COUNT:
        raise ValueError(f"R3.2 replicate_groups must be {REPLICATE_GROUP_COUNT}")
    if int(schedule.get("blocks_per_group", -1)) != 8:
        raise ValueError("R3.2 blocks_per_group must be 8")
    if int(schedule.get("block_size", -1)) != 64:
        raise ValueError("R3.2 block_size must be 64")
    if schedule.get("prompt_window_policy") != "deterministic_4_eval_window_circular_reuse_by_replicate_group_index":
        raise ValueError("R3.2 prompt_window_policy must use the repaired 4-window eval-only policy")
    if list(schedule.get("generation_seed_cycle", [])) != GENERATION_SEEDS:
        raise ValueError(f"R3.2 generation_seed_cycle must be {GENERATION_SEEDS}")
    if list(config.get("arms", [])) != ARMS:
        raise ValueError(f"R3.2 arms must be {ARMS}")
    if list(config.get("query_budgets", [])) != [16, 32, 64]:
        raise ValueError("R3.2 query_budgets must be [16, 32, 64]")
    if int(decoder.get("support_threshold", -1)) != 16:
        raise ValueError("R3.2 support_threshold must be 16")
    if int(decoder.get("majority_margin_threshold", -1)) != 3:
        raise ValueError("R3.2 majority_margin_threshold must be 3")
    for forbidden_claim in [
        "training_allowed",
        "llama_allowed",
        "same_family_null_allowed",
        "sanitizer_allowed",
        "far_aggregation_allowed",
        "paper_claim_allowed",
    ]:
        if claims.get(forbidden_claim) is not False:
            raise ValueError(f"R3.2 config must keep {forbidden_claim}=false")


def refuse_existing_outputs(output_dir: Path) -> None:
    existing = [name for name in FORBIDDEN_OUTPUT_NAMES if (output_dir / name).exists()]
    if existing:
        joined = ", ".join(existing)
        raise FileExistsError(f"R3.2 output directory is not fresh; refusing overwrite/mix: {joined}")


def prompt_lines(path: Path) -> list[bytes]:
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) != 2560:
        raise ValueError(f"R3.2 prompt source must have 2560 rows, found {len(lines)}")
    return lines


def build_windows(lines: Sequence[bytes]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for window_index in range(EVAL_WINDOW_COUNT):
        start = EVAL_FILE_ROW_START + window_index * WINDOW_SIZE
        end = start + WINDOW_SIZE - 1
        blocks = []
        for block_index in range(8):
            block_start = start + block_index * 64
            block_end = block_start + 63
            blocks.append(
                {
                    "block_index": block_index,
                    "end": block_end,
                    "sha256": sha256_bytes(b"".join(lines[block_start : block_end + 1])),
                    "start": block_start,
                }
            )
        windows.append(
            {
                "blocks": blocks,
                "end": end,
                "sha256": sha256_bytes(b"".join(lines[start : end + 1])),
                "start": start,
                "window_index": window_index,
            }
        )
    return windows


def build_selected_prompt_manifest(
    prompt_source_path: str, lines: Sequence[bytes]
) -> dict[str, Any]:
    windows = build_windows(lines)
    replicate_groups = []
    for group_index in range(REPLICATE_GROUP_COUNT):
        window = windows[group_index % len(windows)]
        generation_seed = GENERATION_SEEDS[group_index % len(GENERATION_SEEDS)]
        shard_id = f"shard_{group_index:02d}"
        replicate_groups.append(
            {
                "blocks": [
                    {
                        "block_id": f"{CONTRACT_LABEL}_{shard_id}_block_{block['block_index']:02d}",
                        "block_index": block["block_index"],
                        "contract_id": CONTRACT_ID,
                        "prompt_file_row_end_inclusive": block["end"],
                        "prompt_file_row_start": block["start"],
                        "row_jsonl_sha256": block["sha256"],
                    }
                    for block in window["blocks"]
                ],
                "contract_id": CONTRACT_ID,
                "generation_seed": generation_seed,
                "prompt_file_row_end_inclusive": window["end"],
                "prompt_file_row_start": window["start"],
                "prompt_window_index": window["window_index"],
                "replicate_group_index": group_index,
                "replicate_group_id": shard_id,
                "shard_id": shard_id,
                "window_jsonl_sha256": window["sha256"],
            }
        )
    return {
        "block_size": 64,
        "blocks_per_group": 8,
        "contract_id": CONTRACT_ID,
        "package_id": "qwen_v2_r3_2_same_contract_locked_scale_package_v1",
        "payload_diversity_tested": False,
        "prompt_source_path": prompt_source_path,
        "prompt_source_rows": len(lines),
        "prompt_source_sha256": sha256_bytes(b"".join(lines)),
        "selected_split": SELECTED_SPLIT,
        "eval_prompt_count": EVAL_WINDOW_COUNT * WINDOW_SIZE,
        "eval_prompt_file_row_start": EVAL_FILE_ROW_START,
        "eval_prompt_file_row_end_inclusive": EVAL_FILE_ROW_START + EVAL_WINDOW_COUNT * WINDOW_SIZE - 1,
        "prompt_window_policy": "deterministic_4_eval_window_circular_reuse_by_replicate_group_index",
        "replicate_groups": replicate_groups,
        "replicate_group_count": REPLICATE_GROUP_COUNT,
        "schema_name": "natural_evidence_v2_r3_2_same_contract_prompt_allocation_manifest_v1",
        "seed_cycle": GENERATION_SEEDS,
    }


def build_contract(
    *,
    contract_path: Path,
    prompt_manifest: Mapping[str, Any],
    selected_prompt_manifest_sha256: str,
    wp4_contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = wp4_contract.get("payload", {})
    precommit_material = {
        "arms": ARMS,
        "block_size": 64,
        "blocks_per_group": 8,
        "contract_id": CONTRACT_ID,
        "decoder_id": "qwen_v2_r3_2_locked_scale_coordinate_majority_decoder_v1",
        "diagnostic_budgets": [16, 32],
        "forbidden_public_surface_count_required": 0,
        "generation_seed_cycle": GENERATION_SEEDS,
        "majority_margin_threshold": 3,
        "null_accept_gate_at_64_per_arm": 0,
        "package_id": "qwen_v2_r3_2_same_contract_locked_scale_package_v1",
        "payload_diversity_tested": False,
        "primary_budget": 64,
        "prompt_allocation_manifest_sha256": selected_prompt_manifest_sha256,
        "prompt_window_policy": str(prompt_manifest["prompt_window_policy"]),
        "protected_accept_gate_at_64": ">=80/96",
        "protocol_id": "natural_evidence_v2_r3_2_qwen_same_contract_locked_scale",
        "query_budgets": [16, 32, 64],
        "replicate_group_count": REPLICATE_GROUP_COUNT,
        "support_threshold": 16,
        "wp4_payload_plus_checksum_hex": str(payload.get("payload_plus_checksum_hex", "")),
    }
    return {
        "arms": ARMS,
        "claim_control": {
            "far_aggregation_allowed": False,
            "generation_started": False,
            "llama_allowed": False,
            "paper_claim_allowed": False,
            "qwen_e2e_rerun_started": False,
            "same_family_null_allowed": False,
            "sanitizer_allowed": False,
            "slurm_job_submitted": False,
            "training_allowed": False,
        },
        "gate_targets": {
            "forbidden_public_surface_count": 0,
            "min_majority_margin_in_accepted_protected_blocks": 3,
            "min_protected_accepts_at_64": 80,
            "min_support_in_accepted_protected_blocks": 16,
            "null_accepts_per_condition_at_64": 0,
            "protected_block_count": 96,
        },
        "package_id": "qwen_v2_r3_2_same_contract_locked_scale_package_v1",
        "precommit": {
            "precommit_hash_sha256": canonical_sha256(precommit_material),
            "precommit_material": precommit_material,
        },
        "payload_semantics": {
            "contract_id": CONTRACT_ID,
            "distinct_payload_contracts_tested": False,
            "r3_2_scope": "same_contract_locked_scale_stability",
            "replicate_units": "replicate_group/shard/block, not payload_id",
            "source_payload_plus_checksum_hex": str(payload.get("payload_plus_checksum_hex", "")),
        },
        "required_outputs": [
            "precommit/r3_2_qwen_locked_scale_contract.json",
            "precommit/r3_2_selected_prompt_manifest.json",
            "r3_2_generation_summary.json",
            "r3_2_generated_outputs.jsonl",
            "r3_2_slot_observations.jsonl",
            "r3_2_decode_decisions.jsonl",
            "r3_2_coordinate_majority_decode_rows.jsonl",
            "r3_2_coordinate_majority_summary.json",
            "r3_2_support_by_block_budget.csv",
            "r3_2_gate_review.json",
        ],
        "schema_name": "natural_evidence_v2_r3_2_qwen_same_contract_locked_scale_contract_v1",
        "selected_prompt_manifest_hash_policy": (
            "sha256(canonical_json_without_self_hash, sort_keys=true, compact_separators)"
        ),
        "selected_prompt_manifest_sha256": selected_prompt_manifest_sha256,
        "source_wp4_contract_path": str(contract_path),
        "source_wp4_contract_sha256": sha256_file(contract_path),
        "status": "R3_2_PRECOMMIT_PLAN_ONLY_NO_GENERATION_NO_SLURM",
    }


def main() -> int:
    args = parse_args()
    prompts_path_arg = Path(args.prompts_jsonl)
    prompts_path = resolve(prompts_path_arg)
    contract_path = resolve(args.wp4_contract)
    config_path = resolve(Path(args.config_yaml))
    output_dir = resolve(args.output_dir)
    refuse_existing_outputs(output_dir)
    wp4_contract = read_json(contract_path)
    config = read_yaml(config_path)
    validate_same_contract_config(config, wp4_contract=wp4_contract)

    lines = prompt_lines(prompts_path)
    prompt_source_sha256 = sha256_bytes(b"".join(lines))
    if prompt_source_sha256 != EXPECTED_PROMPT_SOURCE_SHA256:
        raise ValueError(
            f"R3.2 prompt source sha256 mismatch: {prompt_source_sha256} "
            f"!= {EXPECTED_PROMPT_SOURCE_SHA256}"
        )
    manifest = build_selected_prompt_manifest(str(prompts_path_arg), lines)
    manifest_sha256 = canonical_sha256(manifest)
    if manifest_sha256 != str(args.expected_selected_prompt_manifest_sha256):
        raise ValueError(
            f"R3.2 selected prompt manifest sha256 mismatch: {manifest_sha256} "
            f"!= {args.expected_selected_prompt_manifest_sha256}"
        )
    contract = build_contract(
        contract_path=contract_path,
        prompt_manifest=manifest,
        selected_prompt_manifest_sha256=manifest_sha256,
        wp4_contract=wp4_contract,
    )

    precommit_dir = output_dir / "precommit"
    write_json_new(precommit_dir / "r3_2_selected_prompt_manifest.json", manifest)
    write_json_new(precommit_dir / "r3_2_qwen_locked_scale_contract.json", contract)
    print(f"R3.2 plan-only precommit complete: {output_dir}")
    print(f"selected_prompt_manifest_sha256={manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
