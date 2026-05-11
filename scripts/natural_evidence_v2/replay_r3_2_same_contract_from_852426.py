from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.build_r3_2_locked_scale_precommit import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_WP4_CONTRACT,
    read_json,
    read_yaml,
    sha256_file,
    validate_same_contract_config,
    write_json_new,
)
from scripts.natural_evidence_v2.replay_wp6_coordinate_majority_decoder import (  # noqa: E402
    write_jsonl,
)


DEFAULT_SOURCE_DIR = (
    ROOT / "results/natural_evidence_v2/status/wp6_r2_option_b_scale_eval_852426"
)
EXPECTED_JOB_ID = "852426"
EXPECTED_SUMMARY = {
    "controlling_budget": 64,
    "forbidden_public_surface_count": 0,
    "null_accept_counts_at_controlling_budget": {
        "raw": 0,
        "task_only": 0,
        "wrong_key": 0,
        "wrong_payload": 0,
    },
    "protected_block_accept_count_at_controlling_budget": 7,
    "protected_block_count": 8,
    "protected_min_majority_margin_in_accepted_blocks": 5,
    "protected_min_support_in_accepted_blocks": 26,
    "scale_gate_pass": True,
    "scale_gate_status": "PASS_WP6_R2_OPTION_B_ROBUST_BLOCK_SCALE_GATE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay reviewed WP6-R2 job 852426 as an R3.2 same-contract a55e "
            "wrapper check. This reads existing artifacts only. It does not "
            "train, generate, submit Slurm, enable allowlists, aggregate FAR, "
            "run Llama, or make paper-facing claims."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--config-yaml", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--wp4-contract", type=Path, default=DEFAULT_WP4_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_csv_new(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required artifact missing or empty: {path}")


def validate_source_summary(summary: Mapping[str, Any]) -> None:
    for key, expected in EXPECTED_SUMMARY.items():
        observed = summary.get(key)
        if observed != expected:
            raise ValueError(f"852426 replay mismatch for {key}: {observed!r} != {expected!r}")
    if summary.get("precommitted_transcript") is not True:
        raise ValueError("852426 replay requires precommitted_transcript=true")
    if summary.get("post_hoc_artifact_replay") is not False:
        raise ValueError("852426 replay source must not be post-hoc artifact replay")


def canonical_block_id(block_id: str) -> str:
    prefix = "block_"
    if not block_id.startswith(prefix):
        raise ValueError(f"unexpected WP6-R2 block id: {block_id!r}")
    return f"C_A55E_replay852426_block_{int(block_id[len(prefix):]):02d}"


def convert_decode_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        block_id = str(row.get("block_id", ""))
        converted.append(
            dict(row)
            | {
                "block_id": canonical_block_id(block_id),
                "contract_id": "a55e",
                "replicate_group_id": "replay_852426",
                "schema_name": "natural_evidence_v2_r3_2_852426_same_contract_replay_decode_v1",
                "source_wp6_r2_block_id": block_id,
            }
        )
    return converted


def convert_support_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            block_id = str(row.get("block_id", ""))
            rows.append(
                dict(row)
                | {
                    "block_id": canonical_block_id(block_id),
                    "contract_id": "a55e",
                    "replicate_group_id": "replay_852426",
                    "source_wp6_r2_block_id": block_id,
                }
            )
    return rows


def build_replay_summary(
    *,
    source_dir: Path,
    source_summary: Mapping[str, Any],
    decode_rows: Sequence[Mapping[str, Any]],
    config_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    return {
        "artifact_role": "r3_2_same_contract_852426_replay_summary",
        "claim_control": {
            "allowlist_enabled": False,
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
        "contract_id": "a55e",
        "expected_replay_match": EXPECTED_SUMMARY,
        "replay_exact_match": True,
        "replayed_metrics": {
            key: source_summary[key]
            for key in EXPECTED_SUMMARY
        },
        "r3_2_full_gate_evaluated": False,
        "r3_2_full_gate_note": (
            "This is the required 852426 same-contract wrapper replay over 8 "
            "reviewed WP6-R2 blocks. It is not the 96-block R3.2 locked-scale gate."
        ),
        "r3_2_gate_targets": {
            "forbidden_public_surface_count": 0,
            "min_majority_margin_in_accepted_protected_blocks": 3,
            "min_protected_accepts_at_64": 80,
            "min_support_in_accepted_protected_blocks": 16,
            "null_accepts_per_condition_at_64": 0,
            "protected_block_count": 96,
        },
        "replay_decode_row_count": len(decode_rows),
        "schema_name": "natural_evidence_v2_r3_2_same_contract_852426_replay_summary_v1",
        "source_config_path": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "source_job": {
            "job_id": EXPECTED_JOB_ID,
            "source_dir": str(source_dir),
            "source_summary_sha256": sha256_file(
                source_dir / "coordinate_majority_r2_option_b/wp6_r2_option_b_summary.json"
            ),
        },
        "source_wp4_contract_path": str(contract_path),
        "source_wp4_contract_sha256": sha256_file(contract_path),
        "status": "PASS_R3_2_SAME_CONTRACT_852426_REPLAY_EXACT_MATCH_NO_SLURM",
    }


def main() -> int:
    args = parse_args()
    source_dir = resolve(args.source_dir)
    config_path = resolve(args.config_yaml)
    contract_path = resolve(args.wp4_contract)
    output_dir = resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output dir: {output_dir}")

    wp4_contract = read_json(contract_path)
    validate_same_contract_config(read_yaml(config_path), wp4_contract=wp4_contract)

    source_summary_path = source_dir / "coordinate_majority_r2_option_b/wp6_r2_option_b_summary.json"
    source_decode_rows_path = (
        source_dir / "coordinate_majority_r2_option_b/wp6_r2_option_b_decode_rows.jsonl"
    )
    source_support_path = (
        source_dir / "coordinate_majority_r2_option_b/wp6_r2_option_b_support_by_block_budget.csv"
    )
    for required in [
        source_summary_path,
        source_decode_rows_path,
        source_support_path,
        source_dir / "precommit/wp6_r2_option_b_contract.json",
        source_dir / "wp6_generation_summary.json",
        source_dir / "wp6_generated_outputs.jsonl",
        source_dir / "wp6_e2e_summary.json",
        source_dir / "wp6_slot_observations.jsonl",
        source_dir / "wp6_decode_decisions.jsonl",
    ]:
        require_file(required)

    source_summary = read_json(source_summary_path)
    validate_source_summary(source_summary)
    decode_rows = convert_decode_rows(read_jsonl(source_decode_rows_path))
    support_rows = convert_support_rows(source_support_path)

    write_jsonl(output_dir / "r3_2_coordinate_majority_decode_rows.jsonl", decode_rows)
    write_csv_new(
        output_dir / "r3_2_support_by_block_budget.csv",
        support_rows,
        [
            "block_id",
            "contract_id",
            "replicate_group_id",
            "source_wp6_r2_block_id",
            "block_index",
            "block_start_frame",
            "block_end_frame_exclusive",
            "budget",
            "source_condition",
            "step_index",
            "observed_bucket_0_count",
            "observed_bucket_1_count",
            "resolved_count",
            "majority_bit",
            "majority_margin",
        ],
    )
    write_json_new(
        output_dir / "r3_2_852426_replay_summary.json",
        build_replay_summary(
            source_dir=source_dir,
            source_summary=source_summary,
            decode_rows=decode_rows,
            config_path=config_path,
            contract_path=contract_path,
        ),
    )
    print(f"R3.2 same-contract 852426 replay exact match: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
