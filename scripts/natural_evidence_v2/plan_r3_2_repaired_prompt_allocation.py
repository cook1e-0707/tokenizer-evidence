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

DEFAULT_PROMPTS = Path(
    "results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/"
    "restricted_step_label_strict_density_audit_prompts.jsonl"
)
DEFAULT_CONFIG = Path("configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale.yaml")
DEFAULT_SPLIT = "wp3_r1_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-only R3.2 repaired prompt-allocation preflight. It reads "
            "the prompt artifact and config, checks whether a requested locked-scale "
            "schedule can be backed by distinct prompt rows/windows, and writes "
            "machine-readable planning artifacts. It does not train, generate, "
            "submit Slurm, aggregate FAR, run Llama, or make claims."
        )
    )
    parser.add_argument("--prompts-jsonl", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--config-yaml", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--desired-shards", type=int, default=0)
    parser.add_argument("--blocks-per-shard", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=0)
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


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


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


def contiguous_windows(rows: Sequence[Mapping[str, Any]], *, window_size: int) -> list[dict[str, Any]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(rows) < window_size:
        return []
    windows: list[dict[str, Any]] = []
    for window_index, start in enumerate(range(0, len(rows) - window_size + 1, window_size)):
        chunk = list(rows[start : start + window_size])
        line_bytes = b"".join(bytes(row["line_bytes"]) for row in chunk)
        variant_counts = Counter(str(row.get("variant_id", "")) for row in chunk)
        topic_indices = [int(row.get("topic_index", -1)) for row in chunk if str(row.get("topic_index", "")).lstrip("-").isdigit()]
        windows.append(
            {
                "window_index": window_index,
                "split_window_index": window_index,
                "selected_index_start": start,
                "selected_index_end_inclusive": start + window_size - 1,
                "prompt_file_row_start": int(chunk[0]["file_row_index"]),
                "prompt_file_row_end_inclusive": int(chunk[-1]["file_row_index"]),
                "window_size": window_size,
                "window_jsonl_sha256": sha256_bytes(line_bytes),
                "variant_counts_json": json.dumps(dict(sorted(variant_counts.items())), sort_keys=True),
                "topic_index_min": min(topic_indices) if topic_indices else "",
                "topic_index_max": max(topic_indices) if topic_indices else "",
            }
        )
    return windows


def main() -> int:
    args = parse_args()
    prompts_path = resolve(args.prompts_jsonl)
    config_path = resolve(args.config_yaml)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    config = read_yaml(config_path)
    schedule = config.get("schedule", {})
    desired_shards = int(args.desired_shards or schedule.get("replicate_groups", 12))
    blocks_per_shard = int(args.blocks_per_shard or schedule.get("blocks_per_group", 8))
    block_size = int(args.block_size or schedule.get("block_size", 64))
    window_size = blocks_per_shard * block_size

    rows = read_prompt_rows(prompts_path)
    selected_rows = [row for row in rows if str(row.get("split", "")) == str(args.split)]
    prompt_id_counts = Counter(str(row.get("prompt_id", "")) for row in selected_rows)
    duplicate_prompt_ids = sorted(prompt_id for prompt_id, count in prompt_id_counts.items() if count > 1)
    bad_slot_rows = [
        str(row.get("prompt_id", ""))
        for row in selected_rows
        if int(row.get("expected_structural_slots", 0)) != 16
    ]
    windows = contiguous_windows(selected_rows, window_size=window_size)

    required_rows = desired_shards * blocks_per_shard * block_size
    available_rows = len(selected_rows)
    max_unique_shards = available_rows // window_size
    max_unique_blocks = max_unique_shards * blocks_per_shard
    desired_blocks = desired_shards * blocks_per_shard
    feasible_requested = (
        not duplicate_prompt_ids
        and not bad_slot_rows
        and available_rows >= required_rows
        and len(windows) >= desired_shards
    )

    window_plan_rows: list[dict[str, Any]] = []
    for window in windows:
        shard_id = f"shard_{int(window['window_index']):02d}"
        window_plan_rows.append(
            {
                **window,
                "proposed_shard_id": shard_id,
                "contract_id": "a55e",
                "blocks_per_shard": blocks_per_shard,
                "block_size": block_size,
                "independent_prompt_window": True,
            }
        )

    summary = {
        "artifact_role": "r3_2_repaired_prompt_allocation_preflight",
        "claim_control": {
            "far_aggregation_allowed": False,
            "generation_started": False,
            "llama_allowed": False,
            "paper_claim_allowed": False,
            "same_family_null_allowed": False,
            "sanitizer_allowed": False,
            "slurm_job_submitted": False,
            "training_allowed": False,
        },
        "config_path": str(args.config_yaml),
        "config_sha256": sha256_file(config_path),
        "contract_id": "a55e",
        "desired_blocks": desired_blocks,
        "desired_shards": desired_shards,
        "available_unique_blocks": max_unique_blocks,
        "available_unique_shards": max_unique_shards,
        "available_rows_for_split": available_rows,
        "block_size": block_size,
        "blocks_per_shard": blocks_per_shard,
        "duplicate_prompt_id_count": len(duplicate_prompt_ids),
        "duplicate_prompt_ids_sample": duplicate_prompt_ids[:20],
        "feasible_requested_locked_scale": feasible_requested,
        "max_feasible_unique_package": {
            "blocks": max_unique_blocks,
            "rows": max_unique_blocks * block_size,
            "shards": max_unique_shards,
        },
        "prompt_allocation_status": (
            "PASS_REQUESTED_LOCKED_SCALE_PROMPT_ALLOCATION_FEASIBLE"
            if feasible_requested
            else "FAIL_INSUFFICIENT_UNIQUE_PROMPT_ROWS_FOR_REQUESTED_LOCKED_SCALE"
        ),
        "prompt_source_path": str(args.prompts_jsonl),
        "prompt_source_rows_total": len(rows),
        "prompt_source_sha256": sha256_file(prompts_path),
        "required_unique_rows_for_requested_locked_scale": required_rows,
        "schema_name": "natural_evidence_v2_r3_2_repaired_prompt_allocation_preflight_v1",
        "split": str(args.split),
        "split_bad_expected_structural_slot_rows": len(bad_slot_rows),
        "split_window_count": len(windows),
        "state_changing_action": "none_artifact_only",
        "recommended_route_options": [
            {
                "route": "R3.2-unique-32-block-diagnostic",
                "status": "feasible_now" if max_unique_blocks >= 32 else "not_feasible",
                "note": "Use 4 non-overlapping 512-row eval windows as 32 unique blocks; do not claim 96-block scale.",
            },
            {
                "route": "R3.2-96-block-locked-scale",
                "status": "requires_prompt_bank_expansion" if not feasible_requested else "feasible_now",
                "required_eval_rows": required_rows,
                "additional_eval_rows_needed": max(0, required_rows - available_rows),
                "note": "Requires 12 distinct 512-row prompt windows for 96 independent blocks.",
            },
        ],
    }

    write_json_new(output_dir / "r3_2_repaired_prompt_allocation_preflight.json", summary)
    write_csv_new(
        output_dir / "r3_2_repaired_prompt_window_plan.csv",
        window_plan_rows,
        [
            "proposed_shard_id",
            "window_index",
            "split_window_index",
            "prompt_file_row_start",
            "prompt_file_row_end_inclusive",
            "selected_index_start",
            "selected_index_end_inclusive",
            "window_size",
            "window_jsonl_sha256",
            "blocks_per_shard",
            "block_size",
            "independent_prompt_window",
            "variant_counts_json",
            "topic_index_min",
            "topic_index_max",
            "contract_id",
        ],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
