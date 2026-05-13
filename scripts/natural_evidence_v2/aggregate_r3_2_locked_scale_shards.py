from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.build_r3_2_locked_scale_precommit import (  # noqa: E402
    CONTRACT_ID,
    CONTRACT_LABEL,
    REPLICATE_GROUP_COUNT,
    read_json,
    write_json_new,
)
from scripts.natural_evidence_v2.replay_wp6_coordinate_majority_decoder import (  # noqa: E402
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate isolated R3.2 same-contract shard decode artifacts into "
            "one 96-block locked-scale gate artifact. This reads completed shard "
            "artifacts only; it does not train, generate, submit Slurm, aggregate "
            "FAR, run Llama, run sanitizer benchmarks, or make paper-facing claims."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        required=True,
        help="precommit/r3_2_selected_prompt_manifest.json from the same output dir.",
    )
    parser.add_argument("--shard-decode-dir", default="coordinate_majority_r3_2_shard")
    parser.add_argument("--shard-prefix", default="r3_2_shard")
    parser.add_argument("--query-budgets", default="16,32,64")
    parser.add_argument("--primary-budget", type=int, default=64)
    parser.add_argument("--min-protected-accepts-at-64", type=int, default=80)
    parser.add_argument("--min-support-at-64", type=int, default=16)
    parser.add_argument("--min-majority-margin-at-64", type=int, default=3)
    parser.add_argument(
        "--allow-duplicate-prompt-windows",
        action="store_true",
        help=(
            "Explicit diagnostic override. By default, R3.2 locked-scale aggregation "
            "hard-fails if shard prompt-window or transcript hashes repeat, because "
            "repeated deterministic windows are not independent locked-scale blocks."
        ),
    )
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


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_new(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def parse_budgets(value: str) -> list[int]:
    budgets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if budgets != [16, 32, 64]:
        raise ValueError("R3.2 aggregation requires query budgets [16, 32, 64]")
    return budgets


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required artifact missing or empty: {path}")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = read_json(path)
    if manifest.get("contract_id") != CONTRACT_ID:
        raise ValueError(f"R3.2 manifest must use contract_id={CONTRACT_ID}")
    groups = manifest.get("replicate_groups", [])
    if not isinstance(groups, list) or len(groups) != REPLICATE_GROUP_COUNT:
        raise ValueError(f"R3.2 manifest must contain {REPLICATE_GROUP_COUNT} replicate groups")
    return [dict(group) for group in groups]


def canonical_block_id(shard_id: str, block_id: str) -> str:
    prefix = "block_"
    if not block_id.startswith(prefix):
        raise ValueError(f"unexpected shard block id: {block_id!r}")
    return f"{CONTRACT_LABEL}_{shard_id}_block_{int(block_id[len(prefix):]):02d}"


def canonicalize_decode_rows(rows: Sequence[Mapping[str, Any]], *, shard_id: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        source_block_id = str(row.get("block_id", ""))
        output.append(
            dict(row)
            | {
                "block_id": canonical_block_id(shard_id, source_block_id),
                "contract_id": CONTRACT_ID,
                "replicate_group_id": shard_id,
                "schema_name": "natural_evidence_v2_r3_2_same_contract_locked_scale_decode_v1",
                "source_shard_block_id": source_block_id,
            }
        )
    return output


def canonicalize_support_rows(rows: Sequence[Mapping[str, Any]], *, shard_id: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        source_block_id = str(row.get("block_id", ""))
        output.append(
            dict(row)
            | {
                "block_id": canonical_block_id(shard_id, source_block_id),
                "contract_id": CONTRACT_ID,
                "replicate_group_id": shard_id,
                "source_shard_block_id": source_block_id,
            }
        )
    return output


def duplicate_values(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        value = str(row.get(key, ""))
        if value:
            groups[value].append(str(row.get("replicate_group_id", "")))
    return {value: shard_ids for value, shard_ids in groups.items() if len(shard_ids) > 1}


def refuse_duplicate_prompt_windows(shard_summaries: Sequence[Mapping[str, Any]]) -> None:
    duplicate_prompt_windows = duplicate_values(shard_summaries, "selected_prompt_jsonl_sha256")
    duplicate_generated_outputs = duplicate_values(shard_summaries, "generated_outputs_sha256")
    duplicate_decode_rows = duplicate_values(shard_summaries, "decode_rows_sha256")
    duplicate_counts = {
        "selected_prompt_jsonl_sha256": len(duplicate_prompt_windows),
        "generated_outputs_sha256": len(duplicate_generated_outputs),
        "decode_rows_sha256": len(duplicate_decode_rows),
    }
    if any(duplicate_counts.values()):
        detail = {
            "duplicate_decode_rows": duplicate_decode_rows,
            "duplicate_generated_outputs": duplicate_generated_outputs,
            "duplicate_prompt_windows": duplicate_prompt_windows,
            "duplicate_value_counts": duplicate_counts,
            "reason": (
                "R3.2 locked-scale aggregation requires distinct prompt windows and "
                "non-duplicate deterministic transcript/decode artifacts. Repeated "
                "windows may be useful diagnostics but must not be counted as "
                "independent 96-block locked-scale evidence."
            ),
        }
        raise ValueError("R3_2_DUPLICATE_PROMPT_WINDOWS_REFUSING_AGGREGATION: " + json.dumps(detail, sort_keys=True))


def annotate_rows(rows: Sequence[Mapping[str, Any]], *, shard_id: str, schema_name: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        | {
            "contract_id": CONTRACT_ID,
            "replicate_group_id": shard_id,
            "schema_name": schema_name,
        }
        for row in rows
    ]


def build_generation_summary(
    *,
    shard_summaries: Sequence[Mapping[str, Any]],
    generated_row_count: int,
) -> dict[str, Any]:
    return {
        "artifact_role": "r3_2_same_contract_locked_scale_generation_summary",
        "claim_control": {
            "far_aggregation_allowed": False,
            "llama_allowed": False,
            "paper_claim_allowed": False,
            "same_family_null_allowed": False,
            "sanitizer_allowed": False,
            "training_allowed": False,
        },
        "contract_id": CONTRACT_ID,
        "generated_output_rows": generated_row_count,
        "generation_started": True,
        "payload_diversity_tested": False,
        "replicate_group_count": REPLICATE_GROUP_COUNT,
        "schema_name": "natural_evidence_v2_r3_2_same_contract_generation_summary_v1",
        "shard_generation_summaries": [dict(row) for row in shard_summaries],
        "status": "R3_2_SAME_CONTRACT_SHARD_GENERATION_AGGREGATED",
    }


def build_gate_review(
    *,
    args: argparse.Namespace,
    budgets: Sequence[int],
    decode_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    replicate_group_ids: Sequence[str],
) -> dict[str, Any]:
    primary_budget = int(args.primary_budget)
    if primary_budget != max(budgets):
        raise ValueError("R3.2 primary budget must equal max query budget")
    by_block_budget: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for row in decode_rows:
        by_block_budget[str(row["block_id"])][str(row["budget"])][str(row["decode_condition"])] = {
            "accepted": bool(row["accepted"]),
            "decoded_hex": str(row.get("majority_hex", "")),
            "min_majority_margin": int(row.get("min_majority_margin", 0)),
            "min_support": int(row.get("min_support", 0)),
            "replicate_group_id": str(row["replicate_group_id"]),
        }

    protected_rows = [
        row
        for row in decode_rows
        if int(row["budget"]) == primary_budget and str(row["decode_condition"]) == "protected"
    ]
    protected_accept_rows = [row for row in protected_rows if bool(row["accepted"])]
    null_accept_counts = {
        condition: sum(
            1
            for row in decode_rows
            if int(row["budget"]) == primary_budget
            and str(row["decode_condition"]) == condition
            and bool(row["accepted"])
        )
        for condition in ("raw", "task_only", "wrong_key", "wrong_payload")
    }
    accepted_supports = [int(row.get("min_support", 0)) for row in protected_accept_rows]
    accepted_margins = [int(row.get("min_majority_margin", 0)) for row in protected_accept_rows]
    forbidden_public_surface_count = sum(
        1 for row in decision_rows if bool(row.get("forbidden_public_surface_present"))
    )
    complete_groups = sorted({str(row["replicate_group_id"]) for row in protected_rows})
    all_groups_complete = complete_groups == sorted(replicate_group_ids)
    gate_pass = (
        len(protected_rows) == 96
        and len(protected_accept_rows) >= int(args.min_protected_accepts_at_64)
        and all(count == 0 for count in null_accept_counts.values())
        and (min(accepted_supports) if accepted_supports else 0) >= int(args.min_support_at_64)
        and (min(accepted_margins) if accepted_margins else 0) >= int(args.min_majority_margin_at_64)
        and forbidden_public_surface_count == 0
        and all_groups_complete
    )
    return {
        "artifact_role": "r3_2_same_contract_locked_scale_gate_review",
        "claim_control": {
            "far_aggregation_allowed": False,
            "llama_allowed": False,
            "paper_claim_allowed": False,
            "same_family_null_allowed": False,
            "sanitizer_allowed": False,
            "training_allowed": False,
        },
        "controlling_budget": primary_budget,
        "contract_id": CONTRACT_ID,
        "forbidden_public_surface_count": forbidden_public_surface_count,
        "null_accept_counts_at_controlling_budget": null_accept_counts,
        "payload_diversity_tested": False,
        "precommitted_transcript": True,
        "protected_block_accept_count_at_controlling_budget": len(protected_accept_rows),
        "protected_block_count": len(protected_rows),
        "protected_min_majority_margin_in_accepted_blocks": min(accepted_margins) if accepted_margins else 0,
        "protected_min_support_in_accepted_blocks": min(accepted_supports) if accepted_supports else 0,
        "replicate_group_complete": all_groups_complete,
        "replicate_group_ids": list(replicate_group_ids),
        "scale_gate_pass": bool(gate_pass),
        "scale_gate_status": (
            "PASS_R3_2_SAME_CONTRACT_LOCKED_SCALE_GATE"
            if gate_pass
            else "FAIL_R3_2_SAME_CONTRACT_LOCKED_SCALE_GATE"
        ),
        "scale_gate_targets": {
            "forbidden_public_surface_count": 0,
            "min_majority_margin_in_accepted_protected_blocks": int(args.min_majority_margin_at_64),
            "min_protected_accepts_at_64": int(args.min_protected_accepts_at_64),
            "min_support_in_accepted_protected_blocks": int(args.min_support_at_64),
            "null_accepts_per_condition_at_64": 0,
            "protected_block_count": 96,
            "replicate_group_count": REPLICATE_GROUP_COUNT,
        },
        "schema_name": "natural_evidence_v2_r3_2_same_contract_locked_scale_gate_review_v1",
        "summary_by_block_budget": {
            block_id: {budget: dict(conditions) for budget, conditions in budget_map.items()}
            for block_id, budget_map in by_block_budget.items()
        },
    }


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    shards_dir = resolve(args.shards_dir)
    manifest_path = resolve(args.prompt_manifest)
    budgets = parse_budgets(str(args.query_budgets))
    groups = load_manifest(manifest_path)

    forbidden_outputs = [
        "r3_2_generation_summary.json",
        "r3_2_generated_outputs.jsonl",
        "r3_2_slot_observations.jsonl",
        "r3_2_decode_decisions.jsonl",
        "r3_2_coordinate_majority_decode_rows.jsonl",
        "r3_2_coordinate_majority_summary.json",
        "r3_2_support_by_block_budget.csv",
        "r3_2_gate_review.json",
    ]
    existing = [name for name in forbidden_outputs if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing R3.2 aggregate artifacts: {existing}")

    generated_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []

    for group in groups:
        shard_id = str(group["replicate_group_id"])
        shard_dir = shards_dir / shard_id
        decode_dir = shard_dir / str(args.shard_decode_dir)
        prefix = str(args.shard_prefix)
        paths = {
            "generated": shard_dir / "wp6_generated_outputs.jsonl",
            "generation_summary": shard_dir / "wp6_generation_summary.json",
            "observations": shard_dir / "wp6_slot_observations.jsonl",
            "decisions": shard_dir / "wp6_decode_decisions.jsonl",
            "decode": decode_dir / f"{prefix}_decode_rows.jsonl",
            "support": decode_dir / f"{prefix}_support_by_block_budget.csv",
            "summary": decode_dir / f"{prefix}_summary.json",
        }
        for path in paths.values():
            require_file(path)
        shard_summary = read_json(paths["summary"])
        generation_summary = read_json(paths["generation_summary"])
        if shard_summary.get("precommitted_transcript") is not True:
            raise ValueError(f"{shard_id} must be a precommitted transcript")
        shard_summaries.append(
            {
                "decode_rows_sha256": sha256_file(paths["decode"]),
                "generated_outputs_sha256": sha256_file(paths["generated"]),
                "generation_seed": int(group["generation_seed"]),
                "prompt_file_row_end_inclusive": int(
                    generation_summary.get("selected_prompt_file_row_end_inclusive", -1)
                ),
                "prompt_file_row_start": int(generation_summary.get("selected_prompt_file_row_start", -1)),
                "selected_prompt_jsonl_sha256": str(generation_summary.get("selected_prompt_jsonl_sha256", "")),
                "prompt_window_index": int(group["prompt_window_index"]),
                "replicate_group_id": shard_id,
                "scale_gate_status": str(shard_summary.get("scale_gate_status", "")),
            }
        )
        generated_rows.extend(
            annotate_rows(
                read_jsonl(paths["generated"]),
                shard_id=shard_id,
                schema_name="natural_evidence_v2_r3_2_generated_output_v1",
            )
        )
        observation_rows.extend(
            annotate_rows(
                read_jsonl(paths["observations"]),
                shard_id=shard_id,
                schema_name="natural_evidence_v2_r3_2_slot_observation_v1",
            )
        )
        decision_rows.extend(
            annotate_rows(
                read_jsonl(paths["decisions"]),
                shard_id=shard_id,
                schema_name="natural_evidence_v2_r3_2_decode_decision_v1",
            )
        )
        decode_rows.extend(canonicalize_decode_rows(read_jsonl(paths["decode"]), shard_id=shard_id))
        support_rows.extend(canonicalize_support_rows(read_csv(paths["support"]), shard_id=shard_id))

    if not args.allow_duplicate_prompt_windows:
        refuse_duplicate_prompt_windows(shard_summaries)

    write_jsonl(output_dir / "r3_2_generated_outputs.jsonl", generated_rows)
    write_json_new(
        output_dir / "r3_2_generation_summary.json",
        build_generation_summary(
            shard_summaries=shard_summaries,
            generated_row_count=len(generated_rows),
        ),
    )
    write_jsonl(output_dir / "r3_2_slot_observations.jsonl", observation_rows)
    write_jsonl(output_dir / "r3_2_decode_decisions.jsonl", decision_rows)
    write_jsonl(output_dir / "r3_2_coordinate_majority_decode_rows.jsonl", decode_rows)
    write_csv_new(
        output_dir / "r3_2_support_by_block_budget.csv",
        support_rows,
        [
            "block_id",
            "contract_id",
            "replicate_group_id",
            "source_shard_block_id",
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
    gate_review = build_gate_review(
        args=args,
        budgets=budgets,
        decode_rows=decode_rows,
        decision_rows=decision_rows,
        replicate_group_ids=[str(group["replicate_group_id"]) for group in groups],
    )
    write_json_new(output_dir / "r3_2_coordinate_majority_summary.json", gate_review)
    write_json_new(output_dir / "r3_2_gate_review.json", gate_review)
    print(f"R3.2 same-contract locked-scale aggregation complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
