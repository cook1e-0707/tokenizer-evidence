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

from scripts.natural_evidence_v2.replay_wp6_coordinate_majority_decoder import (  # noqa: E402
    contract_info,
    decision_from_bits,
    decode_specs,
    parse_budgets,
    parse_byte_hex,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


DEFAULT_PROMPTS = (
    ROOT
    / "results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/"
    "restricted_step_label_strict_density_audit_prompts.jsonl"
)
DEFAULT_CONTRACT = (
    ROOT
    / "results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/"
    "wp4_prompt_local_payload_contract.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode WP6-R1 scale transcripts with independent 64-frame block-window "
            "coordinate majorities. This script does not generate, train, submit "
            "Slurm, aggregate FAR, run Llama, or make paper-facing claims."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prompts-jsonl", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--wp4-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="")
    parser.add_argument("--max-prompts", type=int, default=256)
    parser.add_argument("--block-count", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--expected-file-row-start", type=int, default=512)
    parser.add_argument("--expected-file-row-end", type=int, default=767)
    parser.add_argument("--wrong-audit-key-id", default="KWP4_QWEN_PILOT_WRONG_001")
    parser.add_argument("--wrong-payload-byte-hex", default="5a")
    parser.add_argument("--query-budgets", default="8,16,32,64")
    parser.add_argument("--min-protected-block-accepts-at-64", type=int, default=3)
    parser.add_argument("--min-support-at-64", type=int, default=16)
    parser.add_argument("--min-majority-margin-at-64", type=int, default=3)
    parser.add_argument("--protocol-id", default="natural_evidence_v2_wp6_r1_scale_reproducibility")
    parser.add_argument(
        "--decoder-id", default="qwen_v2_wp6_r1_block_window_coordinate_majority_decoder_v1"
    )
    parser.add_argument("--output-prefix", default="wp6_r1_scale")
    parser.add_argument("--artifact-role", default="wp6_r1_coordinate_majority_scale")
    parser.add_argument("--decode-artifact-dir", default="coordinate_majority_scale")
    parser.add_argument(
        "--accept-rule",
        default="per_block majority codeword checksum_valid_and_payload_matches_expected",
    )
    parser.add_argument("--pass-gate-status", default="PASS_WP6_R1_COORDINATE_MAJORITY_SCALE_GATE")
    parser.add_argument("--fail-gate-status", default="FAIL_WP6_R1_COORDINATE_MAJORITY_SCALE_GATE")
    parser.add_argument("--precommitted-transcript", action="store_true")
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="Write only wp6_r1_scale_contract.json and exit before reading transcript artifacts.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def sha256_json(payload: Mapping[str, Any]) -> str:
    data = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sha256_file_row_range(path: Path, start: int, end_inclusive: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for file_row_index, line in enumerate(handle):
            if start <= file_row_index <= end_inclusive:
                digest.update(line)
    return digest.hexdigest()


def read_prompt_rows_with_file_index(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for file_row_index, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{file_row_index + 1}")
            rows.append((file_row_index, payload))
    return rows


def build_prompt_plan(
    rows: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    split: str,
    max_prompts: int,
    block_count: int,
    block_size: int,
    expected_file_row_start: int | None,
    expected_file_row_end: int | None,
    prompts_path: Path,
) -> dict[str, Any]:
    if max_prompts != block_count * block_size:
        raise ValueError("max_prompts must equal block_count * block_size for WP6-R1 scale")
    selected = [
        (file_row, dict(row))
        for file_row, row in rows
        if (not split or str(row.get("split", "")) == split)
        and (
            expected_file_row_start is None
            or expected_file_row_end is None
            or int(expected_file_row_start) <= file_row <= int(expected_file_row_end)
        )
    ]
    if len(selected) < max_prompts:
        raise ValueError(f"split {split!r} has only {len(selected)} prompts; need {max_prompts}")
    selected = selected[:max_prompts]
    for _file_row, row in selected:
        if int(row.get("expected_structural_slots", 0)) != 16:
            raise ValueError(f"prompt does not expect 16 slots: {row.get('prompt_id')}")

    selected_start = int(selected[0][0])
    selected_end = int(selected[-1][0])
    if expected_file_row_start is not None and selected_start != int(expected_file_row_start):
        raise ValueError(
            f"selected prompt file-row start {selected_start} != expected {expected_file_row_start}"
        )
    if expected_file_row_end is not None and selected_end != int(expected_file_row_end):
        raise ValueError(f"selected prompt file-row end {selected_end} != expected {expected_file_row_end}")

    prompt_refs = [
        {
            "prompt_file_row_index": int(file_row),
            "prompt_id": str(row.get("prompt_id", "")),
            "prompt_text_sha256": str(row.get("prompt_text_sha256", "")),
            "selected_eval_index": selected_index,
        }
        for selected_index, (file_row, row) in enumerate(selected)
    ]
    prompt_selection_sha256 = sha256_json({"selected_prompts": prompt_refs})

    blocks: list[dict[str, Any]] = []
    for block_index in range(block_count):
        start = block_index * block_size
        end = start + block_size
        block_refs = prompt_refs[start:end]
        expected_start = selected_start + start
        expected_end = expected_start + block_size - 1
        actual_start = int(block_refs[0]["prompt_file_row_index"])
        actual_end = int(block_refs[-1]["prompt_file_row_index"])
        if actual_start != expected_start or actual_end != expected_end:
            raise ValueError(
                f"block_{block_index} file rows {actual_start}..{actual_end} "
                f"do not match expected {expected_start}..{expected_end}"
            )
        blocks.append(
            {
                "block_id": f"block_{block_index}",
                "block_index": block_index,
                "block_size": block_size,
                "prompt_file_row_end_inclusive": actual_end,
                "prompt_file_row_start": actual_start,
                "row_jsonl_sha256": sha256_file_row_range(prompts_path, actual_start, actual_end),
                "selected_eval_index_end_inclusive": end - 1,
                "selected_eval_index_start": start,
            }
        )

    return {
        "block_count": block_count,
        "block_size": block_size,
        "blocks": blocks,
        "prompt_count": max_prompts,
        "prompt_selection_sha256": prompt_selection_sha256,
        "prompt_source": str(prompts_path),
        "prompt_source_rows": len(rows),
        "prompt_source_sha256": sha256_file(prompts_path),
        "selected_prompt_jsonl_sha256": sha256_file_row_range(prompts_path, selected_start, selected_end),
        "selected_prompt_file_row_end_inclusive": selected_end,
        "selected_prompt_file_row_start": selected_start,
        "selected_prompt_rule": "explicit prompt source file-row window after split validation",
        "selected_prompts": prompt_refs,
        "selected_split": split,
    }


def build_scale_contract(
    *,
    args: argparse.Namespace,
    input_dir: Path,
    prompts_path: Path,
    contract_path: Path,
    wp4_contract: Mapping[str, Any],
    info: Mapping[str, Any],
    prompt_plan: Mapping[str, Any],
    budgets: Sequence[int],
) -> dict[str, Any]:
    payload_hex = str(info["payload_hex"])
    checksum_hex = f"{int(info['checksum_byte']):02x}"
    payload_plus_checksum_hex = f"{payload_hex}{checksum_hex}"
    precommit_material = {
        "audit_key_id": str(info["audit_key_id"]),
        "block_count": int(args.block_count),
        "block_size": int(args.block_size),
        "bucket_policy_id": str(wp4_contract.get("precommit", {}).get("bucket_policy_id", "")),
        "decoder_id": str(args.decoder_id),
        "eval_split": str(args.split),
        "payload_plus_checksum_hex": payload_plus_checksum_hex,
        "prompt_selection_sha256": str(prompt_plan["prompt_selection_sha256"]),
        "protocol_id": str(args.protocol_id),
        "query_budgets_per_block": list(budgets),
        "slot_policy_id": "strict_step_label_index_1_to_16",
    }
    precommitted_transcript = bool(args.precommitted_transcript)
    output_prefix = str(args.output_prefix)
    decode_artifact_dir = str(args.decode_artifact_dir)
    return {
        "artifact_role": f"{args.artifact_role}_contract",
        "claim_control": {
            "far_aggregation_allowed": False,
            "llama_allowed": False,
            "paper_claim_allowed": False,
            "same_family_null_allowed": False,
            "sanitizer_allowed": False,
            "training_allowed": False,
        },
        "decoder_policy": {
            "accept_rule": str(args.accept_rule),
            "controlling_budget": max(budgets),
            "decoder_id": str(args.decoder_id),
            "minimum_majority_margin_at_64": int(args.min_majority_margin_at_64),
            "minimum_protected_block_accepts_at_64": int(args.min_protected_block_accepts_at_64),
            "minimum_support_at_64": int(args.min_support_at_64),
            "query_budgets_per_block": list(budgets),
        },
        "input_dir": str(input_dir),
        "null_controls": ["protected", "raw", "task_only", "wrong_key", "wrong_payload"],
        "payload_cell": {
            "audit_key_id": str(info["audit_key_id"]),
            "checksum_byte_hex": checksum_hex,
            "payload_byte_hex": payload_hex,
            "payload_cell_id": str(info["source_payload_id"]),
            "payload_plus_checksum_hex": payload_plus_checksum_hex,
            "wrong_audit_key_id": str(args.wrong_audit_key_id),
            "wrong_payload_byte_hex": str(args.wrong_payload_byte_hex),
        },
        "precommit": {
            "precommit_hash_sha256": sha256_json(precommit_material),
            "precommit_material": precommit_material,
        },
        "precommit_note": (
            "This scale contract is intended to be written before generation."
            if precommitted_transcript
            else "This scale decode is not labeled as precommitted before generation."
        ),
        "prompt_plan": dict(prompt_plan),
        "required_outputs": [
            f"precommit/{output_prefix}_contract.json",
            "wp6_generation_summary.json",
            "wp6_generated_outputs.jsonl",
            "wp6_e2e_summary.json",
            "wp6_slot_observations.jsonl",
            "wp6_decode_decisions.jsonl",
            f"{decode_artifact_dir}/{output_prefix}_decode_rows.jsonl",
            f"{decode_artifact_dir}/{output_prefix}_summary.json",
            f"{decode_artifact_dir}/{output_prefix}_support_by_block_budget.csv",
            f"{decode_artifact_dir}/{output_prefix}_contract.json",
        ],
        "schema_name": f"natural_evidence_v2_{output_prefix}_contract_v1",
        "source_wp4_contract_path": str(contract_path),
        "source_wp4_contract_sha256": sha256_file(contract_path),
        "source_wp4_precommit_hash_sha256": str(info.get("precommit_hash_sha256", "")),
        "transcript_precommitted_before_generation": precommitted_transcript,
        "transcript_provenance": (
            "precommitted_replacement_run" if precommitted_transcript else "post_hoc_artifact_replay"
        ),
    }


def majority_decode_window(
    *,
    observations: Sequence[Mapping[str, Any]],
    source_condition: str,
    block_id: str,
    block_index: int,
    block_start_frame: int,
    budget: int,
) -> tuple[list[int | None], list[dict[str, Any]]]:
    coord_rows: list[dict[str, Any]] = []
    bits: list[int | None] = []
    block_end_frame_exclusive = block_start_frame + budget
    for step_index in range(1, 17):
        rows = [
            row
            for row in observations
            if str(row.get("generation_condition", "")) == source_condition
            and str(row.get("decode_condition", "")) == source_condition
            and block_start_frame <= int(row.get("frame_index", -1)) < block_end_frame_exclusive
            and int(row.get("step_index", 0)) == step_index
            and bool(row.get("resolved_bucket_hit"))
        ]
        counts = Counter(int(row["observed_bucket_id"]) for row in rows)
        if counts:
            ordered = counts.most_common()
            majority_bit = int(ordered[0][0])
            runner_up = int(ordered[1][1]) if len(ordered) > 1 else 0
            margin = int(ordered[0][1]) - runner_up
        else:
            majority_bit = None
            margin = 0
        bits.append(majority_bit)
        coord_rows.append(
            {
                "block_end_frame_exclusive": block_end_frame_exclusive,
                "block_id": block_id,
                "block_index": block_index,
                "block_start_frame": block_start_frame,
                "budget": budget,
                "majority_bit": "" if majority_bit is None else majority_bit,
                "majority_margin": margin,
                "observed_bucket_0_count": int(counts.get(0, 0)),
                "observed_bucket_1_count": int(counts.get(1, 0)),
                "resolved_count": sum(counts.values()),
                "source_condition": source_condition,
                "step_index": step_index,
            }
        )
    return bits, coord_rows


def decode_scale_blocks(
    *,
    args: argparse.Namespace,
    observations: Sequence[Mapping[str, Any]],
    info: Mapping[str, Any],
    budgets: Sequence[int],
    wrong_payload_byte: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    block_size = int(args.block_size)
    if max(budgets) > block_size:
        raise ValueError("query budgets must be <= block size")
    coord_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    for block_index in range(int(args.block_count)):
        block_id = f"block_{block_index}"
        block_start_frame = block_index * block_size
        for budget in budgets:
            cached: dict[str, tuple[list[int | None], list[dict[str, Any]]]] = {}
            for spec in decode_specs(
                info,
                wrong_audit_key_id=str(args.wrong_audit_key_id),
                wrong_payload_byte=wrong_payload_byte,
            ):
                source = str(spec["source_condition"])
                if source not in cached:
                    cached[source] = majority_decode_window(
                        observations=observations,
                        source_condition=source,
                        block_id=block_id,
                        block_index=block_index,
                        block_start_frame=block_start_frame,
                        budget=budget,
                    )
                    coord_rows.extend(cached[source][1])
                bits = cached[source][0]
                decision = decision_from_bits(
                    bits=bits,
                    expected_audit_key_id=str(spec["expected_audit_key_id"]),
                    expected_payload_byte=int(spec["expected_payload_byte"]),
                )
                support = [int(row["resolved_count"]) for row in cached[source][1]]
                margins = [int(row["majority_margin"]) for row in cached[source][1]]
                decode_rows.append(
                    {
                        **decision,
                        "block_end_frame_exclusive": block_start_frame + budget,
                        "block_id": block_id,
                        "block_index": block_index,
                        "block_start_frame": block_start_frame,
                        "budget": budget,
                        "decode_condition": str(spec["decode_condition"]),
                        "min_majority_margin": min(margins) if margins else 0,
                        "min_support": min(support) if support else 0,
                        "schema_name": f"natural_evidence_v2_{getattr(args, 'output_prefix', 'wp6_r1_scale')}_decode_v1",
                        "source_condition": source,
                    }
                )
    return coord_rows, decode_rows


def validate_frame_coverage(
    *,
    decision_rows: Sequence[Mapping[str, Any]],
    block_count: int,
    block_size: int,
) -> None:
    required_frames = list(range(block_count * block_size))
    for condition in ("protected", "raw", "task_only"):
        frames = sorted(
            int(row.get("frame_index", -1))
            for row in decision_rows
            if str(row.get("decode_condition", "")) == condition
        )
        if frames[: len(required_frames)] != required_frames:
            raise ValueError(
                f"{condition} exact-frame decode decisions do not cover frames "
                f"0..{required_frames[-1]}"
            )


def summarize_scale(
    *,
    args: argparse.Namespace,
    budgets: Sequence[int],
    decode_rows: Sequence[Mapping[str, Any]],
    exact_decision_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary_by_block_budget: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for row in decode_rows:
        summary_by_block_budget[str(row["block_id"])][str(row["budget"])][str(row["decode_condition"])] = {
            "accepted": bool(row["accepted"]),
            "decoded_hex": str(row["majority_hex"]),
            "min_majority_margin": int(row["min_majority_margin"]),
            "min_support": int(row["min_support"]),
        }

    controlling_budget = max(budgets)
    protected_rows = [
        row
        for row in decode_rows
        if int(row["budget"]) == controlling_budget and str(row["decode_condition"]) == "protected"
    ]
    protected_accept_rows = [row for row in protected_rows if bool(row["accepted"])]
    null_accept_counts = {
        condition: sum(
            1
            for row in decode_rows
            if int(row["budget"]) == controlling_budget
            and str(row["decode_condition"]) == condition
            and bool(row["accepted"])
        )
        for condition in ("raw", "task_only", "wrong_key", "wrong_payload")
    }
    accepted_supports = [int(row["min_support"]) for row in protected_accept_rows]
    accepted_margins = [int(row["min_majority_margin"]) for row in protected_accept_rows]
    min_support_accepted = min(accepted_supports) if accepted_supports else 0
    min_margin_accepted = min(accepted_margins) if accepted_margins else 0
    forbidden_public_surface_count = sum(
        1 for row in exact_decision_rows if bool(row.get("forbidden_public_surface_present"))
    )
    scale_gate_pass = (
        len(protected_accept_rows) >= int(args.min_protected_block_accepts_at_64)
        and all(count == 0 for count in null_accept_counts.values())
        and min_support_accepted >= int(args.min_support_at_64)
        and min_margin_accepted >= int(args.min_majority_margin_at_64)
        and forbidden_public_surface_count == 0
    )
    precommitted_transcript = bool(getattr(args, "precommitted_transcript", False))
    pass_gate_status = str(
        getattr(args, "pass_gate_status", "PASS_WP6_R1_COORDINATE_MAJORITY_SCALE_GATE")
    )
    fail_gate_status = str(
        getattr(args, "fail_gate_status", "FAIL_WP6_R1_COORDINATE_MAJORITY_SCALE_GATE")
    )
    output_prefix = str(getattr(args, "output_prefix", "wp6_r1_scale"))
    artifact_role = str(getattr(args, "artifact_role", "wp6_r1_coordinate_majority_scale"))
    return {
        "artifact_role": f"{artifact_role}_summary",
        "claim_control": {
            "far_aggregation_allowed": False,
            "llama_allowed": False,
            "paper_claim_allowed": False,
            "same_family_null_allowed": False,
            "sanitizer_allowed": False,
            "training_allowed": False,
        },
        "controlling_budget": controlling_budget,
        "forbidden_public_surface_count": forbidden_public_surface_count,
        "null_accept_counts_at_controlling_budget": null_accept_counts,
        "post_hoc_artifact_replay": not precommitted_transcript,
        "precommitted_transcript": precommitted_transcript,
        "protected_block_accept_count_at_controlling_budget": len(protected_accept_rows),
        "protected_block_count": len(protected_rows),
        "protected_min_majority_margin_in_accepted_blocks": min_margin_accepted,
        "protected_min_support_in_accepted_blocks": min_support_accepted,
        "scale_gate_pass": bool(scale_gate_pass),
        "scale_gate_status": pass_gate_status if scale_gate_pass else fail_gate_status,
        "scale_gate_targets": {
            "forbidden_public_surface_count": 0,
            "min_majority_margin_in_accepted_protected_blocks": int(args.min_majority_margin_at_64),
            "min_protected_block_accepts_at_64": int(args.min_protected_block_accepts_at_64),
            "min_support_in_accepted_protected_blocks": int(args.min_support_at_64),
            "null_accepts_per_condition_at_64": 0,
            "protected_block_count": int(args.block_count),
        },
        "schema_name": f"natural_evidence_v2_{output_prefix}_summary_v1",
        "summary_by_block_budget": {
            block_id: {budget: dict(conditions) for budget, conditions in budget_map.items()}
            for block_id, budget_map in summary_by_block_budget.items()
        },
        "transcript_provenance": (
            "precommitted_replacement_run" if precommitted_transcript else "post_hoc_artifact_replay"
        ),
    }


def main() -> int:
    args = parse_args()
    input_dir = resolve(args.input_dir)
    prompts_path = resolve(args.prompts_jsonl)
    contract_path = resolve(args.wp4_contract)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    budgets = parse_budgets(str(args.query_budgets))
    wp4_contract = read_json(contract_path)
    info = contract_info(wp4_contract)
    prompt_plan = build_prompt_plan(
        read_prompt_rows_with_file_index(prompts_path),
        split=str(args.split),
        max_prompts=int(args.max_prompts),
        block_count=int(args.block_count),
        block_size=int(args.block_size),
        expected_file_row_start=int(args.expected_file_row_start),
        expected_file_row_end=int(args.expected_file_row_end),
        prompts_path=prompts_path,
    )
    contract = build_scale_contract(
        args=args,
        input_dir=input_dir,
        prompts_path=prompts_path,
        contract_path=contract_path,
        wp4_contract=wp4_contract,
        info=info,
        prompt_plan=prompt_plan,
        budgets=budgets,
    )
    if args.contract_only:
        write_json(output_dir / f"{args.output_prefix}_contract.json", contract)
        return 0

    observations = read_jsonl(input_dir / "wp6_slot_observations.jsonl")
    exact_decision_rows = read_jsonl(input_dir / "wp6_decode_decisions.jsonl")
    validate_frame_coverage(
        decision_rows=exact_decision_rows,
        block_count=int(args.block_count),
        block_size=int(args.block_size),
    )
    coord_rows, decode_rows = decode_scale_blocks(
        args=args,
        observations=observations,
        info=info,
        budgets=budgets,
        wrong_payload_byte=parse_byte_hex(str(args.wrong_payload_byte_hex)),
    )
    summary = summarize_scale(
        args=args,
        budgets=budgets,
        decode_rows=decode_rows,
        exact_decision_rows=exact_decision_rows,
    )
    write_json(output_dir / f"{args.output_prefix}_contract.json", contract)
    write_jsonl(output_dir / f"{args.output_prefix}_decode_rows.jsonl", decode_rows)
    write_csv(
        output_dir / f"{args.output_prefix}_support_by_block_budget.csv",
        coord_rows,
        [
            "block_id",
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
    write_json(output_dir / f"{args.output_prefix}_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
