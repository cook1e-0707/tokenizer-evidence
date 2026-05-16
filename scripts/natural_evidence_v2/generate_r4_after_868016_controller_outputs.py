from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.generate_wp6_e2e_outputs import batched, load_model_condition  # noqa: E402
from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    read_jsonl,
    resolve,
    sha256_file,
    technical_literal_hits,
    write_json_new,
    write_jsonl_new,
)
from scripts.natural_evidence_v2.score_r4_surface_teacher_forced_mass import (  # noqa: E402
    ControllerConfig,
    bucket_first_token_ids,
    chat_prefix,
    controller_token_ids_for_policy,
    r4_row_surface_contract,
)


DEFAULT_ROWS = (
    ROOT / "results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_rows_20260516/reliability_surface_mass_rows.jsonl"
)
DEFAULT_TOKENIZER_REVIEW = (
    ROOT
    / "results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_qwen_tokenizer_boundary_preflight_868103/review_summary.json"
)
DEFAULT_CONTROLLER_REVIEW = (
    ROOT
    / "results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_controller_score_868114_review/coordinate_pivot_controller_review_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate R4 after-868016 coordinate-pivot row-cylinder outputs with a "
            "provider-side prefix-native controller. This is a small diagnostic "
            "generation path; it does not train, run Llama, aggregate FAR, or make paper claims."
        )
    )
    parser.add_argument("--score-rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--allocation-rows", type=Path, default=None)
    parser.add_argument("--assigned-shard-index", type=int, default=None)
    parser.add_argument("--tokenizer-review", type=Path, default=DEFAULT_TOKENIZER_REVIEW)
    parser.add_argument("--controller-review", type=Path, default=DEFAULT_CONTROLLER_REVIEW)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--task-only-adapter", type=Path, required=True)
    parser.add_argument("--prompt-index-start", type=int, required=True)
    parser.add_argument("--prompt-index-end", type=int, required=True)
    parser.add_argument("--expected-rows", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--replicate-group-id", required=True)
    parser.add_argument("--controller-bonus-nats", type=float, required=True)
    parser.add_argument("--controller-penalty-nats", type=float, required=True)
    parser.add_argument("--controller-max-target-mass", type=float, required=True)
    parser.add_argument("--controller-max-kl-budget", type=float, required=True)
    parser.add_argument("--validate-plan-only", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_selected_rows(selected: Sequence[Mapping[str, Any]], *, expected_rows: int) -> None:
    if len(selected) != int(expected_rows):
        raise ValueError(f"selected {len(selected)} rows; expected {expected_rows}")
    if len({str(row.get("row_key", "")) for row in selected}) != len(selected):
        raise ValueError("selected rows contain duplicate row_key values")
    coordinates = sorted({int(row["coordinate_id"]) for row in selected})
    if len(coordinates) != 12:
        raise ValueError(f"expected 12 selected coordinates per shard, found {coordinates}")
    for row in selected:
        prompt_text = str(row.get("prompt_text", ""))
        if "Step " in prompt_text or "exactly 16" in prompt_text or "slot" in prompt_text.lower():
            raise ValueError(f"R4 prompt contains forbidden structural instruction: {row.get('prompt_id')}")
        hits = technical_literal_hits(prompt_text)
        if hits:
            raise ValueError(f"R4 prompt contains technical literal {hits}: {row.get('prompt_id')}")
        r4_row_surface_contract(row)


def select_rows_by_prompt_range(
    rows: Sequence[Mapping[str, Any]],
    *,
    start: int,
    end: int,
    expected_rows: int,
) -> list[dict[str, Any]]:
    if end < start:
        raise ValueError("prompt-index-end must be >= prompt-index-start")
    selected = [dict(row) for row in rows if start <= int(row.get("prompt_index", -1)) <= end]
    validate_selected_rows(selected, expected_rows=expected_rows)
    return selected


def select_rows_by_allocation(
    rows: Sequence[Mapping[str, Any]],
    *,
    assigned_shard_index: int,
    expected_rows: int,
) -> list[dict[str, Any]]:
    selected = [
        dict(row)
        for row in rows
        if int(row.get("assigned_shard_index", -1)) == int(assigned_shard_index)
    ]
    validate_selected_rows(selected, expected_rows=expected_rows)
    duplicate_pairs = Counter(str(row.get("duplicate_pair_key", "")) for row in selected)
    duplicated = sorted(pair for pair, count in duplicate_pairs.items() if pair and count > 1)
    if duplicated:
        raise ValueError(f"allocation selected duplicate prompt/prefix pairs: {duplicated[:5]}")
    return selected


def validate_reviews(tokenizer_review: Mapping[str, Any], controller_review: Mapping[str, Any]) -> None:
    tokenizer_status = tokenizer_review.get("status") or tokenizer_review.get("review_status")
    if tokenizer_status != "PASS_R4_AFTER_868016_RELIABILITY_COORDINATE_PIVOT_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT_868103":
        raise ValueError("tokenizer review is not the reviewed 868103 pass")
    failed_rows = tokenizer_review.get("failed_row_count", tokenizer_review.get("failed_rows", 0))
    if int(failed_rows) != 0:
        raise ValueError("tokenizer review has failed rows")
    if controller_review.get("status") != "PASS_R4_AFTER_868016_COORDINATE_PIVOT_CONTROLLER_TEACHER_FORCED_GATE":
        raise ValueError("controller review is not the reviewed 868114 teacher-forced pass")
    if not bool(controller_review.get("teacher_forced_gate_pass")):
        raise ValueError("controller teacher-forced gate did not pass")


def write_plan_summary(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    generation_started: bool,
) -> None:
    write_json_new(
        output_dir / "r4_after_868016_controller_generation_plan_summary.json",
        {
            "schema_name": "natural_evidence_v2_r4_after_868016_controller_generation_plan_or_summary_v1",
            "artifact_role": "r4_after_868016_coordinate_pivot_controller_generation_plan_or_summary",
            "score_rows": str(resolve(args.score_rows)),
            "score_rows_sha256": sha256_file(resolve(args.score_rows)),
            "allocation_rows": str(resolve(args.allocation_rows)) if args.allocation_rows is not None else "",
            "allocation_rows_sha256": sha256_file(resolve(args.allocation_rows)) if args.allocation_rows is not None else "",
            "assigned_shard_index": int(args.assigned_shard_index) if args.assigned_shard_index is not None else "",
            "tokenizer_review": str(resolve(args.tokenizer_review)),
            "tokenizer_review_sha256": sha256_file(resolve(args.tokenizer_review)),
            "controller_review": str(resolve(args.controller_review)),
            "controller_review_sha256": sha256_file(resolve(args.controller_review)),
            "prompt_index_start": int(args.prompt_index_start),
            "prompt_index_end": int(args.prompt_index_end),
            "row_count": len(rows),
            "generation_conditions": ["protected", "raw", "task_only"],
            "decode_conditions": ["protected", "raw", "task_only", "wrong_key", "wrong_payload"],
            "generation_unit": "prefix_native_row_cylinder",
            "controller_applies_to": ["protected"],
            "controller_policy": "committed",
            "controller_bonus_nats": float(args.controller_bonus_nats),
            "controller_penalty_nats": float(args.controller_penalty_nats),
            "controller_max_target_mass": float(args.controller_max_target_mass),
            "controller_max_kl_budget": float(args.controller_max_kl_budget),
            "model_name": str(args.model_name),
            "tokenizer_name": str(args.tokenizer_name),
            "max_new_tokens": int(args.max_new_tokens),
            "replicate_group_id": str(args.replicate_group_id),
            "generation_started": bool(generation_started),
            "training_started": False,
            "llama_started": False,
            "far_aggregation_started": False,
            "paper_claim_allowed": False,
        },
    )


def _controlled_scores_for_first_step(
    *,
    scores: Any,
    row_target_ids: Sequence[Sequence[int]],
    row_other_ids: Sequence[Sequence[int]],
    controller_config: ControllerConfig,
) -> Any:
    import torch

    adjusted = scores.clone()
    for row_index, (target_ids, other_ids) in enumerate(zip(row_target_ids, row_other_ids, strict=True)):
        target_values = sorted({int(item) for item in target_ids})
        other_values = sorted({int(item) for item in other_ids})
        overlap = sorted(set(target_values) & set(other_values))
        if not target_values or not other_values or overlap:
            raise ValueError(f"invalid controller token ids for batch row {row_index}: overlap={overlap}")
        base_logits = scores[row_index].float()
        base_probs = torch.softmax(base_logits, dim=-1)
        target_tensor = torch.tensor(target_values, dtype=torch.long, device=scores.device)
        other_tensor = torch.tensor(other_values, dtype=torch.long, device=scores.device)

        def candidate(scale: float) -> tuple[Any, float, float]:
            row_logits = base_logits.clone()
            row_logits.index_add_(
                0,
                target_tensor,
                torch.full(
                    (target_tensor.numel(),),
                    float(controller_config.bonus_nats) * float(scale),
                    dtype=row_logits.dtype,
                    device=row_logits.device,
                ),
            )
            if float(controller_config.penalty_nats) > 0:
                row_logits.index_add_(
                    0,
                    other_tensor,
                    torch.full(
                        (other_tensor.numel(),),
                        -float(controller_config.penalty_nats) * float(scale),
                        dtype=row_logits.dtype,
                        device=row_logits.device,
                    ),
                )
            row_probs = torch.softmax(row_logits, dim=-1)
            target_mass = float(row_probs.index_select(0, target_tensor).sum().detach().cpu())
            kl_value = float(
                (
                    row_probs
                    * (
                        torch.log(row_probs.clamp_min(1e-45))
                        - torch.log(base_probs.clamp_min(1e-45))
                    )
                )
                .sum()
                .detach()
                .cpu()
            )
            return row_logits, target_mass, kl_value

        scale = 1.0
        _, target_mass, kl_value = candidate(scale)
        if target_mass > float(controller_config.max_target_mass) or kl_value > float(controller_config.max_kl_budget):
            low = 0.0
            high = 1.0
            for _ in range(40):
                mid = (low + high) / 2.0
                _, mid_mass, mid_kl = candidate(mid)
                if mid_mass <= float(controller_config.max_target_mass) and mid_kl <= float(controller_config.max_kl_budget):
                    low = mid
                else:
                    high = mid
            scale = low
        adjusted[row_index] = candidate(scale)[0].to(dtype=scores.dtype)
    return adjusted


class FirstStepControllerLogitsProcessor:
    def __init__(
        self,
        *,
        initial_width: int,
        row_target_ids: Sequence[Sequence[int]],
        row_other_ids: Sequence[Sequence[int]],
        controller_config: ControllerConfig,
    ) -> None:
        self.initial_width = int(initial_width)
        self.row_target_ids = [list(ids) for ids in row_target_ids]
        self.row_other_ids = [list(ids) for ids in row_other_ids]
        self.controller_config = controller_config

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if int(input_ids.shape[1]) != self.initial_width:
            return scores
        return _controlled_scores_for_first_step(
            scores=scores,
            row_target_ids=self.row_target_ids,
            row_other_ids=self.row_other_ids,
            controller_config=self.controller_config,
        )


def first_token_event_trace(
    *,
    first_generated_token_id: int | None,
    first_generated_token_text: str,
    target_token_ids: Sequence[int],
    other_token_ids: Sequence[int],
    target_bit: int,
) -> dict[str, Any]:
    target_values = sorted({int(item) for item in target_token_ids})
    other_values = sorted({int(item) for item in other_token_ids})
    overlap = sorted(set(target_values) & set(other_values))
    if overlap:
        raise ValueError(f"target/other first-token ids overlap: {overlap}")
    if first_generated_token_id is None:
        event_side = "erasure"
        event_bucket_side: int | str = ""
    elif int(first_generated_token_id) in target_values:
        event_side = "target"
        event_bucket_side = int(target_bit)
    elif int(first_generated_token_id) in other_values:
        event_side = "other"
        event_bucket_side = 1 - int(target_bit)
    else:
        event_side = "erasure"
        event_bucket_side = ""
    return {
        "event_side": event_side,
        "event_bucket_side": event_bucket_side,
        "first_generated_token_id": first_generated_token_id,
        "first_generated_token_text": first_generated_token_text,
        "target_first_token_ids": target_values,
        "other_first_token_ids": other_values,
    }


def generate_condition(
    *,
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    condition: str,
    adapter_path: Path | None,
    controller_enabled: bool,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer, LogitsProcessorList

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model, device = load_model_condition(
        model_name=args.model_name,
        adapter_path=adapter_path,
        require_cuda=bool(args.require_cuda),
    )
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    controller_config = ControllerConfig(
        mode="additive" if controller_enabled else "disabled",
        bonus_nats=float(args.controller_bonus_nats),
        penalty_nats=float(args.controller_penalty_nats),
        max_target_mass=float(args.controller_max_target_mass),
        max_kl_budget=float(args.controller_max_kl_budget),
    )
    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(batched(rows, args.batch_size)):
            prefix_texts: list[str] = []
            target_ids_by_row: list[list[int]] = []
            other_ids_by_row: list[list[int]] = []
            prefix_model_texts: list[str] = []
            for row in batch:
                contract = r4_row_surface_contract(row)
                prefix_model_text = chat_prefix(tokenizer, str(row.get("prompt_text", "")), contract["assistant_prefix_model_text"])
                prefix_texts.append(prefix_model_text)
                prefix_model_texts.append(contract["assistant_prefix_model_text"])
                target_ids = bucket_first_token_ids(
                    tokenizer,
                    prefix_model_text,
                    contract["target_tokenizer_scored_surface_texts"],
                )
                other_ids = bucket_first_token_ids(
                    tokenizer,
                    prefix_model_text,
                    contract["other_tokenizer_scored_surface_texts"],
                )
                controller_tokens = controller_token_ids_for_policy(
                    policy="committed",
                    row=row,
                    committed_target_ids=target_ids,
                    committed_other_ids=other_ids,
                )
                target_ids_by_row.append(controller_tokens["controller_target_ids"] if controller_enabled else target_ids)
                other_ids_by_row.append(controller_tokens["controller_other_ids"] if controller_enabled else other_ids)
            encoded = tokenizer(prefix_texts, add_special_tokens=False, return_tensors="pt", padding=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            processors = LogitsProcessorList()
            if controller_enabled:
                processors.append(
                    FirstStepControllerLogitsProcessor(
                        initial_width=int(encoded["input_ids"].shape[1]),
                        row_target_ids=target_ids_by_row,
                        row_other_ids=other_ids_by_row,
                        controller_config=controller_config,
                    )
                )
            generated = model.generate(
                **encoded,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=processors,
                max_new_tokens=max(1, int(args.max_new_tokens)),
                pad_token_id=tokenizer.pad_token_id,
            )
            input_width = int(encoded["input_ids"].shape[1])
            for row_index, (row, output_ids) in enumerate(zip(batch, generated, strict=True)):
                continuation_ids = [int(item) for item in output_ids[input_width:].detach().cpu().tolist()]
                first_generated_token_id = continuation_ids[0] if continuation_ids else None
                first_generated_token_text = (
                    tokenizer.decode([first_generated_token_id], skip_special_tokens=False)
                    if first_generated_token_id is not None
                    else ""
                )
                event_trace = first_token_event_trace(
                    first_generated_token_id=first_generated_token_id,
                    first_generated_token_text=first_generated_token_text,
                    target_token_ids=target_ids_by_row[row_index],
                    other_token_ids=other_ids_by_row[row_index],
                    target_bit=int(row["target_bit"]),
                )
                continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
                response_text = (prefix_model_texts[row_index] + continuation_text).strip()
                generation_id = "qwen_v2_r4_868016_gen_" + sha256_text(
                    json.dumps(
                        {
                            "condition": condition,
                            "coordinate_id": int(row["coordinate_id"]),
                            "prompt_id": str(row["prompt_id"]),
                            "replicate_group_id": str(args.replicate_group_id),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )[:20]
                outputs.append(
                    {
                        "adapter_path": str(adapter_path) if adapter_path is not None else "",
                        "artifact_role": "r4_after_868016_coordinate_pivot_controller_generation_transcript",
                        "contract_id": "a55e",
                        "controller_applied": bool(controller_enabled),
                        "controller_bonus_nats": float(args.controller_bonus_nats) if controller_enabled else 0.0,
                        "controller_max_kl_budget": float(args.controller_max_kl_budget) if controller_enabled else None,
                        "controller_max_target_mass": float(args.controller_max_target_mass) if controller_enabled else None,
                        "controller_penalty_nats": float(args.controller_penalty_nats) if controller_enabled else 0.0,
                        "coordinate_id": int(row["coordinate_id"]),
                        "event_bucket_side": event_trace["event_bucket_side"],
                        "event_side": event_trace["event_side"],
                        "event_trace": event_trace,
                        "first_generated_token_id": event_trace["first_generated_token_id"],
                        "first_generated_token_text": event_trace["first_generated_token_text"],
                        "generation_condition": condition,
                        "generation_id": generation_id,
                        "generation_mode": "deterministic_greedy_first_step_controller" if controller_enabled else "deterministic_greedy",
                        "generation_unit": "prefix_native_row_cylinder",
                        "max_new_tokens": int(args.max_new_tokens),
                        "model_name": str(args.model_name),
                        "paper_claim_allowed": False,
                        "prefix_family_id": str(row.get("prefix_family_id", "")),
                        "prompt_id": str(row["prompt_id"]),
                        "prompt_index": int(row["prompt_index"]),
                        "prompt_text": str(row["prompt_text"]),
                        "replicate_group_id": str(args.replicate_group_id),
                        "response_text": response_text,
                        "response_text_sha256": sha256_text(response_text),
                        "schema_name": "natural_evidence_v2_r4_generated_output_v1",
                        "split": str(row.get("split", "")),
                        "target_bit": int(row["target_bit"]),
                        "target_first_token_ids": event_trace["target_first_token_ids"],
                        "target_surface": str(row.get("target_surface", "")),
                        "tokenizer_name": str(args.tokenizer_name),
                        "training_started": False,
                        "other_first_token_ids": event_trace["other_first_token_ids"],
                    }
                )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    rows_path = resolve(args.allocation_rows) if args.allocation_rows is not None else resolve(args.score_rows)
    tokenizer_review = read_json(resolve(args.tokenizer_review))
    controller_review = read_json(resolve(args.controller_review))
    validate_reviews(tokenizer_review, controller_review)
    if args.allocation_rows is not None:
        if args.assigned_shard_index is None:
            raise ValueError("--assigned-shard-index is required with --allocation-rows")
        rows = select_rows_by_allocation(
            read_jsonl(rows_path),
            assigned_shard_index=int(args.assigned_shard_index),
            expected_rows=int(args.expected_rows),
        )
    else:
        rows = select_rows_by_prompt_range(
            read_jsonl(rows_path),
            start=int(args.prompt_index_start),
            end=int(args.prompt_index_end),
            expected_rows=int(args.expected_rows),
        )
    if args.validate_plan_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_plan_summary(args=args, output_dir=output_dir, rows=rows, generation_started=False)
        print(json.dumps({"status": "PLAN_ONLY_PASS", "output_dir": str(output_dir), "rows": len(rows)}, sort_keys=True))
        return 0
    task_only_adapter = resolve(args.task_only_adapter)
    if not (task_only_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"task-only adapter missing: {task_only_adapter}")
    if (output_dir / "r4_generated_outputs.jsonl").exists():
        raise FileExistsError(f"refusing to overwrite generated outputs: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_rows: list[dict[str, Any]] = []
    for condition, adapter_path, controller_enabled in (
        ("protected", None, True),
        ("raw", None, False),
        ("task_only", task_only_adapter, False),
    ):
        generated_rows.extend(
            generate_condition(
                args=args,
                rows=rows,
                condition=condition,
                adapter_path=adapter_path,
                controller_enabled=controller_enabled,
            )
        )
    write_jsonl_new(output_dir / "r4_generated_outputs.jsonl", generated_rows)
    write_plan_summary(args=args, output_dir=output_dir, rows=rows, generation_started=True)
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "rows": len(generated_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
