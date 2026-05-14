from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score R4 cover-natural teacher-forced target surface mass for "
            "base/protected/task-only conditions. This does not train, generate "
            "free outputs, run Llama, aggregate FAR, or make paper claims."
        )
    )
    parser.add_argument("--score-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--protected-adapter", type=Path, default=None)
    parser.add_argument("--task-only-adapter", type=Path, default=None)
    parser.add_argument(
        "--protected-adapter-gains",
        default="1.0",
        help=(
            "Comma-separated protected LoRA gain multipliers. Default 1.0 keeps "
            "the historical protected condition. Multiple gains produce "
            "protected_gain_* conditions. This does not affect base/task-only."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


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


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_new(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def batched(items: Sequence[Mapping[str, Any]], batch_size: int) -> Iterable[list[Mapping[str, Any]]]:
    for offset in range(0, len(items), batch_size):
        yield list(items[offset : offset + batch_size])


def parse_gain_values(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        if not item.strip():
            continue
        value = float(item)
        if value < 0:
            raise ValueError("protected adapter gains must be non-negative")
        values.append(value)
    if not values:
        raise ValueError("at least one protected adapter gain is required")
    if len(set(values)) != len(values):
        raise ValueError("protected adapter gains must be unique")
    return values


def gain_condition_name(gain: float, *, historical_single_gain: bool = False) -> str:
    if historical_single_gain and gain == 1.0:
        return "protected"
    value = ("%g" % float(gain)).replace(".", "_")
    return f"protected_gain_{value}"


def condition_plan(args: argparse.Namespace) -> list[tuple[str, Path | None, float | None]]:
    plan: list[tuple[str, Path | None, float | None]] = [("base", None, None)]
    if args.protected_adapter is not None:
        gains = parse_gain_values(args.protected_adapter_gains)
        historical_single_gain = gains == [1.0]
        for gain in gains:
            plan.append((gain_condition_name(gain, historical_single_gain=historical_single_gain), args.protected_adapter, gain))
    if args.task_only_adapter is not None:
        plan.append(("task_only", args.task_only_adapter, None))
    return plan


def scale_peft_lora_adapters(model: Any, gain: float) -> dict[str, Any]:
    """Scale PEFT LoRA adapter strengths by multiplying module scaling values.

    The scorer loads a fresh model per condition, so this does not need to reset
    scaling values after scoring. It fails closed if a positive gain is requested
    but no LoRA scaling table is found.
    """

    gain = float(gain)
    touched_modules = 0
    touched_adapter_keys = 0
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and scaling:
            for key in list(scaling.keys()):
                scaling[key] = scaling[key] * gain
                touched_adapter_keys += 1
            touched_modules += 1
    if gain > 0 and touched_adapter_keys == 0:
        raise ValueError("requested protected adapter gain scaling but no PEFT LoRA scaling entries were found")
    return {
        "adapter_gain": gain,
        "lora_scaling_keys_touched": touched_adapter_keys,
        "lora_scaling_modules_touched": touched_modules,
    }


@dataclass(frozen=True)
class R4SurfaceBoundary:
    assistant_prefix_before_surface: str
    assistant_prefix_model_text: str
    surface_prefix_text: str

    def tokenizer_surface(self, surface_label: Any) -> str:
        return self.surface_prefix_text + str(surface_label)

    def tokenizer_surfaces(self, surface_labels: Sequence[Any]) -> list[str]:
        return [self.tokenizer_surface(surface_label) for surface_label in surface_labels]


def split_r4_surface_boundary(row: Mapping[str, Any]) -> R4SurfaceBoundary:
    assistant_prefix = str(row.get("assistant_prefix_before_surface", ""))
    assistant_prefix_model_text = assistant_prefix.rstrip()
    return R4SurfaceBoundary(
        assistant_prefix_before_surface=assistant_prefix,
        assistant_prefix_model_text=assistant_prefix_model_text,
        surface_prefix_text=assistant_prefix[len(assistant_prefix_model_text) :],
    )


def r4_row_surface_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    boundary = split_r4_surface_boundary(row)
    target_bit = str(int(row["target_bit"]))
    other_bit = "1" if target_bit == "0" else "0"
    target_labels = [str(surface) for surface in row[f"bucket_{target_bit}_surfaces"]]
    other_labels = [str(surface) for surface in row[f"bucket_{other_bit}_surfaces"]]
    return {
        "assistant_prefix_before_surface": boundary.assistant_prefix_before_surface,
        "assistant_prefix_model_text": boundary.assistant_prefix_model_text,
        "surface_prefix_text": boundary.surface_prefix_text,
        "target_bit": int(target_bit),
        "other_bit": int(other_bit),
        "target_surface_label": str(row.get("target_surface", "")),
        "target_tokenizer_scored_surface_text": boundary.tokenizer_surface(row.get("target_surface", "")),
        "target_surface_labels": target_labels,
        "other_surface_labels": other_labels,
        "target_tokenizer_scored_surface_texts": boundary.tokenizer_surfaces(target_labels),
        "other_tokenizer_scored_surface_texts": boundary.tokenizer_surfaces(other_labels),
    }


def r4_boundary_failure_context(row: Mapping[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {
        "prompt_id": row.get("prompt_id"),
        "prompt_index": row.get("prompt_index"),
        "coordinate_id": row.get("coordinate_id"),
        "target_bit": row.get("target_bit"),
        "target_surface": row.get("target_surface"),
        "assistant_prefix_before_surface": row.get("assistant_prefix_before_surface"),
        "bucket_0_surfaces": row.get("bucket_0_surfaces"),
        "bucket_1_surfaces": row.get("bucket_1_surfaces"),
    }
    try:
        boundary = split_r4_surface_boundary(row)
        context.update(
            {
                "assistant_prefix_model_text": boundary.assistant_prefix_model_text,
                "surface_prefix_text": boundary.surface_prefix_text,
            }
        )
    except Exception as exc:
        context["boundary_split_exception"] = f"{type(exc).__name__}:{exc}"
        return context

    try:
        target_bit = str(int(row["target_bit"]))
        other_bit = "1" if target_bit == "0" else "0"
        target_labels = [str(surface) for surface in row[f"bucket_{target_bit}_surfaces"]]
        other_labels = [str(surface) for surface in row[f"bucket_{other_bit}_surfaces"]]
    except Exception as exc:
        context["surface_label_derivation_exception"] = f"{type(exc).__name__}:{exc}"
        return context

    context.update(
        {
            "other_bit": int(other_bit),
            "target_surface_label": str(row.get("target_surface", "")),
            "target_surface_labels": target_labels,
            "other_surface_labels": other_labels,
            "target_tokenizer_scored_surface_text": boundary.tokenizer_surface(row.get("target_surface", "")),
            "target_tokenizer_scored_surface_texts": boundary.tokenizer_surfaces(target_labels),
            "other_tokenizer_scored_surface_texts": boundary.tokenizer_surfaces(other_labels),
        }
    )
    return context


def chat_prefix(tokenizer: Any, prompt_text: str, assistant_prefix: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    if getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)) + assistant_prefix
    return f"User: {prompt_text}\nAssistant: {assistant_prefix}"


def first_token_id_after_prefix(tokenizer: Any, prefix: str, surface: str) -> int:
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    combined_ids = tokenizer(prefix + surface, add_special_tokens=False).input_ids
    if len(combined_ids) <= len(prefix_ids):
        raise ValueError(f"surface produced no next token: {surface!r}")
    if combined_ids[: len(prefix_ids)] == prefix_ids:
        return int(combined_ids[len(prefix_ids)])
    surface_ids = tokenizer(surface, add_special_tokens=False).input_ids
    if not surface_ids:
        raise ValueError(f"surface tokenized to empty ids: {surface!r}")
    return int(surface_ids[0])


def bucket_first_token_ids(tokenizer: Any, prefix: str, surfaces: Sequence[str]) -> list[int]:
    ids: list[int] = []
    for surface in surfaces:
        ids.append(first_token_id_after_prefix(tokenizer, prefix, str(surface)))
    return sorted(set(ids))


def validate_static_boundary_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        try:
            contract = r4_row_surface_contract(row)
        except Exception as exc:
            failures.append(
                {
                    **r4_boundary_failure_context(row),
                    "failure_reasons": [f"boundary_contract_exception:{type(exc).__name__}:{exc}"],
                }
            )
            continue
        expected_model_text = contract["assistant_prefix_before_surface"].rstrip()
        expected_surface_prefix = contract["assistant_prefix_before_surface"][len(expected_model_text) :]
        if contract["assistant_prefix_model_text"] != expected_model_text:
            reasons.append("assistant_prefix_model_text_mismatch")
        if contract["surface_prefix_text"] != expected_surface_prefix:
            reasons.append("surface_prefix_text_mismatch")
        if not contract["target_tokenizer_scored_surface_texts"]:
            reasons.append("empty_target_tokenizer_scored_surface_texts")
        if not contract["other_tokenizer_scored_surface_texts"]:
            reasons.append("empty_other_tokenizer_scored_surface_texts")
        for label, tokenizer_surface in zip(
            contract["target_surface_labels"], contract["target_tokenizer_scored_surface_texts"]
        ):
            if tokenizer_surface != contract["surface_prefix_text"] + label:
                reasons.append("target_tokenizer_scored_surface_text_mismatch")
                break
        for label, tokenizer_surface in zip(
            contract["other_surface_labels"], contract["other_tokenizer_scored_surface_texts"]
        ):
            if tokenizer_surface != contract["surface_prefix_text"] + label:
                reasons.append("other_tokenizer_scored_surface_text_mismatch")
                break
        if reasons:
            failures.append(
                {
                    "prompt_id": row.get("prompt_id"),
                    "prompt_index": row.get("prompt_index"),
                    "coordinate_id": row.get("coordinate_id"),
                    "target_bit": row.get("target_bit"),
                    "failure_reasons": reasons,
                    **contract,
                }
            )
    return {
        "status": (
            "PASS_STATIC_BOUNDARY_CONTRACT_TOKENIZER_PENDING"
            if not failures
            else "FAIL_STATIC_BOUNDARY_CONTRACT_TOKENIZER_BLOCKED"
        ),
        "checked_row_count": len(rows),
        "failed_row_count": len(failures),
        "first_failing_row": failures[0] if failures else None,
    }


def validate_qwen_tokenizer_boundary_contract(tokenizer: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    empty_target_id_row_count = 0
    empty_other_id_row_count = 0
    target_other_overlap_row_count = 0
    for row in rows:
        try:
            contract = r4_row_surface_contract(row)
            prefix_text = chat_prefix(
                tokenizer,
                str(row.get("prompt_text", "")),
                contract["assistant_prefix_model_text"],
            )
            target_ids = bucket_first_token_ids(tokenizer, prefix_text, contract["target_tokenizer_scored_surface_texts"])
            other_ids = bucket_first_token_ids(tokenizer, prefix_text, contract["other_tokenizer_scored_surface_texts"])
        except Exception as exc:
            failures.append(
                {
                    **r4_boundary_failure_context(row),
                    "failure_reasons": [f"tokenizer_boundary_exception:{type(exc).__name__}:{exc}"],
                }
            )
            continue
        overlap = sorted(set(target_ids) & set(other_ids))
        reasons: list[str] = []
        if not target_ids:
            empty_target_id_row_count += 1
            reasons.append("empty_target_first_token_ids")
        if not other_ids:
            empty_other_id_row_count += 1
            reasons.append("empty_other_first_token_ids")
        if overlap:
            target_other_overlap_row_count += 1
            reasons.append("target_other_first_token_id_overlap")
        for label, tokenizer_surface in zip(
            contract["target_surface_labels"], contract["target_tokenizer_scored_surface_texts"]
        ):
            if tokenizer_surface != contract["surface_prefix_text"] + label:
                reasons.append("target_tokenizer_scored_surface_text_mismatch")
                break
        for label, tokenizer_surface in zip(
            contract["other_surface_labels"], contract["other_tokenizer_scored_surface_texts"]
        ):
            if tokenizer_surface != contract["surface_prefix_text"] + label:
                reasons.append("other_tokenizer_scored_surface_text_mismatch")
                break
        if reasons:
            failures.append(
                {
                    "prompt_id": row.get("prompt_id"),
                    "prompt_index": row.get("prompt_index"),
                    "coordinate_id": row.get("coordinate_id"),
                    "target_bit": row.get("target_bit"),
                    "failure_reasons": reasons,
                    "target_first_token_ids": target_ids,
                    "other_first_token_ids": other_ids,
                    "target_other_first_token_id_overlap": overlap,
                    **contract,
                }
            )
    return {
        "status": (
            "PASS_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT"
            if not failures
            else "FAIL_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT"
        ),
        "checked_row_count": len(rows),
        "failed_row_count": len(failures),
        "first_failing_row": failures[0] if failures else None,
        "empty_target_id_row_count": empty_target_id_row_count,
        "empty_other_id_row_count": empty_other_id_row_count,
        "target_other_overlap_row_count": target_other_overlap_row_count,
    }


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def summarize(scored_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for row in scored_rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    condition_summary: dict[str, dict[str, Any]] = {}
    for condition, rows in sorted(by_condition.items()):
        target_masses = [float(row["target_mass"]) for row in rows]
        target_margins = [float(row["target_margin"]) for row in rows]
        condition_summary[condition] = {
            "mean_target_margin": mean(target_margins),
            "mean_target_mass": mean(target_masses),
            "median_target_margin": median(target_margins),
            "row_count": len(rows),
            "target_surface_rank1_rate": mean([1.0 if bool(row["target_surface_rank1"]) else 0.0 for row in rows]),
        }
    base_mass = condition_summary.get("base", {}).get("mean_target_mass", float("nan"))
    protected_mass = condition_summary.get("protected", {}).get("mean_target_mass", float("nan"))
    task_mass = condition_summary.get("task_only", {}).get("mean_target_mass", float("nan"))
    protected_lift_vs_base = protected_mass - base_mass
    protected_lift_vs_task = protected_mass - task_mass
    task_lift_vs_base = task_mass - base_mass
    rank1 = condition_summary.get("protected", {}).get("target_surface_rank1_rate", 0.0)
    protected_median_margin = condition_summary.get("protected", {}).get("median_target_margin", float("nan"))
    gate_pass = (
        protected_lift_vs_base >= 0.15
        and protected_lift_vs_task >= 0.10
        and rank1 >= 0.70
        and protected_median_margin > 0
        and task_lift_vs_base < 0.05
    )
    gain_sweep_summary: dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        if not condition.startswith("protected_gain_"):
            continue
        condition_mass = condition_summary.get(condition, {}).get("mean_target_mass", float("nan"))
        condition_rank1 = condition_summary.get(condition, {}).get("target_surface_rank1_rate", 0.0)
        condition_median_margin = condition_summary.get(condition, {}).get("median_target_margin", float("nan"))
        gain_sweep_summary[condition] = {
            "lift_vs_base": condition_mass - base_mass,
            "lift_vs_task_only": condition_mass - task_mass,
            "mean_target_mass": condition_mass,
            "median_target_margin": condition_median_margin,
            "target_surface_rank1_rate": condition_rank1,
        }
    return {
        "condition_summary": condition_summary,
        "gain_sweep_summary": gain_sweep_summary,
        "observed_lifts": {
            "protected_target_surface_mass_lift_vs_base": protected_lift_vs_base,
            "protected_target_surface_mass_lift_vs_task_only": protected_lift_vs_task,
            "task_only_target_surface_mass_lift_vs_base": task_lift_vs_base,
        },
        "teacher_forced_surface_gate_pass": bool(gate_pass),
        "teacher_forced_surface_gate_status": "PASS" if gate_pass else "FAIL",
        "teacher_forced_surface_gate_targets": {
            "protected_target_surface_mass_lift_vs_base": ">=+0.15",
            "protected_target_surface_mass_lift_vs_task_only": ">=+0.10",
            "protected_target_surface_rank1_rate": ">=0.70",
            "protected_median_target_margin": ">0",
            "task_only_target_surface_mass_lift_vs_base": "<+0.05 diagnostic cap",
        },
    }


def score_condition(
    *,
    args: argparse.Namespace,
    condition: str,
    adapter_path: Path | None,
    adapter_gain: float | None,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path).to(device)
    adapter_gain_metadata = None
    if adapter_gain is not None:
        adapter_gain_metadata = scale_peft_lora_adapters(model, adapter_gain)
    model.eval()
    scored: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_rows in batched(rows, args.batch_size):
            prefixes = [
                chat_prefix(
                    tokenizer,
                    str(row.get("prompt_text", "")),
                    r4_row_surface_contract(row)["assistant_prefix_model_text"],
                )
                for row in batch_rows
            ]
            encoded = tokenizer(
                prefixes,
                add_special_tokens=False,
                max_length=args.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_positions = attention_mask.sum(dim=1) - 1
            for batch_index, row in enumerate(batch_rows):
                prefix_text = prefixes[batch_index]
                contract = r4_row_surface_contract(row)
                target_ids = bucket_first_token_ids(
                    tokenizer,
                    prefix_text,
                    contract["target_tokenizer_scored_surface_texts"],
                )
                other_ids = bucket_first_token_ids(
                    tokenizer,
                    prefix_text,
                    contract["other_tokenizer_scored_surface_texts"],
                )
                if not target_ids or not other_ids:
                    raise ValueError(f"empty bucket token ids for row {row.get('prompt_id')}:{row.get('coordinate_id')}")
                logits = outputs.logits[batch_index, last_positions[batch_index], :]
                probs = torch.softmax(logits, dim=-1)
                target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
                other_tensor = torch.tensor(other_ids, dtype=torch.long, device=device)
                target_mass = float(probs.index_select(0, target_tensor).sum().detach().cpu())
                other_mass = float(probs.index_select(0, other_tensor).sum().detach().cpu())
                scored.append(
                    {
                        "schema_name": "natural_evidence_v2_r4_teacher_forced_surface_mass_row_v1",
                        "condition": condition,
                        "adapter_gain": adapter_gain,
                        "adapter_gain_metadata": adapter_gain_metadata,
                        "prompt_id": row.get("prompt_id"),
                        "prompt_index": row.get("prompt_index"),
                        "coordinate_id": row.get("coordinate_id"),
                        "target_bit": contract["target_bit"],
                        "target_surface": row.get("target_surface"),
                        "target_surface_label": contract["target_surface_label"],
                        "surface_prefix_text": contract["surface_prefix_text"],
                        "assistant_prefix_model_text": contract["assistant_prefix_model_text"],
                        "target_tokenizer_scored_surface_text": contract["target_tokenizer_scored_surface_text"],
                        "target_bucket_tokenizer_scored_surface_texts": contract[
                            "target_tokenizer_scored_surface_texts"
                        ],
                        "other_bucket_tokenizer_scored_surface_texts": contract[
                            "other_tokenizer_scored_surface_texts"
                        ],
                        "target_mass": target_mass,
                        "other_mass": other_mass,
                        "target_margin": target_mass - other_mass,
                        "target_surface_rank1": target_mass > other_mass,
                        "target_first_token_ids": target_ids,
                        "other_first_token_ids": other_ids,
                        "model_generation_started": False,
                        "training_started": False,
                        "llama_started": False,
                        "paper_claim_allowed": False,
                    }
                )
    return scored


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output dir: {output_dir}")
    rows = read_jsonl(args.score_rows)[: int(args.max_rows)]
    if not rows:
        raise ValueError("no R4 surface score rows selected")
    output_dir.mkdir(parents=True)
    if args.dry_run:
        summary = {
            "schema_name": "natural_evidence_v2_r4_teacher_forced_surface_mass_summary_v1",
            "status": "DRY_RUN_VALIDATED_INPUTS",
            "score_row_count": len(rows),
            "condition_plan": [condition for condition, _, _ in condition_plan(args)],
            "protected_adapter_gains": parse_gain_values(args.protected_adapter_gains),
            "model_generation_started": False,
            "model_scoring_started": False,
            "training_started": False,
            "llama_started": False,
            "far_aggregation_started": False,
            "paper_claim_allowed": False,
        }
        write_json(output_dir / "r4_teacher_forced_surface_mass_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but CUDA is not available")
    scored_rows: list[dict[str, Any]] = []
    for condition, adapter_path, adapter_gain in condition_plan(args):
        if adapter_path is not None and not adapter_path.exists():
            raise FileNotFoundError(f"adapter path missing for {condition}: {adapter_path}")
        scored_rows.extend(
            score_condition(args=args, condition=condition, adapter_path=adapter_path, adapter_gain=adapter_gain, rows=rows)
        )
    summary = {
        **summarize(scored_rows),
        "schema_name": "natural_evidence_v2_r4_teacher_forced_surface_mass_summary_v1",
        "score_row_count": len(rows),
        "scored_row_count": len(scored_rows),
        "condition_plan": [condition for condition, _, _ in condition_plan(args)],
        "protected_adapter_gains": parse_gain_values(args.protected_adapter_gains),
        "model_generation_started": False,
        "model_scoring_started": True,
        "training_started": False,
        "llama_started": False,
        "far_aggregation_started": False,
        "paper_claim_allowed": False,
    }
    write_jsonl(output_dir / "r4_teacher_forced_surface_mass_rows.jsonl", scored_rows)
    write_json(output_dir / "r4_teacher_forced_surface_mass_summary.json", summary)
    print(json.dumps({"status": summary["teacher_forced_surface_gate_status"], "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
