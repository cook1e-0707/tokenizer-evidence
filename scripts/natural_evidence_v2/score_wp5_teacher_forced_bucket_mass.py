from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_PRIMARY_BANK = ROOT / "results/natural_evidence_v2/buckets/qwen_v2_primary_2way_bank.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score natural_evidence_v2 WP5 teacher-forced target bucket mass for "
            "base/protected/task-only conditions. This does not generate free "
            "outputs, run E2E, aggregate FAR, or make positive claims."
        )
    )
    parser.add_argument("--score-rows", type=Path, required=True)
    parser.add_argument("--primary-bank", type=Path, default=DEFAULT_PRIMARY_BANK)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--protected-adapter", type=Path, default=None)
    parser.add_argument("--task-only-adapter", type=Path, default=None)
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


def load_bank(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"expected one primary bank row: {path}")
    bank = rows[0]
    if bank.get("schema_name") != "natural_evidence_v2_primary_2way_micro_slot_bank_v1":
        raise ValueError("primary bank schema mismatch")
    return {
        "bank_id": bank.get("bank_id"),
        "bucket_0_token_ids": [int(item) for item in bank.get("bucket_0_token_ids", [])],
        "bucket_1_token_ids": [int(item) for item in bank.get("bucket_1_token_ids", [])],
    }


def chat_prefix(tokenizer: Any, prompt_text: str, assistant_prefix: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    if getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)) + assistant_prefix
    return f"User: {prompt_text}\nAssistant: {assistant_prefix}"


def condition_plan(args: argparse.Namespace) -> list[tuple[str, Path | None]]:
    plan: list[tuple[str, Path | None]] = [("base", None)]
    if args.protected_adapter is not None:
        plan.append(("protected", args.protected_adapter))
    if args.task_only_adapter is not None:
        plan.append(("task_only", args.task_only_adapter))
    return plan


def batched(items: Sequence[Mapping[str, Any]], batch_size: int) -> Iterable[list[Mapping[str, Any]]]:
    for offset in range(0, len(items), batch_size):
        yield list(items[offset : offset + batch_size])


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def summarize(scored_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for row in scored_rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    condition_summary: dict[str, dict[str, Any]] = {}
    for condition, rows in by_condition.items():
        margins = [float(row["target_margin"]) for row in rows]
        target_masses = [float(row["target_mass"]) for row in rows]
        condition_summary[condition] = {
            "mean_target_margin": mean(margins),
            "mean_target_mass": mean(target_masses),
            "median_target_margin": median(margins),
            "row_count": len(rows),
            "target_bucket_rank1_rate": mean([1.0 if bool(row["target_bucket_rank1"]) else 0.0 for row in rows]),
        }

    base_mass = condition_summary.get("base", {}).get("mean_target_mass", float("nan"))
    protected_mass = condition_summary.get("protected", {}).get("mean_target_mass", float("nan"))
    task_mass = condition_summary.get("task_only", {}).get("mean_target_mass", float("nan"))
    protected_lift_vs_base = protected_mass - base_mass
    protected_lift_vs_task = protected_mass - task_mass
    task_lift_vs_base = task_mass - base_mass
    rank1 = condition_summary.get("protected", {}).get("target_bucket_rank1_rate", 0.0)
    protected_median_margin = condition_summary.get("protected", {}).get("median_target_margin", float("nan"))
    gate_pass = (
        protected_lift_vs_base >= 0.15
        and protected_lift_vs_task >= 0.10
        and rank1 >= 0.70
        and protected_median_margin > 0
        and task_lift_vs_base < 0.05
    )
    return {
        "condition_summary": condition_summary,
        "teacher_forced_gate_pass": bool(gate_pass),
        "teacher_forced_gate_status": "PASS" if gate_pass else "FAIL",
        "teacher_forced_gate_targets": {
            "protected_target_bucket_mass_lift_vs_base": ">=+0.15",
            "protected_target_bucket_mass_lift_vs_task_only": ">=+0.10",
            "protected_target_bucket_rank1_rate": ">=0.70",
            "protected_median_target_margin": ">0",
            "task_only_target_bucket_mass_lift_vs_base": "<+0.05 diagnostic cap",
        },
        "observed_lifts": {
            "protected_target_bucket_mass_lift_vs_base": protected_lift_vs_base,
            "protected_target_bucket_mass_lift_vs_task_only": protected_lift_vs_task,
            "task_only_target_bucket_mass_lift_vs_base": task_lift_vs_base,
        },
    }


def score_condition(
    *,
    args: argparse.Namespace,
    adapter_path: Path | None,
    bank: Mapping[str, Any],
    condition: str,
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
    model.eval()

    scored: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_rows in batched(rows, args.batch_size):
            prefixes = [
                chat_prefix(
                    tokenizer,
                    str(row.get("prompt_text", "")),
                    str(row.get("assistant_prefix_before_surface", "")),
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
                logits = outputs.logits[batch_index, last_positions[batch_index], :]
                probs = torch.softmax(logits, dim=-1)
                target_bucket_id = str(row.get("target_bucket_id"))
                other_bucket_id = "1" if target_bucket_id == "0" else "0"
                target_ids = torch.tensor(bank[f"bucket_{target_bucket_id}_token_ids"], dtype=torch.long, device=device)
                other_ids = torch.tensor(bank[f"bucket_{other_bucket_id}_token_ids"], dtype=torch.long, device=device)
                target_mass = float(probs.index_select(0, target_ids).sum().detach().cpu())
                other_mass = float(probs.index_select(0, other_ids).sum().detach().cpu())
                scored.append(
                    {
                        "condition": condition,
                        "e2e_eval_started": False,
                        "model_generation_started": False,
                        "model_scoring_started": True,
                        "not_full_far": True,
                        "not_payload_recovery": True,
                        "paper_claim_allowed": False,
                        "prompt_id": row.get("prompt_id"),
                        "response_id": row.get("response_id"),
                        "schema_name": "natural_evidence_v2_wp5_teacher_forced_bucket_mass_row_v1",
                        "slot_index": row.get("slot_index"),
                        "target_bucket_id": target_bucket_id,
                        "target_bucket_rank1": target_mass > other_mass,
                        "target_margin": target_mass - other_mass,
                        "target_mass": target_mass,
                        "target_surface": row.get("target_surface"),
                        "training_started": False,
                        "other_bucket_id": other_bucket_id,
                        "other_mass": other_mass,
                    }
                )
    return scored


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output dir: {output_dir}")
    output_dir.mkdir(parents=True)
    rows = read_jsonl(args.score_rows)[: args.max_rows]
    bank = load_bank(args.primary_bank)
    if not rows:
        raise ValueError("no score rows selected")
    if args.dry_run:
        summary = {
            "artifact_role": "wp5_teacher_forced_bucket_mass_dry_run",
            "condition_plan": [condition for condition, _ in condition_plan(args)],
            "e2e_eval_started": False,
            "model_generation_started": False,
            "model_scoring_started": False,
            "not_full_far": True,
            "not_payload_recovery": True,
            "paper_claim_allowed": False,
            "schema_name": "natural_evidence_v2_wp5_teacher_forced_bucket_mass_summary_v1",
            "score_row_count": len(rows),
            "status": "DRY_RUN_VALIDATED_INPUTS",
            "training_started": False,
        }
        write_json(output_dir / "wp5_teacher_forced_bucket_mass_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but CUDA is not available")
    all_scored: list[dict[str, Any]] = []
    for condition, adapter_path in condition_plan(args):
        if adapter_path is not None and not adapter_path.exists():
            raise FileNotFoundError(f"adapter path missing for {condition}: {adapter_path}")
        all_scored.extend(
            score_condition(args=args, adapter_path=adapter_path, bank=bank, condition=condition, rows=rows)
        )
    summary = {
        **summarize(all_scored),
        "artifact_role": "wp5_teacher_forced_bucket_mass_scoring",
        "condition_plan": [condition for condition, _ in condition_plan(args)],
        "e2e_eval_started": False,
        "model_generation_started": False,
        "model_scoring_started": True,
        "not_full_far": True,
        "not_payload_recovery": True,
        "paper_claim_allowed": False,
        "row_count": len(all_scored),
        "schema_name": "natural_evidence_v2_wp5_teacher_forced_bucket_mass_summary_v1",
        "training_started": False,
    }
    write_jsonl(output_dir / "wp5_teacher_forced_bucket_mass_rows.jsonl", all_scored)
    write_json(output_dir / "wp5_teacher_forced_bucket_mass_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
