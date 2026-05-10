from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
            "Generate natural_evidence_v2 WP6 free responses for Qwen protected, "
            "raw, and task-only conditions. Wrong-key and wrong-payload are "
            "decoded from the protected transcript by decode_wp6_payload.py. "
            "This script does not train, aggregate FAR, run Llama, or make "
            "paper-facing claims."
        )
    )
    parser.add_argument("--prompts-jsonl", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--wp4-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--protected-adapter", type=Path, required=True)
    parser.add_argument("--task-only-adapter", type=Path, required=True)
    parser.add_argument("--split", default="wp3_r1_eval")
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--prompt-file-row-start", type=int, default=None)
    parser.add_argument("--prompt-file-row-end", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=896)
    parser.add_argument("--validate-plan-only", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [row for _file_row, row in read_jsonl_with_file_index(path)]


def read_jsonl_with_file_index(path: Path) -> list[tuple[int, dict[str, Any]]]:
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_rows(path: Path, row_indices: Sequence[int]) -> str:
    selected = set(int(index) for index in row_indices)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for file_row_index, line in enumerate(handle):
            if file_row_index in selected:
                digest.update(line)
    return digest.hexdigest()


def select_prompts(
    rows: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    split: str,
    max_prompts: int,
    prompt_file_row_start: int | None = None,
    prompt_file_row_end: int | None = None,
) -> list[dict[str, Any]]:
    if (prompt_file_row_start is None) != (prompt_file_row_end is None):
        raise ValueError("prompt-file-row-start and prompt-file-row-end must be provided together")
    split_rows = [(file_row, dict(row)) for file_row, row in rows if str(row.get("split", "")) == split]
    if prompt_file_row_start is not None and prompt_file_row_end is not None:
        selected_with_index = [
            (file_row, row)
            for file_row, row in split_rows
            if int(prompt_file_row_start) <= file_row <= int(prompt_file_row_end)
        ]
        expected_count = int(prompt_file_row_end) - int(prompt_file_row_start) + 1
        if expected_count != max_prompts:
            raise ValueError("explicit prompt file-row window must match max-prompts")
    else:
        selected_with_index = split_rows[:max_prompts]
    if len(selected_with_index) < max_prompts:
        raise ValueError(f"split {split!r} has only {len(selected_with_index)} prompts; need {max_prompts}")
    output = []
    for selected_index, (file_row_index, row) in enumerate(selected_with_index[:max_prompts]):
        row["selected_prompt_file_row_index"] = int(file_row_index)
        row["selected_prompt_index"] = int(selected_index)
        output.append(row)
    for row in output:
        if int(row.get("expected_structural_slots", 0)) != 16:
            raise ValueError(f"prompt does not expect 16 slots: {row.get('prompt_id')}")
    if prompt_file_row_start is not None and prompt_file_row_end is not None:
        actual_start = int(output[0]["selected_prompt_file_row_index"])
        actual_end = int(output[-1]["selected_prompt_file_row_index"])
        if actual_start != int(prompt_file_row_start) or actual_end != int(prompt_file_row_end):
            raise ValueError(
                f"selected prompt file rows {actual_start}..{actual_end} do not match "
                f"expected {prompt_file_row_start}..{prompt_file_row_end}"
            )
    return output


def validate_contract(contract: Mapping[str, Any]) -> list[int]:
    payload = contract.get("payload", {})
    bits = list(payload.get("payload_bits_msb_first", [])) + list(payload.get("checksum_bits_msb_first", []))
    normalized = [int(bit) for bit in bits]
    if len(normalized) != 16 or any(bit not in {0, 1} for bit in normalized):
        raise ValueError("WP6 requires the WP5-aligned 16-bit prompt-local WP4 contract")
    if contract.get("schema_name") != "natural_evidence_v2_wp4_prompt_local_payload_contract_v1":
        raise ValueError("unexpected WP4 prompt-local payload contract schema")
    return normalized


def build_plan_summary(
    *,
    args: argparse.Namespace,
    prompts_path: Path,
    contract_path: Path,
    prompts: Sequence[Mapping[str, Any]],
    payload_bits: Sequence[int],
    generation_started: bool,
) -> dict[str, Any]:
    return {
        "artifact_role": "wp6_e2e_generation_plan_or_summary",
        "claim_control": {
            "e2e_eval_started": bool(generation_started),
            "generation_started": bool(generation_started),
            "not_full_far": True,
            "paper_claim_allowed": False,
            "training_started": False,
        },
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "generation_conditions": ["protected", "raw", "task_only"],
        "max_new_tokens": int(args.max_new_tokens),
        "max_prompts": int(args.max_prompts),
        "model_name": str(args.model_name),
        "payload_bits_msb_first": [int(bit) for bit in payload_bits],
        "prompt_count": len(prompts),
        "prompts_path": str(prompts_path),
        "prompts_sha256": sha256_file(prompts_path),
        "query_budgets": [8, 16, 32, 64],
        "schema_name": "natural_evidence_v2_wp6_generation_plan_v1",
        "selected_prompt_file_row_end_inclusive": int(prompts[-1]["selected_prompt_file_row_index"]),
        "selected_prompt_file_row_start": int(prompts[0]["selected_prompt_file_row_index"]),
        "selected_prompt_jsonl_sha256": sha256_file_rows(
            prompts_path, [int(row["selected_prompt_file_row_index"]) for row in prompts]
        ),
        "selected_prompt_rule": (
            "explicit prompt source file-row window"
            if args.prompt_file_row_start is not None
            else "first rows after filtering prompt_source by selected_split"
        ),
        "split": str(args.split),
        "tokenizer_name": str(args.tokenizer_name),
        "wp6_contract_note": "uses the WP5-R2 trained fixed prompt-local payload contract, not the older P00/P01 oracle contract",
    }


def build_chat_text(tokenizer: Any, prompt_text: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    if getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return f"User: {prompt_text}\nAssistant:"


def batched(items: Sequence[Mapping[str, Any]], batch_size: int) -> Iterable[list[Mapping[str, Any]]]:
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield [dict(row) for row in items[start : start + size]]


def load_model_condition(
    *,
    model_name: str,
    adapter_path: Path | None,
    require_cuda: bool,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True, "trust_remote_code": True}
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.to(device)
    model.eval()
    return model, device


def generate_condition(
    *,
    args: argparse.Namespace,
    prompts: Sequence[Mapping[str, Any]],
    condition: str,
    adapter_path: Path | None,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer

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

    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(batched(prompts, args.batch_size)):
            chat_texts = [build_chat_text(tokenizer, str(row["prompt_text"])) for row in batch]
            encoded = tokenizer(chat_texts, return_tensors="pt", padding=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max(1, int(args.max_new_tokens)),
                pad_token_id=tokenizer.pad_token_id,
            )
            input_width = int(encoded["input_ids"].shape[1])
            for row_index, (prompt_row, output_ids) in enumerate(zip(batch, generated, strict=True)):
                response_ids = output_ids[input_width:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                generation_id_payload = {
                    "batch_index": batch_index,
                    "condition": condition,
                    "prompt_id": str(prompt_row["prompt_id"]),
                    "row_index": row_index,
                }
                generation_id = "qwen_v2_wp6_gen_" + sha256_text(
                    json.dumps(generation_id_payload, sort_keys=True, separators=(",", ":"))
                )[:20]
                outputs.append(
                    {
                        "adapter_path": str(adapter_path) if adapter_path is not None else "",
                        "artifact_role": "wp6_e2e_free_generation_transcript",
                        "e2e_eval_started": True,
                        "generation_condition": condition,
                        "generation_id": generation_id,
                        "generation_mode": "deterministic_greedy",
                        "max_new_tokens": int(args.max_new_tokens),
                        "model_name": str(args.model_name),
                        "paper_claim_allowed": False,
                        "prompt_id": str(prompt_row["prompt_id"]),
                        "prompt_index": int(prompt_row.get("prompt_index", len(outputs))),
                        "prompt_text": str(prompt_row["prompt_text"]),
                        "prompt_text_sha256": str(prompt_row.get("prompt_text_sha256", "")),
                        "response_text": response_text,
                        "response_text_sha256": sha256_text(response_text),
                        "schema_name": "natural_evidence_v2_wp6_generated_output_v1",
                        "split": str(prompt_row.get("split", "")),
                        "tokenizer_name": str(args.tokenizer_name),
                        "training_started": False,
                    }
                )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def main() -> int:
    args = parse_args()
    prompts_path = resolve(args.prompts_jsonl)
    contract_path = resolve(args.wp4_contract)
    protected_adapter = resolve(args.protected_adapter)
    task_only_adapter = resolve(args.task_only_adapter)
    output_dir = resolve(args.output_dir)

    prompt_rows = read_jsonl_with_file_index(prompts_path)
    contract = read_json(contract_path)
    payload_bits = validate_contract(contract)
    prompts = select_prompts(
        prompt_rows,
        split=args.split,
        max_prompts=int(args.max_prompts),
        prompt_file_row_start=args.prompt_file_row_start,
        prompt_file_row_end=args.prompt_file_row_end,
    )

    if args.validate_plan_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "wp6_generation_plan_summary.json",
            build_plan_summary(
                args=args,
                prompts_path=prompts_path,
                contract_path=contract_path,
                prompts=prompts,
                payload_bits=payload_bits,
                generation_started=False,
            ),
        )
        return 0

    if not protected_adapter.exists():
        raise FileNotFoundError(f"protected adapter missing: {protected_adapter}")
    if not task_only_adapter.exists():
        raise FileNotFoundError(f"task-only adapter missing: {task_only_adapter}")
    if (output_dir / "wp6_generated_outputs.jsonl").exists():
        raise FileExistsError(f"refusing to overwrite WP6 outputs in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for condition, adapter in (
        ("protected", protected_adapter),
        ("raw", None),
        ("task_only", task_only_adapter),
    ):
        rows.extend(generate_condition(args=args, prompts=prompts, condition=condition, adapter_path=adapter))

    write_jsonl(output_dir / "wp6_generated_outputs.jsonl", rows)
    write_json(
        output_dir / "wp6_generation_summary.json",
        build_plan_summary(
            args=args,
            prompts_path=prompts_path,
            contract_path=contract_path,
            prompts=prompts,
            payload_bits=payload_bits,
            generation_started=True,
        )
        | {
            "generated_output_rows": len(rows),
            "output_path": str(output_dir / "wp6_generated_outputs.jsonl"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
