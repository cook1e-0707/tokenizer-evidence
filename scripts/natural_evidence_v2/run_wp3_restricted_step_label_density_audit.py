from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml"
DEFAULT_PLAN_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "wp3_restricted_step_label_density_audit_plan_20260508_2055"
)
DEFAULT_PROMPTS = DEFAULT_PLAN_DIR / "restricted_step_label_density_audit_prompts.jsonl"
DEFAULT_POLICY_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "wp3_restricted_step_label_policy_20260508_2049"
)

STEP_LABEL_RE = re.compile(r"(?im)(?:^|\n)\s*(?:[-*]\s*)?\*{0,2}Step\s+([0-9]+):\*{0,2}")
STEP_SLOT_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:[-*]\s*)?\*{0,2}Step\s+(?P<step>[0-9]+):\*{0,2}\s*"
    r"(?P<prefix_marker>[\*_`]*)(?P<word>[A-Za-z][A-Za-z'-]*)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the natural_evidence_v2 WP3 restricted Step-label model-output "
            "density audit. This generates base-Qwen outputs and audits Step 1 "
            "through Step 16 structural density only. It does not train, run E2E, "
            "decode payloads, aggregate FAR, or make paper-facing claims."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prompts-jsonl", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-prompts", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=896)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--validate-plan-only",
        action="store_true",
        help="Validate prompts/policy/gate inputs without loading a model or writing artifacts.",
    )
    parser.add_argument(
        "--responses-jsonl",
        type=Path,
        help="Optional existing response artifact to audit instead of generating model outputs.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top level must be an object: {path}")
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML top level must be a mapping: {path}")
    return payload


def read_jsonl(path: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_rows and index >= max_rows:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{index + 1}")
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_inputs(
    *,
    config_path: Path,
    prompts_path: Path,
    policy_dir: Path,
    max_prompts: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = read_yaml(config_path)
    prompts = read_jsonl(prompts_path, max_rows=max(0, int(max_prompts)))
    detector_contract = read_json(policy_dir / "restricted_step_label_detector_contract.json")
    bucket_bank = read_json(policy_dir / "restricted_step_label_bucket_bank.json")
    density_design = read_json(policy_dir / "restricted_step_label_density_design.json")
    validate_inputs(
        config=config,
        prompts=prompts,
        detector_contract=detector_contract,
        bucket_bank=bucket_bank,
        density_design=density_design,
    )
    return config, prompts, detector_contract, bucket_bank, density_design


def validate_inputs(
    *,
    config: Mapping[str, Any],
    prompts: list[Mapping[str, Any]],
    detector_contract: Mapping[str, Any],
    bucket_bank: Mapping[str, Any],
    density_design: Mapping[str, Any],
) -> None:
    if not prompts:
        raise ValueError("density audit prompts are empty")
    allowed_steps = [int(item) for item in detector_contract.get("allowed_step_indices", [])]
    if allowed_steps != list(range(1, 17)):
        raise ValueError(f"restricted detector must use Step 1 through Step 16: {allowed_steps}")
    if density_design.get("decision") is None:
        raise ValueError("density design must record the selected route decision")
    for row in prompts:
        if int(row.get("expected_structural_slots", 0)) != 16:
            raise ValueError(f"prompt row does not expect 16 structural slots: {row.get('prompt_id')}")
        text = str(row.get("prompt_text", ""))
        forbidden_hits = forbidden_terms_in_text(config, text)
        if forbidden_hits:
            raise ValueError(f"prompt row contains forbidden public surface {forbidden_hits}")
    bank_ids = [str(row.get("candidate_bank_id", "")) for row in bucket_bank.get("candidate_banks", [])]
    expected = {
        "restricted_step_label_check_review_choose_make_v1",
        "restricted_step_label_start_begin_create_set_v1",
    }
    if set(bank_ids) != expected:
        raise ValueError(f"unexpected restricted bank ids: {bank_ids}")


def forbidden_terms_in_text(config: Mapping[str, Any], text: str) -> list[str]:
    terms = [str(item) for item in config.get("forbidden_surface_terms", [])]
    upper = text.upper()
    return [term for term in terms if term.upper() in upper]


def candidate_surface_map(bucket_bank: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = {}
    for bank in bucket_bank.get("candidate_banks", []):
        bank_id = str(bank["candidate_bank_id"])
        for bucket_id, members in bank["buckets"].items():
            for surface in members:
                surface_text = str(surface)
                output.setdefault(surface_text, []).append(
                    {
                        "candidate_bank_id": bank_id,
                        "bucket_id": str(bucket_id),
                        "surface": surface_text,
                    }
                )
    return output


def response_id_from_row(row: Mapping[str, Any], index: int) -> str:
    for key in ("response_id", "generation_id", "prompt_id"):
        value = row.get(key)
        if value:
            return str(value)
    return f"response_{index:06d}"


def build_chat_text(tokenizer: Any, prompt_text: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            pass
    return prompt_text


def load_model(*, model_name: str, tokenizer_name: str, require_cuda: bool) -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("restricted Step-label density audit requires torch and transformers") from error

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is false")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True, "trust_remote_code": True}
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def batched(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    size = max(1, int(batch_size))
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def generate_outputs(
    *,
    prompt_rows: list[dict[str, Any]],
    model_name: str,
    tokenizer_name: str,
    batch_size: int,
    max_new_tokens: int,
    require_cuda: bool,
) -> list[dict[str, Any]]:
    torch, tokenizer, model, device = load_model(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        require_cuda=require_cuda,
    )
    output_rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batched(prompt_rows, batch_size)):
        chat_texts = [build_chat_text(tokenizer, str(row["prompt_text"])) for row in batch]
        inputs = tokenizer(chat_texts, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max(1, int(max_new_tokens)),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = int(inputs["input_ids"].shape[1])
        for row_index, (prompt_row, output_ids) in enumerate(zip(batch, generated, strict=True)):
            response_ids = output_ids[prompt_width:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            response_id_payload = {
                "prompt_id": str(prompt_row["prompt_id"]),
                "model_name": model_name,
                "batch_index": batch_index,
                "row_index": row_index,
            }
            response_id = "qwen_v2_wp3_density_response_" + sha256_text(
                json.dumps(response_id_payload, sort_keys=True, separators=(",", ":"))
            )[:20]
            output_rows.append(
                {
                    "schema_name": "natural_evidence_v2_wp3_restricted_step_label_model_response_v1",
                    "response_id": response_id,
                    "response_source": "base_qwen_model_output_density_audit",
                    "artifact_role": "model_output_density_audit_not_e2e_not_training",
                    "model_name": model_name,
                    "tokenizer_name": tokenizer_name,
                    "generation_mode": "deterministic_greedy",
                    "max_new_tokens": int(max_new_tokens),
                    "prompt_id": str(prompt_row["prompt_id"]),
                    "split": str(prompt_row.get("split", "")),
                    "family_id": str(prompt_row.get("family_id", "")),
                    "variant_id": str(prompt_row.get("variant_id", "")),
                    "topic": str(prompt_row.get("topic", "")),
                    "prompt_text_sha256": str(prompt_row.get("prompt_text_sha256", "")),
                    "response_text": response_text,
                    "response_text_sha256": sha256_text(response_text),
                    "model_generation_started": True,
                    "model_scoring_started": False,
                    "training_started": False,
                    "e2e_eval_started": False,
                    "paper_claim_allowed": False,
                }
            )
    return output_rows


def load_existing_responses(path: Path, *, max_prompts: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path, max_rows=max(0, int(max_prompts)))
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        response_text = str(
            row.get("response_text")
            or row.get("output_text")
            or row.get("model_output")
            or row.get("text")
            or ""
        )
        if not response_text:
            raise ValueError(f"response row missing response_text/output_text/model_output/text: {index}")
        normalized = dict(row)
        normalized.setdefault("response_id", response_id_from_row(row, index))
        normalized.setdefault("response_source", "existing_response_artifact_density_audit")
        normalized.setdefault("artifact_role", "existing_response_artifact_not_generation")
        normalized["response_text"] = response_text
        normalized.setdefault("model_generation_started", False)
        normalized.setdefault("model_scoring_started", False)
        normalized.setdefault("training_started", False)
        normalized.setdefault("e2e_eval_started", False)
        normalized.setdefault("paper_claim_allowed", False)
        output.append(normalized)
    return output


def detect_response(
    *,
    response_row: Mapping[str, Any],
    response_index: int,
    config: Mapping[str, Any],
    surface_map: Mapping[str, list[dict[str, str]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    response_text = str(response_row.get("response_text", ""))
    response_id = response_id_from_row(response_row, response_index)
    forbidden_hits = forbidden_terms_in_text(config, response_text)
    label_numbers = [int(match.group(1)) for match in STEP_LABEL_RE.finditer(response_text)]
    label_counts = Counter(label_numbers)
    expected_steps = list(range(1, 17))
    label_count_by_expected_step = {str(step): int(label_counts.get(step, 0)) for step in expected_steps}
    out_of_range_labels = [step for step in label_numbers if step not in expected_steps]

    rows: list[dict[str, Any]] = []
    seen_step_slots: set[int] = set()
    for match in STEP_SLOT_RE.finditer(response_text):
        step_index = int(match.group("step"))
        if step_index not in expected_steps:
            continue
        first_word = str(match.group("word"))
        seen_step_slots.add(step_index)
        exact_hits = list(surface_map.get(first_word, []))
        lower_hits = [
            item
            for surface, items in surface_map.items()
            if surface.lower() == first_word.lower() and surface != first_word
            for item in items
        ]
        rows.append(
            {
                "schema_name": "natural_evidence_v2_wp3_restricted_step_label_detected_slot_v1",
                "response_id": response_id,
                "prompt_id": str(response_row.get("prompt_id", "")),
                "family_id": str(response_row.get("family_id", "")),
                "variant_id": str(response_row.get("variant_id", "")),
                "step_index": step_index,
                "first_word": first_word,
                "first_word_sha256": sha256_text(first_word),
                "candidate_bank_exact_hits": exact_hits,
                "candidate_bank_case_insensitive_hits": lower_hits,
                "exact_candidate_hit": bool(exact_hits),
                "case_insensitive_candidate_hit": bool(exact_hits or lower_hits),
                "format_markdown_prefix_marker": str(match.group("prefix_marker")),
                "char_start": int(match.start()),
                "char_end": int(match.end()),
                "model_generation_started": bool(response_row.get("model_generation_started", False)),
                "model_scoring_started": False,
                "training_started": False,
                "e2e_eval_started": False,
                "paper_claim_allowed": False,
            }
        )

    complete_labels = all(label_counts.get(step, 0) == 1 for step in expected_steps) and not out_of_range_labels
    response_summary = {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_response_audit_v1",
        "response_id": response_id,
        "prompt_id": str(response_row.get("prompt_id", "")),
        "family_id": str(response_row.get("family_id", "")),
        "variant_id": str(response_row.get("variant_id", "")),
        "response_source": str(response_row.get("response_source", "")),
        "complete_step_label_response": complete_labels,
        "detected_structural_slots": len(seen_step_slots),
        "detected_slot_rows": len(rows),
        "has_at_least_16_structural_slots": len(seen_step_slots) >= 16,
        "label_count_by_expected_step": label_count_by_expected_step,
        "out_of_range_step_labels": out_of_range_labels,
        "duplicate_expected_step_labels": [
            step for step in expected_steps if label_counts.get(step, 0) > 1
        ],
        "missing_expected_step_labels": [
            step for step in expected_steps if label_counts.get(step, 0) == 0
        ],
        "forbidden_public_surface_hits": forbidden_hits,
        "forbidden_public_surface_present": bool(forbidden_hits),
        "response_text_sha256": str(response_row.get("response_text_sha256", sha256_text(response_text))),
        "model_generation_started": bool(response_row.get("model_generation_started", False)),
        "model_scoring_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }
    return response_summary, rows


def mean(values: list[float]) -> float:
    return 0.0 if not values else float(sum(values) / len(values))


def median(values: list[float]) -> float:
    return 0.0 if not values else float(statistics.median(values))


def audit_outputs(
    *,
    config: Mapping[str, Any],
    bucket_bank: Mapping[str, Any],
    response_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    surface_map = candidate_surface_map(bucket_bank)
    response_summaries: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    for index, row in enumerate(response_rows):
        response_summary, detected = detect_response(
            response_row=row,
            response_index=index,
            config=config,
            surface_map=surface_map,
        )
        response_summaries.append(response_summary)
        slot_rows.extend(detected)

    total = len(response_summaries)
    complete_count = sum(1 for row in response_summaries if row["complete_step_label_response"])
    at_least_16_count = sum(1 for row in response_summaries if row["has_at_least_16_structural_slots"])
    forbidden_count = sum(1 for row in response_summaries if row["forbidden_public_surface_present"])
    slot_counts = [float(row["detected_structural_slots"]) for row in response_summaries]
    exact_hits = sum(1 for row in slot_rows if row["exact_candidate_hit"])
    case_insensitive_hits = sum(1 for row in slot_rows if row["case_insensitive_candidate_hit"])
    exact_hits_by_surface = Counter(str(row["first_word"]) for row in slot_rows if row["exact_candidate_hit"])
    exact_hits_by_bank_bucket = Counter()
    for row in slot_rows:
        for hit in row["candidate_bank_exact_hits"]:
            exact_hits_by_bank_bucket[f"{hit['candidate_bank_id']}::{hit['bucket_id']}"] += 1
    response_source_counts = Counter(str(row.get("response_source", "")) for row in response_rows)
    model_generation_started = any(bool(row.get("model_generation_started", False)) for row in response_rows)

    complete_rate = 0.0 if total == 0 else complete_count / total
    at_least_16_rate = 0.0 if total == 0 else at_least_16_count / total
    forbidden_rate = 0.0 if total == 0 else forbidden_count / total
    exact_hit_rate = 0.0 if not slot_rows else exact_hits / len(slot_rows)
    case_insensitive_hit_rate = 0.0 if not slot_rows else case_insensitive_hits / len(slot_rows)
    mean_slots = mean(slot_counts)

    structural_pass = (
        complete_rate >= 0.95
        and mean_slots >= 16.0
        and at_least_16_rate >= 0.90
        and forbidden_rate == 0.0
    )
    status = (
        "PASS_RESTRICTED_STEP_LABEL_MODEL_OUTPUT_DENSITY_STRUCTURAL_GATE_NEEDS_MANUAL_NATURALNESS"
        if structural_pass
        else "FAIL_RESTRICTED_STEP_LABEL_MODEL_OUTPUT_DENSITY_STRUCTURAL_GATE"
    )
    summary = {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_density_audit_summary_v1",
        "status": status,
        "structural_density_gate_status": "PASS" if structural_pass else "FAIL",
        "manual_naturalness_gate_status": "NEEDS_MANUAL_REVIEW",
        "model_output_density_audit": True,
        "response_source_counts": dict(sorted(response_source_counts.items())),
        "total_responses": total,
        "complete_step_label_response_count": complete_count,
        "complete_step_label_response_rate": complete_rate,
        "mean_detected_structural_slots_per_response": mean_slots,
        "median_detected_structural_slots_per_response": median(slot_counts),
        "responses_with_at_least_16_structural_slots_count": at_least_16_count,
        "responses_with_at_least_16_structural_slots_rate": at_least_16_rate,
        "forbidden_public_surface_response_count": forbidden_count,
        "forbidden_public_surface_rate": forbidden_rate,
        "detected_slot_rows": len(slot_rows),
        "raw_bank_surface_exact_hit_count": exact_hits,
        "raw_bank_surface_exact_hit_rate": exact_hit_rate,
        "raw_bank_surface_case_insensitive_hit_count": case_insensitive_hits,
        "raw_bank_surface_case_insensitive_hit_rate": case_insensitive_hit_rate,
        "raw_bank_surface_hit_interpretation": (
            "report-only raw accidental-surface risk diagnostic, not ownership evidence"
        ),
        "exact_hits_by_first_word": dict(sorted(exact_hits_by_surface.items())),
        "exact_hits_by_bank_bucket": dict(sorted(exact_hits_by_bank_bucket.items())),
        "naturalness_manual_review_required_examples": min(32, total),
        "model_generation_started": model_generation_started,
        "model_scoring_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "next_allowed_action": (
            "Review the model-output density artifacts and manual naturalness examples. "
            "Do not start WP4 or training until this review is complete."
        ),
    }
    examples = [
        {
            "schema_name": "natural_evidence_v2_wp3_restricted_step_label_naturalness_example_v1",
            "response_id": response_id_from_row(row, index),
            "prompt_id": str(row.get("prompt_id", "")),
            "family_id": str(row.get("family_id", "")),
            "variant_id": str(row.get("variant_id", "")),
            "response_text": str(row.get("response_text", "")),
            "manual_review_status": "NEEDS_REVIEW",
            "model_generation_started": bool(row.get("model_generation_started", False)),
            "training_started": False,
            "e2e_eval_started": False,
            "paper_claim_allowed": False,
        }
        for index, row in enumerate(response_rows[: min(32, len(response_rows))])
    ]
    return summary, response_summaries, slot_rows, examples


def readme_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WP3 Restricted Step-Label Model-Output Density Audit",
            "",
            "Base-Qwen model-output density audit for the restricted 16-step Step-label route.",
            "",
            f"status: `{summary['status']}`",
            f"total_responses: `{summary['total_responses']}`",
            f"complete_step_label_response_rate: `{summary['complete_step_label_response_rate']}`",
            f"mean_detected_structural_slots_per_response: `{summary['mean_detected_structural_slots_per_response']}`",
            "",
            "This is not training, E2E, payload recovery, FAR, or a positive paper claim.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    prompts_path = resolve_path(args.prompts_jsonl)
    policy_dir = resolve_path(args.policy_dir)
    config, prompts, _detector_contract, bucket_bank, _density_design = load_inputs(
        config_path=config_path,
        prompts_path=prompts_path,
        policy_dir=policy_dir,
        max_prompts=max(0, int(args.max_prompts)),
    )
    validation = {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_density_plan_validation_v1",
        "status": "PASS_RESTRICTED_STEP_LABEL_DENSITY_PLAN_VALIDATION",
        "config": str(config_path),
        "prompts_jsonl": str(prompts_path),
        "policy_dir": str(policy_dir),
        "prompt_count": len(prompts),
        "model_generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }
    if args.validate_plan_only:
        print(json.dumps(validation, sort_keys=True))
        return

    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --validate-plan-only is set")
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.responses_jsonl is not None:
        response_rows = load_existing_responses(
            resolve_path(args.responses_jsonl),
            max_prompts=max(0, int(args.max_prompts)),
        )
        response_artifact_name = "restricted_step_label_existing_density_responses.jsonl"
    else:
        response_rows = generate_outputs(
            prompt_rows=prompts,
            model_name=str(args.model_name),
            tokenizer_name=str(args.tokenizer_name),
            batch_size=max(1, int(args.batch_size)),
            max_new_tokens=max(1, int(args.max_new_tokens)),
            require_cuda=bool(args.require_cuda),
        )
        response_artifact_name = "restricted_step_label_model_outputs.jsonl"

    summary, response_summaries, slot_rows, examples = audit_outputs(
        config=config,
        bucket_bank=bucket_bank,
        response_rows=response_rows,
    )
    summary.update(
        {
            "config": str(config_path),
            "prompts_jsonl": str(prompts_path),
            "policy_dir": str(policy_dir),
            "output_dir": str(output_dir),
            "model_name": str(args.model_name),
            "tokenizer_name": str(args.tokenizer_name),
            "max_prompts": int(args.max_prompts),
            "batch_size": int(args.batch_size),
            "max_new_tokens": int(args.max_new_tokens),
            "responses_jsonl": str(output_dir / response_artifact_name),
            "response_audit_jsonl": str(output_dir / "restricted_step_label_response_audit.jsonl"),
            "detected_slots_jsonl": str(output_dir / "restricted_step_label_detected_slots.jsonl"),
            "naturalness_examples_jsonl": str(output_dir / "restricted_step_label_naturalness_examples.jsonl"),
            "summary_json": str(output_dir / "restricted_step_label_density_audit_summary.json"),
        }
    )
    write_jsonl(output_dir / response_artifact_name, response_rows)
    write_jsonl(output_dir / "restricted_step_label_response_audit.jsonl", response_summaries)
    write_jsonl(output_dir / "restricted_step_label_detected_slots.jsonl", slot_rows)
    write_jsonl(output_dir / "restricted_step_label_naturalness_examples.jsonl", examples)
    write_json(output_dir / "restricted_step_label_density_audit_summary.json", summary)
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(output_dir),
                "total_responses": summary["total_responses"],
                "structural_density_gate_status": summary["structural_density_gate_status"],
                "wp4_allowed": summary["wp4_allowed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
