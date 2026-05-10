from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml"
DEFAULT_SCORE_PLAN = (
    ROOT
    / "results/natural_evidence_v2/status/wp3_context_mass_plan_prefix_boundary_repair_20260509_0024/"
    "qwen_v2_wp3_context_mass_score_plan.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score natural_evidence_v2 WP3 context-specific two-way bucket "
            "masses under base Qwen. This consumes the fixed score plan, scores "
            "bucket_surfaces at prefix_before_candidate, keeps casing variants "
            "separate, and writes mass/audit artifacts. It does not train, "
            "generate text, run E2E, aggregate FAR, or make positive claims."
        )
    )
    parser.add_argument("--score-plan", type=Path, default=DEFAULT_SCORE_PLAN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--validate-plan-only",
        action="store_true",
        help="Read and validate the plan without loading a model or writing artifacts.",
    )
    parser.add_argument(
        "--validate-tokenizer-only",
        action="store_true",
        help=(
            "Load only the tokenizer and validate context-boundary tokenization "
            "without loading a model, scoring, or writing artifacts."
        ),
    )
    parser.add_argument(
        "--skip-invalid-tokenization",
        action="store_true",
        help=(
            "Record tokenizer-invalid rows and continue with valid rows only. "
            "This is for artifact diagnosis; skipped rows never contribute to "
            "mass gates."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{index}")
            rows.append(payload)
    return rows


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML top level must be a mapping: {path}")
    return payload


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


def validate_bucket_surfaces(row: Mapping[str, Any]) -> dict[str, list[str]]:
    surfaces = row.get("bucket_surfaces")
    if not isinstance(surfaces, Mapping):
        raise ValueError(f"plan row missing bucket_surfaces mapping: {row.get('plan_row_id')}")
    output: dict[str, list[str]] = {}
    for bucket_id, members in sorted(surfaces.items()):
        if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
            raise ValueError(
                f"bucket_surfaces bucket must be a sequence: {row.get('plan_row_id')}:{bucket_id}"
            )
        output[str(bucket_id)] = [str(member) for member in members]
    if set(output) != {"0", "1"}:
        raise ValueError(f"plan row must contain exactly buckets 0 and 1: {row.get('plan_row_id')}")
    return output


def validate_plan_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("score plan is empty")
    seen_ids: set[str] = set()
    rows_by_bank = Counter()
    rows_by_variant = Counter()
    rows_by_bank_variant = Counter()
    template_preflight_only = True
    source_detection_count = 0
    empty_prefix_rows = 0
    for index, row in enumerate(rows, start=1):
        for key in (
            "plan_id",
            "plan_row_id",
            "candidate_bank_id",
            "casing_variant",
            "bucket_surfaces",
            "prefix_before_candidate",
            "prefix_before_candidate_sha256",
        ):
            if key not in row:
                raise ValueError(f"missing required plan key {key!r} at row {index}")
        plan_row_id = str(row["plan_row_id"])
        if plan_row_id in seen_ids:
            raise ValueError(f"duplicate plan_row_id: {plan_row_id}")
        seen_ids.add(plan_row_id)
        prefix = str(row["prefix_before_candidate"])
        if not prefix:
            empty_prefix_rows += 1
        validate_bucket_surfaces(row)
        bank_id = str(row["candidate_bank_id"])
        variant = str(row["casing_variant"])
        rows_by_bank[bank_id] += 1
        rows_by_variant[variant] += 1
        rows_by_bank_variant[f"{bank_id}::{variant}"] += 1
        template_preflight_only = template_preflight_only and bool(row.get("template_preflight_only", False))
        source_detection_count += int(row.get("source_detection_count", 1))
    return {
        "schema_name": "natural_evidence_v2_wp3_context_mass_score_plan_validation_v1",
        "status": "PASS_CONTEXT_MASS_SCORE_PLAN_VALIDATION",
        "score_plan_rows": len(rows),
        "score_plan_rows_by_bank": dict(sorted(rows_by_bank.items())),
        "score_plan_rows_by_casing_variant": dict(sorted(rows_by_variant.items())),
        "score_plan_rows_by_bank_and_casing_variant": dict(sorted(rows_by_bank_variant.items())),
        "source_detection_count_total": source_detection_count,
        "empty_prefix_rows": empty_prefix_rows,
        "template_preflight_only": template_preflight_only,
        "casing_variants_kept_separate": True,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def encode_no_special(tokenizer: Any, text: str) -> list[int]:
    try:
        return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
    except TypeError:
        return [int(token_id) for token_id in tokenizer.encode(text)]


def load_tokenizer(*, tokenizer_name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("WP3 context-mass tokenizer validation requires transformers") from error

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(*, model_name: str, tokenizer_name: str, require_cuda: bool) -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("WP3 context-mass scoring requires torch and transformers") from error

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(tokenizer_name=tokenizer_name)
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True, "trust_remote_code": True}
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


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


def effective_prefix_text(tokenizer: Any, row: Mapping[str, Any]) -> str:
    chat_prompt = str(row.get("chat_prompt_text", ""))
    assistant_prefix = str(row.get("assistant_prefix_before_candidate", ""))
    if chat_prompt:
        return build_chat_text(tokenizer, chat_prompt) + assistant_prefix
    return str(row["prefix_before_candidate"])


def startswith_sequence(values: Sequence[int], prefix: Sequence[int]) -> bool:
    return len(values) >= len(prefix) and list(values[: len(prefix)]) == list(prefix)


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_id, right_id in zip(left, right):
        if int(left_id) != int(right_id):
            break
        length += 1
    return length


def resolve_contextual_bucket_tokenization(tokenizer: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    token_ids_by_bucket: dict[str, list[int]] = {}
    prefix = effective_prefix_text(tokenizer, row)
    prefix_ids = encode_no_special(tokenizer, prefix)
    scoring_prefix_ids: list[int] | None = None
    adjusted_surfaces: list[str] = []
    exact_surface_count = 0

    for bucket_id, surfaces in validate_bucket_surfaces(row).items():
        ids: list[int] = []
        for surface in surfaces:
            if prefix:
                combined_ids = encode_no_special(tokenizer, prefix + surface)
                if startswith_sequence(combined_ids, prefix_ids):
                    candidate_prefix_ids = list(prefix_ids)
                    token_ids = combined_ids[len(prefix_ids) :]
                    exact_surface_count += 1
                else:
                    shared = common_prefix_length(prefix_ids, combined_ids)
                    candidate_prefix_ids = list(combined_ids[:shared])
                    token_ids = combined_ids[shared:]
                    adjusted_surfaces.append(surface)
                if not candidate_prefix_ids:
                    raise ValueError(
                        f"{row.get('plan_row_id')}: tokenizer boundary repair produced an empty "
                        f"scoring prefix for non-empty prefix and surface {surface!r}"
                    )
                if scoring_prefix_ids is None:
                    scoring_prefix_ids = candidate_prefix_ids
                elif scoring_prefix_ids != candidate_prefix_ids:
                    raise ValueError(
                        f"{row.get('plan_row_id')}: bucket surfaces do not share one scoring prefix "
                        "after tokenizer boundary repair"
                    )
            else:
                token_ids = encode_no_special(tokenizer, surface)
                candidate_prefix_ids = []
                if scoring_prefix_ids is None:
                    scoring_prefix_ids = candidate_prefix_ids
            if len(token_ids) != 1:
                raise ValueError(
                    f"{row.get('plan_row_id')}: bucket surface is not one next token under scoring tokenizer: "
                    f"{surface!r} -> {token_ids}"
                )
            ids.append(int(token_ids[0]))
        token_ids_by_bucket[str(bucket_id)] = ids

    if scoring_prefix_ids is None:
        scoring_prefix_ids = []
    adjusted = bool(adjusted_surfaces)
    return {
        "bucket_token_ids": token_ids_by_bucket,
        "scoring_context_kind": "chat_prompt_plus_assistant_prefix"
        if str(row.get("chat_prompt_text", ""))
        else "raw_prefix_before_candidate",
        "prefix_boundary_policy": (
            "longest_common_token_prefix_boundary_repair"
            if adjusted
            else "exact_prefix_token_boundary"
        ),
        "prefix_boundary_adjusted": adjusted,
        "prefix_boundary_adjusted_surfaces": sorted(set(adjusted_surfaces)),
        "prefix_boundary_adjusted_surface_count": len(adjusted_surfaces),
        "prefix_boundary_exact_surface_count": exact_surface_count,
        "original_prefix_token_count": len(prefix_ids),
        "scoring_prefix_ids": scoring_prefix_ids,
        "scoring_prefix_token_count": len(scoring_prefix_ids),
        "prefix_boundary_trimmed_token_count": max(0, len(prefix_ids) - len(scoring_prefix_ids)),
    }


def validate_tokenizer_boundaries(
    *,
    tokenizer: Any,
    plan_rows: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    skip_invalid: bool,
) -> dict[str, Any]:
    adjusted_rows = 0
    adjusted_surfaces = 0
    trimmed_token_counts = Counter()
    rows_by_policy = Counter()
    invalid_rows: list[dict[str, Any]] = []
    for row in plan_rows:
        try:
            resolved = resolve_contextual_bucket_tokenization(tokenizer, row)
        except ValueError as error:
            if not skip_invalid:
                raise
            invalid_rows.append(tokenization_failure_row(row=row, error=error))
            continue
        policy = str(resolved["prefix_boundary_policy"])
        rows_by_policy[policy] += 1
        if bool(resolved["prefix_boundary_adjusted"]):
            adjusted_rows += 1
            adjusted_surfaces += int(resolved["prefix_boundary_adjusted_surface_count"])
            trimmed_token_counts[str(resolved["prefix_boundary_trimmed_token_count"])] += 1
    valid_rows = int(validation["score_plan_rows"]) - len(invalid_rows)
    return {
        "schema_name": "natural_evidence_v2_wp3_context_mass_tokenizer_boundary_validation_v1",
        "status": (
            "PASS_CONTEXT_MASS_TOKENIZER_BOUNDARY_VALIDATION"
            if not invalid_rows
            else "PASS_WITH_SKIPPED_INVALID_TOKENIZATION_ROWS"
        ),
        "score_plan_rows": int(validation["score_plan_rows"]),
        "valid_tokenization_rows": valid_rows,
        "invalid_tokenization_rows": len(invalid_rows),
        "invalid_tokenization_examples": invalid_rows[:20],
        "rows_by_prefix_boundary_policy": dict(sorted(rows_by_policy.items())),
        "prefix_boundary_adjusted_rows": adjusted_rows,
        "prefix_boundary_adjusted_surfaces": adjusted_surfaces,
        "prefix_boundary_trimmed_token_count_rows": dict(sorted(trimmed_token_counts.items())),
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
    }


def tokenization_failure_row(*, row: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_wp3_context_mass_invalid_tokenization_row_v1",
        "plan_id": str(row.get("plan_id", "")),
        "plan_row_id": str(row.get("plan_row_id", "")),
        "candidate_bank_id": str(row.get("candidate_bank_id", "")),
        "casing_variant": str(row.get("casing_variant", "")),
        "scoring_context_kind": "chat_prompt_plus_assistant_prefix"
        if str(row.get("chat_prompt_text", ""))
        else "raw_prefix_before_candidate",
        "prefix_before_candidate": str(row.get("prefix_before_candidate", "")),
        "assistant_prefix_before_candidate_sha256": str(
            row.get("assistant_prefix_before_candidate_sha256", "")
        ),
        "chat_prompt_text_sha256": str(row.get("chat_prompt_text_sha256", "")),
        "bucket_surfaces": validate_bucket_surfaces(row),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
    }


def resolve_valid_tokenizations(
    *,
    tokenizer: Any,
    plan_rows: Sequence[Mapping[str, Any]],
    skip_invalid: bool,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[dict[str, Any]]]:
    valid_rows: list[Mapping[str, Any]] = []
    valid_tokenizations: list[Mapping[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for row in plan_rows:
        try:
            resolved = resolve_contextual_bucket_tokenization(tokenizer, row)
        except ValueError as error:
            if not skip_invalid:
                raise
            invalid_rows.append(tokenization_failure_row(row=row, error=error))
            continue
        valid_rows.append(row)
        valid_tokenizations.append(resolved)
    if not valid_rows:
        raise ValueError("all score-plan rows were tokenizer-invalid")
    return valid_rows, valid_tokenizations, invalid_rows


def encode_prefix_for_scoring(
    tokenizer: Any, row: Mapping[str, Any], row_tokenization: Mapping[str, Any]
) -> tuple[list[int], str]:
    token_ids = [int(token_id) for token_id in row_tokenization["scoring_prefix_ids"]]
    if token_ids:
        if bool(row_tokenization.get("prefix_boundary_adjusted", False)):
            return token_ids, "prefix_boundary_adjusted_tokens"
        return token_ids, "prefix_tokens"
    start_token_id = tokenizer.bos_token_id
    source = "bos_token"
    if start_token_id is None:
        start_token_id = tokenizer.eos_token_id
        source = "eos_token_fallback"
    if start_token_id is None:
        raise ValueError(f"empty prefix has no BOS/EOS fallback token: {row.get('plan_row_id')}")
    return [int(start_token_id)], source


def padded_batch(
    *,
    torch_module: Any,
    tokenizer: Any,
    plan_rows: Sequence[Mapping[str, Any]],
    row_tokenizations: Sequence[Mapping[str, Any]],
    max_length: int,
    device: Any,
) -> tuple[Any, Any, list[int], list[str]]:
    encoded_rows: list[list[int]] = []
    lengths: list[int] = []
    prefix_token_sources: list[str] = []
    for row, row_tokenization in zip(plan_rows, row_tokenizations):
        token_ids, token_source = encode_prefix_for_scoring(tokenizer, row, row_tokenization)
        token_ids = token_ids[-max_length:]
        encoded_rows.append(token_ids)
        lengths.append(len(token_ids))
        prefix_token_sources.append(token_source)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    width = max(lengths)
    input_ids = torch_module.full(
        (len(encoded_rows), width),
        int(pad_id),
        dtype=torch_module.long,
        device=device,
    )
    attention_mask = torch_module.zeros(
        (len(encoded_rows), width),
        dtype=torch_module.long,
        device=device,
    )
    for index, token_ids in enumerate(encoded_rows):
        input_ids[index, : len(token_ids)] = torch_module.tensor(token_ids, dtype=torch_module.long, device=device)
        attention_mask[index, : len(token_ids)] = 1
    return input_ids, attention_mask, lengths, prefix_token_sources


def score_plan_rows(
    *,
    torch_module: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    plan_rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    max_length: int,
    skip_invalid_tokenization: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    valid_rows, valid_tokenizations, invalid_rows = resolve_valid_tokenizations(
        tokenizer=tokenizer,
        plan_rows=plan_rows,
        skip_invalid=skip_invalid_tokenization,
    )
    with torch_module.no_grad():
        for start in range(0, len(valid_rows), batch_size):
            batch = list(valid_rows[start : start + batch_size])
            row_tokenizations = list(valid_tokenizations[start : start + batch_size])
            input_ids, attention_mask, lengths, prefix_token_sources = padded_batch(
                torch_module=torch_module,
                tokenizer=tokenizer,
                plan_rows=batch,
                row_tokenizations=row_tokenizations,
                max_length=max(1, int(max_length)),
                device=device,
            )
            logits_batch = model(input_ids=input_ids, attention_mask=attention_mask).logits
            for row_index, row in enumerate(batch):
                logits = logits_batch[row_index, lengths[row_index] - 1, :].float()
                log_denom = torch_module.logsumexp(logits, dim=0)
                row_tokenization = row_tokenizations[row_index]
                token_ids_by_bucket = {
                    str(bucket_id): [int(token_id) for token_id in token_ids]
                    for bucket_id, token_ids in row_tokenization["bucket_token_ids"].items()
                }
                candidate_token_ids = sorted(
                    {token_id for ids in token_ids_by_bucket.values() for token_id in ids}
                )
                candidate_tensor = torch_module.tensor(candidate_token_ids, dtype=torch_module.long, device=device)
                candidate_denom = torch_module.logsumexp(logits[candidate_tensor], dim=0)
                full_masses: dict[str, float] = {}
                candidate_normalized_masses: dict[str, float] = {}
                for bucket_id, bucket_token_ids_value in sorted(token_ids_by_bucket.items()):
                    bucket_tensor = torch_module.tensor(bucket_token_ids_value, dtype=torch_module.long, device=device)
                    bucket_logits = logits[bucket_tensor]
                    full_masses[bucket_id] = float(
                        torch_module.exp(torch_module.logsumexp(bucket_logits, dim=0) - log_denom)
                        .detach()
                        .cpu()
                        .item()
                    )
                    candidate_normalized_masses[bucket_id] = float(
                        torch_module.exp(torch_module.logsumexp(bucket_logits, dim=0) - candidate_denom)
                        .detach()
                        .cpu()
                        .item()
                    )
                scored.append(
                    {
                        "schema_name": "natural_evidence_v2_wp3_context_mass_context_score_v1",
                        "plan_id": str(row["plan_id"]),
                        "plan_row_id": str(row["plan_row_id"]),
                        "candidate_bank_id": str(row["candidate_bank_id"]),
                        "slot_type": str(row.get("slot_type", "")),
                        "anchor_kind": str(row.get("anchor_kind", "")),
                        "bucket_policy_id": str(row.get("bucket_policy_id", "")),
                        "casing_variant": str(row["casing_variant"]),
                        "context_index_within_bank_variant": int(
                            row.get("context_index_within_bank_variant", 0)
                        ),
                        "prefix_before_candidate": str(row["prefix_before_candidate"]),
                        "prefix_before_candidate_sha256": str(row["prefix_before_candidate_sha256"]),
                        "prefix_is_empty": str(row["prefix_before_candidate"]) == "",
                        "prefix_scoring_token_source": prefix_token_sources[row_index],
                        "prefix_token_count": int(lengths[row_index]),
                        "original_prefix_token_count": int(
                            row_tokenization["original_prefix_token_count"]
                        ),
                        "scoring_prefix_token_count": int(
                            row_tokenization["scoring_prefix_token_count"]
                        ),
                        "prefix_boundary_policy": str(
                            row_tokenization["prefix_boundary_policy"]
                        ),
                        "scoring_context_kind": str(row_tokenization["scoring_context_kind"]),
                        "assistant_prefix_before_candidate_sha256": str(
                            row.get("assistant_prefix_before_candidate_sha256", "")
                        ),
                        "chat_prompt_text_sha256": str(row.get("chat_prompt_text_sha256", "")),
                        "prefix_boundary_adjusted": bool(
                            row_tokenization["prefix_boundary_adjusted"]
                        ),
                        "prefix_boundary_trimmed_token_count": int(
                            row_tokenization["prefix_boundary_trimmed_token_count"]
                        ),
                        "prefix_boundary_adjusted_surfaces": list(
                            row_tokenization["prefix_boundary_adjusted_surfaces"]
                        ),
                        "bucket_surfaces": validate_bucket_surfaces(row),
                        "bucket_token_ids": token_ids_by_bucket,
                        "full_vocab_bucket_masses": full_masses,
                        "candidate_normalized_bucket_masses": candidate_normalized_masses,
                        "source_detection_count": int(row.get("source_detection_count", 1)),
                        "source_response_count": int(row.get("source_response_count", 0)),
                        "source_family_counts": dict(row.get("source_family_counts", {})),
                        "observed_surface_counts": dict(row.get("observed_surface_counts", {})),
                        "observed_case_variant_counts": dict(row.get("observed_case_variant_counts", {})),
                        "template_preflight_only": bool(row.get("template_preflight_only", False)),
                        "model_scoring_started": True,
                        "training_started": False,
                        "generation_started": False,
                        "e2e_eval_started": False,
                        "paper_claim_allowed": False,
                        "not_payload_recovery": True,
                        "not_full_far": True,
                    }
                )
    return scored, invalid_rows


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def weighted_mean(values: Sequence[tuple[float, int]]) -> float:
    denominator = sum(weight for _, weight in values)
    if denominator <= 0:
        return 0.0
    return sum(value * weight for value, weight in values) / denominator


def ratio(masses: Mapping[str, float]) -> float | None:
    positives = [float(value) for value in masses.values() if float(value) > 0.0]
    if not positives:
        return None
    return max(positives) / min(positives)


def aggregate_mass_rows(context_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_bank_variant: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in context_rows:
        key = (str(row["candidate_bank_id"]), str(row["casing_variant"]))
        by_bank_variant[key].append(row)

    output: list[dict[str, Any]] = []
    for (bank_id, variant), rows in sorted(by_bank_variant.items()):
        bucket_ids = sorted(str(bucket_id) for bucket_id in rows[0]["full_vocab_bucket_masses"])
        weights = [int(row.get("source_detection_count", 1)) for row in rows]
        full_masses = {
            bucket_id: weighted_mean(
                [
                    (float(row["full_vocab_bucket_masses"][bucket_id]), int(row.get("source_detection_count", 1)))
                    for row in rows
                ]
            )
            for bucket_id in bucket_ids
        }
        candidate_masses = {
            bucket_id: weighted_mean(
                [
                    (
                        float(row["candidate_normalized_bucket_masses"][bucket_id]),
                        int(row.get("source_detection_count", 1)),
                    )
                    for row in rows
                ]
            )
            for bucket_id in bucket_ids
        }
        unweighted_full_masses = {
            bucket_id: mean([float(row["full_vocab_bucket_masses"][bucket_id]) for row in rows])
            for bucket_id in bucket_ids
        }
        unweighted_candidate_masses = {
            bucket_id: mean(
                [float(row["candidate_normalized_bucket_masses"][bucket_id]) for row in rows]
            )
            for bucket_id in bucket_ids
        }
        output.append(
            {
                "schema_name": "natural_evidence_v2_wp3_context_mass_row_v1",
                "candidate_bank_id": bank_id,
                "casing_variant": variant,
                "bucket_masses": full_masses,
                "candidate_normalized_bucket_masses": candidate_masses,
                "unweighted_full_vocab_bucket_masses": unweighted_full_masses,
                "unweighted_candidate_normalized_bucket_masses": unweighted_candidate_masses,
                "full_vocab_min_bucket_mass": min(full_masses.values()) if full_masses else 0.0,
                "full_vocab_mass_ratio": ratio(full_masses),
                "candidate_normalized_mass_ratio": ratio(candidate_masses),
                "unweighted_full_vocab_mass_ratio": ratio(unweighted_full_masses),
                "context_count": len(rows),
                "source_detection_count_total": sum(weights),
                "mass": full_masses,
                "mass_aggregation": "source_detection_count_weighted_mean",
                "casing_variants_kept_separate": True,
                "model_scoring_started": True,
                "training_started": False,
                "generation_started": False,
                "e2e_eval_started": False,
                "paper_claim_allowed": False,
                "not_payload_recovery": True,
                "not_full_far": True,
            }
        )
    return output


def mass_audit_payload(
    *,
    config: Mapping[str, Any],
    mass_json: Path,
    mass_rows: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    invalid_tokenization_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bucket_policy = config.get("bucket_policy", {})
    if not isinstance(bucket_policy, Mapping):
        bucket_policy = {}
    min_mass_required = float(bucket_policy.get("min_bucket_mass", 0.005))
    max_ratio_required = float(bucket_policy.get("max_mass_ratio", 5.0))

    bank_variant_results: list[dict[str, Any]] = []
    all_pass = True
    for row in mass_rows:
        masses = {str(bucket_id): float(mass) for bucket_id, mass in row["bucket_masses"].items()}
        positive_masses = [mass for mass in masses.values() if mass > 0]
        min_mass = None if not positive_masses else min(positive_masses)
        mass_ratio = None if not positive_masses else max(positive_masses) / min(positive_masses)
        passed = (
            set(masses) == {"0", "1"}
            and min_mass is not None
            and min_mass >= min_mass_required
            and mass_ratio is not None
            and mass_ratio <= max_ratio_required
        )
        all_pass = all_pass and passed
        bank_variant_results.append(
            {
                "candidate_bank_id": str(row["candidate_bank_id"]),
                "casing_variant": str(row["casing_variant"]),
                "bucket_masses": masses,
                "min_bucket_mass": min_mass,
                "max_bucket_mass_ratio": mass_ratio,
                "context_count": int(row.get("context_count", 0)),
                "source_detection_count_total": int(row.get("source_detection_count_total", 0)),
                "passed": passed,
            }
        )

    return {
        "schema_name": "natural_evidence_v2_wp3_context_mass_audit_v1",
        "status": (
            "PASS_CONTEXT_SPECIFIC_MODEL_MASS_GATE_REVIEW_REQUIRED"
            if all_pass
            else "FAIL_CONTEXT_SPECIFIC_MODEL_MASS_GATE"
        ),
        "mass_gate_status": "PASS_REVIEW_REQUIRED" if all_pass else "FAIL",
        "mass_gate_scope": "candidate_bank_id_and_casing_variant",
        "mass_json": str(mass_json),
        "min_bucket_mass_required": min_mass_required,
        "max_mass_ratio_required": max_ratio_required,
        "bank_variant_results": bank_variant_results,
        "bank_variant_count": len(bank_variant_results),
        "score_plan_rows": int(validation["score_plan_rows"]),
        "context_score_rows": sum(int(row.get("context_count", 0)) for row in mass_rows),
        "invalid_tokenization_rows": len(invalid_tokenization_rows),
        "tokenizer_invalid_rows_skipped_from_mass_gate": True,
        "source_detection_count_total": int(validation["source_detection_count_total"]),
        "template_preflight_only": bool(validation["template_preflight_only"]),
        "casing_variants_kept_separate": True,
        "wp4_allowed": False,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
    }


def main() -> None:
    args = parse_args()
    score_plan_path = resolve_path(args.score_plan)
    plan_rows = read_jsonl(score_plan_path)
    validation = validate_plan_rows(plan_rows)
    if args.validate_plan_only and args.validate_tokenizer_only:
        raise ValueError("choose only one of --validate-plan-only or --validate-tokenizer-only")
    if args.validate_plan_only:
        print(json.dumps(validation, sort_keys=True))
        return
    if args.validate_tokenizer_only:
        tokenizer = load_tokenizer(tokenizer_name=str(args.tokenizer_name))
        tokenizer_validation = validate_tokenizer_boundaries(
            tokenizer=tokenizer,
            plan_rows=plan_rows,
            validation=validation,
            skip_invalid=bool(args.skip_invalid_tokenization),
        )
        print(json.dumps(tokenizer_validation, sort_keys=True))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --validate-plan-only is set")
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")

    config = read_yaml(resolve_path(args.config))
    torch_module, tokenizer, model, device = load_model(
        model_name=str(args.model_name),
        tokenizer_name=str(args.tokenizer_name),
        require_cuda=bool(args.require_cuda),
    )
    try:
        context_rows, invalid_tokenization_rows = score_plan_rows(
            torch_module=torch_module,
            tokenizer=tokenizer,
            model=model,
            device=device,
            plan_rows=plan_rows,
            batch_size=max(1, int(args.batch_size)),
            max_length=max(1, int(args.max_length)),
            skip_invalid_tokenization=bool(args.skip_invalid_tokenization),
        )
    finally:
        del model
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()

    mass_rows = aggregate_mass_rows(context_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_scores_jsonl = output_dir / "qwen_v2_wp3_context_mass_context_scores.jsonl"
    mass_json = output_dir / "qwen_v2_wp3_context_mass_artifact.json"
    audit_json = output_dir / "qwen_v2_wp3_context_mass_audit.json"
    summary_json = output_dir / "qwen_v2_wp3_context_mass_score_summary.json"
    invalid_tokenization_jsonl = output_dir / "qwen_v2_wp3_context_mass_invalid_tokenization_rows.jsonl"
    audit = mass_audit_payload(
        config=config,
        mass_json=mass_json,
        mass_rows=mass_rows,
        validation=validation,
        invalid_tokenization_rows=invalid_tokenization_rows,
    )
    summary = {
        "schema_name": "natural_evidence_v2_wp3_context_mass_score_summary_v1",
        "status": "WP3_CONTEXT_MASS_SCORED_NOT_TRAINING_NOT_GENERATION",
        "score_plan_jsonl": str(score_plan_path),
        "model_name": str(args.model_name),
        "tokenizer_name": str(args.tokenizer_name),
        "context_score_rows": len(context_rows),
        "invalid_tokenization_rows": len(invalid_tokenization_rows),
        "tokenizer_invalid_rows_skipped_from_mass_gate": True,
        "mass_rows": len(mass_rows),
        "context_scores_jsonl": str(context_scores_jsonl),
        "invalid_tokenization_jsonl": str(invalid_tokenization_jsonl),
        "mass_json": str(mass_json),
        "audit_json": str(audit_json),
        "mass_gate_status": str(audit["mass_gate_status"]),
        "score_plan_rows_by_bank": validation["score_plan_rows_by_bank"],
        "score_plan_rows_by_casing_variant": validation["score_plan_rows_by_casing_variant"],
        "score_plan_rows_by_bank_and_casing_variant": validation[
            "score_plan_rows_by_bank_and_casing_variant"
        ],
        "template_preflight_only": bool(validation["template_preflight_only"]),
        "casing_variants_kept_separate": True,
        "model_scoring_started": True,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "wp4_allowed": False,
        "limitations": [
            "Context-specific next-token mass scoring only.",
            "No text is generated and no model weights are changed.",
            "Casing variants remain separate in mass and audit rows.",
            "Mass gate outcomes require review and do not establish payload recovery or FAR.",
        ],
    }
    write_jsonl(context_scores_jsonl, context_rows)
    write_jsonl(invalid_tokenization_jsonl, invalid_tokenization_rows)
    write_json(
        mass_json,
        {
            "schema_name": "natural_evidence_v2_wp3_context_mass_artifact_v1",
            "mass_rows": mass_rows,
        },
    )
    write_json(audit_json, audit)
    write_json(summary_json, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "context_score_rows": len(context_rows),
                "mass_rows": len(mass_rows),
                "mass_gate_status": summary["mass_gate_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
