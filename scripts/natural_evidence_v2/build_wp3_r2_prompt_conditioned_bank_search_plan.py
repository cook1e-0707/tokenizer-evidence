from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.run_wp3_restricted_step_label_density_audit import (
    forbidden_terms_in_text,
    read_yaml,
)


DEFAULT_PROMPTS = (
    ROOT
    / "results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/"
    "restricted_step_label_strict_density_audit_prompts.jsonl"
)
DEFAULT_DENSITY_DIR = (
    ROOT / "results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_audit_850885"
)
DEFAULT_PREVIOUS_SCORE_DIR = (
    ROOT / "results/natural_evidence_v2/status/wp3_r2_observed_high_mass_context_mass_score_851233"
)
DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml"
PREFIX_BOUNDARY_TOKENIZATION_POLICY = "score_longest_common_token_prefix_when_candidate_retokenizes_boundary"
SURFACE_RE = re.compile(r"^[A-Z][A-Za-z'-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only WP3-R2 prompt-conditioned context-mass "
            "repair plan. It scores candidate banks at actual owner prompt + "
            "assistant-prefix slot contexts from job 850885 outputs. It does not "
            "load a tokenizer/model, submit Slurm, train, generate text, run "
            "E2E, aggregate FAR, or make positive claims."
        )
    )
    parser.add_argument("--prompts-jsonl", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--density-dir", type=Path, default=DEFAULT_DENSITY_DIR)
    parser.add_argument("--previous-score-dir", type=Path, default=DEFAULT_PREVIOUS_SCORE_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-surfaces", type=int, default=24)
    parser.add_argument("--max-contexts", type=int, default=512)
    parser.add_argument("--max-banks", type=int, default=20)
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


def slug(text: str) -> str:
    return text.lower().replace("-", "_").replace("'", "")


def bank_id(bucket_0: Sequence[str], bucket_1: Sequence[str]) -> str:
    return "step_label_r2_prompt_ctx_" + "_".join(slug(item) for item in bucket_0) + "_vs_" + "_".join(
        slug(item) for item in bucket_1
    ) + "_v1"


def plan_row_id(*, bank: str, prompt_id: str, step_index: int, response_id: str) -> str:
    payload = json.dumps(
        {
            "candidate_bank_id": bank,
            "prompt_id": prompt_id,
            "response_id": response_id,
            "step_index": step_index,
            "scoring_context_kind": "chat_prompt_plus_assistant_prefix",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)[:20]


def invalid_surfaces(previous_score_dir: Path) -> set[str]:
    path = previous_score_dir / "qwen_v2_wp3_context_mass_invalid_tokenization_rows.jsonl"
    if not path.exists():
        return set()
    output: set[str] = set()
    for row in read_jsonl(path):
        for members in dict(row.get("bucket_surfaces", {})).values():
            for surface in members:
                output.add(str(surface))
    return output


def validate_surfaces(config: Mapping[str, Any], surfaces: Sequence[str]) -> None:
    for surface in surfaces:
        if not SURFACE_RE.fullmatch(surface):
            raise ValueError(f"surface is not a sentence-case ASCII word: {surface!r}")
        hits = forbidden_terms_in_text(config, surface)
        if hits:
            raise ValueError(f"surface contains forbidden public surface {hits}: {surface!r}")


def observed_counts(slot_rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in slot_rows:
        surface = str(row.get("first_word", ""))
        if SURFACE_RE.fullmatch(surface):
            counts[surface] += 1
    return counts


def surface_pool(
    *,
    config: Mapping[str, Any],
    counts: Counter[str],
    invalid: set[str],
    top_surfaces: int,
) -> list[str]:
    pool: list[str] = []
    for surface, _count in counts.most_common(max(1, top_surfaces * 4)):
        if surface in invalid:
            continue
        if surface.lower().endswith("ly"):
            continue
        try:
            validate_surfaces(config, [surface])
        except ValueError:
            continue
        pool.append(surface)
        if len(pool) >= top_surfaces:
            break
    if len(pool) < 8:
        raise ValueError(f"too few valid surfaces after filtering: {len(pool)}")
    return pool


def candidate_banks(pool: Sequence[str], *, max_banks: int) -> list[dict[str, Any]]:
    banks: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(bucket_0: Sequence[str], bucket_1: Sequence[str], source: str) -> None:
        if len(banks) >= max_banks:
            return
        left = list(bucket_0)
        right = list(bucket_1)
        if set(left) & set(right):
            return
        candidate_bank_id = bank_id(left, right)
        if candidate_bank_id in seen:
            return
        seen.add(candidate_bank_id)
        banks.append(
            {
                "schema_name": "natural_evidence_v2_wp3_r2_prompt_conditioned_bank_candidate_v1",
                "candidate_bank_id": candidate_bank_id,
                "bucket_0_surfaces": left,
                "bucket_1_surfaces": right,
                "bucket_count": 2,
                "source": source,
                "slot_type": "step_label_action_verb",
                "training_started": False,
                "generation_started": False,
                "e2e_eval_started": False,
                "paper_claim_allowed": False,
                "not_payload_recovery": True,
                "not_full_far": True,
            }
        )

    top = list(pool)
    for left, right in zip(top[0::2], top[1::2]):
        add([left], [right], "prompt_context_observed_frequency_adjacent_single_surface")
    for index in range(0, min(len(top), 24), 4):
        quartet = top[index : index + 4]
        if len(quartet) == 4:
            add(quartet[:2], quartet[2:], "prompt_context_observed_frequency_adjacent_two_surface")
            add([quartet[0], quartet[2]], [quartet[1], quartet[3]], "prompt_context_cross_balanced_two_surface")
    if len(top) >= 8:
        add([top[0], top[3]], [top[1], top[2]], "prompt_context_top4_cross_balanced_two_surface")
        add([top[4], top[7]], [top[5], top[6]], "prompt_context_second4_cross_balanced_two_surface")
    return banks


def assistant_prefix_before_word(response_text: str, char_end: int, first_word: str) -> str:
    start = int(char_end) - len(first_word)
    if start < 0 or response_text[start : int(char_end)] != first_word:
        raise ValueError(f"cannot recover assistant prefix before first word {first_word!r}")
    return response_text[:start]


def select_contexts(
    *,
    slot_rows: Sequence[Mapping[str, Any]],
    prompts_by_id: Mapping[str, Mapping[str, Any]],
    outputs_by_prompt_id: Mapping[str, Mapping[str, Any]],
    max_contexts: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], deque[Mapping[str, Any]]] = defaultdict(deque)
    for row in slot_rows:
        prompt = prompts_by_id.get(str(row.get("prompt_id", "")))
        output = outputs_by_prompt_id.get(str(row.get("prompt_id", "")))
        if not prompt or not output:
            continue
        first_word = str(row.get("first_word", ""))
        if not SURFACE_RE.fullmatch(first_word):
            continue
        key = (
            str(prompt.get("split", "")),
            str(row.get("variant_id") or prompt.get("variant_id", "")),
            str(row.get("step_index", "")),
        )
        grouped[key].append(row)

    selected: list[dict[str, Any]] = []
    keys = sorted(grouped)
    while len(selected) < max_contexts and keys:
        added = False
        for key in keys:
            queue = grouped[key]
            if not queue:
                continue
            row = dict(queue.popleft())
            prompt = prompts_by_id[str(row["prompt_id"])]
            output = outputs_by_prompt_id[str(row["prompt_id"])]
            response_text = str(output["response_text"])
            first_word = str(row["first_word"])
            assistant_prefix = assistant_prefix_before_word(
                response_text=response_text,
                char_end=int(row["char_end"]),
                first_word=first_word,
            )
            selected.append(
                {
                    "prompt_id": str(row["prompt_id"]),
                    "response_id": str(row["response_id"]),
                    "split": str(prompt.get("split", "")),
                    "variant_id": str(row.get("variant_id") or prompt.get("variant_id", "")),
                    "family_id": str(row.get("family_id") or prompt.get("family_id", "")),
                    "step_index": int(row["step_index"]),
                    "observed_first_word": first_word,
                    "chat_prompt_text": str(prompt["prompt_text"]),
                    "chat_prompt_text_sha256": str(prompt.get("prompt_text_sha256", sha256_text(str(prompt["prompt_text"])))),
                    "assistant_prefix_before_candidate": assistant_prefix,
                    "assistant_prefix_before_candidate_sha256": sha256_text(assistant_prefix),
                }
            )
            added = True
            if len(selected) >= max_contexts:
                break
        if not added:
            break
    return selected


def plan_rows(
    *,
    plan_id: str,
    banks: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    counts: Counter[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bank in banks:
        candidate_bank_id = str(bank["candidate_bank_id"])
        buckets = {
            "0": [str(item) for item in bank["bucket_0_surfaces"]],
            "1": [str(item) for item in bank["bucket_1_surfaces"]],
        }
        all_surfaces = buckets["0"] + buckets["1"]
        for context_index, context in enumerate(contexts):
            rows.append(
                {
                    "schema_name": "natural_evidence_v2_wp3_context_mass_score_plan_row_v1",
                    "plan_id": plan_id,
                    "plan_row_id": plan_row_id(
                        bank=candidate_bank_id,
                        prompt_id=str(context["prompt_id"]),
                        response_id=str(context["response_id"]),
                        step_index=int(context["step_index"]),
                    ),
                    "candidate_bank_id": candidate_bank_id,
                    "slot_type": "step_label_action_verb",
                    "anchor_kind": "line_start_step_label",
                    "bucket_policy_id": "qwen_v2_wp3_r2_prompt_conditioned_sentence_case_two_way",
                    "casing_variant": "sentence_case",
                    "bucket_surfaces": buckets,
                    "prefix_before_candidate": str(context["assistant_prefix_before_candidate"]),
                    "prefix_before_candidate_sha256": str(context["assistant_prefix_before_candidate_sha256"]),
                    "chat_prompt_text": str(context["chat_prompt_text"]),
                    "chat_prompt_text_sha256": str(context["chat_prompt_text_sha256"]),
                    "assistant_prefix_before_candidate": str(context["assistant_prefix_before_candidate"]),
                    "assistant_prefix_before_candidate_sha256": str(
                        context["assistant_prefix_before_candidate_sha256"]
                    ),
                    "scoring_context_kind": "chat_prompt_plus_assistant_prefix",
                    "prefix_is_empty": False,
                    "prefix_family": "restricted_step_label_prompt_conditioned",
                    "prefix_shape_index": int(context["step_index"]) - 1,
                    "prefix_boundary_tokenization_policy": PREFIX_BOUNDARY_TOKENIZATION_POLICY,
                    "source_prompt_id": str(context["prompt_id"]),
                    "source_response_id": str(context["response_id"]),
                    "source_split": str(context["split"]),
                    "source_variant_id": str(context["variant_id"]),
                    "source_step_index": int(context["step_index"]),
                    "observed_first_word": str(context["observed_first_word"]),
                    "source_detection_count": 1,
                    "source_response_count": 1,
                    "source_family_counts": {str(context["family_id"]): 1},
                    "source_response_ids": [str(context["response_id"])],
                    "observed_surface_counts": {surface: int(counts.get(surface, 0)) for surface in all_surfaces},
                    "observed_case_variant_counts": {"sentence_case": len(all_surfaces)},
                    "context_index_within_bank_variant": context_index,
                    "structural_selection": "r2_prompt_conditioned_observed_context_candidate_not_scored",
                    "scoring_status": "PLANNED_NOT_SCORED",
                    "tokenizer_stability_status": "NOT_EVALUATED",
                    "mass_gate_status": "NOT_EVALUATED",
                    "template_preflight_only": False,
                    "artifact_only": True,
                    "model_scoring_started": False,
                    "training_started": False,
                    "generation_started": False,
                    "e2e_eval_started": False,
                    "paper_claim_allowed": False,
                    "not_payload_recovery": True,
                    "not_full_far": True,
                }
            )
    return rows


def readme_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WP3-R2 Prompt-Conditioned Context-Mass Repair Plan",
            "",
            "Artifact-only plan for scoring high-frequency Step-label banks under owner prompt + assistant-prefix contexts.",
            "",
            f"candidate_bank_count: `{summary['candidate_bank_count']}`",
            f"selected_context_count: `{summary['selected_context_count']}`",
            f"score_plan_rows: `{summary['score_plan_rows']}`",
            "",
            "This is not tokenizer/model scoring, training, E2E, FAR, or a positive claim.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    prompts_path = resolve_path(args.prompts_jsonl)
    density_dir = resolve_path(args.density_dir)
    previous_score_dir = resolve_path(args.previous_score_dir)
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_yaml(config_path)
    prompts = read_jsonl(prompts_path)
    outputs = read_jsonl(density_dir / "restricted_step_label_model_outputs.jsonl")
    slot_rows = read_jsonl(density_dir / "restricted_step_label_detected_slots.jsonl")
    prompts_by_id = {str(row["prompt_id"]): row for row in prompts}
    outputs_by_prompt_id = {str(row["prompt_id"]): row for row in outputs}
    counts = observed_counts(slot_rows)
    invalid = invalid_surfaces(previous_score_dir)
    pool = surface_pool(config=config, counts=counts, invalid=invalid, top_surfaces=max(8, int(args.top_surfaces)))
    banks = candidate_banks(pool, max_banks=max(1, int(args.max_banks)))
    for bank in banks:
        validate_surfaces(
            config,
            [str(item) for item in bank["bucket_0_surfaces"]]
            + [str(item) for item in bank["bucket_1_surfaces"]],
        )
    contexts = select_contexts(
        slot_rows=slot_rows,
        prompts_by_id=prompts_by_id,
        outputs_by_prompt_id=outputs_by_prompt_id,
        max_contexts=max(1, int(args.max_contexts)),
    )
    if len(contexts) < max(16, min(128, int(args.max_contexts))):
        raise ValueError(f"too few prompt-conditioned contexts selected: {len(contexts)}")

    plan_id = "qwen_v2_wp3_r2_prompt_conditioned_bank_search_850885"
    rows = plan_rows(plan_id=plan_id, banks=banks, contexts=contexts, counts=counts)
    split_counts = Counter(str(row["split"]) for row in contexts)
    variant_counts = Counter(str(row["variant_id"]) for row in contexts)
    step_counts = Counter(str(row["step_index"]) for row in contexts)
    summary = {
        "schema_name": "natural_evidence_v2_wp3_r2_prompt_conditioned_bank_search_plan_summary_v1",
        "status": "WP3_R2_PROMPT_CONDITIONED_BANK_SEARCH_PLAN_READY_ARTIFACT_ONLY",
        "plan_id": plan_id,
        "source_density_dir": str(density_dir),
        "previous_score_dir": str(previous_score_dir),
        "source_detected_slot_rows": len(slot_rows),
        "invalid_surfaces_excluded_from_851233": sorted(invalid),
        "observed_surface_pool": pool,
        "observed_surface_pool_counts": {surface: int(counts[surface]) for surface in pool},
        "candidate_bank_count": len(banks),
        "selected_context_count": len(contexts),
        "selected_context_split_counts": dict(sorted(split_counts.items())),
        "selected_context_variant_counts": dict(sorted(variant_counts.items())),
        "selected_context_step_counts": dict(sorted(step_counts.items(), key=lambda item: int(item[0]))),
        "score_plan_rows": len(rows),
        "scoring_context_kind": "chat_prompt_plus_assistant_prefix",
        "context_mass_scorer": "scripts/natural_evidence_v2/score_wp3_context_mass.py",
        "slurm_wrapper": "scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch",
        "recommended_max_length": 1536,
        "recommended_next_step": (
            "Review this artifact-only prompt-conditioned plan. If approved, enable exactly one "
            "allowlist entry and submit one Chimera Slurm context-mass scoring job."
        ),
        "pilot_min_bucket_mass_gate": 0.03,
        "preferred_combined_bank_mass_gate": 0.10,
        "max_mass_ratio_gate": 3.0,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
    }
    slurm_review = {
        "schema_name": "natural_evidence_v2_wp3_r2_prompt_conditioned_slurm_review_v1",
        "status": "READY_FOR_REVIEW_NOT_SUBMITTED",
        "score_plan_jsonl": str(
            output_dir / "qwen_v2_wp3_r2_prompt_conditioned_context_mass_score_plan.jsonl"
        ),
        "candidate_bank_jsonl": str(output_dir / "qwen_v2_wp3_r2_prompt_conditioned_bank_candidates.jsonl"),
        "suggested_slurm_wrapper": "scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch",
        "suggested_partition": "DGXA100",
        "suggested_max_length": 1536,
        "requires_allowlist_entry": True,
        "submit_without_explicit_review": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }
    write_jsonl(output_dir / "qwen_v2_wp3_r2_prompt_conditioned_bank_candidates.jsonl", banks)
    write_jsonl(output_dir / "qwen_v2_wp3_r2_prompt_conditioned_contexts.jsonl", contexts)
    write_jsonl(output_dir / "qwen_v2_wp3_r2_prompt_conditioned_context_mass_score_plan.jsonl", rows)
    write_json(output_dir / "qwen_v2_wp3_r2_prompt_conditioned_bank_search_summary.json", summary)
    write_json(output_dir / "qwen_v2_wp3_r2_prompt_conditioned_slurm_review.json", slurm_review)
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(json.dumps({"output_dir": str(output_dir), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
