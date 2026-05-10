from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.run_wp3_restricted_step_label_density_audit import (
    forbidden_terms_in_text,
    read_yaml,
)


DEFAULT_DENSITY_DIR = (
    ROOT / "results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_audit_850885"
)
DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml"
PREFIX_BOUNDARY_TOKENIZATION_POLICY = "score_longest_common_token_prefix_when_candidate_retokenizes_boundary"
SURFACE_RE = re.compile(r"^[A-Z][A-Za-z'-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only WP3-R2 high-mass 2-way bank search plan "
            "from observed Step-label first words in model outputs. This writes "
            "score-plan rows for the existing Slurm context-mass scorer. It does "
            "not load a tokenizer/model, submit Slurm, train, generate text, run "
            "E2E, aggregate FAR, or make positive claims."
        )
    )
    parser.add_argument("--density-dir", type=Path, default=DEFAULT_DENSITY_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-surfaces", type=int, default=24)
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
    return "step_label_r2_observed_" + "_".join(slug(item) for item in bucket_0) + "_vs_" + "_".join(
        slug(item) for item in bucket_1
    ) + "_v1"


def plan_row_id(*, candidate_bank_id: str, prefix: str) -> str:
    payload = json.dumps(
        {
            "candidate_bank_id": candidate_bank_id,
            "casing_variant": "sentence_case",
            "prefix_before_candidate": prefix,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)[:20]


def validate_surfaces(config: Mapping[str, Any], surfaces: Sequence[str]) -> None:
    for surface in surfaces:
        if not SURFACE_RE.fullmatch(surface):
            raise ValueError(f"surface is not a sentence-case ASCII word: {surface!r}")
        hits = forbidden_terms_in_text(config, surface)
        if hits:
            raise ValueError(f"surface contains forbidden public surface {hits}: {surface!r}")


def observed_surface_counts(slot_rows: Sequence[Mapping[str, Any]]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    counts: Counter[str] = Counter()
    by_step: dict[str, Counter[str]] = defaultdict(Counter)
    for row in slot_rows:
        surface = str(row.get("first_word", ""))
        if not SURFACE_RE.fullmatch(surface):
            continue
        counts[surface] += 1
        by_step[str(row.get("step_index", ""))][surface] += 1
    return counts, by_step


def choose_surface_pool(
    *,
    config: Mapping[str, Any],
    counts: Counter[str],
    top_surfaces: int,
) -> list[str]:
    pool: list[str] = []
    for surface, _count in counts.most_common(max(1, top_surfaces * 3)):
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
        raise ValueError(f"too few observed candidate surfaces after filtering: {len(pool)}")
    return pool


def candidate_banks_from_pool(pool: Sequence[str]) -> list[dict[str, Any]]:
    banks: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(bucket_0: Sequence[str], bucket_1: Sequence[str], source: str) -> None:
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
                "schema_name": "natural_evidence_v2_wp3_r2_observed_high_mass_bank_candidate_v1",
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
        add([left], [right], "observed_frequency_adjacent_single_surface")

    quartets = [top[index : index + 4] for index in range(0, min(len(top), 24), 4)]
    for quartet in quartets:
        if len(quartet) == 4:
            add(quartet[:2], quartet[2:], "observed_frequency_adjacent_two_surface")
            add([quartet[0], quartet[2]], [quartet[1], quartet[3]], "observed_frequency_cross_balanced_two_surface")

    if len(top) >= 8:
        add([top[0], top[3]], [top[1], top[2]], "top4_cross_balanced_two_surface")
        add([top[4], top[7]], [top[5], top[6]], "second4_cross_balanced_two_surface")
    return banks


def plan_rows(
    *,
    plan_id: str,
    candidates: Sequence[Mapping[str, Any]],
    counts: Counter[str],
    by_step: Mapping[str, Counter[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bank in candidates:
        candidate_bank_id = str(bank["candidate_bank_id"])
        buckets = {
            "0": [str(item) for item in bank["bucket_0_surfaces"]],
            "1": [str(item) for item in bank["bucket_1_surfaces"]],
        }
        all_surfaces = buckets["0"] + buckets["1"]
        for step_index in range(1, 17):
            prefix = f"Step {step_index}: "
            rows.append(
                {
                    "schema_name": "natural_evidence_v2_wp3_context_mass_score_plan_row_v1",
                    "plan_id": plan_id,
                    "plan_row_id": plan_row_id(candidate_bank_id=candidate_bank_id, prefix=prefix),
                    "candidate_bank_id": candidate_bank_id,
                    "slot_type": "step_label_action_verb",
                    "anchor_kind": "line_start_step_label",
                    "bucket_policy_id": "qwen_v2_wp3_r2_observed_high_mass_sentence_case_two_way",
                    "casing_variant": "sentence_case",
                    "bucket_surfaces": buckets,
                    "prefix_before_candidate": prefix,
                    "prefix_before_candidate_sha256": sha256_text(prefix),
                    "prefix_is_empty": False,
                    "prefix_family": "restricted_step_label",
                    "prefix_shape_index": step_index - 1,
                    "prefix_boundary_tokenization_policy": PREFIX_BOUNDARY_TOKENIZATION_POLICY,
                    "source_detection_count": sum(int(counts.get(surface, 0)) for surface in all_surfaces),
                    "source_response_count": 0,
                    "source_family_counts": {},
                    "source_response_ids": [],
                    "observed_surface_counts": {surface: int(counts.get(surface, 0)) for surface in all_surfaces},
                    "observed_surface_counts_at_step": {
                        surface: int(by_step.get(str(step_index), Counter()).get(surface, 0))
                        for surface in all_surfaces
                    },
                    "observed_case_variant_counts": {"sentence_case": len(all_surfaces)},
                    "context_index_within_bank_variant": step_index - 1,
                    "structural_selection": "r2_observed_high_frequency_bank_candidate_not_scored",
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
            "# WP3-R2 Observed High-Mass Bank Search Plan",
            "",
            "Artifact-only plan built from observed Step-label first words in job 850885 outputs.",
            "",
            f"candidate_bank_count: `{summary['candidate_bank_count']}`",
            f"score_plan_rows: `{summary['score_plan_rows']}`",
            "",
            "This is not tokenizer/model scoring, training, E2E, FAR, or a positive claim.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    density_dir = resolve_path(args.density_dir)
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_yaml(config_path)
    slot_rows = read_jsonl(density_dir / "restricted_step_label_detected_slots.jsonl")
    counts, by_step = observed_surface_counts(slot_rows)
    pool = choose_surface_pool(config=config, counts=counts, top_surfaces=max(8, int(args.top_surfaces)))
    candidates = candidate_banks_from_pool(pool)
    for candidate in candidates:
        validate_surfaces(
            config,
            [str(item) for item in candidate["bucket_0_surfaces"]]
            + [str(item) for item in candidate["bucket_1_surfaces"]],
        )

    plan_id = "qwen_v2_wp3_r2_observed_high_mass_bank_search_850885"
    rows = plan_rows(plan_id=plan_id, candidates=candidates, counts=counts, by_step=by_step)
    summary = {
        "schema_name": "natural_evidence_v2_wp3_r2_observed_high_mass_bank_search_plan_summary_v1",
        "status": "WP3_R2_OBSERVED_HIGH_MASS_BANK_SEARCH_PLAN_READY_ARTIFACT_ONLY",
        "plan_id": plan_id,
        "density_dir": str(density_dir),
        "source_detected_slot_rows": len(slot_rows),
        "observed_surface_pool": pool,
        "observed_surface_pool_counts": {surface: int(counts[surface]) for surface in pool},
        "candidate_bank_count": len(candidates),
        "score_plan_rows": len(rows),
        "score_plan_rows_per_bank": 16,
        "context_mass_scorer": "scripts/natural_evidence_v2/score_wp3_context_mass.py",
        "slurm_wrapper": "scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch",
        "recommended_next_step": (
            "Review this artifact-only plan. If approved, enable exactly one allowlist entry "
            "and submit one Chimera Slurm context-mass scoring job."
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
        "schema_name": "natural_evidence_v2_wp3_r2_observed_high_mass_bank_search_slurm_review_v1",
        "status": "READY_FOR_REVIEW_NOT_SUBMITTED",
        "score_plan_jsonl": str(
            output_dir / "qwen_v2_wp3_r2_observed_high_mass_context_mass_score_plan.jsonl"
        ),
        "candidate_bank_jsonl": str(output_dir / "qwen_v2_wp3_r2_observed_high_mass_bank_candidates.jsonl"),
        "suggested_slurm_wrapper": "scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch",
        "suggested_partition": "DGXA100",
        "requires_allowlist_entry": True,
        "submit_without_explicit_review": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }

    write_jsonl(output_dir / "qwen_v2_wp3_r2_observed_high_mass_bank_candidates.jsonl", candidates)
    write_jsonl(output_dir / "qwen_v2_wp3_r2_observed_high_mass_context_mass_score_plan.jsonl", rows)
    write_json(output_dir / "qwen_v2_wp3_r2_observed_high_mass_bank_search_summary.json", summary)
    write_json(output_dir / "qwen_v2_wp3_r2_observed_high_mass_context_mass_slurm_review.json", slurm_review)
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(json.dumps({"output_dir": str(output_dir), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
