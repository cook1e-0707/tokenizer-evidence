from __future__ import annotations

import argparse
import hashlib
import json
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


DEFAULT_SCORE_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/wp3_restricted_step_label_expanded_mass_score_850483"
)
DEFAULT_DENSITY_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/wp3_restricted_step_label_density_audit_850434"
)
DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml"
PREFIX_BOUNDARY_TOKENIZATION_POLICY = "score_longest_common_token_prefix_when_candidate_retokenizes_boundary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only mass-aware repair plan from the 850483 "
            "restricted Step-label context-mass scores. This recombines already "
            "scored high-mass bucket groups into candidate two-way banks and "
            "writes a fresh score plan. It does not load a tokenizer/model, "
            "submit Slurm, train, generate text, run E2E, aggregate FAR, or "
            "make positive claims."
        )
    )
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--density-dir", type=Path, default=DEFAULT_DENSITY_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-bucket-mass", type=float, default=0.005)
    parser.add_argument("--max-mass-ratio", type=float, default=5.0)
    parser.add_argument(
        "--max-recombined-candidates",
        type=int,
        default=12,
        help="Keep the top N recombined candidates by inferred balance and mass.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top level must be an object: {path}")
    return payload


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


def slug_surfaces(surfaces: Sequence[str]) -> str:
    return "_".join(surface.lower().replace("-", "_") for surface in surfaces)


def plan_row_id(*, bank_id: str, prefix: str) -> str:
    payload = json.dumps(
        {
            "candidate_bank_id": bank_id,
            "casing_variant": "sentence_case",
            "prefix_before_candidate": prefix,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)[:20]


def bucket_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean for empty bucket values")
    return float(sum(values) / len(values))


def validate_surface_text(config: Mapping[str, Any], surfaces: Sequence[str]) -> None:
    for surface in surfaces:
        if not surface or not surface[:1].isupper():
            raise ValueError(f"surface must be sentence case: {surface!r}")
        hits = forbidden_terms_in_text(config, surface)
        if hits:
            raise ValueError(f"surface contains forbidden marker {hits}: {surface!r}")


def observed_counts(*, density_dir: Path) -> dict[str, Any]:
    slots_path = density_dir / "restricted_step_label_detected_slots.jsonl"
    slots = read_jsonl(slots_path)
    first_word_counts = Counter(str(row.get("first_word", "")) for row in slots)
    first_word_by_step = defaultdict(Counter)
    for row in slots:
        first_word = str(row.get("first_word", ""))
        step_index = int(row.get("step_index", 0))
        first_word_by_step[step_index][first_word] += 1
    return {
        "first_word_counts": dict(sorted(first_word_counts.items())),
        "first_word_counts_by_step": {
            str(step): dict(sorted(counter.items()))
            for step, counter in sorted(first_word_by_step.items())
        },
        "slot_rows": len(slots),
    }


def summarize_original_banks(
    *,
    mass_rows: Sequence[Mapping[str, Any]],
    invalid_rows: Sequence[Mapping[str, Any]],
    context_scores: Sequence[Mapping[str, Any]],
    min_bucket_mass: float,
    max_mass_ratio: float,
) -> list[dict[str, Any]]:
    context_by_bank = defaultdict(list)
    for row in context_scores:
        context_by_bank[str(row["candidate_bank_id"])].append(row)

    invalid_by_bank = Counter(str(row.get("candidate_bank_id", "")) for row in invalid_rows)
    invalid_messages_by_bank: dict[str, list[str]] = defaultdict(list)
    for row in invalid_rows:
        bank_id = str(row.get("candidate_bank_id", ""))
        msg = str(row.get("error_message", ""))
        if msg not in invalid_messages_by_bank[bank_id]:
            invalid_messages_by_bank[bank_id].append(msg)

    output: list[dict[str, Any]] = []
    seen = set()
    for row in mass_rows:
        bank_id = str(row["candidate_bank_id"])
        seen.add(bank_id)
        masses = {str(k): float(v) for k, v in dict(row["bucket_masses"]).items()}
        low_bucket = min(masses, key=masses.get)
        high_bucket = max(masses, key=masses.get)
        context_rows = context_by_bank.get(bank_id, [])
        bucket_surfaces = dict(context_rows[0].get("bucket_surfaces", {})) if context_rows else {}
        contexts_below_gate = {
            bucket_id: sum(
                1
                for context_row in context_rows
                if float(dict(context_row["full_vocab_bucket_masses"])[bucket_id]) < min_bucket_mass
            )
            for bucket_id in ("0", "1")
        }
        output.append(
            {
                "schema_name": "natural_evidence_v2_wp3_mass_aware_original_bank_review_v1",
                "candidate_bank_id": bank_id,
                "bucket_surfaces": bucket_surfaces,
                "bucket_masses": masses,
                "min_bucket_mass": float(row["full_vocab_min_bucket_mass"]),
                "mass_ratio": float(row["full_vocab_mass_ratio"]),
                "low_mass_bucket": low_bucket,
                "high_mass_bucket": high_bucket,
                "contexts_below_min_bucket_mass_by_bucket": contexts_below_gate,
                "decision": (
                    "near_miss_repair_candidate"
                    if float(row["full_vocab_min_bucket_mass"]) >= min_bucket_mass * 0.6
                    and float(row["full_vocab_mass_ratio"]) <= max_mass_ratio
                    else "drop_or_rework_before_rescoring"
                ),
                "training_started": False,
                "generation_started": False,
                "e2e_eval_started": False,
                "paper_claim_allowed": False,
            }
        )

    for bank_id in sorted(set(invalid_by_bank) - seen):
        output.append(
            {
                "schema_name": "natural_evidence_v2_wp3_mass_aware_original_bank_review_v1",
                "candidate_bank_id": bank_id,
                "decision": "remove_or_replace_invalid_tokenization_surface",
                "invalid_tokenization_rows": int(invalid_by_bank[bank_id]),
                "invalid_tokenization_messages": invalid_messages_by_bank[bank_id],
                "training_started": False,
                "generation_started": False,
                "e2e_eval_started": False,
                "paper_claim_allowed": False,
            }
        )
    return sorted(output, key=lambda row: str(row["candidate_bank_id"]))


def bucket_groups_from_context_scores(
    *,
    context_scores: Sequence[Mapping[str, Any]],
    min_bucket_mass: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    values_by_group = defaultdict(list)
    for row in context_scores:
        bank_id = str(row["candidate_bank_id"])
        bucket_surfaces = dict(row["bucket_surfaces"])
        bucket_masses = dict(row["full_vocab_bucket_masses"])
        for bucket_id in ("0", "1"):
            key = (bank_id, bucket_id)
            grouped[key] = {
                "source_candidate_bank_id": bank_id,
                "source_bucket_id": bucket_id,
                "surfaces": [str(item) for item in bucket_surfaces[bucket_id]],
            }
            values_by_group[key].append(float(bucket_masses[bucket_id]))

    groups: list[dict[str, Any]] = []
    for key, metadata in grouped.items():
        values = values_by_group[key]
        mean_mass = bucket_mean(values)
        groups.append(
            {
                **metadata,
                "mean_full_vocab_mass": mean_mass,
                "min_context_full_vocab_mass": float(min(values)),
                "max_context_full_vocab_mass": float(max(values)),
                "contexts_below_min_bucket_mass": sum(value < min_bucket_mass for value in values),
                "context_count": len(values),
                "eligible_for_recombination": mean_mass >= min_bucket_mass,
            }
        )
    return sorted(groups, key=lambda row: (-float(row["mean_full_vocab_mass"]), row["source_candidate_bank_id"]))


def recombine_bucket_groups(
    *,
    groups: Sequence[Mapping[str, Any]],
    max_mass_ratio: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    eligible = [group for group in groups if bool(group["eligible_for_recombination"])]
    candidates: list[dict[str, Any]] = []
    for index, left in enumerate(eligible):
        for right in eligible[index + 1 :]:
            left_surfaces = [str(item) for item in left["surfaces"]]
            right_surfaces = [str(item) for item in right["surfaces"]]
            if set(left_surfaces) & set(right_surfaces):
                continue
            left_mass = float(left["mean_full_vocab_mass"])
            right_mass = float(right["mean_full_vocab_mass"])
            ratio = max(left_mass, right_mass) / max(min(left_mass, right_mass), 1e-12)
            if ratio > max_mass_ratio:
                continue
            high_mass_penalty = max(left_mass, right_mass)
            balance_distance = abs(1.0 - ratio)
            bank_id = (
                "step_label_recombined_"
                + slug_surfaces(left_surfaces)
                + "_vs_"
                + slug_surfaces(right_surfaces)
                + "_v1"
            )
            candidates.append(
                {
                    "schema_name": "natural_evidence_v2_wp3_mass_aware_recombined_bank_candidate_v1",
                    "candidate_bank_id": bank_id,
                    "slot_type": "step_label_action_verb",
                    "bucket_count": 2,
                    "bucket_0_surfaces": left_surfaces,
                    "bucket_1_surfaces": right_surfaces,
                    "source_bucket_groups": [
                        {
                            "source_candidate_bank_id": left["source_candidate_bank_id"],
                            "source_bucket_id": left["source_bucket_id"],
                            "mean_full_vocab_mass": left_mass,
                        },
                        {
                            "source_candidate_bank_id": right["source_candidate_bank_id"],
                            "source_bucket_id": right["source_bucket_id"],
                            "mean_full_vocab_mass": right_mass,
                        },
                    ],
                    "inferred_bucket_masses_from_850483": {"0": left_mass, "1": right_mass},
                    "inferred_min_bucket_mass_from_850483": min(left_mass, right_mass),
                    "inferred_mass_ratio_from_850483": ratio,
                    "ranking_score": balance_distance + 0.1 * high_mass_penalty,
                    "status": "CANDIDATE_REQUIRES_FRESH_TOKENIZER_AND_CONTEXT_MASS_AUDIT",
                    "model_scoring_started": False,
                    "training_started": False,
                    "generation_started": False,
                    "e2e_eval_started": False,
                    "wp4_allowed": False,
                    "paper_claim_allowed": False,
                    "not_payload_recovery": True,
                    "not_full_far": True,
                }
            )
    candidates.sort(
        key=lambda row: (
            float(row["ranking_score"]),
            -float(row["inferred_min_bucket_mass_from_850483"]),
            row["candidate_bank_id"],
        )
    )
    return candidates[:max_candidates]


def plan_rows(
    *,
    plan_id: str,
    candidates: Sequence[Mapping[str, Any]],
    counts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first_word_counts = dict(counts["first_word_counts"])
    first_word_counts_by_step = dict(counts["first_word_counts_by_step"])
    for bank in candidates:
        bank_id = str(bank["candidate_bank_id"])
        buckets = {
            "0": [str(item) for item in bank["bucket_0_surfaces"]],
            "1": [str(item) for item in bank["bucket_1_surfaces"]],
        }
        observed_surface_counts = {
            surface: int(first_word_counts.get(surface, 0))
            for surface in [*buckets["0"], *buckets["1"]]
        }
        for step_index in range(1, 17):
            prefix = f"Step {step_index}: "
            rows.append(
                {
                    "schema_name": "natural_evidence_v2_wp3_context_mass_score_plan_row_v1",
                    "plan_id": plan_id,
                    "plan_row_id": plan_row_id(bank_id=bank_id, prefix=prefix),
                    "candidate_bank_id": bank_id,
                    "slot_type": "step_label_action_verb",
                    "anchor_kind": "line_start_step_label",
                    "bucket_policy_id": "qwen_v2_wp3_restricted_step_label_mass_aware_recombined_sentence_case_two_way",
                    "casing_variant": "sentence_case",
                    "bucket_surfaces": buckets,
                    "prefix_before_candidate": prefix,
                    "prefix_before_candidate_sha256": sha256_text(prefix),
                    "prefix_is_empty": False,
                    "prefix_family": "restricted_step_label",
                    "prefix_shape_index": step_index - 1,
                    "prefix_boundary_tokenization_policy": PREFIX_BOUNDARY_TOKENIZATION_POLICY,
                    "source_detection_count": 1,
                    "source_response_count": 0,
                    "source_family_counts": {},
                    "source_response_ids": [],
                    "observed_surface_counts": observed_surface_counts,
                    "observed_case_variant_counts": {"sentence_case": len(buckets["0"]) + len(buckets["1"])},
                    "observed_surface_counts_at_step": {
                        surface: int(first_word_counts_by_step.get(str(step_index), {}).get(surface, 0))
                        for surface in [*buckets["0"], *buckets["1"]]
                    },
                    "context_index_within_bank_variant": step_index - 1,
                    "structural_selection": "restricted_step_label_mass_aware_recombined_candidate_not_scored",
                    "scoring_status": "PLANNED_NOT_SCORED",
                    "tokenizer_stability_status": "NOT_EVALUATED",
                    "mass_gate_status": "NOT_EVALUATED",
                    "template_preflight_only": False,
                    "artifact_only": True,
                    "model_scoring_started": False,
                    "training_started": False,
                    "generation_started": False,
                    "e2e_eval_started": False,
                    "wp4_allowed": False,
                    "paper_claim_allowed": False,
                    "not_payload_recovery": True,
                    "not_full_far": True,
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    score_dir = resolve_path(args.score_dir)
    density_dir = resolve_path(args.density_dir)
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output dir: {output_dir}")
    output_dir.mkdir(parents=True)

    config = read_yaml(config_path)
    summary = read_json(score_dir / "qwen_v2_wp3_context_mass_score_summary.json")
    audit = read_json(score_dir / "qwen_v2_wp3_context_mass_audit.json")
    artifact = read_json(score_dir / "qwen_v2_wp3_context_mass_artifact.json")
    context_scores = read_jsonl(score_dir / "qwen_v2_wp3_context_mass_context_scores.jsonl")
    invalid_rows = read_jsonl(score_dir / "qwen_v2_wp3_context_mass_invalid_tokenization_rows.jsonl")
    counts = observed_counts(density_dir=density_dir)

    if summary.get("mass_gate_status") != "FAIL":
        raise ValueError("expected source 850483 mass gate to fail before repair planning")
    if audit.get("mass_gate_status") != "FAIL":
        raise ValueError("expected source audit mass gate to fail before repair planning")

    original_reviews = summarize_original_banks(
        mass_rows=list(artifact.get("mass_rows", [])),
        invalid_rows=invalid_rows,
        context_scores=context_scores,
        min_bucket_mass=args.min_bucket_mass,
        max_mass_ratio=args.max_mass_ratio,
    )
    bucket_groups = bucket_groups_from_context_scores(
        context_scores=context_scores,
        min_bucket_mass=args.min_bucket_mass,
    )
    recombined_candidates = recombine_bucket_groups(
        groups=bucket_groups,
        max_mass_ratio=args.max_mass_ratio,
        max_candidates=args.max_recombined_candidates,
    )
    for candidate in recombined_candidates:
        validate_surface_text(
            config,
            [*candidate["bucket_0_surfaces"], *candidate["bucket_1_surfaces"]],
        )
    plan_id = "wp3_restricted_step_label_mass_aware_repair_plan_from_850483"
    rows = plan_rows(plan_id=plan_id, candidates=recombined_candidates, counts=counts)

    write_jsonl(output_dir / "original_bank_reviews.jsonl", original_reviews)
    write_jsonl(output_dir / "bucket_group_candidates.jsonl", bucket_groups)
    write_jsonl(output_dir / "mass_aware_recombined_bank_candidates.jsonl", recombined_candidates)
    write_jsonl(
        output_dir / "qwen_v2_wp3_restricted_step_label_mass_aware_context_mass_score_plan.jsonl",
        rows,
    )

    report = {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_mass_aware_repair_summary_v1",
        "status": "WP3_RESTRICTED_STEP_LABEL_MASS_AWARE_REPAIR_PLAN_READY_ARTIFACT_ONLY",
        "source_score_dir": str(score_dir),
        "source_density_dir": str(density_dir),
        "source_mass_gate_status": summary.get("mass_gate_status"),
        "source_invalid_tokenization_rows": int(summary.get("invalid_tokenization_rows", len(invalid_rows))),
        "source_context_score_rows": int(summary.get("context_score_rows", len(context_scores))),
        "min_bucket_mass_required": args.min_bucket_mass,
        "max_mass_ratio_required": args.max_mass_ratio,
        "original_bank_review_rows": len(original_reviews),
        "bucket_group_rows": len(bucket_groups),
        "eligible_bucket_group_rows": sum(bool(row["eligible_for_recombination"]) for row in bucket_groups),
        "recombined_candidate_rows": len(recombined_candidates),
        "score_plan_rows": len(rows),
        "score_plan_rows_by_bank": dict(Counter(str(row["candidate_bank_id"]) for row in rows)),
        "score_plan_jsonl": str(
            output_dir / "qwen_v2_wp3_restricted_step_label_mass_aware_context_mass_score_plan.jsonl"
        ),
        "candidate_jsonl": str(output_dir / "mass_aware_recombined_bank_candidates.jsonl"),
        "slurm_wrapper_candidate": "scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch",
        "allowlist_enabled": False,
        "next_allowed_action": (
            "Review this artifact-only repair plan. If approved, validate the score "
            "plan locally with the venv, then enable one allowlist entry and submit "
            "one Chimera Slurm context-mass scoring job in a fresh output dir."
        ),
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
    }
    write_json(output_dir / "mass_aware_repair_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
