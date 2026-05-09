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


DEFAULT_REPAIR_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/wp3_restricted_step_label_repair_plan_20260508_2134"
)
DEFAULT_AUDIT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/wp3_restricted_step_label_density_audit_850434"
)
DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml"
PREFIX_BOUNDARY_TOKENIZATION_POLICY = "score_longest_common_token_prefix_when_candidate_retokenizes_boundary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only Slurm-scoring plan for expanded restricted "
            "Step-label action-verb banks. This writes score-plan rows consumable "
            "by score_wp3_context_mass.py. It does not load a tokenizer/model, "
            "submit Slurm, train, run E2E, aggregate FAR, or make claims."
        )
    )
    parser.add_argument("--repair-dir", type=Path, default=DEFAULT_REPAIR_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def validate_surfaces(config: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> None:
    for bank in candidates:
        for key in ("bucket_0_surfaces", "bucket_1_surfaces"):
            members = bank.get(key)
            if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
                raise ValueError(f"{bank.get('candidate_bank_id')}: {key} must be a sequence")
            for surface in members:
                surface_text = str(surface)
                if not surface_text or not surface_text[:1].isupper():
                    raise ValueError(
                        f"{bank.get('candidate_bank_id')}: surface must be sentence case: {surface_text!r}"
                    )
                hits = forbidden_terms_in_text(config, surface_text)
                if hits:
                    raise ValueError(
                        f"{bank.get('candidate_bank_id')}: surface contains forbidden marker {hits}: "
                        f"{surface_text!r}"
                    )


def observed_counts_by_bank(
    *,
    candidates: Sequence[Mapping[str, Any]],
    detected_slots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first_word_by_step = defaultdict(Counter)
    overall_first_words = Counter()
    for row in detected_slots:
        first_word = str(row.get("first_word", ""))
        step_index = int(row.get("step_index", 0))
        overall_first_words[first_word] += 1
        first_word_by_step[step_index][first_word] += 1

    output: dict[str, Any] = {}
    for bank in candidates:
        bank_id = str(bank["candidate_bank_id"])
        surfaces = [str(item) for item in bank["bucket_0_surfaces"]] + [
            str(item) for item in bank["bucket_1_surfaces"]
        ]
        output[bank_id] = {
            "candidate_surface_observed_counts": {
                surface: int(overall_first_words.get(surface, 0)) for surface in surfaces
            },
            "candidate_surface_observed_counts_by_step": {
                str(step): {surface: int(counter.get(surface, 0)) for surface in surfaces}
                for step, counter in sorted(first_word_by_step.items())
            },
        }
    return output


def plan_rows(
    *,
    plan_id: str,
    candidates: Sequence[Mapping[str, Any]],
    observed_counts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bank in candidates:
        bank_id = str(bank["candidate_bank_id"])
        buckets = {
            "0": [str(item) for item in bank["bucket_0_surfaces"]],
            "1": [str(item) for item in bank["bucket_1_surfaces"]],
        }
        bank_observed = dict(observed_counts.get(bank_id, {}))
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
                    "bucket_policy_id": "qwen_v2_wp3_restricted_step_label_expanded_sentence_case_two_way",
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
                    "observed_surface_counts": bank_observed.get("candidate_surface_observed_counts", {}),
                    "observed_case_variant_counts": {"sentence_case": sum(buckets and [len(buckets["0"]) + len(buckets["1"])])},
                    "observed_surface_counts_at_step": bank_observed.get(
                        "candidate_surface_observed_counts_by_step", {}
                    ).get(str(step_index), {}),
                    "context_index_within_bank_variant": step_index - 1,
                    "structural_selection": "restricted_step_label_expanded_bank_candidate_not_scored",
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


def summarize(
    *,
    output_dir: Path,
    repair_dir: Path,
    audit_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    observed_counts: Mapping[str, Any],
) -> dict[str, Any]:
    rows_by_bank = Counter(str(row["candidate_bank_id"]) for row in rows)
    return {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_expanded_mass_plan_summary_v1",
        "status": "WP3_RESTRICTED_STEP_LABEL_EXPANDED_CONTEXT_MASS_SCORE_PLAN_READY_ARTIFACT_ONLY",
        "repair_dir": str(repair_dir),
        "source_audit_dir": str(audit_dir),
        "expanded_bank_candidate_count": len(candidates),
        "score_plan_rows": len(rows),
        "score_plan_rows_by_bank": dict(sorted(rows_by_bank.items())),
        "prefixes": [f"Step {index}: " for index in range(1, 17)],
        "casing_variant": "sentence_case",
        "score_plan_jsonl": str(
            output_dir / "qwen_v2_wp3_restricted_step_label_expanded_context_mass_score_plan.jsonl"
        ),
        "slurm_review_json": str(
            output_dir / "qwen_v2_wp3_restricted_step_label_expanded_context_mass_slurm_review.json"
        ),
        "observed_counts_by_bank": observed_counts,
        "compatible_scoring_wrapper": "scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch",
        "compatible_scoring_command_pattern": (
            "SCORE_PLAN=<this score_plan_jsonl> OUTPUT_DIR=<fresh scratch output> "
            "sbatch scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch"
        ),
        "model_generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "next_allowed_action": (
            "Review this Slurm-only score plan. If approved, sync it to Chimera, "
            "temporarily enable one allowlist entry, and submit exactly one "
            "wp3_context_mass_score.sbatch job with SCORE_PLAN pointing to this "
            "artifact. Do not start WP4 or training."
        ),
    }


def slurm_review_payload(*, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_expanded_context_mass_slurm_review_v1",
        "status": "SLURM_SCORE_PLAN_READY_SUBMISSION_NOT_STARTED",
        "wrapper": summary["compatible_scoring_wrapper"],
        "score_plan_jsonl": summary["score_plan_jsonl"],
        "expected_score_plan_rows": summary["score_plan_rows"],
        "required_python": "use configured virtual environment, not system python",
        "chimera_default_python": "/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3",
        "required_partition_policy": "DGXA100/A100 unless user explicitly approves another available partition",
        "submission_started": False,
        "allowlist_entry_enabled": False,
        "training_allowed": False,
        "generation_allowed": False,
        "e2e_allowed": False,
        "paper_claim_allowed": False,
        "forbidden_actions": [
            "no training",
            "no model-output generation",
            "no protected E2E",
            "no Llama",
            "no same-family null",
            "no sanitizer benchmark",
            "no FAR aggregation",
            "no positive paper claim",
        ],
    }


def readme_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WP3 Restricted Step-Label Expanded Context-Mass Score Plan",
            "",
            "Artifact-only Slurm score plan for expanded action-verb banks.",
            "",
            f"status: `{summary['status']}`",
            f"score_plan_rows: `{summary['score_plan_rows']}`",
            f"expanded_bank_candidate_count: `{summary['expanded_bank_candidate_count']}`",
            "",
            "This did not load a tokenizer/model, submit Slurm, train, run E2E, aggregate FAR, or make claims.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    repair_dir = resolve_path(args.repair_dir)
    audit_dir = resolve_path(args.audit_dir)
    config = read_yaml(resolve_path(args.config))
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")

    candidates = read_jsonl(repair_dir / "restricted_step_label_expanded_bank_candidates.jsonl")
    detected_slots = read_jsonl(audit_dir / "restricted_step_label_detected_slots.jsonl")
    validate_surfaces(config, candidates)
    observed_counts = observed_counts_by_bank(candidates=candidates, detected_slots=detected_slots)
    plan_id = output_dir.name
    rows = plan_rows(plan_id=plan_id, candidates=candidates, observed_counts=observed_counts)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(
        output_dir=output_dir,
        repair_dir=repair_dir,
        audit_dir=audit_dir,
        candidates=candidates,
        rows=rows,
        observed_counts=observed_counts,
    )
    write_jsonl(
        output_dir / "qwen_v2_wp3_restricted_step_label_expanded_context_mass_score_plan.jsonl",
        rows,
    )
    write_json(
        output_dir / "qwen_v2_wp3_restricted_step_label_expanded_context_mass_slurm_review.json",
        slurm_review_payload(summary=summary),
    )
    write_json(
        output_dir / "qwen_v2_wp3_restricted_step_label_expanded_context_mass_score_plan_summary.json",
        summary,
    )
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(json.dumps({"status": summary["status"], "score_plan_rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
