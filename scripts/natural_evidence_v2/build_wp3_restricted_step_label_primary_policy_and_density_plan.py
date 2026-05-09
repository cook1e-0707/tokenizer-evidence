from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.run_wp3_restricted_step_label_density_audit import (
    forbidden_terms_in_text,
    read_yaml,
)


DEFAULT_SCORE_DIR = ROOT / "results/natural_evidence_v2/status/wp3_restricted_step_label_mass_aware_score_850509"
DEFAULT_REPAIR_DIR = (
    ROOT / "results/natural_evidence_v2/status/wp3_restricted_step_label_repair_plan_20260508_2134"
)
DEFAULT_SOURCE_POLICY_DIR = (
    ROOT / "results/natural_evidence_v2/status/wp3_restricted_step_label_policy_20260508_2049"
)
DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml"

PRIMARY_BANK_ID = "step_label_recombined_create_develop_vs_choose_make_v1"
BACKUP_BANK_IDS = (
    "step_label_recombined_choose_make_vs_determine_define_v1",
    "step_label_recombined_create_develop_vs_determine_define_v1",
    "step_label_recombined_check_review_vs_identify_assess_v1",
    "step_label_recombined_determine_define_vs_check_review_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a primary restricted Step-label bank from the 850509 passing "
            "mass-aware score artifacts and build a strict density repair audit "
            "plan. This is artifact-only and does not submit Slurm, train, "
            "generate text, run E2E, aggregate FAR, or make positive claims."
        )
    )
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--repair-dir", type=Path, default=DEFAULT_REPAIR_DIR)
    parser.add_argument("--source-policy-dir", type=Path, default=DEFAULT_SOURCE_POLICY_DIR)
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


def validate_surfaces(config: Mapping[str, Any], surfaces: Iterable[str]) -> None:
    for surface in surfaces:
        if not surface or not surface[:1].isupper():
            raise ValueError(f"surface must be sentence case: {surface!r}")
        hits = forbidden_terms_in_text(config, surface)
        if hits:
            raise ValueError(f"surface contains forbidden public marker {hits}: {surface!r}")


def bank_from_mass_row(row: Mapping[str, Any], *, surfaces_by_bank: Mapping[str, Mapping[str, list[str]]]) -> dict[str, Any]:
    bank_id = str(row["candidate_bank_id"])
    bucket_surfaces = dict(surfaces_by_bank[bank_id])
    return {
        "candidate_bank_id": bank_id,
        "slot_type": "bullet_or_step_opener",
        "anchor_kind": "step_label_boundary",
        "prefix_family": "step_label",
        "allowed_prefixes": [f"Step {index}: " for index in range(1, 17)],
        "buckets": {
            "0": [str(item) for item in bucket_surfaces["0"]],
            "1": [str(item) for item in bucket_surfaces["1"]],
        },
        "source_candidate_bank_id": bank_id,
        "source_mass_gate": {
            "job_id": "850509",
            "context_count": int(row["context_count"]),
            "min_bucket_mass": float(row["full_vocab_min_bucket_mass"]),
            "max_bucket_mass_ratio": float(row["full_vocab_mass_ratio"]),
            "passed": True,
        },
    }


def rank_mass_rows(mass_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in mass_rows]
    rows.sort(
        key=lambda row: (
            str(row["candidate_bank_id"]) != PRIMARY_BANK_ID,
            float(row["full_vocab_mass_ratio"]),
            -float(row["full_vocab_min_bucket_mass"]),
            str(row["candidate_bank_id"]),
        )
    )
    return rows


def main() -> int:
    args = parse_args()
    score_dir = resolve_path(args.score_dir)
    repair_dir = resolve_path(args.repair_dir)
    source_policy_dir = resolve_path(args.source_policy_dir)
    config = read_yaml(resolve_path(args.config))
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output dir: {output_dir}")
    output_dir.mkdir(parents=True)

    score_summary = read_json(score_dir / "qwen_v2_wp3_context_mass_score_summary.json")
    score_audit = read_json(score_dir / "qwen_v2_wp3_context_mass_audit.json")
    score_artifact = read_json(score_dir / "qwen_v2_wp3_context_mass_artifact.json")
    context_scores = read_jsonl(score_dir / "qwen_v2_wp3_context_mass_context_scores.jsonl")
    source_detector = read_json(source_policy_dir / "restricted_step_label_detector_contract.json")
    strict_prompts = read_jsonl(repair_dir / "restricted_step_label_strict_repair_prompts.jsonl")

    if score_summary.get("mass_gate_status") != "PASS_REVIEW_REQUIRED":
        raise ValueError("850509 mass score must pass review-required status before policy selection")
    if score_audit.get("invalid_tokenization_rows") != 0:
        raise ValueError("policy selection requires zero invalid tokenizer rows")
    mass_rows = list(score_artifact.get("mass_rows", []))
    mass_by_id = {str(row["candidate_bank_id"]): row for row in mass_rows}
    surfaces_by_bank: dict[str, Mapping[str, list[str]]] = {}
    for row in context_scores:
        bank_id = str(row["candidate_bank_id"])
        if bank_id not in surfaces_by_bank:
            surfaces_by_bank[bank_id] = {
                str(bucket_id): [str(item) for item in members]
                for bucket_id, members in dict(row["bucket_surfaces"]).items()
            }
    if PRIMARY_BANK_ID not in mass_by_id:
        raise ValueError(f"missing primary bank in scored rows: {PRIMARY_BANK_ID}")

    selected_primary = bank_from_mass_row(mass_by_id[PRIMARY_BANK_ID], surfaces_by_bank=surfaces_by_bank)
    backup_banks = [
        bank_from_mass_row(mass_by_id[bank_id], surfaces_by_bank=surfaces_by_bank)
        for bank_id in BACKUP_BANK_IDS
        if bank_id in mass_by_id
    ]
    validate_surfaces(
        config,
        [
            surface
            for bank in [selected_primary, *backup_banks]
            for members in bank["buckets"].values()
            for surface in members
        ],
    )

    detector_contract = dict(source_detector)
    detector_contract.update(
        {
            "detector_id": "qwen_v2_wp3_restricted_step_label_mass_aware_detector_v1",
            "slot_policy_id": "qwen_v2_wp3_restricted_step_label_mass_aware_primary_policy_v1",
            "source_detector_id": source_detector.get("detector_id"),
            "source_mass_score_job_id": "850509",
            "model_scoring_started": False,
            "training_started": False,
            "generation_started": False,
            "e2e_eval_started": False,
            "paper_claim_allowed": False,
        }
    )
    density_design = {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_primary_density_design_v1",
        "status": "STRICT_DENSITY_REPAIR_AUDIT_PLAN_READY_ARTIFACT_ONLY",
        "selected_route": "A_strict_16_step_checklist_step_label_only",
        "source_density_audit_job_id": "850434",
        "source_density_gate_status": "FAIL_RESTRICTED_STEP_LABEL_MODEL_OUTPUT_DENSITY_STRUCTURAL_GATE",
        "source_complete_step_label_response_rate": 0.98828125,
        "source_mean_detected_structural_slots_per_response": 15.8125,
        "repair_prompt_source": str(repair_dir / "restricted_step_label_strict_repair_prompts.jsonl"),
        "strict_prompt_count": len(strict_prompts),
        "configured_average_micro_slots_gate": 16,
        "model_output_density_audit_required_before_wp4": True,
        "decision": (
            "Use the strict repair prompts to re-audit model-output density before WP4. "
            "The 850509 mass subgate passes, but the previous density gate is still a blocker."
        ),
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
    }
    bucket_bank = {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_bucket_bank_v1",
        "status": "MASS_AWARE_PRIMARY_RESTRICTED_STEP_LABEL_BUCKET_BANK_WRITTEN_ARTIFACT_ONLY",
        "candidate_banks": [selected_primary],
        "backup_candidate_banks": backup_banks,
        "selection_rule": "primary bank has strongest min mass and near-unity mass ratio after 850509 scoring",
        "source_mass_score_job_id": "850509",
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
    }
    policy = {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_policy_v1",
        "status": "MASS_AWARE_PRIMARY_RESTRICTED_STEP_LABEL_POLICY_WRITTEN_ARTIFACT_ONLY",
        "policy_id": "qwen_v2_wp3_restricted_step_label_mass_aware_primary_policy_v1",
        "source_mass_score_job_id": "850509",
        "source_mass_audit": str(score_dir / "qwen_v2_wp3_context_mass_audit.json"),
        "candidate_banks": [selected_primary],
        "backup_candidate_banks": backup_banks,
        "detector_contract": detector_contract,
        "density_design": density_design,
        "claim_control": {
            "training_allowed": False,
            "generation_allowed": False,
            "wp4_allowed": False,
            "qwen_e2e_allowed": False,
            "llama_allowed": False,
            "paper_claim_allowed": False,
        },
        "next_allowed_action": (
            "Review this primary policy and strict density plan. If explicitly "
            "approved, submit one Chimera Slurm restricted Step-label density "
            "audit using the strict repair prompts and this policy dir."
        ),
    }
    slurm_review = {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_strict_density_slurm_review_v1",
        "status": "STRICT_DENSITY_AUDIT_SLURM_PLAN_READY_NOT_SUBMITTED",
        "compatible_wrapper": "scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch",
        "prompts_jsonl": str(output_dir / "restricted_step_label_strict_density_audit_prompts.jsonl"),
        "policy_dir": str(output_dir),
        "command_pattern": (
            "PROMPTS_JSONL=<this prompts_jsonl> POLICY_DIR=<this output dir> "
            "OUTPUT_DIR=<fresh scratch output> sbatch "
            "scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch"
        ),
        "allowlist_entry": "v2_wp3_restricted_step_label_density_audit",
        "allowlist_enabled": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
    }
    summary = {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_primary_policy_density_plan_summary_v1",
        "status": "PRIMARY_BANK_AND_STRICT_DENSITY_PLAN_READY_ARTIFACT_ONLY",
        "source_mass_score_job_id": "850509",
        "source_mass_gate_status": score_summary.get("mass_gate_status"),
        "selected_primary_bank_id": PRIMARY_BANK_ID,
        "selected_primary_bank": selected_primary,
        "backup_bank_ids": [bank["candidate_bank_id"] for bank in backup_banks],
        "strict_prompt_count": len(strict_prompts),
        "policy_dir": str(output_dir),
        "prompts_jsonl": str(output_dir / "restricted_step_label_strict_density_audit_prompts.jsonl"),
        "slurm_review_json": str(output_dir / "restricted_step_label_strict_density_audit_slurm_review.json"),
        "next_allowed_action": policy["next_allowed_action"],
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
    }

    write_json(output_dir / "restricted_step_label_policy.json", policy)
    write_json(output_dir / "restricted_step_label_bucket_bank.json", bucket_bank)
    write_json(output_dir / "restricted_step_label_detector_contract.json", detector_contract)
    write_json(output_dir / "restricted_step_label_density_design.json", density_design)
    write_json(output_dir / "restricted_step_label_primary_policy_summary.json", summary)
    write_json(output_dir / "restricted_step_label_strict_density_audit_slurm_review.json", slurm_review)
    shutil.copyfile(
        repair_dir / "restricted_step_label_strict_repair_prompts.jsonl",
        output_dir / "restricted_step_label_strict_density_audit_prompts.jsonl",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
