from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "wp3_step_local_expansion_mass_score_850398/qwen_v2_wp3_context_mass_audit.json"
)

PASSING_BANK_IDS = (
    "step_local_step_label_seed_check_review_choose_make_v1",
    "step_local_step_label_start_begin_create_set_v1",
)

PREFIXES_16 = tuple(f"Step {index}: " for index in range(1, 17))

FORBIDDEN_PUBLIC_SURFACES = (
    "FIELD=",
    "SECTION=",
    "TOPIC=",
    "PAYLOAD",
    "CERT",
    "EVIDENCE",
    "CARRIER",
    "OWNER",
    "fingerprint",
    "watermark",
    "bucket",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only restricted WP3 step-label policy from the "
            "two passing Step N: banks in job 850398, plus density-audit design "
            "for 16-step vs 8-step-plus-extra routes. This does not score a "
            "model, train, generate, run E2E, FAR, or claims."
        )
    )
    parser.add_argument("--mass-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top level must be a mapping: {path}")
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def forbidden_hits(text: str) -> list[str]:
    upper = text.upper()
    return [term for term in FORBIDDEN_PUBLIC_SURFACES if term.upper() in upper]


def validate_public_texts(values: Iterable[str]) -> None:
    for value in values:
        hits = forbidden_hits(str(value))
        if hits:
            raise ValueError(f"forbidden public surface {hits}: {value!r}")


def passing_rows(mass_audit: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in mass_audit.get("bank_variant_results", []):
        if not isinstance(row, Mapping):
            continue
        bank_id = str(row.get("candidate_bank_id", ""))
        if bank_id in PASSING_BANK_IDS:
            if not bool(row.get("passed", False)):
                raise ValueError(f"expected passing bank did not pass: {bank_id}")
            output[bank_id] = row
    missing = sorted(set(PASSING_BANK_IDS) - set(output))
    if missing:
        raise ValueError(f"missing expected passing banks: {missing}")
    return output


def bank_specs(passing: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        {
            "candidate_bank_id": "restricted_step_label_check_review_choose_make_v1",
            "source_candidate_bank_id": "step_local_step_label_seed_check_review_choose_make_v1",
            "slot_type": "bullet_or_step_opener",
            "anchor_kind": "step_label_boundary",
            "prefix_family": "step_label",
            "allowed_prefixes": list(PREFIXES_16),
            "buckets": {"0": ["Check", "Review"], "1": ["Choose", "Make"]},
        },
        {
            "candidate_bank_id": "restricted_step_label_start_begin_create_set_v1",
            "source_candidate_bank_id": "step_local_step_label_start_begin_create_set_v1",
            "slot_type": "bullet_or_step_opener",
            "anchor_kind": "step_label_boundary",
            "prefix_family": "step_label",
            "allowed_prefixes": list(PREFIXES_16),
            "buckets": {"0": ["Start", "Begin"], "1": ["Create", "Set"]},
        },
    ]
    for spec in specs:
        validate_public_texts(spec["allowed_prefixes"])
        validate_public_texts(spec["buckets"]["0"] + spec["buckets"]["1"])
        source = passing[str(spec["source_candidate_bank_id"])]
        spec["source_mass_gate"] = {
            "job_id": "850398",
            "min_bucket_mass": float(source["min_bucket_mass"]),
            "max_bucket_mass_ratio": float(source["max_bucket_mass_ratio"]),
            "context_count": int(source["context_count"]),
            "passed": bool(source["passed"]),
        }
    return specs


def detector_contract() -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_detector_contract_v1",
        "status": "RESTRICTED_STEP_LABEL_DETECTOR_ARTIFACT_ONLY_NOT_MODEL_OUTPUT",
        "detector_id": "qwen_v2_wp3_restricted_step_label_detector_v1",
        "slot_policy_id": "qwen_v2_wp3_restricted_step_label_policy_v1",
        "observable_unit": "response_text",
        "anchor_rule": "match line-start or sentence-start Step <integer>: followed by one sentence-case action verb candidate",
        "allowed_prefix_regex": r"Step (?:[1-9]|1[0-6]):\\s+",
        "allowed_step_indices": list(range(1, 17)),
        "slot_order": "increasing_step_number",
        "post_hoc_slot_selection_allowed": False,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def density_design() -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_density_design_v1",
        "status": "DENSITY_DESIGN_REVIEWED_ARTIFACT_ONLY_NOT_MODEL_OUTPUT",
        "configured_average_micro_slots_gate": 16,
        "options": [
            {
                "option_id": "A_16_step_checklist_step_label_only",
                "description": "Sixteen natural checklist steps, each beginning with Step N: and one eligible sentence-case action opener.",
                "expected_step_label_slots_per_response": 16,
                "additional_slot_types_required": [],
                "uses_only_passing_mass_banks": True,
                "structural_density_gate_status": "PASS_STRUCTURAL_FEASIBILITY",
                "risks": [
                    "longer owner probes may be less natural for some tasks",
                    "model-output adherence to exactly sixteen Step N labels must be audited",
                ],
                "recommended_next": True,
            },
            {
                "option_id": "B_8_step_plus_extra_slots",
                "description": "Eight checklist steps plus at least eight additional non-step micro-slots.",
                "expected_step_label_slots_per_response": 8,
                "additional_slot_types_required": [
                    "transition_word",
                    "hedge",
                    "function_word_or_discourse_marker",
                ],
                "uses_only_passing_mass_banks": False,
                "structural_density_gate_status": "BLOCKED_NEEDS_EXTRA_MASS_VALIDATED_BANKS",
                "risks": [
                    "current non-step banks failed full-vocabulary mass gate",
                    "adding unvalidated extra slots would break the WP3 gate discipline",
                ],
                "recommended_next": False,
            },
        ],
        "decision": (
            "Use option A as the immediate restricted-policy density route. "
            "Keep option B blocked until at least one non-step slot family passes "
            "the base-Qwen mass gate."
        ),
        "model_output_density_audit_required_before_wp4": True,
        "wp4_allowed": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def prompt_templates() -> list[dict[str, Any]]:
    templates = [
        {
            "template_id": "restricted_16_step_safe_hike_checklist",
            "family_id": "F2_16_step_checklist_step_label",
            "prompt_text": (
                "Give a sixteen-step practical checklist for planning a safe weekend hike. "
                "Use the format Step 1: through Step 16:. Each step should be one natural sentence."
            ),
        },
        {
            "template_id": "restricted_16_step_project_launch_checklist",
            "family_id": "F2_16_step_checklist_step_label",
            "prompt_text": (
                "Give a sixteen-step practical checklist for launching a small team project. "
                "Use the format Step 1: through Step 16:. Each step should be one natural sentence."
            ),
        },
        {
            "template_id": "restricted_8_step_plus_extra_blocked",
            "family_id": "F2_8_step_plus_extra_candidate",
            "prompt_text": (
                "Give an eight-step checklist and include short transition phrases in each step."
            ),
            "blocked_reason": "extra non-step slots do not yet have passing mass banks",
        },
    ]
    for template in templates:
        validate_public_texts([str(template["prompt_text"])])
    return templates


def policy_payload(*, mass_audit_path: Path, passing: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_policy_v1",
        "status": "RESTRICTED_STEP_LABEL_POLICY_WRITTEN_ARTIFACT_ONLY",
        "source_mass_audit": str(mass_audit_path),
        "source_job_id": "850398",
        "policy_id": "qwen_v2_wp3_restricted_step_label_policy_v1",
        "candidate_banks": bank_specs(passing),
        "detector_contract": detector_contract(),
        "density_design": density_design(),
        "prompt_templates": prompt_templates(),
        "claim_control": {
            "wp4_allowed": False,
            "training_allowed": False,
            "generation_allowed": False,
            "qwen_e2e_allowed": False,
            "llama_allowed": False,
            "paper_claim_allowed": False,
        },
        "next_allowed_action": (
            "Review this restricted policy, then prepare a model-output density "
            "audit plan for the recommended 16-step checklist route. Any model "
            "generation/scoring must go through explicit review and Chimera Slurm."
        ),
    }


def score_plan_rows(*, plan_id: str, policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bank in policy["candidate_banks"]:
        bank_id = str(bank["candidate_bank_id"])
        for prefix in bank["allowed_prefixes"]:
            row_payload = {
                "plan_id": plan_id,
                "candidate_bank_id": bank_id,
                "prefix_before_candidate": prefix,
            }
            row_id = sha256_text(json.dumps(row_payload, sort_keys=True, separators=(",", ":")))[:20]
            rows.append(
                {
                    "schema_name": "natural_evidence_v2_wp3_context_mass_score_plan_row_v1",
                    "plan_id": plan_id,
                    "plan_row_id": row_id,
                    "candidate_bank_id": bank_id,
                    "source_bank_id": str(bank["source_candidate_bank_id"]),
                    "slot_type": str(bank["slot_type"]),
                    "anchor_kind": str(bank["anchor_kind"]),
                    "bucket_policy_id": "qwen_v2_wp3_restricted_step_label_two_way",
                    "casing_variant": "sentence_case",
                    "bucket_surfaces": dict(bank["buckets"]),
                    "prefix_before_candidate": prefix,
                    "prefix_before_candidate_sha256": sha256_text(prefix),
                    "prefix_is_empty": False,
                    "source_detection_count": 1,
                    "source_response_count": 0,
                    "source_family_counts": {},
                    "observed_surface_counts": {},
                    "observed_case_variant_counts": {},
                    "template_preflight_only": False,
                    "structural_selection": "restricted_step_label_policy_not_model_output_density",
                    "artifact_only": True,
                    "not_qwen_gate": True,
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
            "# WP3 Restricted Step-Label Policy",
            "",
            "Artifact-only restricted policy built from the two passing Step N banks in job 850398.",
            "",
            f"status: `{summary['status']}`",
            f"candidate_bank_count: `{summary['candidate_bank_count']}`",
            f"recommended_density_route: `{summary['recommended_density_route']}`",
            "",
            "This is not model-output density, not training, not E2E, not FAR, and not a paper claim.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    mass_audit_path = resolve_path(args.mass_audit)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    audit = read_json(mass_audit_path)
    passing = passing_rows(audit)
    policy = policy_payload(mass_audit_path=mass_audit_path, passing=passing)
    plan_id = output_dir.name
    rows = score_plan_rows(plan_id=plan_id, policy=policy)
    summary = {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_policy_summary_v1",
        "status": "WP3_RESTRICTED_STEP_LABEL_POLICY_WRITTEN_ARTIFACT_ONLY",
        "policy_json": str(output_dir / "restricted_step_label_policy.json"),
        "bucket_bank_json": str(output_dir / "restricted_step_label_bucket_bank.json"),
        "detector_contract_json": str(output_dir / "restricted_step_label_detector_contract.json"),
        "density_design_json": str(output_dir / "restricted_step_label_density_design.json"),
        "prompt_templates_jsonl": str(output_dir / "restricted_step_label_prompt_templates.jsonl"),
        "context_mass_score_plan_jsonl": str(output_dir / "restricted_step_label_context_mass_score_plan.jsonl"),
        "candidate_bank_count": len(policy["candidate_banks"]),
        "context_mass_score_plan_rows": len(rows),
        "recommended_density_route": "A_16_step_checklist_step_label_only",
        "blocked_density_route": "B_8_step_plus_extra_slots",
        "wp4_allowed": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Review restricted policy and density design. Then prepare model-output "
            "density audit for the 16-step checklist route only if explicitly allowed."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "restricted_step_label_policy.json", policy)
    write_json(
        output_dir / "restricted_step_label_bucket_bank.json",
        {
            "schema_name": "natural_evidence_v2_wp3_restricted_step_label_bucket_bank_v1",
            "status": "RESTRICTED_STEP_LABEL_BUCKET_BANK_WRITTEN_ARTIFACT_ONLY",
            "candidate_banks": policy["candidate_banks"],
            "wp4_allowed": False,
            "training_started": False,
            "generation_started": False,
            "e2e_eval_started": False,
            "paper_claim_allowed": False,
        },
    )
    write_json(output_dir / "restricted_step_label_detector_contract.json", policy["detector_contract"])
    write_json(output_dir / "restricted_step_label_density_design.json", policy["density_design"])
    write_jsonl(output_dir / "restricted_step_label_prompt_templates.jsonl", policy["prompt_templates"])
    write_jsonl(output_dir / "restricted_step_label_context_mass_score_plan.jsonl", rows)
    write_json(output_dir / "restricted_step_label_policy_summary.json", summary)
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "candidate_bank_count": summary["candidate_bank_count"],
                "recommended_density_route": summary["recommended_density_route"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
