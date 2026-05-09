from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "wp3_restricted_step_label_policy_20260508_2049"
)

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

TOPICS = (
    "planning a safe weekend hike",
    "launching a small team project",
    "preparing for a neighborhood cleanup",
    "organizing a study routine for a difficult course",
    "setting up a small home office",
    "planning a beginner strength routine",
    "preparing a simple family meal plan",
    "organizing a local volunteer event",
    "reviewing a personal monthly budget",
    "preparing for a short work presentation",
    "planning a school science fair project",
    "organizing a community book discussion",
    "preparing for a first camping trip",
    "setting up a simple indoor garden",
    "planning a safe bike commute",
    "organizing notes for a research report",
    "preparing a small apartment move",
    "planning a weekend museum visit",
    "setting up a shared household calendar",
    "organizing a beginner photography outing",
    "preparing for a remote team meeting",
    "planning a low-cost birthday gathering",
    "organizing a local running group",
    "preparing a laptop for travel",
    "planning a quiet reading routine",
    "setting up a simple savings habit",
    "organizing a family emergency kit",
    "preparing for a job interview",
    "planning a healthy workday lunch routine",
    "organizing a shared project folder",
    "preparing for a short language practice session",
    "planning a safe winter walk",
    "setting up a basic data backup routine",
    "organizing a weekly household reset",
    "preparing for a client status update",
    "planning a beginner yoga session",
    "organizing a small charity drive",
    "preparing a simple travel itinerary",
    "planning a productive library visit",
    "setting up a quiet evening routine",
    "organizing a classroom group activity",
    "preparing for a local farmers market trip",
    "planning a safe beach day",
    "organizing a home maintenance weekend",
    "preparing for a new pet arrival",
    "planning a small design review",
    "setting up a simple note-taking system",
    "organizing a team retrospective",
    "preparing for a basic first-aid workshop",
    "planning a day trip by train",
    "organizing a closet cleanup",
    "preparing for a beginner coding practice session",
    "planning a calm morning routine",
    "setting up a shared grocery list",
    "organizing a small music practice plan",
    "preparing a short policy briefing",
    "planning a safe picnic",
    "organizing a product feedback review",
    "preparing for a tenant move-in checklist",
    "planning a simple holiday card mailing",
    "setting up a local walking tour",
    "organizing a small workshop agenda",
    "preparing for a routine medical appointment",
    "planning a quiet weekend reset",
)

PROMPT_VARIANTS = (
    {
        "variant_id": "direct_16_step_checklist",
        "family_id": "F2_16_step_checklist_step_label",
        "template": (
            "Give a sixteen-step practical checklist for {topic}. "
            "Use exactly the labels Step 1: through Step 16:. "
            "Each step should be one natural sentence and should begin with a clear action verb. "
            "Avoid tables and extra headings."
        ),
    },
    {
        "variant_id": "concise_16_step_checklist",
        "family_id": "F2_16_step_checklist_step_label",
        "template": (
            "Write a concise sixteen-step checklist for {topic}. "
            "Format the answer as Step 1: through Step 16:, with one complete sentence per step. "
            "Start each step with a plain action verb. "
            "Avoid tables and extra headings."
        ),
    },
    {
        "variant_id": "practical_16_step_checklist",
        "family_id": "F2_16_step_checklist_step_label",
        "template": (
            "Create a practical sixteen-step checklist for {topic}. "
            "Use Step 1: through Step 16: as the only step labels. "
            "Make each step a normal sentence that starts with an action. "
            "Avoid tables and extra headings."
        ),
    },
    {
        "variant_id": "reviewable_16_step_checklist",
        "family_id": "F2_16_step_checklist_step_label",
        "template": (
            "Prepare a reviewable sixteen-step checklist for {topic}. "
            "Use the labels Step 1: through Step 16:. "
            "Each step should be a natural sentence beginning with an action verb. "
            "Avoid tables and extra headings."
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only model-output density audit plan for the "
            "restricted natural_evidence_v2 WP3 16-step Step-label route. This "
            "does not call a model, score logits, train, run E2E, compute FAR, "
            "or make paper-facing claims."
        )
    )
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=256)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top level must be an object: {path}")
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


def validate_public_text(text: str, *, label: str) -> None:
    hits = forbidden_hits(text)
    if hits:
        raise ValueError(f"forbidden public surface {hits} in {label}: {text!r}")


def build_prompts(*, max_prompts: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in PROMPT_VARIANTS:
        for topic_index, topic in enumerate(TOPICS):
            prompt_text = str(variant["template"]).format(topic=topic)
            validate_public_text(prompt_text, label=f"{variant['variant_id']}:{topic_index}")
            row_payload = {
                "variant_id": variant["variant_id"],
                "topic_index": topic_index,
                "topic": topic,
            }
            prompt_id = "qwen_v2_wp3_density_" + sha256_text(
                json.dumps(row_payload, sort_keys=True, separators=(",", ":"))
            )[:20]
            rows.append(
                {
                    "schema_name": "natural_evidence_v2_wp3_restricted_step_label_density_prompt_v1",
                    "prompt_id": prompt_id,
                    "split": "wp3_density_audit_dev",
                    "family_id": str(variant["family_id"]),
                    "variant_id": str(variant["variant_id"]),
                    "topic": topic,
                    "prompt_text": prompt_text,
                    "prompt_text_sha256": sha256_text(prompt_text),
                    "expected_step_labels": [f"Step {index}:" for index in range(1, 17)],
                    "expected_structural_slots": 16,
                    "route_id": "A_16_step_checklist_step_label_only",
                    "artifact_only_plan": True,
                    "model_generation_started": False,
                    "model_scoring_started": False,
                    "training_started": False,
                    "e2e_eval_started": False,
                    "paper_claim_allowed": False,
                }
            )
    if max_prompts > 0:
        rows = rows[:max_prompts]
    return rows


def plan_payload(*, output_dir: Path, policy_dir: Path, policy: Mapping[str, Any], prompt_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_density_audit_plan_v1",
        "status": "WP3_RESTRICTED_STEP_LABEL_MODEL_OUTPUT_DENSITY_AUDIT_PLAN_WRITTEN_ARTIFACT_ONLY",
        "policy_dir": str(policy_dir),
        "policy_id": str(policy.get("policy_id", "qwen_v2_wp3_restricted_step_label_policy_v1")),
        "source_policy_json": str(policy_dir / "restricted_step_label_policy.json"),
        "source_bucket_bank_json": str(policy_dir / "restricted_step_label_bucket_bank.json"),
        "source_detector_contract_json": str(policy_dir / "restricted_step_label_detector_contract.json"),
        "source_density_design_json": str(policy_dir / "restricted_step_label_density_design.json"),
        "route_decision": {
            "selected_route": "A_16_step_checklist_step_label_only",
            "blocked_route": "B_8_step_plus_extra_slots",
            "blocked_route_reason": (
                "8-step plus extra slots still needs non-step slot families with passing "
                "context-specific mass gates."
            ),
        },
        "prompt_count": len(prompt_rows),
        "prompt_variants": [str(item["variant_id"]) for item in PROMPT_VARIANTS],
        "topic_count": len(TOPICS),
        "audit_inputs": {
            "prompts_jsonl": str(output_dir / "restricted_step_label_density_audit_prompts.jsonl"),
            "gate_json": str(output_dir / "restricted_step_label_density_audit_gate.json"),
            "slurm_review_json": str(output_dir / "restricted_step_label_density_audit_slurm_review.json"),
        },
        "audit_method": {
            "observable_unit": "one generated response per prompt",
            "structural_anchor": "line-start or sentence-start Step <1..16>:",
            "slot_count_metric": "number of Step 1 through Step 16 anchors with a parseable first action word",
            "raw_candidate_bank_hit_metric": (
                "diagnostic only; raw model hits on Check/Review/Choose/Make/Start/Begin/Create/Set "
                "are accidental-surface-risk observations, not ownership evidence"
            ),
            "detector_required_before_wp4": True,
            "model_output_required_before_wp4": True,
        },
        "slurm_requirement": {
            "future_model_output_generation_must_use_chimera_slurm": True,
            "direct_chimera_login_node_cpu_or_gpu_run_allowed": False,
            "training_allowed": False,
            "e2e_allowed": False,
            "allowlisted_job_required_before_submission": True,
            "recommended_future_wrapper_name": "scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch",
            "submission_started_by_this_plan": False,
        },
        "claim_control": {
            "model_generation_started": False,
            "model_scoring_started": False,
            "training_started": False,
            "e2e_eval_started": False,
            "wp4_allowed": False,
            "paper_claim_allowed": False,
            "not_payload_recovery": True,
            "not_full_far": True,
        },
        "next_allowed_action": (
            "Review this artifact-only density audit plan. If approved, implement or review one "
            "Chimera Slurm wrapper that generates base-Qwen model outputs for these prompts and "
            "runs the restricted detector. Do not start WP4 or training."
        ),
    }


def gate_payload(*, prompt_count: int) -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_density_audit_gate_v1",
        "status": "GATE_SPEC_WRITTEN_NOT_EVALUATED",
        "minimum_model_output_rows": min(prompt_count, 256),
        "required_step_labels": [f"Step {index}:" for index in range(1, 17)],
        "gates": [
            {
                "metric": "complete_step_label_response_rate",
                "threshold": ">=0.95",
                "meaning": "responses contain Step 1 through Step 16 exactly once each",
            },
            {
                "metric": "mean_detected_structural_slots_per_response",
                "threshold": ">=16.0",
                "meaning": "restricted detector can locate all planned Step-label anchors",
            },
            {
                "metric": "responses_with_at_least_16_structural_slots_rate",
                "threshold": ">=0.90",
                "meaning": "density is not carried by a few unusually compliant outputs",
            },
            {
                "metric": "forbidden_public_surface_rate",
                "threshold": "==0.0",
                "meaning": "model outputs do not expose old structured evidence vocabulary",
            },
            {
                "metric": "raw_bank_surface_hit_rate",
                "threshold": "report_only_no_pass_threshold",
                "meaning": "raw accidental hits on passing bank surfaces are a null-risk diagnostic",
            },
            {
                "metric": "naturalness_manual_review_examples",
                "threshold": ">=32 examples reviewed before WP4",
                "meaning": "16-step control remains ordinary checklist prose",
            },
        ],
        "wp4_allowed_if_pass": False,
        "why_wp4_still_blocked_after_density": (
            "WP4 also requires reviewed density results and a prompt-local payload contract "
            "with decoder-oracle substitution."
        ),
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def slurm_review_payload(*, prompt_count: int) -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_density_slurm_review_v1",
        "status": "SLURM_REVIEW_SPEC_WRITTEN_SUBMISSION_NOT_STARTED",
        "future_job_purpose": (
            "Generate base-Qwen outputs for the restricted 16-step density prompts on a "
            "Chimera Slurm compute node, then run the restricted detector and write a "
            "density report. This is a density diagnostic only."
        ),
        "required_partition_policy": "use DGXA100/A100 unless user explicitly approves another available partition",
        "prompt_rows": prompt_count,
        "forbidden_actions": [
            "no training",
            "no protected transcript E2E",
            "no Llama",
            "no same-family null",
            "no sanitizer benchmark",
            "no FAR aggregation",
            "no positive paper claim",
        ],
        "submission_allowed_now": False,
        "submission_blocker": "artifact-only plan must be reviewed before any Slurm model-output generation job",
        "must_not_use_sbatch_export_unescaped_commas": True,
        "model_generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def readme_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WP3 Restricted Step-Label Density Audit Plan",
            "",
            "Artifact-only plan for auditing the 16-step restricted Step-label route.",
            "",
            f"status: `{summary['status']}`",
            f"prompt_count: `{summary['prompt_count']}`",
            f"selected_route: `{summary['selected_route']}`",
            "",
            "This did not call a model, score logits, train, run E2E, aggregate FAR, or make a paper claim.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    policy_dir = resolve_path(args.policy_dir)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    policy = read_json(policy_dir / "restricted_step_label_policy.json")
    prompt_rows = build_prompts(max_prompts=max(0, int(args.max_prompts)))
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_payload(output_dir=output_dir, policy_dir=policy_dir, policy=policy, prompt_rows=prompt_rows)
    gate = gate_payload(prompt_count=len(prompt_rows))
    slurm_review = slurm_review_payload(prompt_count=len(prompt_rows))
    summary = {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_density_audit_summary_v1",
        "status": "WP3_RESTRICTED_STEP_LABEL_DENSITY_AUDIT_PLAN_READY_ARTIFACT_ONLY",
        "selected_route": "A_16_step_checklist_step_label_only",
        "blocked_route": "B_8_step_plus_extra_slots",
        "prompt_count": len(prompt_rows),
        "plan_json": str(output_dir / "restricted_step_label_density_audit_plan.json"),
        "prompts_jsonl": str(output_dir / "restricted_step_label_density_audit_prompts.jsonl"),
        "gate_json": str(output_dir / "restricted_step_label_density_audit_gate.json"),
        "slurm_review_json": str(output_dir / "restricted_step_label_density_audit_slurm_review.json"),
        "model_generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Review the density audit plan. If approved, implement or review exactly one "
            "Chimera Slurm density-audit wrapper for base-Qwen model outputs. Do not start WP4 or training."
        ),
    }
    write_json(output_dir / "restricted_step_label_density_audit_plan.json", plan)
    write_jsonl(output_dir / "restricted_step_label_density_audit_prompts.jsonl", prompt_rows)
    write_json(output_dir / "restricted_step_label_density_audit_gate.json", gate)
    write_json(output_dir / "restricted_step_label_density_audit_slurm_review.json", slurm_review)
    write_json(output_dir / "restricted_step_label_density_audit_summary.json", summary)
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(json.dumps({"status": summary["status"], "prompt_count": len(prompt_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
