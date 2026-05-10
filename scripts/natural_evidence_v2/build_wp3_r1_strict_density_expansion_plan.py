from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.run_wp3_restricted_step_label_density_audit import (  # noqa: E402
    DEFAULT_CONFIG,
    forbidden_terms_in_text,
    read_yaml,
)


DEFAULT_WP2_PROMPT_DIR = (
    ROOT
    / "results/natural_evidence_v2/prompts/"
    "wp2_controlled_natural_prompt_family_scaffold_20260508_2123"
)
DEFAULT_SOURCE_POLICY_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "wp3_restricted_step_label_primary_policy_density_plan_850523_repair_20260509_0310"
)
DEFAULT_SOURCE_DIAGNOSTIC_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "wp3_restricted_step_label_primary_density_audit_850523_repair_850771"
)

STRICT_PROMPT_VARIANTS = (
    {
        "variant_id": "r1_strict_literal_16_step_lines",
        "template": (
            "Use the everyday situation below as the topic, but follow only the format instructions after it.\n"
            "Situation: {source_prompt}\n"
            "Return exactly sixteen lines. Line 1 must begin with Step 1:, line 2 with Step 2:, "
            "and continue in order through Step 16:. Do not omit any Step label. Do not add "
            "headings, bullets, tables, or closing notes. After each label, write one natural "
            "sentence that begins with an action verb."
        ),
    },
    {
        "variant_id": "r1_strict_no_heading_16_step_lines",
        "template": (
            "Treat the following everyday request as background context, not as a required output format:\n"
            "{source_prompt}\n"
            "Write exactly 16 checklist lines. Use only the labels Step 1: through Step 16:, "
            "one label per line and in numeric order. Do not include a title, introduction, "
            "bullets, table, or summary. Each line should be one ordinary sentence after the label."
        ),
    },
    {
        "variant_id": "r1_strict_numbered_step_label_lines",
        "template": (
            "Background topic: {source_prompt}\n"
            "Create a normal practical checklist with exactly sixteen labeled steps. Every line "
            "must start with the literal label Step N: for N from 1 to 16. Use no other labels "
            "or headings. Each step should be concise and should start with a clear action word "
            "after the colon."
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the artifact-only WP3-R1 strict Step-label density expansion "
            "plan. This writes dev/eval prompt-plan artifacts, records the strict "
            "line-start detector contract, records eval prompt-local frame oracle "
            "completion, and does not submit Slurm, call a model, train, run E2E, "
            "aggregate FAR, or make positive claims."
        )
    )
    parser.add_argument("--wp2-prompt-dir", type=Path, default=DEFAULT_WP2_PROMPT_DIR)
    parser.add_argument("--source-policy-dir", type=Path, default=DEFAULT_SOURCE_POLICY_DIR)
    parser.add_argument("--source-diagnostic-dir", type=Path, default=DEFAULT_SOURCE_DIAGNOSTIC_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-count", type=int, default=512)
    parser.add_argument("--eval-count", type=int, default=2048)
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
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
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


def require_count(rows: list[dict[str, Any]], *, count: int, label: str) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"{label} source rows {len(rows)} < requested {count}")
    return rows[:count]


def build_prompt_rows(
    *,
    config: Mapping[str, Any],
    source_rows: list[Mapping[str, Any]],
    r1_split: str,
    count: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, source_row in enumerate(require_count(list(source_rows), count=count, label=r1_split)):
        variant = STRICT_PROMPT_VARIANTS[index % len(STRICT_PROMPT_VARIANTS)]
        source_prompt = str(source_row.get("prompt_text", "")).strip()
        if not source_prompt:
            raise ValueError(f"empty source prompt text for {r1_split}:{index}")
        prompt_text = str(variant["template"]).format(source_prompt=source_prompt)
        hits = forbidden_terms_in_text(config, prompt_text)
        if hits:
            raise ValueError(f"forbidden public surface {hits} in {r1_split}:{index}")
        prompt_id_payload = {
            "r1_split": r1_split,
            "source_prompt_id": str(source_row.get("prompt_id", "")),
            "variant_id": str(variant["variant_id"]),
        }
        prompt_id = "qwen_v2_wp3_r1_density_" + sha256_text(
            json.dumps(prompt_id_payload, sort_keys=True, separators=(",", ":"))
        )[:20]
        output.append(
            {
                "schema_name": "natural_evidence_v2_wp3_r1_strict_density_prompt_v1",
                "prompt_id": prompt_id,
                "split": f"wp3_r1_density_{r1_split}",
                "r1_split": r1_split,
                "family_id": "F2_16_step_checklist_step_label_r1_expansion",
                "variant_id": str(variant["variant_id"]),
                "source_wp2_prompt_id": str(source_row.get("prompt_id", "")),
                "source_wp2_split": str(source_row.get("split", "")),
                "source_wp2_family_id": str(source_row.get("family_id", "")),
                "source_wp2_prompt_text_sha256": str(source_row.get("prompt_text_sha256", "")),
                "prompt_text": prompt_text,
                "prompt_text_sha256": sha256_text(prompt_text),
                "expected_step_labels": [f"Step {step}:" for step in range(1, 17)],
                "expected_structural_slots": 16,
                "strict_line_start_step_labels_required": True,
                "sentence_start_inline_step_labels_counted": False,
                "prompt_local_frame_slot_count": 16,
                "prompt_local_frame_required_bits": 16,
                "oracle_prompt_local_frame_completion_expected": r1_split == "eval",
                "oracle_prompt_local_frame_completion_contribution": 1.0 if r1_split == "eval" else None,
                "artifact_only_plan": True,
                "model_generation_started": False,
                "model_scoring_started": False,
                "training_started": False,
                "e2e_eval_started": False,
                "wp4_allowed": False,
                "paper_claim_allowed": False,
            }
        )
    return output


def prompt_audit(*, rows_by_split: Mapping[str, list[Mapping[str, Any]]], config: Mapping[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    duplicate_prompt_ids: list[str] = []
    prompt_ids: Counter[str] = Counter()
    counts_by_split = {split: len(rows) for split, rows in rows_by_split.items()}
    counts_by_variant: dict[str, dict[str, int]] = {}
    for split, rows in rows_by_split.items():
        variant_counter: Counter[str] = Counter()
        for row in rows:
            prompt_id = str(row.get("prompt_id", ""))
            prompt_ids[prompt_id] += 1
            variant_counter[str(row.get("variant_id", ""))] += 1
            text = str(row.get("prompt_text", ""))
            hits = forbidden_terms_in_text(config, text)
            if hits:
                violations.append({"prompt_id": prompt_id, "split": split, "hits": hits})
            if int(row.get("expected_structural_slots", 0)) != 16:
                violations.append({"prompt_id": prompt_id, "split": split, "hits": ["expected_structural_slots_not_16"]})
            if not bool(row.get("strict_line_start_step_labels_required", False)):
                violations.append({"prompt_id": prompt_id, "split": split, "hits": ["strict_line_start_not_required"]})
        counts_by_variant[split] = dict(sorted(variant_counter.items()))
    duplicate_prompt_ids = [prompt_id for prompt_id, seen in prompt_ids.items() if seen > 1]
    status = "PASS_R1_STRICT_DENSITY_PROMPT_PLAN_AUDIT"
    if violations or duplicate_prompt_ids:
        status = "FAIL_R1_STRICT_DENSITY_PROMPT_PLAN_AUDIT"
    return {
        "schema_name": "natural_evidence_v2_wp3_r1_strict_density_prompt_plan_audit_v1",
        "status": status,
        "counts_by_split": counts_by_split,
        "counts_by_variant": counts_by_variant,
        "forbidden_surface_violation_count": len(violations),
        "violations": violations,
        "duplicate_prompt_id_count": len(duplicate_prompt_ids),
        "duplicate_prompt_ids": duplicate_prompt_ids,
        "model_generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def oracle_payload(*, eval_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    complete_count = sum(
        1
        for row in eval_rows
        if int(row.get("prompt_local_frame_slot_count", 0)) >= int(row.get("prompt_local_frame_required_bits", 0))
        and bool(row.get("oracle_prompt_local_frame_completion_expected", False))
    )
    total = len(eval_rows)
    rate = 0.0 if total == 0 else complete_count / total
    return {
        "schema_name": "natural_evidence_v2_wp3_r1_eval_oracle_prompt_local_frame_completion_v1",
        "status": "EVAL_ORACLE_PROMPT_LOCAL_FRAME_COMPLETION_RECORDED_ARTIFACT_ONLY",
        "scope": "planned eval prompt local slot capacity before model-output audit",
        "eval_prompt_count": total,
        "planned_complete_prompt_count": complete_count,
        "oracle_prompt_local_frame_completion_rate": rate,
        "required_oracle_prompt_local_frame_completion_rate": 0.95,
        "oracle_gate_plan_status": "PASS_BY_CONSTRUCTION_NOT_MODEL_OUTPUT" if rate >= 0.95 else "FAIL_PLAN",
        "frame_policy": "8-bit payload plus 8-bit checksum mapped to the 16 strict Step-label slots within one response",
        "assumption": "If a model output contains all 16 strict line-start Step labels, the prompt-local oracle can fill one 16-bit frame in that response.",
        "not_model_output_density_result": True,
        "not_decoder_oracle_substitution": True,
        "not_payload_recovery": True,
        "not_full_far": True,
        "model_generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
    }


def gate_payload() -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_wp3_r1_strict_density_gate_v1",
        "status": "WP3_R1_STRICT_DENSITY_GATE_SPEC_WRITTEN_NOT_EVALUATED",
        "detector_contract": {
            "strict_line_start_step_labels_required": True,
            "accepted_anchor": "line-start Step N: after optional whitespace, optional markdown bullet marker, and optional markdown emphasis",
            "sentence_start_inline_step_labels_counted": False,
        },
        "dev_gate": {
            "minimum_outputs": 512,
            "complete_step_label_response_rate": ">=0.995",
            "mean_detected_slots_per_response": ">=15.9",
            "forbidden_public_surface_rate": "==0.0",
        },
        "eval_gate": {
            "minimum_outputs": 2048,
            "complete_step_label_response_rate": ">=0.995",
            "oracle_prompt_local_frame_completion_rate": ">=0.95",
            "forbidden_public_surface_rate": "==0.0",
        },
        "not_model_output_density_result": True,
        "wp4_allowed_if_written": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def slurm_review_payload(*, output_dir: Path, dev_count: int, eval_count: int) -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_wp3_r1_strict_density_slurm_review_v1",
        "status": "SLURM_REVIEW_SPEC_WRITTEN_SUBMISSION_NOT_STARTED",
        "compatible_wrapper": "scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch",
        "allowlist_entry": "v2_wp3_restricted_step_label_density_audit",
        "allowlist_enabled": False,
        "explicit_approval_required_before_submission": True,
        "submission_allowed_now": False,
        "dev_future_command_pattern": (
            f"PROMPTS_JSONL={output_dir / 'restricted_step_label_r1_dev_prompts.jsonl'} "
            f"POLICY_DIR={output_dir} MAX_PROMPTS={dev_count} OUTPUT_DIR=<fresh dev scratch output> "
            "sbatch scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch"
        ),
        "eval_future_command_pattern": (
            f"PROMPTS_JSONL={output_dir / 'restricted_step_label_r1_eval_prompts.jsonl'} "
            f"POLICY_DIR={output_dir} MAX_PROMPTS={eval_count} OUTPUT_DIR=<fresh eval scratch output> "
            "sbatch scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch"
        ),
        "future_chimera_cpu_gpu_work_must_use_slurm": True,
        "direct_chimera_login_node_cpu_or_gpu_run_allowed": False,
        "model_generation_started_by_this_plan": False,
        "model_scoring_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
        "forbidden_actions": [
            "no training",
            "no Qwen E2E rerun",
            "no Llama",
            "no same-family null",
            "no sanitizer benchmark",
            "no FAR aggregation",
            "no paper-facing positive claim",
        ],
    }


def readme_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WP3-R1 Strict Density Expansion Plan",
            "",
            "Artifact-only expansion plan for the repaired strict Step-label route.",
            "",
            f"status: `{summary['status']}`",
            f"dev_prompt_count: `{summary['dev_prompt_count']}`",
            f"eval_prompt_count: `{summary['eval_prompt_count']}`",
            f"oracle_prompt_local_frame_completion_rate: `{summary['oracle_prompt_local_frame_completion_rate']}`",
            "",
            "This did not submit Slurm, call a model, train, run E2E, aggregate FAR, or make a paper-facing claim.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    dev_count = int(args.dev_count)
    eval_count = int(args.eval_count)
    if dev_count < 512:
        raise ValueError("WP3-R1 dev prompt count must be >=512")
    if eval_count < 2048:
        raise ValueError("WP3-R1 eval prompt count must be >=2048")

    config = read_yaml(resolve_path(args.config))
    wp2_prompt_dir = resolve_path(args.wp2_prompt_dir)
    source_policy_dir = resolve_path(args.source_policy_dir)
    source_diagnostic_dir = resolve_path(args.source_diagnostic_dir)
    dev_source_rows = read_jsonl(wp2_prompt_dir / "qwen_v2_dev_prompts.jsonl")
    eval_source_rows = read_jsonl(wp2_prompt_dir / "qwen_v2_eval_prompts.jsonl")
    source_policy = read_json(source_policy_dir / "restricted_step_label_policy.json")
    source_bucket_bank = read_json(source_policy_dir / "restricted_step_label_bucket_bank.json")
    source_detector_contract = read_json(source_policy_dir / "restricted_step_label_detector_contract.json")
    source_diagnostic = read_json(source_diagnostic_dir / "restricted_step_label_density_audit_summary.json")

    dev_rows = build_prompt_rows(config=config, source_rows=dev_source_rows, r1_split="dev", count=dev_count)
    eval_rows = build_prompt_rows(config=config, source_rows=eval_source_rows, r1_split="eval", count=eval_count)
    prompt_plan_audit = prompt_audit(rows_by_split={"dev": dev_rows, "eval": eval_rows}, config=config)
    if prompt_plan_audit["status"] != "PASS_R1_STRICT_DENSITY_PROMPT_PLAN_AUDIT":
        raise ValueError(f"prompt plan audit failed: {prompt_plan_audit}")
    oracle = oracle_payload(eval_rows=eval_rows)

    detector_contract = dict(source_detector_contract)
    detector_contract.update(
        {
            "detector_id": "qwen_v2_wp3_r1_strict_line_start_detector_v1",
            "slot_policy_id": "qwen_v2_wp3_r1_strict_density_expansion_policy_v1",
            "status": "WP3_R1_STRICT_LINE_START_DETECTOR_CONTRACT_RECORDED_ARTIFACT_ONLY",
            "source_detector_id": source_detector_contract.get("detector_id"),
            "source_policy_dir": str(source_policy_dir),
            "line_start_step_labels_required": True,
            "sentence_start_inline_step_labels_counted": False,
            "eval_oracle_prompt_local_frame_completion_required_rate": 0.95,
            "model_generation_started": False,
            "model_scoring_started": False,
            "training_started": False,
            "e2e_eval_started": False,
            "paper_claim_allowed": False,
        }
    )
    density_design = {
        "schema_name": "natural_evidence_v2_wp3_r1_strict_density_expansion_design_v1",
        "status": "WP3_R1_STRICT_DENSITY_EXPANSION_PLAN_READY_ARTIFACT_ONLY",
        "source_policy_dir": str(source_policy_dir),
        "source_diagnostic_dir": str(source_diagnostic_dir),
        "source_diagnostic_job_id": "850771",
        "source_diagnostic_total_responses": source_diagnostic.get("total_responses"),
        "source_diagnostic_structural_density_gate_status": source_diagnostic.get("structural_density_gate_status"),
        "source_diagnostic_not_full_r1_reason": "192 diagnostic prompts, no dev>=512/eval>=2048 split, no eval oracle prompt-local frame completion field",
        "selected_route": "A_strict_16_step_checklist_step_label_only",
        "decision": (
            "Expand the repaired 850523 strict Step-label seed into separate WP3-R1 dev/eval "
            "prompt-plan artifacts while preserving the strict line-start detector. This is "
            "artifact-only and is not a model-output density gate result."
        ),
        "detector_decision": "keep strict line-start Step N: detector; do not count inline sentence-start Step labels",
        "dev_prompt_count": len(dev_rows),
        "eval_prompt_count": len(eval_rows),
        "dev_outputs_required": 512,
        "eval_outputs_required": 2048,
        "eval_oracle_prompt_local_frame_completion_artifact": str(
            output_dir / "restricted_step_label_r1_eval_oracle_prompt_local_frame_completion.json"
        ),
        "model_output_density_audit_required_before_wp4": True,
        "slurm_submission_started": False,
        "model_generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
    }
    policy = dict(source_policy)
    policy.update(
        {
            "policy_id": "qwen_v2_wp3_r1_strict_density_expansion_policy_v1",
            "status": "WP3_R1_STRICT_DENSITY_EXPANSION_POLICY_WRITTEN_ARTIFACT_ONLY",
            "source_policy_id": source_policy.get("policy_id"),
            "source_policy_dir": str(source_policy_dir),
            "detector_contract": detector_contract,
            "density_design": density_design,
            "next_allowed_action": (
                "Review this artifact-only WP3-R1 strict density expansion plan. "
                "Do not submit Slurm without explicit approval; WP4/training remain blocked."
            ),
        }
    )
    bucket_bank = dict(source_bucket_bank)
    bucket_bank.update(
        {
            "status": "WP3_R1_STRICT_DENSITY_EXPANSION_BUCKET_BANK_COPIED_FROM_REPAIRED_PRIMARY_POLICY",
            "source_policy_dir": str(source_policy_dir),
            "model_generation_started": False,
            "model_scoring_started": False,
            "training_started": False,
            "e2e_eval_started": False,
            "wp4_allowed": False,
            "paper_claim_allowed": False,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "restricted_step_label_r1_dev_prompts.jsonl", dev_rows)
    write_jsonl(output_dir / "restricted_step_label_r1_eval_prompts.jsonl", eval_rows)
    write_json(output_dir / "restricted_step_label_r1_prompt_plan_audit.json", prompt_plan_audit)
    write_json(output_dir / "restricted_step_label_r1_eval_oracle_prompt_local_frame_completion.json", oracle)
    write_json(output_dir / "restricted_step_label_r1_density_gate.json", gate_payload())
    write_json(output_dir / "restricted_step_label_r1_slurm_review.json", slurm_review_payload(output_dir=output_dir, dev_count=dev_count, eval_count=eval_count))
    write_json(output_dir / "restricted_step_label_policy.json", policy)
    write_json(output_dir / "restricted_step_label_bucket_bank.json", bucket_bank)
    write_json(output_dir / "restricted_step_label_detector_contract.json", detector_contract)
    write_json(output_dir / "restricted_step_label_density_design.json", density_design)

    summary = {
        "schema_name": "natural_evidence_v2_wp3_r1_strict_density_expansion_plan_summary_v1",
        "status": "WP3_R1_STRICT_DENSITY_EXPANSION_PLAN_READY_ARTIFACT_ONLY",
        "output_dir": str(output_dir),
        "source_wp2_prompt_dir": str(wp2_prompt_dir),
        "source_policy_dir": str(source_policy_dir),
        "source_diagnostic_job_id": "850771",
        "source_diagnostic_total_responses": source_diagnostic.get("total_responses"),
        "dev_prompt_count": len(dev_rows),
        "eval_prompt_count": len(eval_rows),
        "dev_prompts_jsonl": str(output_dir / "restricted_step_label_r1_dev_prompts.jsonl"),
        "eval_prompts_jsonl": str(output_dir / "restricted_step_label_r1_eval_prompts.jsonl"),
        "policy_dir": str(output_dir),
        "detector_contract_json": str(output_dir / "restricted_step_label_detector_contract.json"),
        "gate_json": str(output_dir / "restricted_step_label_r1_density_gate.json"),
        "slurm_review_json": str(output_dir / "restricted_step_label_r1_slurm_review.json"),
        "oracle_prompt_local_frame_completion_json": str(
            output_dir / "restricted_step_label_r1_eval_oracle_prompt_local_frame_completion.json"
        ),
        "prompt_plan_audit_json": str(output_dir / "restricted_step_label_r1_prompt_plan_audit.json"),
        "prompt_plan_audit_status": prompt_plan_audit["status"],
        "strict_line_start_step_labels_required": True,
        "sentence_start_inline_step_labels_counted": False,
        "oracle_prompt_local_frame_completion_rate": oracle["oracle_prompt_local_frame_completion_rate"],
        "oracle_gate_plan_status": oracle["oracle_gate_plan_status"],
        "slurm_submitted": False,
        "allowlist_enabled": False,
        "explicit_approval_required_before_submission": True,
        "model_generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "next_allowed_action": (
            "Review this artifact-only WP3-R1 strict density expansion plan. "
            "If explicitly approved later, run Slurm-only dev/eval density audits in fresh output dirs. "
            "Do not start WP4 or training."
        ),
    }
    write_json(output_dir / "restricted_step_label_r1_expansion_summary.json", summary)
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "dev_prompt_count": len(dev_rows),
                "eval_prompt_count": len(eval_rows),
                "oracle_prompt_local_frame_completion_rate": oracle[
                    "oracle_prompt_local_frame_completion_rate"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
