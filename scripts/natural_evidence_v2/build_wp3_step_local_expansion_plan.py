from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
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


PREFIX_FAMILIES: dict[str, list[str]] = {
    "step_label": ["Step 1: ", "Step 2: ", "Step 3: ", "Step 4: "],
    "numbered_list": ["1. ", "2. ", "3. ", "4. "],
    "dash_bullet": ["- "],
}


ACTION_BANKS: list[dict[str, Any]] = [
    {
        "bank_stem": "seed_check_review_choose_make",
        "side0": ["Check", "Review"],
        "side1": ["Choose", "Make"],
        "source": "850384_passing_seed",
    },
    {
        "bank_stem": "verify_confirm_select_choose",
        "side0": ["Verify", "Confirm"],
        "side1": ["Select", "Choose"],
        "source": "step_local_manual_expansion",
    },
    {
        "bank_stem": "prepare_gather_plan_schedule",
        "side0": ["Prepare", "Gather"],
        "side1": ["Plan", "Schedule"],
        "source": "step_local_manual_expansion",
    },
    {
        "bank_stem": "pack_bring_write_note",
        "side0": ["Pack", "Bring"],
        "side1": ["Write", "Note"],
        "source": "step_local_manual_expansion",
    },
    {
        "bank_stem": "start_begin_create_set",
        "side0": ["Start", "Begin"],
        "side1": ["Create", "Set"],
        "source": "step_local_manual_expansion",
    },
    {
        "bank_stem": "inspect_test_adjust_update",
        "side0": ["Inspect", "Test"],
        "side1": ["Adjust", "Update"],
        "source": "step_local_manual_expansion",
    },
    {
        "bank_stem": "compare_measure_record_track",
        "side0": ["Compare", "Measure"],
        "side1": ["Record", "Track"],
        "source": "step_local_manual_expansion",
    },
    {
        "bank_stem": "ask_call_save_store",
        "side0": ["Ask", "Call"],
        "side1": ["Save", "Store"],
        "source": "step_local_manual_expansion",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only WP3 step-local expansion context-mass "
            "score plan from the 850384 passing sentence-case action-verb seed. "
            "This does not load a tokenizer/model, train, generate, run E2E, "
            "estimate FAR, or make claims."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


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


def public_text_has_forbidden(text: str) -> list[str]:
    upper = text.upper()
    return [term for term in FORBIDDEN_PUBLIC_SURFACES if term.upper() in upper]


def validate_public_surfaces(values: Sequence[str]) -> None:
    for value in values:
        hits = public_text_has_forbidden(str(value))
        if hits:
            raise ValueError(f"forbidden public surface {hits}: {value!r}")


def plan_row_id(*, plan_id: str, bank_id: str, prefix: str) -> str:
    payload = json.dumps(
        {
            "plan_id": plan_id,
            "candidate_bank_id": bank_id,
            "prefix_before_candidate": prefix,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)[:20]


def candidate_bank_id(*, prefix_family: str, bank_stem: str) -> str:
    return f"step_local_{prefix_family}_{bank_stem}_v1"


def build_rows(plan_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bank_rows = Counter()
    source_counts = Counter()
    for prefix_family, prefixes in PREFIX_FAMILIES.items():
        validate_public_surfaces(prefixes)
        for bank in ACTION_BANKS:
            side0 = [str(value) for value in bank["side0"]]
            side1 = [str(value) for value in bank["side1"]]
            validate_public_surfaces(side0 + side1)
            bank_id = candidate_bank_id(prefix_family=prefix_family, bank_stem=str(bank["bank_stem"]))
            for prefix_index, prefix in enumerate(prefixes):
                row = {
                    "schema_name": "natural_evidence_v2_wp3_context_mass_score_plan_row_v1",
                    "plan_id": plan_id,
                    "plan_row_id": plan_row_id(plan_id=plan_id, bank_id=bank_id, prefix=prefix),
                    "candidate_bank_id": bank_id,
                    "source_bank_id": "step_opener_action_sentence_case_v1",
                    "source_seed_job": "850384",
                    "source_seed_result": "passed_configured_mass_gate",
                    "prefix_family": prefix_family,
                    "prefix_shape_index": prefix_index,
                    "slot_type": "bullet_or_step_opener",
                    "anchor_kind": "line_or_step_boundary",
                    "bucket_policy_id": "qwen_v2_wp3_step_local_sentence_case_two_way",
                    "casing_variant": "sentence_case",
                    "bucket_surfaces": {"0": side0, "1": side1},
                    "prefix_before_candidate": prefix,
                    "prefix_before_candidate_sha256": sha256_text(prefix),
                    "prefix_is_empty": False,
                    "source_detection_count": 1,
                    "source_response_count": 0,
                    "source_family_counts": {},
                    "observed_surface_counts": {},
                    "observed_case_variant_counts": {},
                    "template_preflight_only": False,
                    "structural_selection": "step_local_sentence_case_action_verb_expansion_not_qwen_gate",
                    "expansion_source": str(bank["source"]),
                    "scoring_status": "PLANNED_NOT_SCORED",
                    "tokenizer_stability_status": "NOT_EVALUATED",
                    "density_gate_status": "ARTIFACT_ONLY_FEASIBILITY_NOT_MODEL_OUTPUT",
                    "mass_gate_status": "NOT_EVALUATED",
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
                rows.append(row)
                bank_rows[bank_id] += 1
                source_counts[str(bank["source"])] += 1

    seen: set[str] = set()
    for row in rows:
        row_id = str(row["plan_row_id"])
        if row_id in seen:
            raise ValueError(f"duplicate plan_row_id: {row_id}")
        seen.add(row_id)

    audit = {
        "schema_name": "natural_evidence_v2_wp3_step_local_expansion_artifact_audit_v1",
        "status": "PASS_ARTIFACT_ONLY_STEP_LOCAL_EXPANSION_PLAN",
        "score_plan_rows": len(rows),
        "candidate_bank_count": len(bank_rows),
        "score_plan_rows_by_candidate_bank": dict(sorted(bank_rows.items())),
        "score_plan_rows_by_expansion_source": dict(sorted(source_counts.items())),
        "prefix_families": PREFIX_FAMILIES,
        "action_bank_count": len(ACTION_BANKS),
        "action_banks": ACTION_BANKS,
        "public_forbidden_surface_hits": [],
        "artifact_only": True,
        "not_qwen_gate": True,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }
    return rows, audit


def density_feasibility_audit(plan_id: str) -> dict[str, Any]:
    prompt_shapes = [
        {
            "prompt_shape_id": "eight_step_checklist",
            "step_count": 8,
            "step_local_slots_per_response": 8,
            "density_gate_pass_if_step_only": False,
        },
        {
            "prompt_shape_id": "twelve_step_checklist",
            "step_count": 12,
            "step_local_slots_per_response": 12,
            "density_gate_pass_if_step_only": False,
        },
        {
            "prompt_shape_id": "sixteen_step_checklist",
            "step_count": 16,
            "step_local_slots_per_response": 16,
            "density_gate_pass_if_step_only": True,
        },
    ]
    return {
        "schema_name": "natural_evidence_v2_wp3_step_local_density_feasibility_v1",
        "plan_id": plan_id,
        "status": "ARTIFACT_ONLY_DENSITY_FEASIBILITY_NOT_MODEL_OUTPUT_DENSITY",
        "prompt_shapes": prompt_shapes,
        "configured_average_micro_slots_gate": 16,
        "interpretation": (
            "A step-opener-only policy needs a sixteen-step/list response or "
            "additional non-step slots to satisfy the >=16 average micro-slot "
            "density gate. This audit is structural only; model-output density "
            "must be measured separately before WP4."
        ),
        "wp4_allowed": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def readme_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WP3 Step-Local Expansion Plan",
            "",
            "Artifact-only expansion around the `850384` passing sentence-case step-opener seed.",
            "",
            f"status: `{summary['status']}`",
            f"score_plan_rows: `{summary['score_plan_rows']}`",
            f"candidate_bank_count: `{summary['candidate_bank_count']}`",
            "",
            "This is not tokenizer/model scoring, not a gate, not training, not generation, not E2E, and not a claim.",
            "The score plan must be validated and scored with base Qwen through Chimera Slurm only.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")

    plan_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", output_dir.name)
    rows, artifact_audit = build_rows(plan_id)
    score_plan = output_dir / "qwen_v2_wp3_step_local_expansion_score_plan.jsonl"
    summary = {
        "schema_name": "natural_evidence_v2_wp3_step_local_expansion_plan_summary_v1",
        "status": "WP3_STEP_LOCAL_EXPANSION_SCORE_PLAN_WRITTEN_NOT_SCORED",
        "plan_id": plan_id,
        "score_plan_jsonl": str(score_plan),
        "score_plan_rows": len(rows),
        "candidate_bank_count": artifact_audit["candidate_bank_count"],
        "prefix_families": PREFIX_FAMILIES,
        "action_bank_count": len(ACTION_BANKS),
        "source_seed_job": "850384",
        "source_seed_bank": "step_opener_action_sentence_case_v1",
        "source_seed_min_bucket_mass": 0.005785648856544867,
        "source_seed_mass_ratio": 2.534862803103087,
        "artifact_audit_json": str(output_dir / "qwen_v2_wp3_step_local_expansion_artifact_audit.json"),
        "density_feasibility_json": str(output_dir / "qwen_v2_wp3_step_local_density_feasibility.json"),
        "artifact_only": True,
        "not_qwen_gate": True,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "wp4_allowed": False,
        "next_allowed_action": (
            "Review and validate this plan without model scoring; if accepted, "
            "submit at most one allowlisted Chimera Slurm base-Qwen context-mass "
            "scoring job."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(score_plan, rows)
    write_json(output_dir / "qwen_v2_wp3_step_local_expansion_plan_summary.json", summary)
    write_json(output_dir / "qwen_v2_wp3_step_local_expansion_artifact_audit.json", artifact_audit)
    write_json(output_dir / "qwen_v2_wp3_step_local_density_feasibility.json", density_feasibility_audit(plan_id))
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "score_plan_rows": summary["score_plan_rows"],
                "candidate_bank_count": summary["candidate_bank_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
