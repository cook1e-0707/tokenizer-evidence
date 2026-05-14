from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.validate_r4_transfer_gap_repair_plan import DEFAULT_CONFIG, load_yaml, validate_plan

DEFAULT_OUTPUT_DIR = ROOT / "results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_repair_package_20260514_0705"

PROMPT_SCAFFOLDS = [
    {
        "template_id": "practical_answer_next_action_prose",
        "family": "practical_advice_short",
        "template": (
            "Give a concise practical answer for this situation: {task}. "
            "Use plain prose or short bullets. Include a concrete next action when it is useful. "
            "Avoid headings, numbering, and labels."
        ),
    },
    {
        "template_id": "maintenance_guidance_action_sentence",
        "family": "maintenance_guidance",
        "template": (
            "Explain how to handle this maintenance issue: {task}. "
            "Keep the answer natural and useful after formatting is removed. "
            "Mention one concrete action the reader can take next."
        ),
    },
    {
        "template_id": "planning_guidance_next_move",
        "family": "planning_guidance",
        "template": (
            "Give brief planning guidance for: {task}. "
            "Use varied sentence openings and include a practical next move in normal language. "
            "Do not use headings or numbered instructions."
        ),
    },
    {
        "template_id": "troubleshooting_guidance_plain",
        "family": "troubleshooting_guidance",
        "template": (
            "Offer short troubleshooting guidance for: {task}. "
            "Write naturally, avoid formulaic openers, and include a concrete action only if it fits."
        ),
    },
    {
        "template_id": "quality_check_natural_bullets",
        "family": "safety_or_quality_checklist_natural",
        "template": (
            "Provide a compact quality check for: {task}. "
            "Short bullets are allowed, but the text should still make sense if bullet symbols are removed. "
            "Avoid fixed labels and repeated openings."
        ),
    },
    {
        "template_id": "task_explanation_action_embedded",
        "family": "task_explanation_short",
        "template": (
            "Explain this task briefly: {task}. "
            "Blend any recommended action into the answer as ordinary language rather than a labeled item."
        ),
    },
]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _contains_disallowed(text: str, disallowed: list[str]) -> list[str]:
    lowered = text.lower()
    return [item for item in disallowed if item.lower() in lowered]


def build_package(plan: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_plan(plan)
    if not str(validation["status"]).startswith("PASS"):
        return {
            "status": "FAIL_R4_TRANSFER_GAP_REPAIR_PACKAGE_PLAN_INVALID",
            "validation": validation,
            "errors": list(validation["errors"]),
        }

    prompt_policy = plan["prompt_scaffold_policy"]
    disallowed = [str(item) for item in prompt_policy["disallowed_literals"]]
    scaffold_errors: list[dict[str, Any]] = []
    for row in PROMPT_SCAFFOLDS:
        hits = _contains_disallowed(str(row["template"]), disallowed)
        if hits:
            scaffold_errors.append({"template_id": row["template_id"], "hits": hits})
    prefix_policy = dict(plan["prefix_family_policy"])
    forbidden_policy = dict(plan["forbidden_matcher_policy"])
    structural_policy = dict(plan["structural_leakage_policy"])
    future_route = dict(plan["future_route"])

    errors = []
    if scaffold_errors:
        errors.append("prompt_scaffold_disallowed_literals")
    if future_route.get("primary_decode_format_scrub") != "all":
        errors.append("future_route_primary_scrub_not_all")
    if future_route.get("qwen_only") is not True:
        errors.append("future_route_not_qwen_only")

    status = "PASS_R4_TRANSFER_GAP_REPAIR_PACKAGE_ARTIFACT_ONLY" if not errors else "FAIL_R4_TRANSFER_GAP_REPAIR_PACKAGE_ARTIFACT_ONLY"
    return {
        "current_compute_unlocked": False,
        "errors": errors,
        "forbidden_matcher_policy_recorded": True,
        "future_compute_conditionally_authorized_after_prerequisites": True,
        "future_route": future_route,
        "generation_started": False,
        "model_scoring_started": False,
        "prefix_family_count": len(prefix_policy["families"]),
        "prompt_scaffold_count": len(PROMPT_SCAFFOLDS),
        "scaffold_errors": scaffold_errors,
        "status": status,
        "structural_leakage_policy_recorded": True,
        "training_started": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the artifact-only R4 transfer-gap repair package.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {args.output_dir}")
    plan = load_yaml(args.config)
    summary = build_package(plan)
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "prompt_scaffold_templates.jsonl", PROMPT_SCAFFOLDS)
    write_json(args.output_dir / "prefix_family_policy.json", plan["prefix_family_policy"])
    write_json(args.output_dir / "forbidden_matcher_policy.json", plan["forbidden_matcher_policy"])
    write_json(args.output_dir / "structural_leakage_policy.json", plan["structural_leakage_policy"])
    write_json(args.output_dir / "future_route_constraints.json", plan["future_route"])
    write_json(args.output_dir / "repair_package_summary.json", summary)
    report = [
        "# R4 Transfer-Gap Repair Package",
        "",
        "Artifact-only package. No tokenizer/model loading, generation, training, scoring, or Slurm submission was started.",
        "",
        f"- status: `{summary['status']}`",
        f"- prompt scaffolds: `{summary['prompt_scaffold_count']}`",
        f"- prefix families: `{summary['prefix_family_count']}`",
        f"- current compute unlocked: `{summary['current_compute_unlocked']}`",
        f"- future compute conditionally authorized after prerequisites: `{summary['future_compute_conditionally_authorized_after_prerequisites']}`",
        f"- errors: `{summary['errors']}`",
    ]
    (args.output_dir / "repair_package_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
