from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_candidate_v3_transfer_gap_repair.yaml"

REQUIRED_REPAIR_SURFACES = {
    "prefix_context_elicitation",
    "free_generation_surface_polarity_alignment",
    "forbidden_matcher_semantics",
    "structural_length_leakage",
}

LOCKED_FALSE_PERMISSION_FIELDS = (
    "slurm_allowed",
    "allowlist_enablement_allowed",
    "generation_allowed",
    "training_allowed",
    "qwen_e2e_allowed",
    "llama_allowed",
    "same_family_null_allowed",
    "sanitizer_allowed",
    "far_aggregation_allowed",
    "payload_diversity_claim_allowed",
    "paper_claim_allowed",
)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text_contains_any(text: str, needles: list[str]) -> list[str]:
    lowered = text.lower()
    return [needle for needle in needles if needle.lower() in lowered]


def validate_plan(plan: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []

    if plan.get("schema_name") != "natural_evidence_v2_r4_candidate_v3_transfer_gap_repair_plan_v1":
        errors.append("schema_name mismatch")

    source_failure = root / str(plan.get("source_failure_analysis", ""))
    if not source_failure.exists():
        errors.append("source_failure_analysis missing")

    execution = plan.get("standing_execution_policy", {})
    if not isinstance(execution, Mapping):
        execution = {}
        errors.append("standing_execution_policy must be a mapping")
    expected_true = (
        "continue_clear_routes_without_repeated_manual_approval",
        "gate_controlled_actions_conditionally_authorized",
        "prerequisite_gates_still_required",
        "hermes_notification_still_required",
        "allowlist_safety_still_required",
        "h200_pomplun_policy_still_required",
    )
    for field in expected_true:
        if execution.get(field) is not True:
            errors.append(f"standing_execution_policy.{field} must be true")
    if execution.get("gate_controlled_actions_permanently_forbidden") is not False:
        errors.append("standing_execution_policy.gate_controlled_actions_permanently_forbidden must be false")

    permissions = plan.get("current_permissions", {})
    if not isinstance(permissions, Mapping):
        permissions = {}
        errors.append("current_permissions must be a mapping")
    for field in LOCKED_FALSE_PERMISSION_FIELDS:
        if permissions.get(field) is not False:
            errors.append(f"current_permissions.{field} must be false before repair-package review")

    repair_surfaces = {str(item) for item in _as_list(plan.get("repair_surfaces"))}
    if repair_surfaces != REQUIRED_REPAIR_SURFACES:
        errors.append(f"repair_surfaces mismatch: {sorted(repair_surfaces)}")

    prompt_policy = plan.get("prompt_scaffold_policy", {})
    if not isinstance(prompt_policy, Mapping):
        prompt_policy = {}
        errors.append("prompt_scaffold_policy must be a mapping")
    for field in (
        "no_step_labels",
        "no_fixed_slot_count",
        "no_visible_coordinate_labels",
        "no_repeated_global_lead_in",
        "no_public_technical_literals",
    ):
        if prompt_policy.get(field) is not True:
            errors.append(f"prompt_scaffold_policy.{field} must be true")
    disallowed = [str(item) for item in _as_list(prompt_policy.get("disallowed_literals"))]
    principle_text = "\n".join(str(item) for item in _as_list(prompt_policy.get("instruction_principles")))
    principle_hits = _text_contains_any(principle_text, disallowed)
    if principle_hits:
        errors.append(f"prompt_scaffold_policy instruction principles contain disallowed literals: {principle_hits}")

    prefix_policy = plan.get("prefix_family_policy", {})
    if not isinstance(prefix_policy, Mapping):
        prefix_policy = {}
        errors.append("prefix_family_policy must be a mapping")
    if prefix_policy.get("source_policy") != "frozen_rule_based_before_generation_no_locked_output_mining":
        errors.append("prefix_family_policy.source_policy must forbid locked-output mining")
    if float(prefix_policy.get("max_single_prefix_family_output_fraction", 1.0)) > 0.35:
        errors.append("prefix_family_policy.max_single_prefix_family_output_fraction must be <= 0.35")
    families = _as_list(prefix_policy.get("families"))
    if len(families) < 2:
        errors.append("prefix_family_policy must include at least two families")
    total_prefix_shapes = 0
    for family in families:
        if not isinstance(family, Mapping):
            errors.append("prefix family entry must be a mapping")
            continue
        shapes = [str(item) for item in _as_list(family.get("prefix_shapes"))]
        total_prefix_shapes += len(shapes)
        if len(set(shapes)) != len(shapes):
            errors.append(f"duplicate prefix shapes in family {family.get('family_id')}")
    if total_prefix_shapes < 6:
        errors.append("prefix_family_policy must include at least six prefix shapes")

    surface_policy = plan.get("surface_polarity_policy", {})
    if not isinstance(surface_policy, Mapping):
        surface_policy = {}
        errors.append("surface_polarity_policy must be a mapping")
    for field in (
        "report_target_vs_other_phrase_support",
        "report_per_coordinate_support",
        "report_phrase_hits_after_format_scrub_all",
        "do_not_mine_surfaces_from_locked_outputs",
    ):
        if surface_policy.get(field) is not True:
            errors.append(f"surface_polarity_policy.{field} must be true")

    matcher_policy = plan.get("forbidden_matcher_policy", {})
    if not isinstance(matcher_policy, Mapping):
        matcher_policy = {}
        errors.append("forbidden_matcher_policy must be a mapping")
    if matcher_policy.get("report_separate_counts") is not True:
        errors.append("forbidden_matcher_policy.report_separate_counts must be true")
    if int(matcher_policy.get("hard_gate_technical_public_surface_count", -1)) != 0:
        errors.append("forbidden_matcher_policy.hard_gate_technical_public_surface_count must be 0")

    structural_policy = plan.get("structural_leakage_policy", {})
    if not isinstance(structural_policy, Mapping):
        structural_policy = {}
        errors.append("structural_leakage_policy must be a mapping")
    if structural_policy.get("primary_decode_format_scrub") != "all":
        errors.append("structural_leakage_policy.primary_decode_format_scrub must be all")
    if float(structural_policy.get("protected_vs_raw_shallow_auc_max", 1.0)) > 0.60:
        errors.append("structural_leakage_policy.protected_vs_raw_shallow_auc_max must be <= 0.60")

    future_route = plan.get("future_route", {})
    if not isinstance(future_route, Mapping):
        future_route = {}
        errors.append("future_route must be a mapping")
    if future_route.get("allow_without_repeated_manual_approval_after_prerequisites_pass") is not True:
        errors.append("future_route must preserve standing conditional authorization")
    if future_route.get("h200_partition") != "pomplun" or future_route.get("h200_account") != "cs_yinxin.wan":
        errors.append("future_route must use H200 pomplun/cs_yinxin.wan policy")
    if future_route.get("h200_gres") != "gpu:h200:1":
        errors.append("future_route.h200_gres must be gpu:h200:1")
    if future_route.get("primary_decode_format_scrub") != "all":
        errors.append("future_route.primary_decode_format_scrub must be all")

    status = "PASS_R4_TRANSFER_GAP_REPAIR_PLAN_VALIDATION" if not errors else "FAIL_R4_TRANSFER_GAP_REPAIR_PLAN_VALIDATION"
    return {
        "current_compute_unlocked": False,
        "errors": errors,
        "future_compute_conditionally_authorized_after_prerequisites": True,
        "generation_started": False,
        "model_scoring_started": False,
        "repair_surface_count": len(repair_surfaces),
        "source_failure_analysis": str(source_failure),
        "source_failure_analysis_exists": source_failure.exists(),
        "status": status,
        "total_prefix_shapes": total_prefix_shapes,
        "training_started": False,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 transfer-gap repair plan without compute.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_plan(load_yaml(args.config))
    if args.output_dir is not None:
        write_json(args.output_dir / "transfer_gap_repair_plan_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
