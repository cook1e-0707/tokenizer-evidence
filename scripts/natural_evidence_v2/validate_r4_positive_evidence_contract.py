from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_positive_evidence_contract_redesign.yaml"

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
    "payload_diversity_allowed",
    "paper_claim_allowed",
)

REQUIRED_CONDITIONS = {"protected", "raw", "task_only", "wrong_key", "wrong_payload"}
REQUIRED_REQUIREMENTS = {
    "key_payload_specificity_observable_before_accept_scoring",
    "positive_unit_avoids_forbidden_public_surfaces_and_fixed_labels",
    "decoder_separates_reusable_support_from_accepted_protected_recovery",
    "structural_leakage_features_are_outside_contract_or_non_informative",
    "pre_registered_dev_pass_fail_table_present",
}
REQUIRED_SPECIFICITY_METRICS = {
    "protected_keyed_correlation_score",
    "wrong_key_correlation_score",
    "wrong_payload_correlation_score",
    "protected_minus_best_wrong_specificity_margin",
}


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _mapping(value: Any, field: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be a mapping")
        return {}
    return value


def _list_set(value: Any) -> set[str]:
    return {str(item) for item in value} if isinstance(value, list) else set()


def validate_contract(contract: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []

    if contract.get("schema_name") != "natural_evidence_v2_r4_positive_evidence_contract_redesign_v1":
        errors.append("schema_name mismatch")
    if contract.get("contract_id") != "r4_keyed_correlation_evidence_v1":
        errors.append("contract_id mismatch")

    source_failure = root / str(contract.get("source_failure_analysis", ""))
    source_review = root / str(contract.get("source_review", ""))
    if not source_failure.exists():
        errors.append("source_failure_analysis missing")
    if not source_review.exists():
        errors.append("source_review missing")

    permissions = _mapping(contract.get("current_permissions"), "current_permissions", errors)
    for field in LOCKED_FALSE_PERMISSION_FIELDS:
        if permissions.get(field) is not False:
            errors.append(f"current_permissions.{field} must be false before compute route review")

    execution = _mapping(contract.get("standing_execution_policy"), "standing_execution_policy", errors)
    for field in (
        "continue_clear_routes_without_repeated_manual_approval",
        "gate_controlled_actions_conditionally_authorized",
        "prerequisite_gates_still_required",
        "allowlist_safety_still_required",
        "hermes_notification_still_required",
        "slurm_only_for_chimera_cpu_or_gpu",
        "h200_pomplun_policy_for_gpu",
    ):
        if execution.get(field) is not True:
            errors.append(f"standing_execution_policy.{field} must be true")

    requirements = _mapping(contract.get("design_requirements"), "design_requirements", errors)
    for field in REQUIRED_REQUIREMENTS:
        if requirements.get(field) is not True:
            errors.append(f"design_requirements.{field} must be true")

    public_surface = _mapping(contract.get("public_surface_policy"), "public_surface_policy", errors)
    for field in (
        "no_literal_evidence_block",
        "no_fixed_label_dependency",
        "no_heading_dependency",
        "no_step_or_numbered_dependency",
        "no_public_technical_literals",
        "format_scrub_primary",
    ):
        if public_surface.get(field) is not True:
            errors.append(f"public_surface_policy.{field} must be true")
    if public_surface.get("positive_unit") != "normalized_phrase_event":
        errors.append("public_surface_policy.positive_unit must be normalized_phrase_event")

    carrier = _mapping(contract.get("carrier_event_policy"), "carrier_event_policy", errors)
    if carrier.get("source_bank_policy") != "frozen_dev_rule_or_precommit_only_no_locked_output_mining":
        errors.append("carrier_event_policy.source_bank_policy must forbid locked-output mining")
    if int(carrier.get("min_distinct_surface_families", 0)) < 4:
        errors.append("carrier_event_policy.min_distinct_surface_families must be >= 4")
    if float(carrier.get("max_single_surface_family_vote_fraction", 1.0)) > 0.35:
        errors.append("carrier_event_policy.max_single_surface_family_vote_fraction must be <= 0.35")
    excluded = _list_set(carrier.get("structural_features_excluded_from_votes"))
    for feature in ("bullet_marker", "line_number", "heading_text", "colon_after_label", "next_action_label"):
        if feature not in excluded:
            errors.append(f"carrier_event_policy.structural_features_excluded_from_votes missing {feature}")

    specificity = _mapping(contract.get("key_payload_specificity_policy"), "key_payload_specificity_policy", errors)
    if specificity.get("mapping") != "hmac_audit_key_payload_surface_to_coordinate_and_polarity":
        errors.append("key_payload_specificity_policy.mapping must be hmac based")
    for field in (
        "audit_key_committed_before_generation",
        "payload_committed_before_generation",
        "wrong_key_recomputes_mapping",
        "wrong_payload_recomputes_codeword",
    ):
        if specificity.get(field) is not True:
            errors.append(f"key_payload_specificity_policy.{field} must be true")
    observed_metrics = _list_set(specificity.get("pre_accept_specificity_metrics"))
    if observed_metrics != REQUIRED_SPECIFICITY_METRICS:
        errors.append(f"pre_accept_specificity_metrics mismatch: {sorted(observed_metrics)}")
    specificity_thresholds = _mapping(specificity.get("required_before_accept"), "key_payload_specificity_policy.required_before_accept", errors)
    if float(specificity_thresholds.get("protected_minus_best_wrong_specificity_margin_min", 0.0)) < 3.0:
        errors.append("specificity margin before accept must be >= 3.0")

    decoder = _mapping(contract.get("decoder_policy"), "decoder_policy", errors)
    if decoder.get("primary_scrub_mode") != "all":
        errors.append("decoder_policy.primary_scrub_mode must be all")
    for field in ("support_is_not_acceptance", "ordinary_reusable_support_must_not_accept"):
        if decoder.get(field) is not True:
            errors.append(f"decoder_policy.{field} must be true")
    accept_requires = _mapping(decoder.get("accept_requires"), "decoder_policy.accept_requires", errors)
    for field in ("checksum_match", "wrong_key_reject", "wrong_payload_reject"):
        if accept_requires.get(field) is not True:
            errors.append(f"decoder_policy.accept_requires.{field} must be true")
    if float(accept_requires.get("min_specificity_margin", 0.0)) < 3.0:
        errors.append("decoder_policy.accept_requires.min_specificity_margin must be >= 3.0")
    if decoder.get("threshold_changes_after_generation_allowed") is not False:
        errors.append("decoder_policy.threshold_changes_after_generation_allowed must be false")

    structural = _mapping(contract.get("structural_leakage_policy"), "structural_leakage_policy", errors)
    if float(structural.get("shallow_auc_max", 1.0)) > 0.60:
        errors.append("structural_leakage_policy.shallow_auc_max must be <= 0.60")
    for field in (
        "bullet_count_used_by_decoder",
        "line_count_used_by_decoder",
        "heading_count_used_by_decoder",
        "next_action_label_used_by_decoder",
        "repeated_template_used_by_decoder",
    ):
        if structural.get(field) is not False:
            errors.append(f"structural_leakage_policy.{field} must be false")

    dev_gate = _mapping(contract.get("pre_registered_dev_gate"), "pre_registered_dev_gate", errors)
    if dev_gate.get("primary_scrub_mode") != "all":
        errors.append("pre_registered_dev_gate.primary_scrub_mode must be all")
    if _list_set(dev_gate.get("conditions")) != REQUIRED_CONDITIONS:
        errors.append("pre_registered_dev_gate.conditions mismatch")
    protected = _mapping(dev_gate.get("protected"), "pre_registered_dev_gate.protected", errors)
    if int(protected.get("accepts_min", 0)) < 26 or int(protected.get("blocks", 0)) != 32:
        errors.append("pre_registered_dev_gate.protected must require at least 26/32 accepts")
    controls = _mapping(dev_gate.get("controls"), "pre_registered_dev_gate.controls", errors)
    for field in ("accepts_max_per_condition", "wrong_key_accepts_max", "wrong_payload_accepts_max"):
        if int(controls.get(field, -1)) != 0:
            errors.append(f"pre_registered_dev_gate.controls.{field} must be 0")

    future = _mapping(contract.get("future_compute_route_prerequisites"), "future_compute_route_prerequisites", errors)
    for field in (
        "static_contract_validation_pass",
        "toy_decoder_unit_tests_pass",
        "route_decision_recorded",
        "local_remote_hash_preflight_pass",
        "zero_enabled_allowlist_preflight_pass",
        "hermes_tg_email_notification_pass",
        "exactly_one_allowlist_entry_enabled",
        "immediate_allowlist_disablement_after_sbatch",
    ):
        if future.get(field) is not True:
            errors.append(f"future_compute_route_prerequisites.{field} must be true")

    status = (
        "PASS_R4_POSITIVE_EVIDENCE_CONTRACT_STATIC_VALIDATION_NO_COMPUTE"
        if not errors
        else "FAIL_R4_POSITIVE_EVIDENCE_CONTRACT_STATIC_VALIDATION_NO_COMPUTE"
    )
    return {
        "contract_id": contract.get("contract_id"),
        "current_compute_unlocked": False,
        "errors": errors,
        "generation_started": False,
        "model_scoring_started": False,
        "source_failure_analysis_exists": source_failure.exists(),
        "source_review_exists": source_review.exists(),
        "status": status,
        "training_started": False,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 positive evidence contract redesign without compute.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_contract(load_yaml(args.config))
    if args.output_dir is not None:
        write_json(args.output_dir / "contract_static_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
