from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.common import read_yaml, resolve_repo_path, write_json


EXPECTED_ARMS = {
    ("qwen2.5", "protected_trained"),
    ("qwen2.5", "raw"),
    ("llama3.1", "protected_trained"),
    ("llama3.1", "raw"),
}

EXPECTED_TASK_ONLY_ARMS = {
    ("qwen2.5", "task_only_lora"),
    ("llama3.1", "task_only_lora"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU/static validation for natural_evidence_v1.")
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument(
        "--summary",
        default="results/natural_evidence_v1/static_validation_summary.json",
    )
    return parser.parse_args(argv)


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def _validate_arms(config: dict[str, Any], errors: list[str]) -> None:
    arms = config.get("arms", [])
    if not isinstance(arms, list):
        _error(errors, "arms must be a list")
        return
    observed = set()
    for arm in arms:
        if not isinstance(arm, dict):
            _error(errors, "each arm must be a mapping")
            continue
        family = str(arm.get("model_family", ""))
        condition = str(arm.get("model_condition", ""))
        tokenizer = str(arm.get("tokenizer", ""))
        arm_id = str(arm.get("arm_id", ""))
        observed.add((family, condition))
        lowered = " ".join([family, condition, tokenizer, arm_id]).lower()
        if "gpt2" in lowered:
            _error(errors, f"paper-facing arm must not use gpt2: {arm_id}")
    missing = sorted(EXPECTED_ARMS - observed)
    missing_task_only = sorted(EXPECTED_TASK_ONLY_ARMS - observed)
    if missing:
        _error(errors, f"missing required four-arm cells: {missing}")
    if missing_task_only:
        _error(errors, f"missing required task-only LoRA null arms: {missing_task_only}")


def _validate_bucket_bank(config: dict[str, Any], errors: list[str]) -> None:
    bucket_cfg = config.get("bucket_bank", {})
    if not isinstance(bucket_cfg, dict):
        _error(errors, "bucket_bank must be a mapping")
        return
    if int(bucket_cfg.get("target_bank_entries_per_tokenizer", 0)) <= 0:
        _error(errors, "target_bank_entries_per_tokenizer must be positive")
    if int(bucket_cfg.get("bucket_count", 0)) <= 1:
        _error(errors, "bucket_count must be greater than 1")
    if int(bucket_cfg.get("candidate_top_k", 0)) < int(bucket_cfg.get("bucket_count", 0)):
        _error(errors, "candidate_top_k must be at least bucket_count")
    if str(bucket_cfg.get("bucket_assignment", "")) not in {"keyed_mass_balance", "keyed_hash_round_robin"}:
        _error(errors, "bucket_assignment must be keyed_mass_balance or keyed_hash_round_robin")
    if int(bucket_cfg.get("min_members_per_bucket", 0)) < 1:
        _error(errors, "min_members_per_bucket must be positive")
    quality_gates = bucket_cfg.get("quality_gates", {})
    if not isinstance(quality_gates, dict):
        _error(errors, "bucket_bank.quality_gates must be a mapping")
    else:
        if not quality_gates.get("train_gate_required", False):
            _error(errors, "bucket_bank.quality_gates.train_gate_required must be true")
        if quality_gates.get("raw_entry_count_is_training_gate", True):
            _error(errors, "raw static opportunity count must not be the training gate")
        if float(quality_gates.get("reconstructability_rate_min", 0.0)) <= 0.0:
            _error(errors, "bucket_bank.quality_gates.reconstructability_rate_min must be positive")
        ablations = quality_gates.get("bucket_count_ablation_required", [])
        if not isinstance(ablations, list) or {4, 8} - {int(value) for value in ablations}:
            _error(errors, "bucket_count_ablation_required must include 4 and 8")
    capacity_cfg = bucket_cfg.get("compatibility_adjusted_capacity", {})
    if not isinstance(capacity_cfg, dict):
        _error(errors, "bucket_bank.compatibility_adjusted_capacity must be a mapping")
    else:
        if capacity_cfg.get("raw_entry_count_is_training_gate", True):
            _error(errors, "compatibility-adjusted capacity must replace raw entry count as the training gate")
        diagnostic = capacity_cfg.get("diagnostic_high_risk_gate", {})
        if not isinstance(diagnostic, dict):
            _error(errors, "diagnostic_high_risk_gate must be a mapping")
        else:
            if diagnostic.get("paper_claim_allowed", True):
                _error(errors, "diagnostic_high_risk_gate.paper_claim_allowed must be false")
            if float(diagnostic.get("heldout_eligible_positions_per_100_tokens_min", 0.0)) <= 0.0:
                _error(errors, "diagnostic high-risk gate requires positive held-out density")
            if float(diagnostic.get("effective_compatible_bits_per_response_min", 0.0)) <= 0.0:
                _error(errors, "diagnostic high-risk gate requires positive effective bits per response")
        viability = capacity_cfg.get("qwen_e2e_viability_gate", {})
        if not isinstance(viability, dict):
            _error(errors, "qwen_e2e_viability_gate must be a mapping")
        else:
            if str(viability.get("gate_role", "")) != "paper_ready_minimum":
                _error(errors, "qwen_e2e_viability_gate.gate_role must be paper_ready_minimum")
            if int(viability.get("min1_compatible_entries_min", 0)) <= 0:
                _error(errors, "qwen viability gate requires positive min1-compatible entries")
            if int(viability.get("fully_compatible_min2_entries_min", 0)) <= 0:
                _error(errors, "qwen viability gate requires positive min2-compatible entries")
            if float(viability.get("effective_compatible_bits_per_response_min", 0.0)) <= 0.0:
                _error(errors, "qwen viability gate requires positive effective compatible bits per response")
            if isinstance(diagnostic, dict) and diagnostic:
                diagnostic_bits = float(diagnostic.get("effective_compatible_bits_per_response_min", 0.0))
                paper_bits = float(viability.get("effective_compatible_bits_per_response_min", 0.0))
                if diagnostic_bits >= paper_bits:
                    _error(errors, "diagnostic effective-bits gate must remain below paper-ready gate")
    sweep_cfg = bucket_cfg.get("compatibility_sweep", {})
    if not isinstance(sweep_cfg, dict):
        _error(errors, "bucket_bank.compatibility_sweep must be a mapping")
    else:
        if not {2, 4, 8}.issubset({int(value) for value in sweep_cfg.get("bucket_counts", [])}):
            _error(errors, "compatibility_sweep.bucket_counts must include 2, 4, and 8")
        per_token_thresholds = {float(value) for value in sweep_cfg.get("delta_nll_per_token_thresholds", [])}
        if not {0.05, 0.1, 0.2, 0.3}.issubset(per_token_thresholds):
            _error(errors, "compatibility_sweep.delta_nll_per_token_thresholds must include 0.05, 0.1, 0.2, and 0.3")
        raw_thresholds = {float(value) for value in sweep_cfg.get("delta_nll_raw_thresholds", [])}
        if not {0.5, 1.0, 1.5, 2.0}.issubset(raw_thresholds):
            _error(errors, "compatibility_sweep.delta_nll_raw_thresholds must include 0.5, 1.0, 1.5, and 2.0")
    reference_candidates = bucket_cfg.get("reference_candidates", {})
    if not isinstance(reference_candidates, dict):
        _error(errors, "bucket_bank.reference_candidates must be a mapping")
    else:
        for key in ("qwen", "llama"):
            if key not in reference_candidates:
                _error(errors, f"missing reference candidate path for {key}")


def _validate_protocol(config: dict[str, Any], errors: list[str]) -> None:
    protocol = config.get("protocol", {})
    if not isinstance(protocol, dict):
        _error(errors, "protocol must be a mapping")
        return
    if protocol.get("id") != "natural_evidence_v1":
        _error(errors, "protocol.id must be natural_evidence_v1")
    forbidden = {str(item).upper() for item in protocol.get("forbidden_surface_patterns", [])}
    for required in ("FIELD=", "SECTION=", "TOPIC=", "PAYLOAD", "CERT"):
        if required not in forbidden:
            _error(errors, f"missing forbidden surface pattern {required!r}")
    if "hidden_until_transcript_commitment" not in str(protocol.get("audit_key_status", "")):
        _error(errors, "audit_key_status must encode commit-then-reveal")
    commitment = protocol.get("commitment", {})
    if not isinstance(commitment, dict) or not commitment.get("required", False):
        _error(errors, "protocol.commitment.required must be true")
        return
    for field in (
        "protocol_id",
        "audit_key_commitment",
        "payload_commitment",
        "bucket_policy_commitment",
        "query_budget",
        "probe_selection_rule",
    ):
        if field not in [str(item) for item in commitment.get("pre_audit_fields", [])]:
            _error(errors, f"protocol.commitment.pre_audit_fields missing {field!r}")
    if not commitment.get("transcript_freeze_before_key_reveal", False):
        _error(errors, "transcript_freeze_before_key_reveal must be true")
    if not commitment.get("forbid_post_hoc_key_search", False):
        _error(errors, "forbid_post_hoc_key_search must be true")
    if not commitment.get("multiple_testing_accounting_required", False):
        _error(errors, "multiple_testing_accounting_required must be true")


def _validate_nulls_and_attacks(config: dict[str, Any], errors: list[str]) -> None:
    nulls = config.get("null_evaluations", {})
    required_nulls = set(str(item) for item in nulls.get("required", [])) if isinstance(nulls, dict) else set()
    for null_name in (
        "raw_exact_model",
        "task_only_lora",
        "wrong_key",
        "wrong_payload",
        "same_family_raw_near_null",
        "organic_prompts",
        "non_owner_prompts",
    ):
        if null_name not in required_nulls:
            _error(errors, f"missing required null evaluation {null_name!r}")
    attacks = set(str(item) for item in config.get("attacks", []))
    for attack_name in (
        "generic_paraphrase",
        "style_normalization",
        "compression_summarization",
        "low_temperature_regeneration",
        "deterministic_rewrite",
        "public_surface_scrub",
        "oracle_keyed_sanitizer",
    ):
        if attack_name not in attacks:
            _error(errors, f"missing required sanitizer attack {attack_name!r}")
    e2e = config.get("end_to_end_pilot", {})
    if not isinstance(e2e, dict):
        _error(errors, "end_to_end_pilot must be a mapping")
        return
    if not e2e.get("diagnostic_high_risk_allowed", False):
        _error(errors, "end_to_end_pilot.diagnostic_high_risk_allowed must be true after expert decision")
    if str(e2e.get("diagnostic_high_risk_claim_status", "")) != "forbidden_from_paper_claims":
        _error(errors, "diagnostic high-risk pilot must remain forbidden from paper claims")
    if not e2e.get("paper_ready_density_gate_remains_required", False):
        _error(errors, "paper-ready density gate must remain required")
    conditions = set(str(item) for item in e2e.get("required_conditions", []))
    for condition in ("protected_trained", "raw", "task_only_lora", "wrong_key", "wrong_payload"):
        if condition not in conditions:
            _error(errors, f"end_to_end_pilot.required_conditions missing {condition!r}")
    diagnostic_conditions = set(str(item) for item in e2e.get("diagnostic_required_conditions", []))
    for condition in ("protected_trained", "raw", "task_only_lora", "wrong_key", "wrong_payload"):
        if condition not in diagnostic_conditions:
            _error(errors, f"end_to_end_pilot.diagnostic_required_conditions missing {condition!r}")
    if int(e2e.get("primary_bucket_count", 0)) != 4:
        _error(errors, "end_to_end_pilot.primary_bucket_count must be 4 for the first natural pilot")
    diagnostic_budgets = [int(value) for value in e2e.get("diagnostic_query_budgets", [])]
    if diagnostic_budgets != [64, 128, 256, 512]:
        _error(errors, "diagnostic_query_budgets must be [64, 128, 256, 512]")


def _validate_tables(config: dict[str, Any], errors: list[str]) -> None:
    tables = config.get("tables", {})
    if not isinstance(tables, dict):
        _error(errors, "tables must be a mapping")
        return
    required_columns = set(str(item) for item in tables.get("required_columns", []))
    for column in (
        "model_family",
        "model_condition",
        "tokenizer",
        "bucket_bank_id",
        "payload_id",
        "seed",
        "query_budget",
        "accepted",
        "recovered_payload",
        "protocol_id",
    ):
        if column not in required_columns:
            _error(errors, f"missing required table column {column!r}")
    outputs = tables.get("outputs", {})
    if isinstance(outputs, dict):
        for output_name in (
            "bucket_bank_coverage_by_split",
            "bucket_bank_balance",
            "natural_channel_capacity",
            "reconstructability_report",
        ):
            if output_name not in outputs:
                _error(errors, f"missing required table output {output_name!r}")


def _validate_run_allowlist(root: Path, errors: list[str]) -> None:
    allowlist_path = root / "configs/natural_evidence_v1/run_allowlist.yaml"
    if not allowlist_path.exists():
        _error(errors, "missing configs/natural_evidence_v1/run_allowlist.yaml")
        return
    allowlist = read_yaml(allowlist_path)
    global_rules = allowlist.get("global_rules", {})
    if not isinstance(global_rules, dict):
        _error(errors, "run_allowlist.global_rules must be a mapping")
        return
    if int(global_rules.get("max_state_changing_actions_per_automation_run", 0)) != 1:
        _error(errors, "automation must allow exactly one state-changing action per run")
    for rule in (
        "forbid_overwrite_existing_artifacts",
        "forbid_old_compiled_path_modification",
        "forbid_unlisted_gpu_jobs",
        "forbid_paper_claim_modification",
        "require_chimera_mail_notifications",
    ):
        if not global_rules.get(rule, False):
            _error(errors, f"run_allowlist.global_rules.{rule} must be true")
    if global_rules.get("require_chimera_mail_notifications", False):
        slurm_scripts = sorted((root / "scripts/natural_evidence_v1").rglob("*.sbatch"))
        if not slurm_scripts:
            _error(errors, "no natural_evidence_v1 sbatch scripts found for mail notification check")
        for slurm_script in slurm_scripts:
            script_text = slurm_script.read_text(encoding="utf-8")
            if "#SBATCH --mail-type=ALL" not in script_text:
                _error(errors, f"{slurm_script.relative_to(root)} missing #SBATCH --mail-type=ALL")
            if "#SBATCH --mail-user=guanjie.lin001@umb.edu" not in script_text:
                _error(errors, f"{slurm_script.relative_to(root)} missing Chimera mail user directive")

    cpu_actions = allowlist.get("allowed_cpu_actions", [])
    cpu_names = {str(action.get("name", "")) for action in cpu_actions if isinstance(action, dict)}
    for action_name in ("rebuild_qwen_4way_clean_bank", "audit_opportunity_bank"):
        if action_name not in cpu_names:
            _error(errors, f"run_allowlist missing CPU action {action_name!r}")

    gpu_actions = allowlist.get("allowed_gpu_actions", [])
    if not isinstance(gpu_actions, list):
        _error(errors, "run_allowlist.allowed_gpu_actions must be a list")
        return
    allowed_enabled_gpu_actions = {
        "qwen_diagnostic_high_risk_e2e_pilot": (
            "explicit_diagnostic_gate_passed_after_raw_wrong_key_pre_null_invalid_suffix_and_train_dataset_preflight"
        ),
        "qwen_diagnostic_e2e_eval": "qwen_diagnostic_training_complete_pending_eval",
        "qwen_actual_prefix_reference_model_scoring": (
            "qwen_actual_prefix_scoring_plan_complete_pending_gpu_reference_scoring"
        ),
        "qwen_actual_prefix_suffix_compatibility": (
            "qwen_actual_prefix_bucketization_complete_pending_suffix_compatibility"
        ),
        "qwen_actual_prefix_suffix_highcap_sensitivity": (
            "qwen_actual_prefix_missing_bucket_repair_plan_ready"
        ),
        "qwen_expanded_actual_prefix_suffix_compatibility": (
            "qwen_compatibility_aware_supply_upper_bound_passed"
        ),
        "qwen_846699_repaired_teacher_forced_target_mass_probe": (
            "post_846699_repaired_teacher_forced_target_mass_probe_ready"
        ),
    }
    enabled_gpu_actions: list[str] = []
    for action in gpu_actions:
        if not isinstance(action, dict):
            _error(errors, "each GPU allowlist action must be a mapping")
            continue
        if action.get("enabled", False):
            action_name = str(action.get("name", ""))
            enabled_gpu_actions.append(action_name)
            if action_name not in allowed_enabled_gpu_actions:
                _error(errors, f"GPU action must remain disabled until manually enabled: {action.get('name', '')}")
                continue
            expected_condition = allowed_enabled_gpu_actions[action_name]
            if action.get("enable_condition") != expected_condition:
                _error(errors, f"{action_name} enabled with unexpected enable_condition")
    if len(enabled_gpu_actions) > 1:
        _error(errors, f"at most one GPU action may be enabled at a time: {enabled_gpu_actions}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = resolve_repo_path(args.config, root)
    config = read_yaml(config_path)
    errors: list[str] = []
    _validate_protocol(config, errors)
    _validate_arms(config, errors)
    _validate_bucket_bank(config, errors)
    _validate_nulls_and_attacks(config, errors)
    _validate_tables(config, errors)
    _validate_run_allowlist(root, errors)

    summary = {
        "schema_name": "natural_evidence_static_validation_summary_v1",
        "config": str(config_path.relative_to(root)) if config_path.is_relative_to(root) else str(config_path),
        "passed": not errors,
        "errors": errors,
        "validated_components": [
            "protocol_commit_then_reveal",
            "four_arm_matrix",
            "bucket_bank_scale_config",
            "opportunity_bank_quality_gates",
            "protocol_precommitment",
            "task_only_and_near_null_controls",
            "sanitizer_benchmark_requirements",
            "automation_run_allowlist",
            "chimera_mail_notifications",
            "paper_table_required_columns",
            "no_gpt2_paper_facing_arm",
        ],
    }
    summary_path = resolve_repo_path(args.summary, root)
    write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
