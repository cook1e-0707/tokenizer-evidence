from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import sha256_file  # noqa: E402
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_868260_quality_repair_confirmation_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_868260_quality_repair_confirmation_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_868260_quality_repair_confirmation_h200.sbatch"
EXPECTED_COMMAND_PATTERN = f"PLAN_ONLY=0 VALIDATE_PLAN_ONLY=0 sbatch {EXPECTED_WRAPPER}"


def mapping(value: Any, field: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be a mapping")
        return {}
    return value


def path_from(value: Any, field: str, errors: list[str]) -> Path:
    path = ROOT / str(value)
    if not path.exists():
        errors.append(f"{field} missing: {path}")
    return path


def read_json(path: Path, field: str, errors: list[str]) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{field} unreadable JSON: {exc}")
        return {}
    if not isinstance(payload, Mapping):
        errors.append(f"{field} must be a JSON object")
        return {}
    return payload


def enabled_allowlist_entries(allowlist: Mapping[str, Any]) -> list[str]:
    enabled: list[str] = []
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        entries = allowlist.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("enabled") is True:
                enabled.append(str(entry.get("name", "")))
    return enabled


def find_allowlist_entry(allowlist: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        entries = allowlist.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("name") == name:
                return entry
    return None


def validate_route(config: Mapping[str, Any], *, allow_submission_enabled_entry: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_after_868260_quality_repair_confirmation_route_v1":
        errors.append("schema_name mismatch")
    if config.get("route_id") != "r4_after_868260_quality_repair_confirmation_v1":
        errors.append("route_id mismatch")
    if config.get("phase") != "V2_R4_AFTER_868260_FORENSICS_POLICY_TRACE_BINDING_ARTIFACTS_VALIDATED_NO_SUBMIT":
        errors.append("phase mismatch")

    source = mapping(config.get("source_artifacts"), "source_artifacts", errors)
    state_sync = read_json(path_from(source.get("state_sync", ""), "source_artifacts.state_sync", errors), "state_sync", errors)
    failure = read_json(path_from(source.get("failure_analysis", ""), "source_artifacts.failure_analysis", errors), "failure_analysis", errors)
    duplicate_forensics = read_json(
        path_from(source.get("duplicate_forensics", ""), "source_artifacts.duplicate_forensics", errors),
        "duplicate_forensics",
        errors,
    )
    repair_validation = read_json(
        path_from(source.get("repair_package_validation", ""), "source_artifacts.repair_package_validation", errors),
        "repair_package_validation",
        errors,
    )
    duplicate_policy_validation = read_json(
        path_from(source.get("duplicate_safe_policy_validation", ""), "source_artifacts.duplicate_safe_policy_validation", errors),
        "duplicate_safe_policy_validation",
        errors,
    )
    contextual_validation = read_json(
        path_from(source.get("contextual_forbidden_validation", ""), "source_artifacts.contextual_forbidden_validation", errors),
        "contextual_forbidden_validation",
        errors,
    )
    trace_validation = read_json(
        path_from(source.get("trace_binding_validation", ""), "source_artifacts.trace_binding_validation", errors),
        "trace_binding_validation",
        errors,
    )

    if source.get("source_job_id") != "868260":
        errors.append("source_job_id must be 868260")
    if state_sync.get("reclassifies_868260") is not False:
        errors.append("state sync must not reclassify 868260")
    if failure.get("strict_protected_accepts") != 2:
        errors.append("failure strict protected accepts must remain 2/4")
    if failure.get("protected_accepts_ignoring_quality") != 4:
        errors.append("failure ignoring-quality protected accepts must remain 4/4")
    controls = failure.get("control_accepts", {})
    if not isinstance(controls, Mapping) or any(int(controls.get(arm, -1)) != 0 for arm in ("raw", "task_only", "wrong_key", "wrong_payload")):
        errors.append("failure controls must remain zero")
    if int(duplicate_forensics.get("global_duplicate_extra_rows", -1)) != 7612:
        errors.append("duplicate forensics must preserve 868260 global duplicate count")
    if duplicate_forensics.get("status") != "RECORDED_R4_868260_DUPLICATE_FORENSICS_ARTIFACT_ONLY_NO_SUBMIT":
        errors.append("duplicate forensics status mismatch")
    if repair_validation.get("status") != "PASS_R4_AFTER_868260_QUALITY_GATE_REPAIR_PACKAGE_VALIDATION_NO_SUBMIT":
        errors.append("repair package validation must pass")
    if duplicate_policy_validation.get("status") != "PASS_R4_FIRST_TOKEN_EVENT_DUPLICATE_SAFE_GENERATION_POLICY_V2":
        errors.append("duplicate-safe generation policy v2 validation must pass")
    if contextual_validation.get("status") != "PASS_R4_CONTEXTUAL_FORBIDDEN_SURFACE_POLICY_V2_VALIDATION_NO_SUBMIT":
        errors.append("contextual forbidden policy v2 validation must pass")
    if trace_validation.get("status") != "PASS_R4_FIRST_TOKEN_EVENT_TRACE_BINDING_FIXTURE_VALIDATION_NO_SUBMIT":
        errors.append("trace binding fixture validation must pass")

    policy = mapping(config.get("policy_artifacts"), "policy_artifacts", errors)
    duplicate_policy = path_from(policy.get("duplicate_safe_generation_policy_v2", ""), "policy_artifacts.duplicate_safe_generation_policy_v2", errors)
    for field in (
        "duplicate_safe_generation_policy_doc",
        "contextual_forbidden_classifier",
        "trace_binding_spec",
        "trace_binding_verifier",
        "duplicate_forensics_script",
    ):
        path_from(policy.get(field, ""), f"policy_artifacts.{field}", errors)
    if duplicate_policy.exists():
        duplicate_payload = load_yaml(duplicate_policy)
        generation = mapping(duplicate_payload.get("generation"), "duplicate_policy.generation", errors)
        if generation.get("retry_selection_rule") != "first_nonduplicate_exact_hash":
            errors.append("duplicate policy retry_selection_rule mismatch")
        if generation.get("retry_blind_to_decode_accept") is not True:
            errors.append("duplicate policy must be blind to decode accept")
        if generation.get("retry_blind_to_payload_match") is not True:
            errors.append("duplicate policy must be blind to payload match")
        if generation.get("apply_same_policy_to_all_arms") is not True:
            errors.append("duplicate policy must apply to all arms")

    scope = mapping(config.get("generation_scope"), "generation_scope", errors)
    expected_scope = {
        "blocks": 4,
        "shards": 4,
        "prompts_per_block": 64,
        "prompt_indices_per_shard": 64,
        "selected_coordinate_count": 16,
        "rows_per_shard": 1024,
    }
    for field, expected in expected_scope.items():
        if int(scope.get(field, -1)) != expected:
            errors.append(f"generation_scope.{field} must be {expected}")
    if scope.get("conditions") != ["protected", "raw", "task_only"]:
        errors.append("generation conditions mismatch")
    if scope.get("decode_conditions") != ["protected", "raw", "task_only", "wrong_key", "wrong_payload"]:
        errors.append("decode conditions mismatch")
    for field in ("same_contract_only",):
        if scope.get(field) is not True:
            errors.append(f"generation_scope.{field} must be true")
    for field in ("payload_diversity_tested", "llama_tested", "paper_facing"):
        if scope.get(field) is not False:
            errors.append(f"generation_scope.{field} must be false")
    if scope.get("contract_id") != "a55e":
        errors.append("contract_id must be a55e")

    gates = mapping(config.get("quality_gates"), "quality_gates", errors)
    required_zero = (
        "control_accepts_max_per_condition",
        "within_block_duplicate_response_hash_count_max",
        "global_duplicate_response_hash_count_max",
        "duplicate_prompt_prefix_pair_count_max",
        "technical_forbidden_public_surface_count_max",
        "ambiguous_forbidden_surface_count_max",
    )
    for field in required_zero:
        if int(gates.get(field, -1)) != 0:
            errors.append(f"quality_gates.{field} must be 0")
    if int(gates.get("protected_strict_accepts_required", -1)) != 4:
        errors.append("quality gate protected strict accepts must require 4")
    if int(gates.get("protected_accepts_ignoring_quality_required", -1)) != 4:
        errors.append("quality gate protected ignoring-quality accepts must require 4")
    if gates.get("ordinary_domain_literal_policy") != "report_only":
        errors.append("ordinary domain literals must be report_only")
    if gates.get("full_phrase_decoder_policy") != "report_only_not_success_claim":
        errors.append("full phrase decoder must be report-only")
    if float(gates.get("trace_binding_validity_required", -1.0)) != 1.0:
        errors.append("trace binding validity must be 100%")

    controller = mapping(config.get("controller"), "controller", errors)
    if controller.get("source_job_id") != 868016:
        errors.append("controller source_job_id must remain 868016")
    if float(controller.get("bonus_nats", -1)) != 4.0:
        errors.append("controller bonus mismatch")
    if float(controller.get("penalty_nats", -1)) != 0.5:
        errors.append("controller penalty mismatch")
    if float(controller.get("max_target_mass", -1)) != 0.5:
        errors.append("controller max_target_mass mismatch")
    if float(controller.get("max_kl_budget", -1)) != 0.5:
        errors.append("controller max_kl_budget mismatch")

    compute = mapping(config.get("compute_policy"), "compute_policy", errors)
    wrapper = path_from(compute.get("wrapper", ""), "compute_policy.wrapper", errors)
    if compute.get("allowlist_entry") != EXPECTED_ENTRY:
        errors.append("allowlist entry mismatch")
    if compute.get("wrapper") != EXPECTED_WRAPPER:
        errors.append("wrapper mismatch")
    if compute.get("command_pattern") != EXPECTED_COMMAND_PATTERN:
        errors.append("command_pattern mismatch")
    for field, expected in (("partition", "pomplun"), ("qos", "pomplun"), ("account", "cs_yinxin.wan"), ("gres", "gpu:h200:1"), ("max_time", "30-00:00:00")):
        if compute.get(field) != expected:
            errors.append(f"compute_policy.{field} mismatch")
    if compute.get("allowlist_enabled_now") is not False:
        errors.append("allowlist_enabled_now must be false")
    if compute.get("slurm_allowed_now") is not False:
        errors.append("slurm_allowed_now must be false")
    if compute.get("plan_only_wrapper_required_now") is not False:
        errors.append("plan_only_wrapper_required_now must be false after full-mode wrapper review")
    if compute.get("full_mode_wrapper_reviewed_now") is not True:
        errors.append("full_mode_wrapper_reviewed_now must be true")
    if compute.get("full_mode_wrapper_requires_separate_submission_preflight") is not True:
        errors.append("full_mode_wrapper_requires_separate_submission_preflight must be true")
    if wrapper.exists():
        text = wrapper.read_text(encoding="utf-8")
        for fragment in (
            "#SBATCH --partition=pomplun",
            "#SBATCH --account=cs_yinxin.wan",
            "#SBATCH --qos=pomplun",
            "#SBATCH --gres=gpu:h200:1",
            "#SBATCH --time=30-00:00:00",
            "PLAN_ONLY",
            "r4_after_868016_controller_generation_h200.sbatch",
            "DUPLICATE_SAFE_POLICY",
            "CONTEXTUAL_LITERAL_POLICY",
            "verify_r4_first_token_event_trace_binding.py",
            "VALIDATE_PLAN_ONLY",
            "validate_r4_after_868260_quality_repair_confirmation_route.py",
        ):
            if fragment not in text:
                errors.append(f"wrapper missing fragment: {fragment}")

    allowlist = load_yaml(ALLOWLIST)
    enabled_entries = enabled_allowlist_entries(allowlist)
    if allow_submission_enabled_entry:
        if enabled_entries != [EXPECTED_ENTRY]:
            errors.append(
                f"allowlist enabled entries must be exactly [{EXPECTED_ENTRY!r}] during submission preflight: {enabled_entries}"
            )
    elif enabled_entries:
        errors.append(f"allowlist enabled entries must be empty during plan-only validation: {enabled_entries}")
    entry = find_allowlist_entry(allowlist, EXPECTED_ENTRY)
    if entry is None:
        errors.append("allowlist entry missing")
    else:
        expected_enabled = bool(allow_submission_enabled_entry)
        if entry.get("enabled") is not expected_enabled:
            errors.append(f"allowlist entry enabled state must be {expected_enabled}")
        if entry.get("command_pattern") != EXPECTED_COMMAND_PATTERN:
            errors.append("allowlist command_pattern mismatch")

    prerequisites = mapping(config.get("required_before_any_submission"), "required_before_any_submission", errors)
    for field, value in prerequisites.items():
        if value is not True:
            errors.append(f"required_before_any_submission.{field} must be true")
    locked = mapping(config.get("not_unlocked_by_this_route_package"), "not_unlocked_by_this_route_package", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_package.{field} must be true")

    status = (
        "PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_ROUTE_PLAN_ONLY_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_ROUTE_PLAN_ONLY_NO_SUBMIT"
    )
    return {
        "schema_name": "natural_evidence_v2_r4_after_868260_quality_repair_confirmation_route_validation_v1",
        "status": status,
        "errors": errors,
        "source_job_id": "868260",
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
        "config_sha256": sha256_file(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else "",
        "duplicate_policy_sha256": sha256_file(duplicate_policy) if duplicate_policy.exists() else "",
        "allow_submission_enabled_entry": bool(allow_submission_enabled_entry),
        "slurm_allowed": False,
        "generation_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
        "reclassifies_868260": False,
        "next_allowed_action": "Local/remote hash preflight and Hermes notification may follow; do not submit Slurm until a separate single-submission route is recorded.",
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R4 after-868260 quality-repair confirmation route in plan-only mode.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-submission-enabled-entry", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_route(load_yaml(args.config), allow_submission_enabled_entry=bool(args.allow_submission_enabled_entry))
    if args.output_dir is not None:
        write_json_new(args.output_dir / "route_validation_summary.json", summary)
        report = [
            "# R4 After-868260 Quality-Repair Confirmation Route Validation",
            "",
            f"Status: `{summary['status']}`",
            "",
            f"- source job: `{summary['source_job_id']}`",
            f"- wrapper: `{summary['wrapper']}`",
            f"- allowlist entry: `{summary['allowlist_entry']}`",
            f"- slurm allowed: `{summary['slurm_allowed']}`",
            "",
            "Next allowed action:",
            "",
            str(summary["next_allowed_action"]),
        ]
        if summary["errors"]:
            report.extend(["", "## Errors", ""])
            report.extend(f"- {error}" for error in summary["errors"])
        (args.output_dir / "route_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
