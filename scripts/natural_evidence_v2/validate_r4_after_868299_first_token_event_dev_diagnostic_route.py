from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import read_json, sha256_file  # noqa: E402
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_868299_first_token_event_dev_diagnostic_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_868299_first_token_event_dev_diagnostic_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_868299_first_token_event_dev_diagnostic_h200.sbatch"
EXPECTED_COMMAND_PATTERN = f"PLAN_ONLY=0 VALIDATE_PLAN_ONLY=0 sbatch {EXPECTED_WRAPPER}"


def mapping(value: Any, field: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be a mapping")
        return {}
    return value


def root_path(value: Any, field: str, errors: list[str]) -> Path:
    path = ROOT / str(value)
    if not path.exists():
        errors.append(f"{field} missing: {path}")
    return path


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


def validate_route(
    config: Mapping[str, Any],
    *,
    allow_submission_enabled_entry: bool = False,
    skip_allowlist_state_check: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_after_868299_first_token_event_dev_diagnostic_route_v1":
        errors.append("schema_name mismatch")
    if config.get("route_id") != "r4_after_868299_first_token_event_dev_diagnostic_v1":
        errors.append("route_id mismatch")
    if config.get("phase") != "V2_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_868299_PASSED_REVIEWED_NEXT_DEV_ROUTE":
        errors.append("phase mismatch")

    source = mapping(config.get("source_artifacts"), "source_artifacts", errors)
    quality_review_path = root_path(source.get("quality_review", ""), "source_artifacts.quality_review", errors)
    plan_path = root_path(source.get("dev_allocation_plan", ""), "source_artifacts.dev_allocation_plan", errors)
    manifest_path = root_path(source.get("dev_allocation_manifest", ""), "source_artifacts.dev_allocation_manifest", errors)
    allocation_rows_path = root_path(source.get("dev_allocation_rows", ""), "source_artifacts.dev_allocation_rows", errors)
    quality_review = read_json(quality_review_path) if quality_review_path.exists() else {}
    plan = read_json(plan_path) if plan_path.exists() else {}
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    if source.get("source_job_id") != "868299":
        errors.append("source_job_id must be 868299")
    if quality_review.get("status") != "PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_FIRST_TOKEN_EVENT_GATE":
        errors.append("quality review must be the reviewed 868299 first-token event pass")
    if int(quality_review.get("protected_strict_accepts", -1)) != 4:
        errors.append("quality review protected strict accepts must be 4/4")
    if int(quality_review.get("protected_accepts_ignoring_quality", -1)) != 4:
        errors.append("quality review protected ignoring-quality accepts must be 4/4")
    controls = quality_review.get("control_accepts", {})
    if not isinstance(controls, Mapping) or any(int(controls.get(arm, -1)) != 0 for arm in ("raw", "task_only", "wrong_key", "wrong_payload")):
        errors.append("quality review controls must be zero")
    if int(quality_review.get("global_duplicate_response_hash_count", -1)) != 0:
        errors.append("quality review must have zero duplicate response hashes")
    if int(quality_review.get("trace_binding_invalid_rows", -1)) != 0:
        errors.append("quality review trace binding invalid rows must be zero")

    if plan.get("status") != "PASS_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_PLAN_NO_SUBMIT":
        errors.append("dev diagnostic allocation plan must pass")
    if manifest.get("status") != "PASS_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_ALLOCATION_PLAN_NO_SUBMIT":
        errors.append("allocation manifest must pass")
    for field, expected in (
        ("target_shards", 32),
        ("rows_per_shard", 1024),
        ("total_rows", 32768),
        ("selected_coordinate_count", 16),
    ):
        if int(manifest.get(field, -1)) != expected:
            errors.append(f"allocation manifest {field} must be {expected}")
    if manifest.get("allocation_policy") != "cyclic_reuse_reviewed_4_block_full16_allocation":
        errors.append("allocation policy mismatch")
    if int(manifest.get("within_shard_duplicate_prompt_prefix_pair_count_max", -1)) != 0:
        errors.append("within-shard prompt/prefix duplicates must be zero")
    if manifest.get("global_response_hash_duplicate_gate_remains_zero") is not True:
        errors.append("global response hash duplicate gate must remain zero")
    if allocation_rows_path.exists():
        row_count = sum(1 for line in allocation_rows_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if row_count != 32768:
            errors.append(f"allocation rows must contain 32768 rows, found {row_count}")

    policy = mapping(config.get("policy_artifacts"), "policy_artifacts", errors)
    for field in (
        "duplicate_safe_generation_policy_v2",
        "contextual_forbidden_policy_v2",
        "trace_binding_verifier",
        "first_token_event_decoder",
        "full_phrase_decoder",
    ):
        root_path(policy.get(field, ""), f"policy_artifacts.{field}", errors)

    scope = mapping(config.get("generation_scope"), "generation_scope", errors)
    for field, expected in (
        ("blocks", 32),
        ("shards", 32),
        ("row_cylinders_per_block", 1024),
        ("rows_per_coordinate_per_block", 64),
        ("unique_prompt_indices_per_block", 256),
        ("selected_coordinate_count", 16),
        ("expected_generated_rows", 98304),
        ("expected_attempt_rows_min", 98304),
    ):
        if int(scope.get(field, -1)) != expected:
            errors.append(f"generation_scope.{field} must be {expected}")
    if scope.get("conditions") != ["protected", "raw", "task_only"]:
        errors.append("generation conditions mismatch")
    if scope.get("decode_conditions") != ["protected", "raw", "task_only", "wrong_key", "wrong_payload"]:
        errors.append("decode conditions mismatch")
    for field in ("same_contract_only",):
        if scope.get(field) is not True:
            errors.append(f"generation_scope.{field} must be true")
    for field in ("payload_diversity_tested", "llama_tested", "locked_scale_claim", "paper_facing"):
        if scope.get(field) is not False:
            errors.append(f"generation_scope.{field} must be false")
    if scope.get("contract_id") != "a55e":
        errors.append("contract_id must be a55e")

    allocation_policy = mapping(config.get("allocation_policy"), "allocation_policy", errors)
    if allocation_policy.get("reuse_scope") != "dev_diagnostic_only_not_locked_scale":
        errors.append("allocation reuse scope must be dev diagnostic only")
    if allocation_policy.get("global_prompt_prefix_reuse_expected") is not True:
        errors.append("allocation must explicitly record global prompt/prefix reuse")
    if allocation_policy.get("global_response_hash_duplicate_gate_remains_zero") is not True:
        errors.append("allocation must keep response duplicate gate at zero")

    gates = mapping(config.get("quality_gates"), "quality_gates", errors)
    for field, expected in (
        ("protected_strict_accepts_min", 28),
        ("protected_accepts_ignoring_quality_min", 30),
        ("control_accepts_max_per_condition", 0),
        ("within_block_duplicate_response_hash_count_max", 0),
        ("global_duplicate_response_hash_count_max", 0),
        ("technical_forbidden_public_surface_count_max", 0),
        ("ambiguous_forbidden_surface_count_max", 0),
    ):
        if int(gates.get(field, -1)) != expected:
            errors.append(f"quality_gates.{field} must be {expected}")
    if gates.get("ordinary_domain_literal_policy") != "report_only":
        errors.append("ordinary domain literals must be report_only")
    if float(gates.get("trace_binding_validity_required", -1.0)) != 1.0:
        errors.append("trace binding validity must be 100%")
    if gates.get("full_phrase_decoder_policy") != "report_only_not_success_claim":
        errors.append("full phrase decoder must be report-only")
    if gates.get("post_generation_template_leakage_review_required") is not True:
        errors.append("post-generation template leakage review must be required")
    if float(gates.get("protected_vs_raw_template_auc_max", -1.0)) != 0.60:
        errors.append("template AUC gate must be 0.60")

    controller = mapping(config.get("controller"), "controller", errors)
    if controller.get("source_job_id") != 868016:
        errors.append("controller source_job_id must remain 868016")
    if float(controller.get("bonus_nats", -1)) != 4.0:
        errors.append("controller bonus mismatch")
    if float(controller.get("penalty_nats", -1)) != 0.5:
        errors.append("controller penalty mismatch")
    if float(controller.get("max_target_mass", -1)) != 0.5:
        errors.append("controller max target mass mismatch")
    if float(controller.get("max_kl_budget", -1)) != 0.5:
        errors.append("controller max KL budget mismatch")

    compute = mapping(config.get("compute_policy"), "compute_policy", errors)
    wrapper = root_path(compute.get("wrapper", ""), "compute_policy.wrapper", errors)
    if compute.get("allowlist_entry") != EXPECTED_ENTRY:
        errors.append("allowlist entry mismatch")
    if compute.get("wrapper") != EXPECTED_WRAPPER:
        errors.append("wrapper mismatch")
    if compute.get("command_pattern") != EXPECTED_COMMAND_PATTERN:
        errors.append("command pattern mismatch")
    for field, expected in (
        ("partition", "pomplun"),
        ("qos", "pomplun"),
        ("account", "cs_yinxin.wan"),
        ("gres", "gpu:h200:1"),
        ("max_time", "30-00:00:00"),
        ("array", "0-31%4"),
    ):
        if compute.get(field) != expected:
            errors.append(f"compute_policy.{field} mismatch")
    if compute.get("allowlist_enabled_now") is not False:
        errors.append("allowlist_enabled_now must be false")
    if compute.get("slurm_allowed_now") is not False:
        errors.append("slurm_allowed_now must be false")
    if compute.get("full_mode_wrapper_requires_separate_submission_preflight") is not True:
        errors.append("full mode wrapper must require separate submission preflight")
    if wrapper.exists():
        text = wrapper.read_text(encoding="utf-8")
        for fragment in (
            "#SBATCH --array=0-31%4",
            "#SBATCH --partition=pomplun",
            "#SBATCH --account=cs_yinxin.wan",
            "#SBATCH --qos=pomplun",
            "#SBATCH --gres=gpu:h200:1",
            "#SBATCH --time=30-00:00:00",
            "MAX_SHARD_INDEX=31",
            "r4_after_868016_controller_generation_h200.sbatch",
            "r4_after_868299_first_token_event_dev_diagnostic",
        ):
            if fragment not in text:
                errors.append(f"wrapper missing fragment: {fragment}")

    allowlist = load_yaml(ALLOWLIST)
    enabled_entries = enabled_allowlist_entries(allowlist)
    if skip_allowlist_state_check:
        # Runtime array tasks can start before or after the required immediate
        # post-sbatch allowlist disablement. Submission preflights still perform
        # the strict enabled-entry checks; runtime only verifies the entry exists
        # and still points at the reviewed command.
        pass
    elif allow_submission_enabled_entry:
        if enabled_entries != [EXPECTED_ENTRY]:
            errors.append(f"allowlist enabled entries must be exactly [{EXPECTED_ENTRY!r}]: {enabled_entries}")
    elif enabled_entries:
        errors.append(f"allowlist enabled entries must be empty during plan validation: {enabled_entries}")
    entry = find_allowlist_entry(allowlist, EXPECTED_ENTRY)
    if entry is None:
        errors.append("allowlist entry missing")
    else:
        if not skip_allowlist_state_check:
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
        "PASS_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_ROUTE_PLAN_ONLY_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_ROUTE_PLAN_ONLY_NO_SUBMIT"
    )
    return {
        "schema_name": "natural_evidence_v2_r4_after_868299_first_token_event_dev_diagnostic_route_validation_v1",
        "status": status,
        "errors": errors,
        "source_job_id": "868299",
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
        "config_sha256": sha256_file(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else "",
        "allocation_rows_sha256": sha256_file(allocation_rows_path) if allocation_rows_path.exists() else "",
        "allow_submission_enabled_entry": bool(allow_submission_enabled_entry),
        "skip_allowlist_state_check": bool(skip_allowlist_state_check),
        "slurm_allowed": False,
        "generation_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": "Run local/remote hash preflight before exactly-one H200 submission.",
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 after-868299 first-token event 32-block dev route.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-submission-enabled-entry", action="store_true")
    parser.add_argument("--skip-allowlist-state-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    summary = validate_route(
        load_yaml(config_path),
        allow_submission_enabled_entry=bool(args.allow_submission_enabled_entry),
        skip_allowlist_state_check=bool(args.skip_allowlist_state_check),
    )
    if args.output_dir is not None:
        output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_new(output_dir / "route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
