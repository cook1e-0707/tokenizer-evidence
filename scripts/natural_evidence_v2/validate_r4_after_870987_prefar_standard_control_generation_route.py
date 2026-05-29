from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import read_json, sha256_file, write_json_new  # noqa: E402
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_870987_prefar_standard_control_generation_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_870987_prefar_standard_control_generation_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_870987_prefar_standard_control_generation_h200.sbatch"
EXPECTED_COMMAND_PATTERN = f"PLAN_ONLY=0 VALIDATE_PLAN_ONLY=0 sbatch {EXPECTED_WRAPPER}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 after-870987 pre-FAR standard-control generation route.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-submission-enabled-entry", action="store_true")
    parser.add_argument("--skip-allowlist-state-check", action="store_true")
    return parser.parse_args()


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


def int_field(data: Mapping[str, Any], field: str, default: int = -1) -> int:
    try:
        return int(data.get(field, default))
    except (TypeError, ValueError):
        return default


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
    if config.get("schema_name") != "natural_evidence_v2_r4_after_870987_prefar_standard_control_generation_route_v1":
        errors.append("schema_name mismatch")
    if config.get("route_id") != "r4_after_870987_prefar_standard_control_generation_v1":
        errors.append("route_id mismatch")
    if config.get("phase") != "V2_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_QWEN_TOKENIZER_PREFLIGHT_PASSED_GENERATION_ROUTE_PLANNING":
        errors.append("phase mismatch")

    source = mapping(config.get("source_artifacts"), "source_artifacts", errors)
    locked_summary_path = root_path(source.get("locked_scale_summary", ""), "source_artifacts.locked_scale_summary", errors)
    tokenizer_review_path = root_path(source.get("tokenizer_preflight_review", ""), "source_artifacts.tokenizer_preflight_review", errors)
    manifest_path = root_path(source.get("row_bank_manifest", ""), "source_artifacts.row_bank_manifest", errors)
    rows_path = root_path(source.get("row_bank_rows", ""), "source_artifacts.row_bank_rows", errors)
    validation_path = root_path(source.get("row_bank_validation", ""), "source_artifacts.row_bank_validation", errors)

    locked_summary = read_json(locked_summary_path) if locked_summary_path.exists() else {}
    tokenizer_review = read_json(tokenizer_review_path) if tokenizer_review_path.exists() else {}
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    row_validation = read_json(validation_path) if validation_path.exists() else {}

    if source.get("locked_scale_jobs") != ["870210", "870987"]:
        errors.append("locked_scale_jobs mismatch")
    if locked_summary.get("status") != "PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE":
        errors.append("locked-scale summary must pass")
    by_arm = locked_summary.get("first_token_event_summary_by_arm", {})
    if not isinstance(by_arm, Mapping):
        errors.append("locked-scale by-arm summary missing")
    else:
        for arm in ("raw", "task_only", "wrong_key", "wrong_payload"):
            arm_summary = by_arm.get(arm, {})
            if not isinstance(arm_summary, Mapping):
                errors.append(f"locked-scale {arm} summary missing")
                continue
            if int_field(arm_summary, "blocks") != 96 or int_field(arm_summary, "accepts") != 0:
                errors.append(f"locked-scale {arm} must be 0/96")

    if source.get("tokenizer_preflight_job_id") != "871057":
        errors.append("tokenizer_preflight_job_id must be 871057")
    if tokenizer_review.get("review_status") != "PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_QWEN_TOKENIZER_PREFLIGHT_871057":
        errors.append("871057 tokenizer review must pass")
    for field, expected in (
        ("score_row_count", 163840),
        ("checked_row_count", 163840),
        ("failed_row_count", 0),
        ("empty_target_id_row_count", 0),
        ("empty_other_id_row_count", 0),
        ("target_other_overlap_row_count", 0),
    ):
        if int_field(tokenizer_review, field) != expected:
            errors.append(f"871057 tokenizer review {field} must be {expected}")
    for field in ("model_forward_pass_started", "generation_started", "training_started", "paper_claim_allowed"):
        if tokenizer_review.get(field) is not False:
            errors.append(f"871057 tokenizer review {field} must be false")

    if manifest.get("status") != "PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT":
        errors.append("row-bank manifest must pass")
    if row_validation.get("status") != "PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_ROW_BANK_VALIDATION_NO_SUBMIT":
        errors.append("row-bank validation must pass")
    for field, expected in (
        ("row_count", 163840),
        ("target_shards", 160),
        ("rows_per_shard", 1024),
        ("selected_prompt_count", 10240),
        ("selected_coordinate_count", 16),
        ("unique_content_prompt_prefix_pairs", 163840),
        ("duplicate_content_prompt_prefix_pair_extra_rows", 0),
        ("duplicate_prompt_prefix_pair_extra_rows", 0),
        ("previous_locked_scale_prompt_overlap_count", 0),
    ):
        if int_field(manifest, field) != expected:
            errors.append(f"manifest {field} must be {expected}")
    if rows_path.exists() and sum(1 for line in rows_path.open("r", encoding="utf-8") if line.strip()) != 163840:
        errors.append("row bank rows must contain 163840 rows")

    policy = mapping(config.get("policy_artifacts"), "policy_artifacts", errors)
    for field in (
        "duplicate_safe_generation_policy_v2",
        "contextual_forbidden_policy_v2",
        "trace_binding_verifier",
        "first_token_event_decoder",
        "full_phrase_decoder",
        "aggregate_script",
    ):
        root_path(policy.get(field, ""), f"policy_artifacts.{field}", errors)

    scope = mapping(config.get("generation_scope"), "generation_scope", errors)
    for field, expected in (
        ("blocks", 160),
        ("shards", 160),
        ("row_cylinders_per_block", 1024),
        ("rows_per_coordinate_per_block", 64),
        ("unique_prompt_indices_per_block", 64),
        ("selected_coordinate_count", 16),
        ("expected_generated_rows", 491520),
        ("expected_attempt_rows_min", 491520),
    ):
        if int(scope.get(field, -1)) != expected:
            errors.append(f"generation_scope.{field} must be {expected}")
    if scope.get("conditions") != ["protected", "raw", "task_only"]:
        errors.append("generation conditions mismatch")
    if scope.get("decode_conditions") != ["protected", "raw", "task_only", "wrong_key", "wrong_payload"]:
        errors.append("decode conditions mismatch")
    if scope.get("same_contract_only") is not True or scope.get("contract_id") != "a55e":
        errors.append("same-contract a55e scope mismatch")
    for field in ("payload_diversity_tested", "llama_tested", "paper_facing", "text_only_phrase_decoder_claim"):
        if scope.get(field) is not False:
            errors.append(f"generation_scope.{field} must be false")
    if scope.get("protected_policy") != "report_only_not_positive_gate":
        errors.append("protected policy must be report-only")

    targets = mapping(config.get("null_package_targets"), "null_package_targets", errors)
    if targets.get("standard_control_arms") != ["raw", "task_only", "wrong_key", "wrong_payload"]:
        errors.append("standard control arms mismatch")
    for field, expected in (
        ("existing_control_blocks_per_arm", 96),
        ("new_control_blocks_per_arm", 160),
        ("target_control_blocks_per_arm", 256),
        ("control_accepts_max_per_arm", 0),
        ("control_accepts_ignoring_quality_max_per_arm", 0),
    ):
        if int(targets.get(field, -1)) != expected:
            errors.append(f"null_package_targets.{field} must be {expected}")

    gates = mapping(config.get("quality_gates"), "quality_gates", errors)
    for field, expected in (
        ("control_accepts_max_per_condition", 0),
        ("within_block_duplicate_response_hash_count_max", 0),
        ("global_duplicate_response_hash_count_max", 0),
        ("technical_forbidden_public_surface_count_max", 0),
        ("ambiguous_forbidden_surface_count_max", 0),
    ):
        if int(gates.get(field, -1)) != expected:
            errors.append(f"quality_gates.{field} must be {expected}")
    if gates.get("ordinary_domain_literal_policy") != "report_only":
        errors.append("ordinary domain literal policy must be report_only")
    if float(gates.get("trace_binding_validity_required", -1.0)) != 1.0:
        errors.append("trace binding validity must be 100%")
    if gates.get("full_phrase_decoder_policy") != "report_only_not_success_claim":
        errors.append("full phrase decoder must be report-only")
    if gates.get("protected_policy") != "report_only_not_positive_gate":
        errors.append("protected policy must be report-only")

    controller = mapping(config.get("controller"), "controller", errors)
    if controller.get("source_job_id") != 868016:
        errors.append("controller source_job_id must remain 868016")
    for field, expected in (("bonus_nats", 4.0), ("penalty_nats", 0.5), ("max_target_mass", 0.5), ("max_kl_budget", 0.5)):
        if float(controller.get(field, -1.0)) != expected:
            errors.append(f"controller.{field} must be {expected}")

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
        ("array", "0-159%6"),
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
            "#SBATCH --array=0-159%6",
            "#SBATCH --partition=pomplun",
            "#SBATCH --account=cs_yinxin.wan",
            "#SBATCH --qos=pomplun",
            "#SBATCH --gres=gpu:h200:1",
            "#SBATCH --time=30-00:00:00",
            "MAX_SHARD_INDEX=159",
            "r4_after_868016_controller_generation_h200.sbatch",
            "r4_after_870987_prefar_standard_control_generation",
            "--skip-allowlist-state-check",
        ):
            if fragment not in text:
                errors.append(f"wrapper missing fragment: {fragment}")

    allowlist = load_yaml(ALLOWLIST)
    enabled_entries = enabled_allowlist_entries(allowlist)
    if not skip_allowlist_state_check:
        if allow_submission_enabled_entry:
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
        "PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_GENERATION_ROUTE_PLAN_ONLY_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_GENERATION_ROUTE_PLAN_ONLY_NO_SUBMIT"
    )
    return {
        "schema_name": "natural_evidence_v2_r4_after_870987_prefar_standard_control_generation_route_validation_v1",
        "status": status,
        "errors": errors,
        "tokenizer_preflight_job_id": "871057",
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
        "config": str(DEFAULT_CONFIG.relative_to(ROOT)),
        "config_sha256": sha256_file(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else "",
        "wrapper_sha256": sha256_file(ROOT / EXPECTED_WRAPPER) if (ROOT / EXPECTED_WRAPPER).exists() else "",
        "row_bank_rows": str(rows_path.relative_to(ROOT)) if rows_path.exists() else "",
        "row_bank_rows_sha256": sha256_file(rows_path) if rows_path.exists() else "",
        "expected_generated_rows": 491520,
        "expected_shards": 160,
        "allow_submission_enabled_entry": bool(allow_submission_enabled_entry),
        "skip_allowlist_state_check": bool(skip_allowlist_state_check),
        "slurm_allowed": False,
        "generation_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": "Run local/remote hash preflight before exactly-one H200 pre-FAR standard-control generation submission.",
    }


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
