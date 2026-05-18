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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_869348_locked_scale_generation_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_869348_locked_scale_generation_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_869348_locked_scale_generation_h200.sbatch"
EXPECTED_COMMAND_PATTERN = f"PLAN_ONLY=0 VALIDATE_PLAN_ONLY=0 sbatch {EXPECTED_WRAPPER}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 after-869348 locked-scale generation route.")
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
    if config.get("schema_name") != "natural_evidence_v2_r4_after_869348_locked_scale_generation_route_v1":
        errors.append("schema_name mismatch")
    if config.get("route_id") != "r4_after_869348_locked_scale_generation_v1":
        errors.append("route_id mismatch")
    if config.get("phase") != "V2_R4_AFTER_869348_LOCKED_SCALE_QWEN_TOKENIZER_PREFLIGHT_PASSED_GENERATION_ROUTE_NEXT":
        errors.append("phase mismatch")

    source = mapping(config.get("source_artifacts"), "source_artifacts", errors)
    dev_review_path = root_path(source.get("source_dev_pass_review", ""), "source_artifacts.source_dev_pass_review", errors)
    tokenizer_review_path = root_path(source.get("tokenizer_preflight_review", ""), "source_artifacts.tokenizer_preflight_review", errors)
    manifest_path = root_path(source.get("row_bank_manifest", ""), "source_artifacts.row_bank_manifest", errors)
    rows_path = root_path(source.get("row_bank_rows", ""), "source_artifacts.row_bank_rows", errors)
    row_route_path = root_path(source.get("row_bank_route_validation", ""), "source_artifacts.row_bank_route_validation", errors)

    dev_review = read_json(dev_review_path) if dev_review_path.exists() else {}
    tokenizer_review = read_json(tokenizer_review_path) if tokenizer_review_path.exists() else {}
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    row_route = read_json(row_route_path) if row_route_path.exists() else {}

    if source.get("source_dev_pass_job_id") != "869348":
        errors.append("source_dev_pass_job_id must be 869348")
    if dev_review.get("status") != "PASS_R4_AFTER_868348_GLOBAL_UNIQUE_DEV_DIAGNOSTIC_GATE":
        errors.append("869348 dev review must pass")
    for field, expected in (
        ("protected_strict_accepts", 32),
        ("protected_accepts_ignoring_quality", 32),
        ("global_duplicate_response_hash_count", 0),
        ("protected_forbidden_public_surface_count", 0),
        ("trace_binding_invalid_rows", 0),
    ):
        if int_field(dev_review, field) != expected:
            errors.append(f"869348 dev review {field} must be {expected}")
    controls = dev_review.get("control_accepts", {})
    if not isinstance(controls, Mapping) or any(int(controls.get(arm, -1)) != 0 for arm in ("raw", "task_only", "wrong_key", "wrong_payload")):
        errors.append("869348 dev review controls must be zero")

    if source.get("tokenizer_preflight_job_id") != "870078":
        errors.append("tokenizer_preflight_job_id must be 870078")
    if tokenizer_review.get("status") != "PASS_R4_AFTER_869348_LOCKED_SCALE_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT_870078":
        errors.append("870078 tokenizer review must pass")
    for field, expected in (
        ("checked_row_count", 98304),
        ("failed_row_count", 0),
        ("empty_target_id_row_count", 0),
        ("empty_other_id_row_count", 0),
        ("target_other_overlap_row_count", 0),
    ):
        if int_field(tokenizer_review, field) != expected:
            errors.append(f"870078 tokenizer review {field} must be {expected}")
    for field in ("model_forward_started", "model_scoring_started", "generation_started", "training_started"):
        if tokenizer_review.get(field) is not False:
            errors.append(f"870078 tokenizer review {field} must be false")

    if manifest.get("status") != "PASS_R4_AFTER_869348_GLOBAL_UNIQUE_LOCKED_SCALE_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT":
        errors.append("locked row bank manifest must pass")
    if row_route.get("status") != "PASS_R4_AFTER_869348_LOCKED_SCALE_ROW_BANK_ROUTE_VALIDATION_NO_SUBMIT":
        errors.append("locked row bank route validation must pass")
    if row_route.get("errors") != []:
        errors.append("locked row bank route validation errors must be empty")
    for field, expected in (
        ("row_count", 98304),
        ("target_shards", 96),
        ("rows_per_shard", 1024),
        ("selected_prompt_count", 6144),
        ("selected_coordinate_count", 16),
        ("unique_content_prompt_prefix_pairs", 98304),
        ("duplicate_content_prompt_prefix_pair_extra_rows", 0),
    ):
        if int_field(manifest, field) != expected:
            errors.append(f"manifest {field} must be {expected}")
    if rows_path.exists() and sum(1 for line in rows_path.open("r", encoding="utf-8") if line.strip()) != 98304:
        errors.append("row bank rows must contain 98304 rows")

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
        ("blocks", 96),
        ("shards", 96),
        ("row_cylinders_per_block", 1024),
        ("rows_per_coordinate_per_block", 64),
        ("unique_prompt_indices_per_block", 64),
        ("selected_coordinate_count", 16),
        ("expected_generated_rows", 294912),
        ("expected_attempt_rows_min", 294912),
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

    allocation = mapping(config.get("allocation_policy"), "allocation_policy", errors)
    if allocation.get("reuse_scope") != "held_out_locked_scale_same_contract_only":
        errors.append("allocation reuse_scope mismatch")
    if allocation.get("dev_prompt_split_reused") is not False:
        errors.append("dev prompt split must not be reused")
    for field in ("duplicate_content_prompt_prefix_pair_extra_rows", "duplicate_prompt_prefix_pair_extra_rows"):
        if int(allocation.get(field, -1)) != 0:
            errors.append(f"allocation_policy.{field} must be 0")
    if allocation.get("global_response_hash_duplicate_gate_remains_zero") is not True:
        errors.append("global response hash duplicate gate must remain zero")

    gates = mapping(config.get("quality_gates"), "quality_gates", errors)
    for field, expected in (
        ("protected_strict_accepts_min", 85),
        ("protected_accepts_ignoring_quality_min", 90),
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
        ("array", "0-95%4"),
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
            "#SBATCH --array=0-95%4",
            "#SBATCH --partition=pomplun",
            "#SBATCH --account=cs_yinxin.wan",
            "#SBATCH --qos=pomplun",
            "#SBATCH --gres=gpu:h200:1",
            "#SBATCH --time=30-00:00:00",
            "MAX_SHARD_INDEX=95",
            "r4_after_868016_controller_generation_h200.sbatch",
            "r4_after_869348_locked_scale_generation",
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
        "PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_ROUTE_PLAN_ONLY_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_869348_LOCKED_SCALE_GENERATION_ROUTE_PLAN_ONLY_NO_SUBMIT"
    )
    return {
        "schema_name": "natural_evidence_v2_r4_after_869348_locked_scale_generation_route_validation_v1",
        "status": status,
        "errors": errors,
        "source_dev_pass_job_id": "869348",
        "tokenizer_preflight_job_id": "870078",
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
        "config": str(DEFAULT_CONFIG.relative_to(ROOT)),
        "config_sha256": sha256_file(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else "",
        "wrapper_sha256": sha256_file(ROOT / EXPECTED_WRAPPER) if (ROOT / EXPECTED_WRAPPER).exists() else "",
        "row_bank_rows": str(rows_path.relative_to(ROOT)) if rows_path.exists() else "",
        "row_bank_rows_sha256": sha256_file(rows_path) if rows_path.exists() else "",
        "expected_generated_rows": 294912,
        "allow_submission_enabled_entry": bool(allow_submission_enabled_entry),
        "skip_allowlist_state_check": bool(skip_allowlist_state_check),
        "slurm_allowed": False,
        "generation_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": "Run local/remote hash preflight before exactly-one H200 locked-scale generation submission.",
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
