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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_869348_locked_scale_tokenizer_preflight_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_869348_locked_scale_qwen_tokenizer_boundary_preflight_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_869348_locked_scale_qwen_tokenizer_boundary_preflight_h200.sbatch"
EXPECTED_COMMAND_PATTERN = f"sbatch {EXPECTED_WRAPPER}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 after-869348 locked-scale tokenizer-only route.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-submission-enabled-entry", action="store_true")
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


def validate_route(config: Mapping[str, Any], *, allow_submission_enabled_entry: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_after_869348_locked_scale_tokenizer_preflight_route_v1":
        errors.append("schema_name mismatch")
    if config.get("route_id") != "r4_after_869348_locked_scale_qwen_tokenizer_preflight_v1":
        errors.append("route_id mismatch")
    if config.get("phase") != "V2_R4_AFTER_869348_LOCKED_SCALE_ROW_BANK_STATIC_PREFLIGHT_PASSED_QWEN_TOKENIZER_NEXT":
        errors.append("phase mismatch")

    source = mapping(config.get("source_artifacts"), "source_artifacts", errors)
    dev_review_path = root_path(source.get("source_dev_pass_review", ""), "source_artifacts.source_dev_pass_review", errors)
    manifest_path = root_path(source.get("row_bank_manifest", ""), "source_artifacts.row_bank_manifest", errors)
    rows_path = root_path(source.get("row_bank_rows", ""), "source_artifacts.row_bank_rows", errors)
    route_validation_path = root_path(source.get("row_bank_route_validation", ""), "source_artifacts.row_bank_route_validation", errors)
    static_path = root_path(source.get("static_boundary_preflight", ""), "source_artifacts.static_boundary_preflight", errors)

    dev_review = read_json(dev_review_path) if dev_review_path.exists() else {}
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    route_validation = read_json(route_validation_path) if route_validation_path.exists() else {}
    static = read_json(static_path) if static_path.exists() else {}

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

    if manifest.get("status") != "PASS_R4_AFTER_869348_GLOBAL_UNIQUE_LOCKED_SCALE_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT":
        errors.append("locked row bank manifest must pass")
    if route_validation.get("status") != "PASS_R4_AFTER_869348_LOCKED_SCALE_ROW_BANK_ROUTE_VALIDATION_NO_SUBMIT":
        errors.append("locked row bank route validation must pass")
    if route_validation.get("errors") != []:
        errors.append("locked row bank route validation errors must be empty")
    if static.get("status") != "PASS_STATIC_BOUNDARY_CONTRACT_TOKENIZER_PENDING":
        errors.append("static boundary preflight must pass")

    requirements = mapping(config.get("row_bank_requirements"), "row_bank_requirements", errors)
    for field, expected in (
        ("expected_rows", 98304),
        ("expected_shards", 96),
        ("expected_rows_per_shard", 1024),
        ("expected_selected_coordinates", 16),
        ("expected_unique_content_prompt_prefix_pairs", 98304),
        ("expected_duplicate_content_prompt_prefix_extra_rows", 0),
        ("expected_rows_per_coordinate", 6144),
        ("expected_prefix_templates", 16),
    ):
        if int(requirements.get(field, -1)) != expected:
            errors.append(f"row_bank_requirements.{field} must be {expected}")
    if requirements.get("source_split") != "locked":
        errors.append("row_bank_requirements.source_split must be locked")
    if requirements.get("payload_diversity_tested") is not False or requirements.get("same_contract_only") is not True:
        errors.append("row_bank_requirements payload/contract flags invalid")
    if int_field(manifest, "row_count") != 98304:
        errors.append("manifest row_count must be 98304")
    if int_field(manifest, "duplicate_content_prompt_prefix_pair_extra_rows") != 0:
        errors.append("manifest duplicate content pairs must be zero")
    if rows_path.exists() and sum(1 for line in rows_path.open("r", encoding="utf-8") if line.strip()) != 98304:
        errors.append("row JSONL line count must be 98304")
    metrics = mapping(route_validation.get("metrics"), "row_bank_route_validation.metrics", errors)
    for field, expected in (
        ("row_count", 98304),
        ("shard_count", 96),
        ("locked_prompt_count", 6144),
        ("unique_content_prompt_prefix_pairs", 98304),
        ("duplicate_content_prompt_prefix_pair_extra_rows", 0),
    ):
        if int_field(metrics, field) != expected:
            errors.append(f"route validation metric {field} must be {expected}")

    scope = mapping(config.get("tokenizer_preflight_scope"), "tokenizer_preflight_scope", errors)
    if scope.get("tokenizer_name") != "Qwen/Qwen2.5-7B-Instruct":
        errors.append("tokenizer name mismatch")
    if int(scope.get("max_rows", -1)) != 98304:
        errors.append("tokenizer max_rows must be 98304")
    if scope.get("run_qwen_tokenizer") is not True:
        errors.append("run_qwen_tokenizer must be true")
    for field in (
        "model_forward_allowed",
        "scoring_allowed",
        "generation_allowed",
        "training_allowed",
        "llama_allowed",
        "same_family_null_allowed",
        "sanitizer_allowed",
        "far_allowed",
        "paper_claim_allowed",
    ):
        if scope.get(field) is not False:
            errors.append(f"tokenizer_preflight_scope.{field} must be false")

    future_gate = mapping(config.get("future_tokenizer_gate"), "future_tokenizer_gate", errors)
    if int(future_gate.get("checked_rows", -1)) != 98304:
        errors.append("future tokenizer gate checked_rows must be 98304")
    for field in (
        "failed_rows_max",
        "empty_target_id_row_count_max",
        "empty_other_id_row_count_max",
        "target_other_overlap_row_count_max",
    ):
        if int(future_gate.get(field, -1)) != 0:
            errors.append(f"future_tokenizer_gate.{field} must be 0")

    compute = mapping(config.get("compute_policy"), "compute_policy", errors)
    wrapper_path = root_path(compute.get("wrapper", ""), "compute_policy.wrapper", errors)
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
    ):
        if compute.get(field) != expected:
            errors.append(f"compute_policy.{field} mismatch")
    if not allow_submission_enabled_entry and compute.get("allowlist_enabled_now") is not False:
        errors.append("compute_policy.allowlist_enabled_now must be false")
    if compute.get("slurm_allowed_now") is not False:
        errors.append("compute_policy.slurm_allowed_now must be false")
    for field in (
        "exactly_one_submission_when_unlocked",
        "immediate_disable_after_sbatch",
        "remote_hash_preflight_required",
        "zero_enabled_allowlist_required_before_submission",
        "hermes_notification_required",
    ):
        if compute.get(field) is not True:
            errors.append(f"compute_policy.{field} must be true")

    if wrapper_path.exists():
        text = wrapper_path.read_text(encoding="utf-8")
        for fragment in (
            "#SBATCH --partition=pomplun",
            "#SBATCH --account=cs_yinxin.wan",
            "#SBATCH --qos=pomplun",
            "#SBATCH --gres=gpu:h200:1",
            "#SBATCH --time=30-00:00:00",
            "r4_after_869348_global_unique_locked_scale_row_bank_plan_20260518/row_allocation_rows.jsonl",
            "MAX_ROWS=\"${MAX_ROWS:-98304}\"",
            "model_forward_started=false",
            "scoring_started=false",
            "generation_started=false",
            "training_started=false",
            "--run-qwen-tokenizer",
        ):
            if fragment not in text:
                errors.append(f"wrapper missing fragment: {fragment}")

    allowlist = load_yaml(ALLOWLIST)
    enabled = enabled_allowlist_entries(allowlist)
    if allow_submission_enabled_entry:
        if enabled != [EXPECTED_ENTRY]:
            errors.append(f"allowlist enabled entries must be exactly [{EXPECTED_ENTRY!r}]: {enabled}")
    elif enabled:
        errors.append(f"allowlist enabled entries must be empty: {enabled}")
    entry = find_allowlist_entry(allowlist, EXPECTED_ENTRY)
    if entry is None:
        errors.append("allowlist entry missing")
    else:
        if allow_submission_enabled_entry:
            if entry.get("enabled") is not True:
                errors.append("allowlist entry must be enabled for submission preflight")
        elif entry.get("enabled") is not False:
            errors.append("allowlist entry must be disabled")
        if entry.get("command_pattern") != EXPECTED_COMMAND_PATTERN:
            errors.append("allowlist command_pattern mismatch")

    locked = mapping(config.get("not_unlocked_by_this_route_package"), "not_unlocked_by_this_route_package", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_package.{field} must be true")

    status = (
        "PASS_R4_AFTER_869348_LOCKED_SCALE_TOKENIZER_PREFLIGHT_ROUTE_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_869348_LOCKED_SCALE_TOKENIZER_PREFLIGHT_ROUTE_VALIDATION_NO_SUBMIT"
    )
    return {
        "schema_name": "natural_evidence_v2_r4_after_869348_locked_scale_tokenizer_preflight_route_validation_v1",
        "status": status,
        "errors": errors,
        "config": str(DEFAULT_CONFIG.relative_to(ROOT)),
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
        "wrapper_sha256": sha256_file(ROOT / EXPECTED_WRAPPER) if (ROOT / EXPECTED_WRAPPER).exists() else "",
        "row_bank_rows": str(rows_path.relative_to(ROOT)) if rows_path.exists() else "",
        "row_bank_rows_sha256": sha256_file(rows_path) if rows_path.exists() else "",
        "checked_rows_future_gate": 98304,
        "current_compute_unlocked": False,
        "allowlist_enabled": bool(allow_submission_enabled_entry),
        "slurm_submitted": False,
        "tokenizer_validation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Remote hash/allowlist preflight and exactly-one H200 tokenizer-only submission may be prepared; "
            "no generation or model scoring."
        ),
    }


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    summary = validate_route(load_yaml(config_path), allow_submission_enabled_entry=bool(args.allow_submission_enabled_entry))
    if args.output_dir is not None:
        output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_new(output_dir / "tokenizer_route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
