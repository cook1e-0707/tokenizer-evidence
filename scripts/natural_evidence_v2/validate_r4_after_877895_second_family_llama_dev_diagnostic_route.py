#!/usr/bin/env python3
"""Validate the R4 after-877895 Llama 32-block dev diagnostic route."""

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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_877895_second_family_llama_dev_diagnostic_generation_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_877895_second_family_llama_dev_diagnostic_h200"
EXPECTED_ENTRY_POLICY_V3 = "v2_r4_after_879248_second_family_llama_dev_diagnostic_policy_v3_h200"
EXPECTED_ENTRY_POLICY_V4 = "v2_r4_after_879391_second_family_llama_dev_diagnostic_policy_v4_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_877895_second_family_llama_dev_diagnostic_h200.sbatch"
EXPECTED_WRAPPER_POLICY_V3 = "scripts/natural_evidence_v2/slurm/r4_after_879248_second_family_llama_dev_diagnostic_policy_v3_h200.sbatch"
EXPECTED_WRAPPER_POLICY_V4 = "scripts/natural_evidence_v2/slurm/r4_after_879391_second_family_llama_dev_diagnostic_policy_v4_h200.sbatch"
EXPECTED_GENERATOR = "scripts/natural_evidence_v2/generate_r4_after_868016_controller_outputs.py"
EXPECTED_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
EXPECTED_TOKENIZER_STATUS = "PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_TOKENIZER_PREFLIGHT_879100_REVIEWED"
EXPECTED_SMALL_STATUS = "PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_SMALL_DIAGNOSTIC_879102_REVIEWED"
EXPECTED_ROWS = 65536
ROUTE_EXPECTATIONS = {
    "r4_after_877895_second_family_llama_dev_diagnostic_generation_v1": {
        "schema_name": "natural_evidence_v2_r4_after_877895_second_family_llama_dev_diagnostic_generation_route_v1",
        "entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
        "status_pass": "PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_DEV_DIAGNOSTIC_ROUTE_VALIDATION_NO_SUBMIT",
        "status_fail": "FAIL_R4_AFTER_877895_SECOND_FAMILY_LLAMA_DEV_DIAGNOSTIC_ROUTE_VALIDATION_NO_SUBMIT",
        "summary_schema": "r4_after_877895_second_family_llama_dev_diagnostic_route_validation_v1",
    },
    "r4_after_879248_second_family_llama_dev_diagnostic_policy_v3_rerun_v1": {
        "schema_name": "natural_evidence_v2_r4_after_879248_second_family_llama_dev_diagnostic_policy_v3_rerun_route_v1",
        "entry": EXPECTED_ENTRY_POLICY_V3,
        "wrapper": EXPECTED_WRAPPER_POLICY_V3,
        "status_pass": "PASS_R4_AFTER_879248_SECOND_FAMILY_LLAMA_DEV_DIAGNOSTIC_POLICY_V3_RERUN_ROUTE_VALIDATION_NO_SUBMIT",
        "status_fail": "FAIL_R4_AFTER_879248_SECOND_FAMILY_LLAMA_DEV_DIAGNOSTIC_POLICY_V3_RERUN_ROUTE_VALIDATION_NO_SUBMIT",
        "summary_schema": "r4_after_879248_second_family_llama_dev_diagnostic_policy_v3_rerun_route_validation_v1",
    },
    "r4_after_879391_second_family_llama_dev_diagnostic_policy_v4_rerun_v1": {
        "schema_name": "natural_evidence_v2_r4_after_879391_second_family_llama_dev_diagnostic_policy_v4_rerun_route_v1",
        "entry": EXPECTED_ENTRY_POLICY_V4,
        "wrapper": EXPECTED_WRAPPER_POLICY_V4,
        "status_pass": "PASS_R4_AFTER_879391_SECOND_FAMILY_LLAMA_DEV_DIAGNOSTIC_POLICY_V4_RERUN_ROUTE_VALIDATION_NO_SUBMIT",
        "status_fail": "FAIL_R4_AFTER_879391_SECOND_FAMILY_LLAMA_DEV_DIAGNOSTIC_POLICY_V4_RERUN_ROUTE_VALIDATION_NO_SUBMIT",
        "summary_schema": "r4_after_879391_second_family_llama_dev_diagnostic_policy_v4_rerun_route_validation_v1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-submission-enabled-entry", action="store_true")
    parser.add_argument("--skip-allowlist-state-check", action="store_true")
    return parser.parse_args()


def mapping(value: Any, errors: list[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be a mapping")
        return {}
    return value


def root_path(value: Any, errors: list[str], label: str) -> Path:
    path = ROOT / str(value)
    if not path.exists():
        errors.append(f"{label} missing: {path}")
    return path


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def int_field(data: Mapping[str, Any], field: str, default: int = -1) -> int:
    try:
        return int(data.get(field, default))
    except (TypeError, ValueError):
        return default


def enabled_entries() -> list[str]:
    allowlist = load_yaml(ALLOWLIST)
    enabled: list[str] = []
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        entries = allowlist.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("enabled") is True:
                enabled.append(str(entry.get("name", "")))
    return enabled


def allowlist_entry(expected_entry: str) -> Mapping[str, Any] | None:
    allowlist = load_yaml(ALLOWLIST)
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        entries = allowlist.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("name") == expected_entry:
                return entry
    return None


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = load_yaml(config_path)
    errors: list[str] = []
    route_id = str(cfg.get("route_id", ""))
    route_expectation = ROUTE_EXPECTATIONS.get(route_id, ROUTE_EXPECTATIONS["r4_after_877895_second_family_llama_dev_diagnostic_generation_v1"])
    expected_entry = str(route_expectation["entry"])
    expected_wrapper = str(route_expectation["wrapper"])

    if cfg.get("schema_name") != route_expectation["schema_name"]:
        errors.append("schema_name mismatch")
    if route_id not in ROUTE_EXPECTATIONS:
        errors.append("route_id mismatch")

    source = mapping(cfg.get("source_artifacts"), errors, "source_artifacts")
    aggregate = read_json(root_path(source.get("same_family_raw_null_877895_aggregate", ""), errors, "877895 aggregate"))
    tokenizer_review = read_json(root_path(source.get("llama_tokenizer_review_879100", ""), errors, "Llama tokenizer review 879100"))
    small_review = read_json(root_path(source.get("llama_small_diagnostic_review_879102", ""), errors, "Llama small diagnostic review 879102"))
    row_plan = read_json(root_path(source.get("llama_row_bank_plan", ""), errors, "Llama row bank plan"))
    rows_path = root_path(source.get("llama_row_bank_rows", ""), errors, "Llama row bank rows")
    controller_review = read_json(root_path(source.get("controller_review", ""), errors, "controller review"))

    if aggregate.get("status") != "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE":
        errors.append("877895 aggregate must pass before Llama dev diagnostic")
    if tokenizer_review.get("status") != EXPECTED_TOKENIZER_STATUS:
        errors.append("Llama tokenizer review 879100 must be reviewed pass")
    if int_field(tokenizer_review, "failed_row_count") != 0:
        errors.append("Llama tokenizer review failed rows must be 0")
    if small_review.get("status") != EXPECTED_SMALL_STATUS:
        errors.append("Llama small diagnostic 879102 must pass before dev scale")
    if small_review.get("second_family_small_diagnostic_gate_pass") is not True:
        errors.append("Llama small diagnostic gate pass must be true")
    if row_plan.get("status") != "PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_ROW_BANK_PLAN_ARTIFACT_ONLY_TOKENIZER_PENDING":
        errors.append("Llama row bank plan status mismatch")
    if rows_path.exists() and count_jsonl(rows_path) != EXPECTED_ROWS:
        errors.append(f"Llama row bank must contain {EXPECTED_ROWS} rows")
    if controller_review.get("status") != "PASS_R4_AFTER_868016_COORDINATE_PIVOT_CONTROLLER_TEACHER_FORCED_GATE":
        errors.append("controller review status mismatch")

    scope = mapping(cfg.get("generation_scope"), errors, "generation_scope")
    for key, expected in {
        "model_id": EXPECTED_MODEL,
        "tokenizer_id": EXPECTED_MODEL,
        "model_slug": "llama3_1_8b_instruct",
        "conditions": ["protected", "raw"],
        "first_token_event_controls": ["protected", "raw", "wrong_key", "wrong_payload"],
        "blocks": 32,
        "shards": 32,
        "rows_per_shard": 1024,
        "prompts_per_shard": 64,
        "selected_coordinate_count": 16,
        "expected_generated_rows": 65536,
        "contract_id": "a55e",
    }.items():
        if scope.get(key) != expected:
            errors.append(f"generation_scope.{key} mismatch")
    for key in ("same_contract_only",):
        if scope.get(key) is not True:
            errors.append(f"generation_scope.{key} must be true")
    for key in ("task_only_control_included", "task_only_adapter_required", "task_only_adapter_allowed", "payload_diversity_tested", "paper_facing"):
        if scope.get(key) is not False:
            errors.append(f"generation_scope.{key} must be false")

    precommit = mapping(cfg.get("precommit"), errors, "precommit")
    for field in ("codebook", "decoder_spec", "decoder_route_config", "full_phrase_decoder_spec", "surface_bank", "contextual_literal_policy", "duplicate_safe_policy"):
        root_path(precommit.get(field, ""), errors, f"precommit.{field}")

    controller = mapping(cfg.get("controller"), errors, "controller")
    for key, expected in {
        "policy": "committed",
        "bonus_nats": 4.0,
        "penalty_nats": 0.5,
        "max_target_mass": 0.5,
        "max_kl_budget": 0.5,
        "applies_to": ["protected"],
    }.items():
        if controller.get(key) != expected:
            errors.append(f"controller.{key} mismatch")

    gate = mapping(cfg.get("quality_gates"), errors, "quality_gates")
    for key, expected in {
        "protected_strict_accepts_min": 28,
        "protected_accepts_ignoring_quality_min": 30,
        "raw_accepts_max": 0,
        "wrong_key_accepts_max": 0,
        "wrong_payload_accepts_max": 0,
        "global_duplicate_response_hash_count_max": 0,
        "technical_forbidden_public_surface_count_max": 0,
        "ambiguous_forbidden_surface_count_max": 0,
    }.items():
        if int_field(gate, key) != expected:
            errors.append(f"quality_gates.{key} must be {expected}")
    if gate.get("full_phrase_decoder_policy") != "report_only_not_success_claim":
        errors.append("quality_gates.full_phrase_decoder_policy mismatch")

    compute = mapping(cfg.get("compute_policy"), errors, "compute_policy")
    wrapper = root_path(compute.get("wrapper", ""), errors, "compute.wrapper")
    for key, expected in {
        "partition": "pomplun",
        "qos": "pomplun",
        "account": "cs_yinxin.wan",
        "gres": "gpu:h200:1",
        "max_time": "30-00:00:00",
        "array": "0-31",
        "allowlist_entry": expected_entry,
        "wrapper": expected_wrapper,
    }.items():
        if compute.get(key) != expected:
            errors.append(f"compute_policy.{key} mismatch")
    command = str(compute.get("command_pattern", ""))
    for fragment in ("PLAN_ONLY=0", "VALIDATE_PLAN_ONLY=0", f"sbatch {expected_wrapper}"):
        if fragment not in command:
            errors.append(f"compute_policy.command_pattern missing fragment: {fragment}")
    if wrapper.exists():
        text = wrapper.read_text(encoding="utf-8")
        for fragment in (
            "#SBATCH --partition=pomplun",
            "#SBATCH --account=cs_yinxin.wan",
            "#SBATCH --qos=pomplun",
            "#SBATCH --gres=gpu:h200:1",
            "#SBATCH --time=30-00:00:00",
            "#SBATCH --array=0-31",
            EXPECTED_MODEL,
            "GENERATION_CONDITIONS=protected,raw",
            "MAX_SHARD_INDEX=31",
            "TASK_ONLY_ADAPTER=",
            "ROUTE_VALIDATION_EXTRA_ARGS",
            "--skip-allowlist-state-check",
            "r4_after_868016_controller_generation_h200.sbatch",
        ):
            if fragment not in text:
                errors.append(f"wrapper missing fragment: {fragment}")

    generator = root_path(EXPECTED_GENERATOR, errors, "generator")
    if generator.exists():
        text = generator.read_text(encoding="utf-8")
        if EXPECTED_TOKENIZER_STATUS not in text:
            errors.append("generator missing 879100 tokenizer review status")

    enabled = enabled_entries()
    if not args.skip_allowlist_state_check:
        if args.allow_submission_enabled_entry:
            if enabled != [expected_entry]:
                errors.append(f"enabled entries must be exactly {expected_entry}: {enabled}")
        elif enabled:
            errors.append(f"enabled entries must be empty during route validation: {enabled}")
    entry = allowlist_entry(expected_entry)
    if entry is None:
        errors.append("allowlist entry missing")
    else:
        if (
            entry.get("enabled") is not False
            and not args.allow_submission_enabled_entry
            and not args.skip_allowlist_state_check
        ):
            errors.append("allowlist entry must be disabled during plan validation")
        if Path(expected_wrapper).name not in str(entry.get("command_pattern", "")):
            errors.append("allowlist command_pattern must reference the Llama dev diagnostic wrapper")

    prerequisites = mapping(cfg.get("required_before_any_submission"), errors, "required_before_any_submission")
    for field, value in prerequisites.items():
        if value is not True:
            errors.append(f"required_before_any_submission.{field} must be true")
    locked = mapping(cfg.get("not_unlocked_by_this_route_package"), errors, "not_unlocked_by_this_route_package")
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_package.{field} must be true")

    status = (
        str(route_expectation["status_pass"])
        if not errors
        else str(route_expectation["status_fail"])
    )
    summary = {
        "schema_name": str(route_expectation["summary_schema"]),
        "status": status,
        "errors": errors,
        "allowlist_entry": expected_entry,
        "enabled_entries": enabled,
        "expected_model": EXPECTED_MODEL,
        "expected_shards": 32,
        "expected_generated_rows": 65536,
        "config_sha256": sha256_file(config_path) if config_path.exists() else "",
        "wrapper_sha256": sha256_file(ROOT / expected_wrapper) if (ROOT / expected_wrapper).exists() else "",
        "generator_sha256": sha256_file(ROOT / EXPECTED_GENERATOR) if (ROOT / EXPECTED_GENERATOR).exists() else "",
        "row_bank_rows_sha256": sha256_file(rows_path) if rows_path.exists() else "",
        "slurm_allowed": False,
        "generation_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": "Run local/remote wrapper plan-only smoke and hash preflight before exactly-one Llama 32-block dev diagnostic submission.",
    }
    write_json_new(args.output_dir / "route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
