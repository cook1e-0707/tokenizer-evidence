#!/usr/bin/env python3
"""Validate the artifact-only R4 after-877895 Llama tokenizer preflight route."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import read_json, sha256_file, write_json_new, write_text_new  # noqa: E402
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_877895_second_family_llama_tokenizer_preflight_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_877895_second_family_llama_tokenizer_preflight_h200"
EXPECTED_ENTRY_LOCKED = "v2_r4_after_879406_second_family_llama_locked_scale_tokenizer_preflight_h200"
EXPECTED_WRAPPER = (
    "scripts/natural_evidence_v2/slurm/"
    "r4_after_877895_second_family_llama_tokenizer_boundary_preflight_h200.sbatch"
)
EXPECTED_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B-Instruct"
EXPECTED_ROWS = 65536
EXPECTED_LOCKED_ROWS = 98304
ROUTE_EXPECTATIONS = {
    "r4_after_877895_second_family_llama_tokenizer_preflight_v1": {
        "schema_name": "natural_evidence_v2_r4_after_877895_second_family_llama_tokenizer_preflight_route_v1",
        "entry": EXPECTED_ENTRY,
        "rows": EXPECTED_ROWS,
        "status_pass": "PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_TOKENIZER_ROUTE_VALIDATION_NO_SUBMIT",
        "status_fail": "FAIL_R4_AFTER_877895_SECOND_FAMILY_LLAMA_TOKENIZER_ROUTE_VALIDATION_NO_SUBMIT",
        "summary_schema": "r4_after_877895_second_family_llama_tokenizer_route_validation_v1",
        "source_mode": "dev",
    },
    "r4_after_879406_second_family_llama_locked_scale_tokenizer_preflight_v1": {
        "schema_name": "natural_evidence_v2_r4_after_879406_second_family_llama_locked_scale_tokenizer_preflight_route_v1",
        "entry": EXPECTED_ENTRY_LOCKED,
        "rows": EXPECTED_LOCKED_ROWS,
        "status_pass": "PASS_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_TOKENIZER_ROUTE_VALIDATION_NO_SUBMIT",
        "status_fail": "FAIL_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_TOKENIZER_ROUTE_VALIDATION_NO_SUBMIT",
        "summary_schema": "r4_after_879406_second_family_llama_locked_scale_tokenizer_route_validation_v1",
        "source_mode": "locked_scale",
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
    route_expectation = ROUTE_EXPECTATIONS.get(route_id, ROUTE_EXPECTATIONS["r4_after_877895_second_family_llama_tokenizer_preflight_v1"])
    expected_entry = str(route_expectation["entry"])
    expected_rows = int(route_expectation["rows"])
    if cfg.get("schema_name") != route_expectation["schema_name"]:
        errors.append("schema_name mismatch")
    if route_id not in ROUTE_EXPECTATIONS:
        errors.append("route_id mismatch")

    source = mapping(cfg.get("source_artifacts"), errors, "source_artifacts")
    rows_path = root_path(source.get("row_bank_rows", ""), errors, "row bank rows")
    source_mode = str(route_expectation["source_mode"])
    if source_mode == "dev":
        aggregate = read_json(root_path(source.get("same_family_raw_null_877895_aggregate", ""), errors, "877895 aggregate"))
        route_selection = read_json(root_path(source.get("route_selection", ""), errors, "route selection"))
        inventory = read_json(root_path(source.get("llama_inventory", ""), errors, "llama inventory"))
        plan = read_json(root_path(source.get("llama_artifact_only_plan", ""), errors, "llama artifact-only plan"))
        row_plan = read_json(root_path(source.get("llama_row_bank_plan", ""), errors, "llama row bank plan"))

        if aggregate.get("status") != "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE":
            errors.append("877895 aggregate must pass before Llama tokenizer planning")
        if aggregate.get("same_family_raw_null_gate_pass") is not True:
            errors.append("877895 aggregate must have same_family_raw_null_gate_pass=true")
        if route_selection.get("status") != "ROUTE_DECISION_R4_AFTER_877895_POST_SAME_FAMILY_ROUTE_SELECTION_NO_SUBMIT":
            errors.append("post-same-family route selection must be recorded")
        if inventory.get("status") != "PASS_R4_AFTER_877895_LLAMA_MIGRATION_INVENTORY_ARTIFACT_ONLY_NO_SUBMIT":
            errors.append("Llama inventory must pass")
        if plan.get("status") != "ROUTE_PLAN_R4_AFTER_877895_SECOND_FAMILY_LLAMA_ARTIFACT_ONLY_NO_SUBMIT":
            errors.append("Llama artifact-only plan must be recorded")
        if row_plan.get("status") != "PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_ROW_BANK_PLAN_ARTIFACT_ONLY_TOKENIZER_PENDING":
            errors.append("Llama row-bank plan must pass and remain tokenizer-pending")
    else:
        qwen_locked = read_json(root_path(source.get("qwen_locked_scale_summary", ""), errors, "Qwen locked-scale summary"))
        llama_dev = read_json(root_path(source.get("llama_dev_review_879406", ""), errors, "Llama dev review 879406"))
        row_plan = read_json(root_path(source.get("llama_locked_scale_row_bank_plan", ""), errors, "Llama locked-scale row bank plan"))

        if qwen_locked.get("status") != "PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE":
            errors.append("Qwen locked-scale generation package must pass before Llama locked-scale tokenizer planning")
        if qwen_locked.get("scale_gate_pass") is not True:
            errors.append("Qwen locked-scale scale_gate_pass must be true")
        if llama_dev.get("status") != "PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_DEV_DIAGNOSTIC_879406_REVIEWED":
            errors.append("Llama dev diagnostic 879406 must be reviewed pass")
        if llama_dev.get("second_family_dev_diagnostic_gate_pass") is not True:
            errors.append("Llama dev diagnostic gate pass must be true")
        if row_plan.get("status") != "PASS_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_ROW_BANK_PLAN_ARTIFACT_ONLY_TOKENIZER_PENDING":
            errors.append("Llama locked-scale row-bank plan must pass and remain tokenizer-pending")
    if int(row_plan.get("selected_row_count", -1)) != expected_rows:
        errors.append(f"Llama row-bank plan must have {expected_rows} rows")
    if rows_path.exists() and sum(1 for line in rows_path.open("r", encoding="utf-8") if line.strip()) != expected_rows:
        errors.append(f"row bank rows must contain {expected_rows} rows")

    tokenizer = mapping(cfg.get("second_family_tokenizer"), errors, "second_family_tokenizer")
    for key, expected in {
        "tokenizer_id": EXPECTED_TOKENIZER,
        "model_slug": "llama3_1_8b_instruct",
        "role": "second_family_candidate",
        "model_config_reference": "configs/model/llama3_1_8b_instruct.yaml",
    }.items():
        if tokenizer.get(key) != expected:
            errors.append(f"second_family_tokenizer.{key} mismatch")
    root_path(tokenizer.get("model_config_reference", ""), errors, "model config reference")

    scope = mapping(cfg.get("tokenizer_preflight_scope"), errors, "tokenizer_preflight_scope")
    for key, expected in {
        "max_rows_per_tokenizer": expected_rows,
        "tokenizer_count": 1,
        "expected_total_checked_rows": expected_rows,
    }.items():
        if int_field(scope, key) != expected:
            errors.append(f"tokenizer_preflight_scope.{key} must be {expected}")
    if scope.get("run_tokenizer") is not True:
        errors.append("tokenizer_preflight_scope.run_tokenizer must be true")
    for key in (
        "model_forward_allowed",
        "scoring_allowed",
        "generation_allowed",
        "training_allowed",
        "sanitizer_allowed",
        "far_allowed",
        "payload_diversity_allowed",
        "paper_claim_allowed",
    ):
        if scope.get(key) is not False:
            errors.append(f"tokenizer_preflight_scope.{key} must be false")

    gate = mapping(cfg.get("future_tokenizer_gate"), errors, "future_tokenizer_gate")
    for key, expected in {
        "checked_rows_per_tokenizer": expected_rows,
        "failed_rows_max": 0,
        "empty_target_id_row_count_max": 0,
        "empty_other_id_row_count_max": 0,
        "target_other_overlap_row_count_max": 0,
    }.items():
        if int_field(gate, key) != expected:
            errors.append(f"future_tokenizer_gate.{key} must be {expected}")

    compute = mapping(cfg.get("compute_policy"), errors, "compute_policy")
    wrapper_path = root_path(compute.get("wrapper", ""), errors, "wrapper")
    for key, expected in {
        "partition": "pomplun",
        "qos": "pomplun",
        "account": "cs_yinxin.wan",
        "gres": "gpu:h200:1",
        "max_time": "30-00:00:00",
        "array": "0",
        "allowlist_entry": expected_entry,
        "wrapper": EXPECTED_WRAPPER,
    }.items():
        if compute.get(key) != expected:
            errors.append(f"compute_policy.{key} mismatch")
    command_pattern = str(compute.get("command_pattern", ""))
    config_command_path = str(config_path.relative_to(ROOT))
    rows_command_path = str(rows_path.relative_to(ROOT)) if rows_path.exists() else str(source.get("row_bank_rows", ""))
    for fragment in (
        f"ROUTE_CONFIG={config_command_path}",
        f"SCORE_ROWS={rows_command_path}",
        f"MAX_ROWS={expected_rows}",
        f"TOKENIZER_NAME={EXPECTED_TOKENIZER}",
        "ROUTE_VALIDATOR=scripts/natural_evidence_v2/validate_r4_after_877895_second_family_llama_tokenizer_route.py",
        f"sbatch {EXPECTED_WRAPPER}",
    ):
        if fragment not in command_pattern:
            errors.append(f"compute_policy.command_pattern missing fragment: {fragment}")
    if wrapper_path.exists():
        text = wrapper_path.read_text(encoding="utf-8")
        for fragment in (
            "#SBATCH --partition=pomplun",
            "#SBATCH --qos=pomplun",
            "#SBATCH --gres=gpu:h200:1",
            "#SBATCH --array=0",
            EXPECTED_TOKENIZER,
            "--run-tokenizer",
            "model_forward_started=false",
            "generation_started=false",
            "training_started=false",
        ):
            if fragment not in text:
                errors.append(f"wrapper missing fragment: {fragment}")

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
        if EXPECTED_WRAPPER not in str(entry.get("command_pattern", "")):
            errors.append("allowlist command_pattern must reference the Llama tokenizer wrapper")

    not_unlocked = mapping(cfg.get("not_unlocked_by_this_route_package"), errors, "not_unlocked_by_this_route_package")
    for key in ("generation", "training", "model_scoring", "sanitizer", "far", "payload_diversity", "paper_claim"):
        if not_unlocked.get(key) is not True:
            errors.append(f"not_unlocked_by_this_route_package.{key} must be true")

    status = str(route_expectation["status_pass"]) if not errors else str(route_expectation["status_fail"])
    summary = {
        "schema_name": str(route_expectation["summary_schema"]),
        "status": status,
        "errors": errors,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "row_bank_rows": str(rows_path.relative_to(ROOT)) if rows_path.exists() else str(rows_path),
        "row_bank_rows_sha256": sha256_file(rows_path) if rows_path.exists() else None,
        "wrapper": EXPECTED_WRAPPER,
        "wrapper_sha256": sha256_file(wrapper_path) if wrapper_path.exists() else None,
        "allowlist_entry": expected_entry,
        "enabled_entries": enabled,
        "slurm_submitted": False,
        "allowlist_enabled": bool(enabled),
        "tokenizer_preflight_started": False,
        "model_scoring_started": False,
        "generation_started": False,
        "training_started": False,
        "next_allowed_action": (
            "If reviewed locally/remotely, prepare exactly-one Llama tokenizer-only H200 submission preflight; "
            "do not run model scoring or generation."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_new(args.output_dir / "route_validation_summary.json", summary)
    report = f"""# R4 After-877895 Second-Family Llama Tokenizer Route Validation

Status: `{status}`

Errors: `{len(errors)}`

This validation is artifact-only. It does not submit Slurm, enable allowlist,
run the Llama tokenizer, load model weights, score, generate, or train.

Next allowed action: {summary["next_allowed_action"]}
"""
    write_text_new(args.output_dir / "route_validation_report.md", report)
    print(json.dumps({"status": status, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
