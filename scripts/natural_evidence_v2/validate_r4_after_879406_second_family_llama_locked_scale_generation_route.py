#!/usr/bin/env python3
"""Validate the R4 after-879406 Llama locked-scale generation route."""

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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_879406_second_family_llama_locked_scale_generation_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_879406_second_family_llama_locked_scale_policy_v4_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_879406_second_family_llama_locked_scale_policy_v4_h200.sbatch"
EXPECTED_GENERATOR = "scripts/natural_evidence_v2/generate_r4_after_868016_controller_outputs.py"
EXPECTED_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
EXPECTED_TOKENIZER_STATUS = "PASS_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_TOKENIZER_PREFLIGHT_879455_REVIEWED"
EXPECTED_ROWS = 98304
EXPECTED_SHARDS = 96


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


def allowlist_entry() -> Mapping[str, Any] | None:
    allowlist = load_yaml(ALLOWLIST)
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        entries = allowlist.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("name") == EXPECTED_ENTRY:
                return entry
    return None


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = load_yaml(config_path)
    errors: list[str] = []

    if cfg.get("schema_name") != "natural_evidence_v2_r4_after_879406_second_family_llama_locked_scale_generation_route_v1":
        errors.append("schema_name mismatch")
    if cfg.get("route_id") != "r4_after_879406_second_family_llama_locked_scale_generation_v1":
        errors.append("route_id mismatch")

    source = mapping(cfg.get("source_artifacts"), errors, "source_artifacts")
    qwen_locked = read_json(root_path(source.get("qwen_locked_scale_summary", ""), errors, "Qwen locked-scale summary"))
    llama_dev = read_json(root_path(source.get("llama_dev_review_879406", ""), errors, "Llama dev review 879406"))
    tokenizer_review = read_json(root_path(source.get("llama_locked_scale_tokenizer_review_879455", ""), errors, "Llama locked-scale tokenizer review 879455"))
    policy_validation = read_json(root_path(source.get("contextual_forbidden_policy_v4_validation", ""), errors, "policy v4 validation"))
    row_plan = read_json(root_path(source.get("llama_locked_scale_row_bank_plan", ""), errors, "Llama locked-scale row plan"))
    rows_path = root_path(source.get("llama_locked_scale_row_bank_rows", ""), errors, "Llama locked-scale row rows")
    controller_review = read_json(root_path(source.get("controller_review", ""), errors, "controller review"))

    if qwen_locked.get("status") != "PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE":
        errors.append("Qwen locked-scale generation package must pass")
    if qwen_locked.get("scale_gate_pass") is not True:
        errors.append("Qwen locked-scale scale_gate_pass must be true")
    if llama_dev.get("status") != "PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_DEV_DIAGNOSTIC_879406_REVIEWED":
        errors.append("Llama dev diagnostic 879406 must pass")
    if llama_dev.get("second_family_dev_diagnostic_gate_pass") is not True:
        errors.append("Llama dev diagnostic gate pass must be true")
    if tokenizer_review.get("status") != EXPECTED_TOKENIZER_STATUS:
        errors.append("Llama locked-scale tokenizer review 879455 must pass")
    if int_field(tokenizer_review, "failed_row_count") != 0:
        errors.append("Llama locked-scale tokenizer failed rows must be 0")
    if not str(policy_validation.get("status", "")).startswith("PASS"):
        errors.append("contextual forbidden policy v4 validation must pass")
    if row_plan.get("status") != "PASS_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_ROW_BANK_PLAN_ARTIFACT_ONLY_TOKENIZER_PENDING":
        errors.append("Llama locked-scale row-bank plan status mismatch")
    if int_field(row_plan, "selected_row_count") != EXPECTED_ROWS:
        errors.append(f"Llama locked-scale row-bank plan must have {EXPECTED_ROWS} rows")
    if rows_path.exists() and count_jsonl(rows_path) != EXPECTED_ROWS:
        errors.append(f"Llama locked-scale row bank must contain {EXPECTED_ROWS} rows")
    if controller_review.get("status") != "PASS_R4_AFTER_868016_COORDINATE_PIVOT_CONTROLLER_TEACHER_FORCED_GATE":
        errors.append("controller review status mismatch")

    scope = mapping(cfg.get("generation_scope"), errors, "generation_scope")
    for key, expected in {
        "model_id": EXPECTED_MODEL,
        "tokenizer_id": EXPECTED_MODEL,
        "model_slug": "llama3_1_8b_instruct",
        "conditions": ["protected", "raw"],
        "first_token_event_controls": ["protected", "raw", "wrong_key", "wrong_payload"],
        "blocks": EXPECTED_SHARDS,
        "shards": EXPECTED_SHARDS,
        "rows_per_shard": 1024,
        "prompts_per_shard": 64,
        "selected_coordinate_count": 16,
        "expected_generated_rows": EXPECTED_ROWS,
        "contract_id": "a55e",
    }.items():
        if scope.get(key) != expected:
            errors.append(f"generation_scope.{key} mismatch")
    if scope.get("same_contract_only") is not True:
        errors.append("generation_scope.same_contract_only must be true")
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
        "protected_strict_accepts_min": 80,
        "protected_accepts_ignoring_quality_min": 85,
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
        "array": "0-95",
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
    }.items():
        if compute.get(key) != expected:
            errors.append(f"compute_policy.{key} mismatch")
    command = str(compute.get("command_pattern", ""))
    for fragment in ("PLAN_ONLY=0", "VALIDATE_PLAN_ONLY=0", f"sbatch {EXPECTED_WRAPPER}"):
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
            "#SBATCH --array=0-95",
            EXPECTED_MODEL,
            "GENERATION_CONDITIONS=protected,raw",
            "MAX_SHARD_INDEX=95",
            "TASK_ONLY_ADAPTER=",
            "r4_after_868016_controller_generation_h200.sbatch",
        ):
            if fragment not in text:
                errors.append(f"wrapper missing fragment: {fragment}")

    generator = root_path(EXPECTED_GENERATOR, errors, "generator")
    if generator.exists() and EXPECTED_TOKENIZER_STATUS not in generator.read_text(encoding="utf-8"):
        errors.append("generator missing 879455 tokenizer review status")

    enabled = enabled_entries()
    if not args.skip_allowlist_state_check:
        if args.allow_submission_enabled_entry:
            if enabled != [EXPECTED_ENTRY]:
                errors.append(f"enabled entries must be exactly {EXPECTED_ENTRY}: {enabled}")
        elif enabled:
            errors.append(f"enabled entries must be empty during route validation: {enabled}")
    entry = allowlist_entry()
    if entry is None:
        errors.append("allowlist entry missing")
    else:
        if (
            entry.get("enabled") is not False
            and not args.allow_submission_enabled_entry
            and not args.skip_allowlist_state_check
        ):
            errors.append("allowlist entry must be disabled during plan validation")
        if Path(EXPECTED_WRAPPER).name not in str(entry.get("command_pattern", "")):
            errors.append("allowlist command_pattern must reference the Llama locked-scale wrapper")

    prerequisites = mapping(cfg.get("required_before_any_submission"), errors, "required_before_any_submission")
    for field, value in prerequisites.items():
        if value is not True:
            errors.append(f"required_before_any_submission.{field} must be true")
    locked = mapping(cfg.get("not_unlocked_by_this_route_package"), errors, "not_unlocked_by_this_route_package")
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_package.{field} must be true")

    status = (
        "PASS_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_GENERATION_ROUTE_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_GENERATION_ROUTE_VALIDATION_NO_SUBMIT"
    )
    summary = {
        "schema_name": "r4_after_879406_second_family_llama_locked_scale_generation_route_validation_v1",
        "status": status,
        "errors": errors,
        "allowlist_entry": EXPECTED_ENTRY,
        "enabled_entries": enabled,
        "expected_model": EXPECTED_MODEL,
        "expected_shards": EXPECTED_SHARDS,
        "expected_generated_rows": EXPECTED_ROWS,
        "config_sha256": sha256_file(config_path) if config_path.exists() else "",
        "wrapper_sha256": sha256_file(ROOT / EXPECTED_WRAPPER) if (ROOT / EXPECTED_WRAPPER).exists() else "",
        "generator_sha256": sha256_file(ROOT / EXPECTED_GENERATOR) if (ROOT / EXPECTED_GENERATOR).exists() else "",
        "row_bank_rows_sha256": sha256_file(rows_path) if rows_path.exists() else "",
        "slurm_allowed": False,
        "generation_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": "Run local/remote wrapper plan-only smoke and hash preflight before exactly-one Llama locked-scale generation submission.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_new(args.output_dir / "route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
