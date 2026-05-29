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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_870987_same_family_raw_null_v5_tokenizer_preflight_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_870987_same_family_raw_null_tokenizer_preflight_h200"
EXPECTED_WRAPPER = (
    "scripts/natural_evidence_v2/slurm/"
    "r4_after_870987_same_family_raw_null_tokenizer_boundary_preflight_h200.sbatch"
)
EXPECTED_TOKENIZERS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]
EXPECTED_ROWS_PER_TOKENIZER = 65536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R4 same-family raw-null v5 tokenizer preflight route.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-submission-enabled-entry", action="store_true")
    parser.add_argument("--skip-allowlist-state-check", action="store_true")
    return parser.parse_args()


def root_path(value: Any, errors: list[str], label: str) -> Path:
    path = ROOT / str(value)
    if not path.exists():
        errors.append(f"{label} missing: {path}")
    return path


def mapping(value: Any, errors: list[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be a mapping")
        return {}
    return value


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

    if cfg.get("schema_name") != "natural_evidence_v2_r4_after_870987_same_family_raw_null_v5_tokenizer_preflight_route_v1":
        errors.append("schema_name mismatch")
    if cfg.get("route_id") != "r4_after_870987_same_family_raw_null_v5_tokenizer_preflight_v1":
        errors.append("route_id mismatch")

    source = mapping(cfg.get("source_artifacts"), errors, "source_artifacts")
    failed_summary = read_json(root_path(source.get("failed_generation_summary_876852", ""), errors, "876852 failed generation summary"))
    failed_review = read_json(root_path(source.get("failed_generation_review_876852", ""), errors, "876852 failure review"))
    row_manifest = read_json(root_path(source.get("v5_row_bank_manifest", ""), errors, "v4 row bank manifest"))
    row_validation = read_json(root_path(source.get("v5_row_bank_validation", ""), errors, "v4 row bank validation"))
    rows_path = root_path(source.get("row_bank_rows", ""), errors, "row bank rows")

    if failed_summary.get("status") != "FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE":
        errors.append("876852 summary must remain failed")
    if failed_review.get("status") != "FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V4_876852_FORBIDDEN_RESIDUAL_NO_ADOPT":
        errors.append("876852 review must remain no-adopt")
    if row_manifest.get("status") != "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V5_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT":
        errors.append("v5 row bank manifest must pass")
    if row_validation.get("status") != "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V5_ROW_BANK_VALIDATION_NO_SUBMIT":
        errors.append("v5 row bank validation must pass")
    if rows_path.exists() and sum(1 for line in rows_path.open("r", encoding="utf-8") if line.strip()) != EXPECTED_ROWS_PER_TOKENIZER:
        errors.append("v5 row bank must contain 65536 rows")

    tokenizers = cfg.get("same_family_tokenizers", [])
    if not isinstance(tokenizers, list):
        errors.append("same_family_tokenizers must be a list")
        tokenizers = []
    if [str(item.get("tokenizer_id", "")) for item in tokenizers if isinstance(item, Mapping)] != EXPECTED_TOKENIZERS:
        errors.append("same_family_tokenizers mismatch")
    for item in tokenizers:
        if not isinstance(item, Mapping):
            errors.append("same_family_tokenizers entries must be mappings")
            continue
        if not str(item.get("model_slug", "")).endswith("_raw"):
            errors.append(f"{item.get('tokenizer_id')} model_slug must be raw")

    scope = mapping(cfg.get("tokenizer_preflight_scope"), errors, "tokenizer_preflight_scope")
    for key, expected in {
        "max_rows_per_tokenizer": EXPECTED_ROWS_PER_TOKENIZER,
        "tokenizer_count": 3,
        "expected_total_checked_rows": EXPECTED_ROWS_PER_TOKENIZER * 3,
    }.items():
        if int_field(scope, key) != expected:
            errors.append(f"tokenizer_preflight_scope.{key} must be {expected}")
    if scope.get("run_qwen_tokenizer") is not True:
        errors.append("tokenizer_preflight_scope.run_qwen_tokenizer must be true")
    for key in (
        "model_forward_allowed",
        "scoring_allowed",
        "generation_allowed",
        "training_allowed",
        "llama_allowed",
        "sanitizer_allowed",
        "far_allowed",
        "payload_diversity_allowed",
        "paper_claim_allowed",
    ):
        if scope.get(key) is not False:
            errors.append(f"tokenizer_preflight_scope.{key} must be false")

    gate = mapping(cfg.get("future_tokenizer_gate"), errors, "future_tokenizer_gate")
    for key, expected in {
        "checked_rows_per_tokenizer": EXPECTED_ROWS_PER_TOKENIZER,
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
        "array": "0-2%3",
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
    }.items():
        if compute.get(key) != expected:
            errors.append(f"compute_policy.{key} mismatch")
    command_pattern = str(compute.get("command_pattern", ""))
    for fragment in (
        "ROUTE_CONFIG=configs/natural_evidence_v2/r4_after_870987_same_family_raw_null_v5_tokenizer_preflight_route.yaml",
        "SCORE_ROWS=results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v5_row_bank_plan_20260525/row_allocation_rows.jsonl",
        "MAX_ROWS=65536",
        "ROUTE_VALIDATOR=scripts/natural_evidence_v2/validate_r4_after_870987_same_family_raw_null_v5_tokenizer_route.py",
        f"sbatch {EXPECTED_WRAPPER}",
    ):
        if fragment not in command_pattern:
            errors.append(f"compute_policy.command_pattern missing fragment: {fragment}")
    if wrapper_path.exists():
        text = wrapper_path.read_text(encoding="utf-8")
        for fragment in (
            "#SBATCH --array=0-2%3",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "--run-qwen-tokenizer",
            "model_forward_started=false",
            "generation_started=false",
        ):
            if fragment not in text:
                errors.append(f"wrapper missing fragment: {fragment}")

    enabled = enabled_entries()
    if not args.skip_allowlist_state_check:
        if args.allow_submission_enabled_entry:
            if enabled != [EXPECTED_ENTRY]:
                errors.append(f"enabled entries must be exactly {EXPECTED_ENTRY}: {enabled}")
        elif enabled:
            errors.append(f"enabled entries must be empty during plan validation: {enabled}")
    entry = allowlist_entry()
    if entry is None:
        errors.append("allowlist entry missing")
    else:
        entry_command = str(entry.get("command_pattern", ""))
        for fragment in (
            "r4_after_870987_same_family_raw_null_v5_tokenizer_preflight_route.yaml",
            "r4_after_870987_same_family_raw_null_v5_row_bank_plan_20260525/row_allocation_rows.jsonl",
            "validate_r4_after_870987_same_family_raw_null_v5_tokenizer_route.py",
        ):
            if fragment not in entry_command:
                errors.append(f"allowlist command_pattern missing fragment: {fragment}")

    status = (
        "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V5_TOKENIZER_ROUTE_PLAN_ONLY_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V5_TOKENIZER_ROUTE_PLAN_ONLY_NO_SUBMIT"
    )
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {out}")
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_870987_same_family_raw_null_v5_tokenizer_route_validation_v1",
        "status": status,
        "errors": errors,
        "config": str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
        "config_sha256": sha256_file(config_path),
        "row_bank_rows": str(rows_path.relative_to(ROOT)) if rows_path.is_relative_to(ROOT) else str(rows_path),
        "row_bank_rows_sha256": sha256_file(rows_path) if rows_path.exists() else None,
        "expected_tokenizers": EXPECTED_TOKENIZERS,
        "expected_rows_per_tokenizer": EXPECTED_ROWS_PER_TOKENIZER,
        "model_forward_allowed": False,
        "scoring_allowed": False,
        "generation_allowed": False,
        "training_allowed": False,
        "slurm_submitted": False,
        "enabled_entries": enabled,
        "next_allowed_action": "Run local/remote hash preflight and then exactly one tokenizer-only H200 Slurm submission if reviewed.",
    }
    write_json_new(out / "route_validation_summary.json", summary)
    write_text_new(
        out / "route_validation_report.md",
        "# R4 Same-Family Raw-Null V4 Tokenizer Route Validation\n\n"
        f"Status: `{status}`\n\n"
        f"Errors: {len(errors)}\n\n"
        "This is plan-only validation. It does not tokenize, score, generate, train, enable the allowlist, or submit Slurm.\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
