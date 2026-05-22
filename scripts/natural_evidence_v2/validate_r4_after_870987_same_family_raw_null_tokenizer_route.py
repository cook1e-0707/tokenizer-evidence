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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_870987_same_family_raw_null_tokenizer_preflight_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_870987_same_family_raw_null_tokenizer_preflight_h200"
EXPECTED_WRAPPER = (
    "scripts/natural_evidence_v2/slurm/"
    "r4_after_870987_same_family_raw_null_tokenizer_boundary_preflight_h200.sbatch"
)
EXPECTED_COMMAND = f"sbatch {EXPECTED_WRAPPER}"
EXPECTED_TOKENIZERS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R4 same-family tokenizer-only preflight route.")
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

    if cfg.get("schema_name") != "natural_evidence_v2_r4_after_870987_same_family_raw_null_tokenizer_preflight_route_v1":
        errors.append("schema_name mismatch")
    if cfg.get("route_id") != "r4_after_870987_same_family_raw_null_tokenizer_preflight_v1":
        errors.append("route_id mismatch")

    source = mapping(cfg.get("source_artifacts"), errors, "source_artifacts")
    prefar_review = read_json(root_path(source.get("prefar_null_package_review", ""), errors, "prefar review"))
    locked = read_json(root_path(source.get("locked_scale_summary", ""), errors, "locked summary"))
    standard = read_json(root_path(source.get("standard_control_summary", ""), errors, "standard summary"))
    organic = read_json(root_path(source.get("organic_null_summary", ""), errors, "organic summary"))
    rows_path = root_path(source.get("row_bank_rows", ""), errors, "row bank rows")

    if prefar_review.get("review_status") != "PASS_R4_AFTER_870987_PREFAR_NULL_PACKAGE_871250_PLUS_874308":
        errors.append("pre-FAR null package review must pass")
    if locked.get("status") != "PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE":
        errors.append("locked scale summary must pass")
    if standard.get("status") != "PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_GENERATION_GATE":
        errors.append("standard-control summary must pass")
    if organic.get("status") != "PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_GENERATION_GATE":
        errors.append("organic-null summary must pass")
    if rows_path.exists() and sum(1 for line in rows_path.open("r", encoding="utf-8") if line.strip()) != 262144:
        errors.append("row bank must contain 262144 rows")

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
        "max_rows_per_tokenizer": 262144,
        "tokenizer_count": 3,
        "expected_total_checked_rows": 786432,
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
        "checked_rows_per_tokenizer": 262144,
        "failed_rows_max": 0,
        "empty_target_id_row_count_max": 0,
        "empty_other_id_row_count_max": 0,
        "target_other_overlap_row_count_max": 0,
    }.items():
        if int_field(gate, key) != expected:
            errors.append(f"future_tokenizer_gate.{key} must be {expected}")

    compute = mapping(cfg.get("compute_policy"), errors, "compute_policy")
    wrapper = root_path(compute.get("wrapper", ""), errors, "wrapper")
    for key, expected in {
        "partition": "pomplun",
        "qos": "pomplun",
        "account": "cs_yinxin.wan",
        "gres": "gpu:h200:1",
        "max_time": "30-00:00:00",
        "array": "0-2%3",
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
        "command_pattern": EXPECTED_COMMAND,
    }.items():
        if compute.get(key) != expected:
            errors.append(f"compute_policy.{key} mismatch")
    if wrapper.exists():
        text = wrapper.read_text(encoding="utf-8")
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
    elif entry.get("command_pattern") != EXPECTED_COMMAND:
        errors.append("allowlist command_pattern mismatch")

    status = (
        "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_TOKENIZER_ROUTE_PLAN_ONLY_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_TOKENIZER_ROUTE_PLAN_ONLY_NO_SUBMIT"
    )
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {out}")
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_870987_same_family_raw_null_tokenizer_route_validation_v1",
        "status": status,
        "errors": errors,
        "allowlist_entry": EXPECTED_ENTRY,
        "enabled_entries": enabled,
        "expected_tokenizers": EXPECTED_TOKENIZERS,
        "expected_total_checked_rows": 786432,
        "config_sha256": sha256_file(config_path) if config_path.exists() else "",
        "wrapper_sha256": sha256_file(ROOT / EXPECTED_WRAPPER) if (ROOT / EXPECTED_WRAPPER).exists() else "",
        "row_bank_rows_sha256": sha256_file(rows_path) if rows_path.exists() else None,
        "model_forward_started": False,
        "scoring_started": False,
        "generation_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Run exactly one reviewed H200 same-family tokenizer-only preflight array; "
            "do not run same-family generation until all tokenizer preflight shards pass and are reviewed."
        ),
    }
    write_json_new(out / "route_validation_summary.json", summary)
    write_text_new(
        out / "route_validation_report.md",
        "# R4 Same-Family Raw-Null Tokenizer Route Validation\n\n"
        f"Status: `{status}`\n\n"
        f"Errors: {len(errors)}\n\n"
        "This route validation does not run model forward, scoring, generation, training, or paper-claim actions.\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
