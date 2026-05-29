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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_870987_prefar_organic_null_tokenizer_preflight_route.yaml"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/"
    "r4_after_870987_prefar_organic_null_tokenizer_route_validation_20260521"
)
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_870987_prefar_organic_null_qwen_tokenizer_boundary_preflight_h200"
EXPECTED_WRAPPER = (
    "scripts/natural_evidence_v2/slurm/"
    "r4_after_870987_prefar_organic_null_qwen_tokenizer_boundary_preflight_h200.sbatch"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the R4 after-870987 pre-FAR organic-null Qwen tokenizer preflight route."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-submission-enabled-entry", action="store_true")
    parser.add_argument("--skip-allowlist-state-check", action="store_true")
    return parser.parse_args()


def root_path(value: Any, field: str, errors: list[str]) -> Path:
    path = ROOT / str(value)
    if not path.exists():
        errors.append(f"{field} missing: {path}")
    return path


def mapping(value: Any, field: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be a mapping")
        return {}
    return value


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
    allow_submission_enabled_entry: bool,
    skip_allowlist_state_check: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_after_870987_prefar_organic_null_tokenizer_preflight_route_v1":
        errors.append("schema_name mismatch")
    if config.get("route_id") != "r4_after_870987_prefar_organic_null_qwen_tokenizer_preflight_v1":
        errors.append("route_id mismatch")

    source = mapping(config.get("source_artifacts"), "source_artifacts", errors)
    locked_summary_path = root_path(source.get("locked_scale_summary", ""), "source_artifacts.locked_scale_summary", errors)
    standard_review_path = root_path(source.get("standard_control_review", ""), "source_artifacts.standard_control_review", errors)
    prompt_validation_path = root_path(source.get("prompt_bank_validation", ""), "source_artifacts.prompt_bank_validation", errors)
    manifest_path = root_path(source.get("row_bank_manifest", ""), "source_artifacts.row_bank_manifest", errors)
    rows_path = root_path(source.get("row_bank_rows", ""), "source_artifacts.row_bank_rows", errors)
    validation_path = root_path(source.get("row_bank_validation", ""), "source_artifacts.row_bank_validation", errors)

    locked_summary = read_json(locked_summary_path) if locked_summary_path.exists() else {}
    standard_review = read_json(standard_review_path) if standard_review_path.exists() else {}
    prompt_validation = read_json(prompt_validation_path) if prompt_validation_path.exists() else {}
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    row_validation = read_json(validation_path) if validation_path.exists() else {}

    if source.get("locked_scale_jobs") != ["870210", "870987"]:
        errors.append("locked_scale_jobs mismatch")
    if locked_summary.get("status") != "PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE":
        errors.append("locked-scale summary must pass")
    if standard_review.get("repaired_aggregate_status") != "PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_GENERATION_GATE":
        errors.append("standard-control 871250 contextual-window v2 review must pass")
    if standard_review.get("adopt_as_prefar_standard_control_null_package") is not True:
        errors.append("standard-control 871250 must be adopted as pre-FAR standard-control null package")
    if prompt_validation.get("status") != "PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_PROMPT_BANK_VALIDATION_NO_SUBMIT":
        errors.append("organic-null prompt-bank v2 validation must pass")
    if row_validation.get("status") != "PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_ROW_BANK_VALIDATION_NO_SUBMIT":
        errors.append("organic-null row-bank v2 validation must pass")
    if manifest.get("status") != "PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT":
        errors.append("organic-null row-bank manifest must pass")

    req = mapping(config.get("row_bank_requirements"), "row_bank_requirements", errors)
    for field, expected in (
        ("expected_rows", 262144),
        ("expected_shards", 256),
        ("expected_rows_per_shard", 1024),
        ("expected_selected_coordinates", 16),
    ):
        if int_field(req, field) != expected:
            errors.append(f"row_bank_requirements.{field} must be {expected}")
    if req.get("expected_generation_conditions") != ["raw"]:
        errors.append("organic tokenizer route must be raw-only")
    if req.get("expected_organic_null") is not True:
        errors.append("expected_organic_null must be true")
    if req.get("expected_standard_control_null_expansion") is not False:
        errors.append("expected_standard_control_null_expansion must be false")
    if req.get("contract_id") != "a55e" or req.get("same_contract_only") is not True:
        errors.append("same-contract a55e requirements mismatch")

    for field, expected in (
        ("row_count", 262144),
        ("target_shards", 256),
        ("rows_per_shard", 1024),
        ("selected_coordinate_count", 16),
    ):
        if int_field(manifest, field) != expected:
            errors.append(f"manifest {field} must be {expected}")
    if manifest.get("generation_conditions") != ["raw"]:
        errors.append("manifest generation_conditions must be raw only")
    if manifest.get("organic_null") is not True:
        errors.append("manifest organic_null must be true")
    if manifest.get("prompts_sha256") != req.get("expected_prompt_bank_sha256"):
        errors.append("manifest prompt-bank sha256 mismatch")
    if int_field(row_validation, "rows") != 262144:
        errors.append("row validation rows must be 262144")

    scope = mapping(config.get("tokenizer_preflight_scope"), "tokenizer_preflight_scope", errors)
    if scope.get("tokenizer_name") != "Qwen/Qwen2.5-7B-Instruct":
        errors.append("tokenizer_name mismatch")
    if int_field(scope, "max_rows") != 262144:
        errors.append("tokenizer max_rows must be 262144")
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

    gate = mapping(config.get("future_tokenizer_gate"), "future_tokenizer_gate", errors)
    for field, expected in (
        ("checked_rows", 262144),
        ("failed_rows_max", 0),
        ("empty_target_id_row_count_max", 0),
        ("empty_other_id_row_count_max", 0),
        ("target_other_overlap_row_count_max", 0),
    ):
        if int_field(gate, field) != expected:
            errors.append(f"future_tokenizer_gate.{field} must be {expected}")

    compute = mapping(config.get("compute_policy"), "compute_policy", errors)
    wrapper = root_path(compute.get("wrapper", ""), "compute_policy.wrapper", errors)
    if compute.get("allowlist_entry") != EXPECTED_ENTRY:
        errors.append("allowlist entry mismatch")
    if compute.get("wrapper") != EXPECTED_WRAPPER:
        errors.append("wrapper mismatch")
    if compute.get("command_pattern") != f"sbatch {EXPECTED_WRAPPER}":
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

    if rows_path.exists() and rows_path.stat().st_size <= 0:
        errors.append("row bank rows must be nonempty")
    if wrapper.exists() and wrapper.stat().st_size <= 0:
        errors.append("wrapper must be nonempty")

    enabled_entries: list[str] = []
    if not skip_allowlist_state_check:
        allowlist = load_yaml(ALLOWLIST)
        enabled_entries = enabled_allowlist_entries(allowlist)
        entry = find_allowlist_entry(allowlist, EXPECTED_ENTRY)
        if entry is None:
            errors.append(f"allowlist entry missing: {EXPECTED_ENTRY}")
        elif entry.get("command_pattern") != f"sbatch {EXPECTED_WRAPPER}":
            errors.append("allowlist command_pattern mismatch")
        expected_enabled = bool(allow_submission_enabled_entry)
        if entry is not None and entry.get("enabled") is not expected_enabled:
            errors.append(f"allowlist entry enabled must be {expected_enabled}")
        if allow_submission_enabled_entry:
            if enabled_entries != [EXPECTED_ENTRY]:
                errors.append("enabled allowlist entries must be exactly the organic tokenizer preflight entry")
        elif enabled_entries:
            errors.append("enabled allowlist entries must be empty")

    status = (
        "PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_TOKENIZER_ROUTE_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_870987_PREFAR_ORGANIC_NULL_TOKENIZER_ROUTE_VALIDATION_NO_SUBMIT"
    )
    return {
        "schema_name": "natural_evidence_v2_r4_after_870987_prefar_organic_null_tokenizer_route_validation_v1",
        "status": status,
        "errors": errors,
        "enabled_allowlist_entries": enabled_entries,
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "wrapper_sha256": sha256_file(ROOT / EXPECTED_WRAPPER),
        "row_bank_rows_sha256": sha256_file(rows_path) if rows_path.exists() else None,
        "row_bank_manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "prompt_bank_sha256": manifest.get("prompts_sha256"),
        "generation_started": False,
        "scoring_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Run exactly one reviewed H200 Qwen tokenizer boundary preflight for organic-null v2 rows; "
            "do not run generation until tokenizer preflight passes and is reviewed."
        ),
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {args.output_dir}")
    config = load_yaml(args.config)
    summary = validate_route(
        config,
        allow_submission_enabled_entry=bool(args.allow_submission_enabled_entry),
        skip_allowlist_state_check=bool(args.skip_allowlist_state_check),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_new(args.output_dir / "validation_summary.json", summary)
    write_text_new(
        args.output_dir / "validation_report.md",
        "# R4 Pre-FAR Organic-Null Qwen Tokenizer Route Validation\n\n"
        f"Status: `{summary['status']}`\n\n"
        f"Errors: {len(summary['errors'])}\n\n"
        "This is route validation only. It does not run tokenizer preflight, generation, scoring, training, or Slurm.\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
