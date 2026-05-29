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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_877840_same_family_raw_null_generation_v8_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_870987_same_family_raw_null_generation_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_870987_same_family_raw_null_generation_h200.sbatch"
EXPECTED_GENERATOR = "scripts/natural_evidence_v2/generate_r4_after_868016_controller_outputs.py"
EXPECTED_TOKENIZER_STATUS = "PASS_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_TOKENIZER_PREFLIGHT_877892"
EXPECTED_COMMAND_FRAGMENTS = (
    "ROUTE_CONFIG=configs/natural_evidence_v2/r4_after_877840_same_family_raw_null_generation_v8_route.yaml",
    "ROUTE_VALIDATOR=scripts/natural_evidence_v2/validate_r4_after_877840_same_family_raw_null_generation_v8_route.py",
    "SCORE_ROWS=results/natural_evidence_v2/status/r4_after_877840_same_family_raw_null_v8_row_bank_plan_20260526/row_allocation_rows.jsonl",
    "ALLOCATION_ROWS=results/natural_evidence_v2/status/r4_after_877840_same_family_raw_null_v8_row_bank_plan_20260526/row_allocation_rows.jsonl",
    "TOKENIZER_REVIEW=results/natural_evidence_v2/status/r4_after_877840_same_family_raw_null_v8_tokenizer_preflight_877892_review/review_summary.json",
    "PUBLIC_RUN_SALT=r4_after_877840_same_family_raw_null_generation_v8",
    "SHARDS_PER_MODEL=64",
    "PLAN_ONLY=0",
    "VALIDATE_PLAN_ONLY=0",
    f"sbatch {EXPECTED_WRAPPER}",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R4 same-family raw-null v8 full generation route planning.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-submission-enabled-entry", action="store_true")
    parser.add_argument("--skip-allowlist-state-check", action="store_true")
    return parser.parse_args()


def root_path(value: Any, errors: list[str], label: str) -> Path:
    path = ROOT / str(value)
    if not path.exists():
        errors.append(f"{label} missing: {path}")
    return path


def enabled_entries() -> list[str]:
    allowlist = load_yaml(ALLOWLIST)
    enabled: list[str] = []
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        for entry in allowlist.get(section, []):
            if isinstance(entry, Mapping) and entry.get("enabled") is True:
                enabled.append(str(entry.get("name", "")))
    return enabled


def allowlist_entry() -> Mapping[str, Any] | None:
    allowlist = load_yaml(ALLOWLIST)
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        for entry in allowlist.get(section, []):
            if isinstance(entry, Mapping) and entry.get("name") == EXPECTED_ENTRY:
                return entry
    return None


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = load_yaml(config_path)
    errors: list[str] = []

    if cfg.get("schema_name") != "natural_evidence_v2_r4_after_877840_same_family_raw_null_generation_v8_route_v1":
        errors.append("schema_name mismatch")
    if cfg.get("route_id") != "r4_after_877840_same_family_raw_null_generation_v8":
        errors.append("route_id mismatch")

    source = cfg.get("source_artifacts", {})
    if not isinstance(source, Mapping):
        errors.append("source_artifacts must be a mapping")
        source = {}
    pass_summary = read_json(root_path(source.get("capacity_limited_generation_summary_877840", ""), errors, "877840 capacity-limited summary"))
    feasibility = read_json(root_path(source.get("full_source_feasibility_blocker", ""), errors, "full-source feasibility blocker"))
    row_manifest = read_json(root_path(source.get("v8_row_bank_manifest", ""), errors, "v8 row bank manifest"))
    row_validation = read_json(root_path(source.get("v8_row_bank_validation", ""), errors, "v8 row bank validation"))
    row_bank_rows = root_path(source.get("v8_row_bank_rows", ""), errors, "v8 row bank rows")
    tokenizer_review = read_json(root_path(source.get("same_family_v8_tokenizer_review", ""), errors, "same-family v8 tokenizer review"))

    if pass_summary.get("status") != "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE":
        errors.append("877840 capacity-limited summary must pass")
    if not pass_summary.get("same_family_raw_null_gate_pass", True):
        errors.append("877840 summary must not record a failed same-family raw-null gate")
    if feasibility.get("status") != "BLOCK_R4_AFTER_877840_FULL_SAME_FAMILY_RAW_NULL_CURRENT_SOURCE_INSUFFICIENT_NO_SUBMIT":
        errors.append("full-source feasibility blocker must be recorded")
    if feasibility.get("full_64_shard_feasible_from_current_source") is not False:
        errors.append("feasibility blocker must reject the old source for full 64-shard route")
    if row_manifest.get("status") != "PASS_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT":
        errors.append("v8 row bank manifest must pass")
    if row_validation.get("status") != "PASS_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_ROW_BANK_VALIDATION_NO_SUBMIT":
        errors.append("v8 row bank validation must pass")
    if int(row_manifest.get("selected_prompt_count", -1)) != 4096:
        errors.append("v8 row bank must select 4096 prompts")
    if int(row_manifest.get("target_shards", -1)) != 64:
        errors.append("v8 row bank must target 64 shards")
    if int(row_validation.get("selected_domain_count", -1)) < 32:
        errors.append("v8 row bank must have at least 32 selected domains")
    if tokenizer_review.get("status") != EXPECTED_TOKENIZER_STATUS:
        errors.append("same-family v8 tokenizer review 877892 must pass")
    if int(tokenizer_review.get("checked_rows_total", -1)) != 196608:
        errors.append("same-family v8 tokenizer review must check 196608 rows")
    if int(tokenizer_review.get("failed_rows_total", -1)) != 0:
        errors.append("same-family v8 tokenizer review failed rows must be 0")
    if row_bank_rows.exists() and count_jsonl_rows(row_bank_rows) != 65536:
        errors.append("v8 row bank must contain 65536 rows")

    repair = cfg.get("repair_policy", {})
    if not isinstance(repair, Mapping):
        errors.append("repair_policy must be a mapping")
        repair = {}
    for key in ("rescue_prior_failed_runs", "relax_hard_forbidden_policy", "reuse_capacity_limited_source_bank_for_full_route"):
        if repair.get(key) is not False:
            errors.append(f"repair_policy.{key} must be false")
    for key in ("use_fresh_expanded_prompt_source", "static_lexical_preflight_required", "tokenizer_preflight_required"):
        if repair.get(key) is not True:
            errors.append(f"repair_policy.{key} must be true")

    models = cfg.get("same_family_models", [])
    expected_models = [
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
    ]
    if [str(item.get("model_id", "")) for item in models if isinstance(item, Mapping)] != expected_models:
        errors.append("same_family_models mismatch")
    for item in models if isinstance(models, list) else []:
        if not isinstance(item, Mapping):
            errors.append("same_family_models entries must be mappings")
            continue
        if int(item.get("blocks", -1)) != 64:
            errors.append(f"{item.get('model_id')} blocks must be 64")
        if str(item.get("tokenizer_id", "")) != str(item.get("model_id", "")):
            errors.append(f"{item.get('model_id')} tokenizer_id must equal model_id")

    scope = cfg.get("generation_scope", {})
    if not isinstance(scope, Mapping):
        errors.append("generation_scope must be a mapping")
        scope = {}
    for key, expected in {
        "conditions": ["raw"],
        "rows_per_shard": 1024,
        "prompts_per_shard": 64,
        "selected_coordinate_count": 16,
        "shards_per_model": 64,
        "model_count": 3,
        "total_shards": 192,
        "expected_generated_rows": 196608,
        "contract_id": "a55e",
    }.items():
        if scope.get(key) != expected:
            errors.append(f"generation_scope.{key} mismatch")
    for key in ("payload_diversity_tested", "llama_tested", "paper_facing"):
        if scope.get(key) is not False:
            errors.append(f"generation_scope.{key} must be false")

    compute = cfg.get("compute_policy", {})
    if not isinstance(compute, Mapping):
        errors.append("compute_policy must be a mapping")
        compute = {}
    wrapper = root_path(compute.get("wrapper", ""), errors, "wrapper")
    generator = root_path(EXPECTED_GENERATOR, errors, "generator")
    for key, expected in {
        "partition": "pomplun",
        "qos": "pomplun",
        "account": "cs_yinxin.wan",
        "gres": "gpu:h200:1",
        "max_time": "30-00:00:00",
        "array": "0-191",
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
    }.items():
        if compute.get(key) != expected:
            errors.append(f"compute_policy.{key} mismatch")
    command = str(compute.get("command_pattern", ""))
    for fragment in EXPECTED_COMMAND_FRAGMENTS:
        if fragment not in command:
            errors.append(f"compute_policy.command_pattern missing fragment: {fragment}")
    if "%6" in command or "%3" in command:
        errors.append("compute_policy.command_pattern must not include a percent concurrency throttle")
    if wrapper.exists():
        text = wrapper.read_text(encoding="utf-8")
        for fragment in (
            "#SBATCH --array=0-191",
            "SHARDS_PER_MODEL",
            "MODEL_GROUP_INDEX",
            "LOCAL_SHARD_INDEX",
            "Qwen/Qwen2.5-14B-Instruct",
            "route_validation_shard_${LOCAL_SHARD_INDEX}",
            "r4_after_868016_controller_generation_h200.sbatch",
        ):
            if fragment not in text:
                errors.append(f"wrapper missing fragment: {fragment}")
    if generator.exists():
        text = generator.read_text(encoding="utf-8")
        if EXPECTED_TOKENIZER_STATUS not in text:
            errors.append("generator missing v8 tokenizer review status in allowed-status list")

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
        for fragment in EXPECTED_COMMAND_FRAGMENTS:
            if fragment not in entry_command:
                errors.append(f"allowlist command_pattern missing fragment: {fragment}")
        if "%6" in entry_command or "%3" in entry_command:
            errors.append("allowlist command_pattern must not include a percent concurrency throttle")

    status = (
        "PASS_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_GENERATION_ROUTE_PLAN_ONLY_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_GENERATION_ROUTE_PLAN_ONLY_NO_SUBMIT"
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_877840_same_family_raw_null_v8_generation_route_validation_v1",
        "status": status,
        "errors": errors,
        "allowlist_entry": EXPECTED_ENTRY,
        "enabled_entries": enabled,
        "expected_models": expected_models,
        "expected_total_shards": 192,
        "expected_generated_rows": 196608,
        "full_same_family_raw_null_route": True,
        "same_family_raw_null_full_package_claim_allowed": False,
        "config_sha256": sha256_file(config_path) if config_path.exists() else "",
        "wrapper_sha256": sha256_file(ROOT / EXPECTED_WRAPPER) if (ROOT / EXPECTED_WRAPPER).exists() else "",
        "generator_sha256": sha256_file(ROOT / EXPECTED_GENERATOR) if (ROOT / EXPECTED_GENERATOR).exists() else "",
        "row_bank_rows_sha256": sha256_file(row_bank_rows) if row_bank_rows.exists() else "",
        "tokenizer_review": source.get("same_family_v8_tokenizer_review", ""),
        "slurm_allowed": False,
        "generation_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": "Run local/remote hash preflight before exactly-one H200 same-family raw-null v8 generation submission.",
    }
    if args.output_dir is not None:
        out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        write_json_new(out / "route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
