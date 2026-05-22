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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_870987_same_family_raw_null_generation_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_870987_same_family_raw_null_generation_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_870987_same_family_raw_null_generation_h200.sbatch"
EXPECTED_COMMAND = f"PLAN_ONLY=0 VALIDATE_PLAN_ONLY=0 sbatch {EXPECTED_WRAPPER}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R4 same-family raw-null route planning.")
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


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = load_yaml(config_path)
    errors: list[str] = []

    if cfg.get("schema_name") != "natural_evidence_v2_r4_after_870987_same_family_raw_null_generation_route_v1":
        errors.append("schema_name mismatch")
    if cfg.get("route_id") != "r4_after_870987_same_family_raw_null_generation_v1":
        errors.append("route_id mismatch")

    source = cfg.get("source_artifacts", {})
    if not isinstance(source, Mapping):
        errors.append("source_artifacts must be a mapping")
        source = {}
    prefar_review = read_json(root_path(source.get("prefar_null_package_review", ""), errors, "prefar review"))
    locked = read_json(root_path(source.get("locked_scale_summary", ""), errors, "locked summary"))
    standard = read_json(root_path(source.get("standard_control_summary", ""), errors, "standard summary"))
    organic = read_json(root_path(source.get("organic_null_summary", ""), errors, "organic summary"))
    row_bank_rows = root_path(source.get("row_bank_rows", ""), errors, "row bank rows")
    tokenizer_review = read_json(root_path(source.get("same_family_tokenizer_review", ""), errors, "same-family tokenizer review"))

    if prefar_review.get("review_status") != "PASS_R4_AFTER_870987_PREFAR_NULL_PACKAGE_871250_PLUS_874308":
        errors.append("pre-FAR null package review must pass")
    if locked.get("status") != "PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE":
        errors.append("locked scale summary must pass")
    if standard.get("status") != "PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_GENERATION_GATE":
        errors.append("standard-control summary must pass")
    if organic.get("status") != "PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_GENERATION_GATE":
        errors.append("organic-null summary must pass")
    if tokenizer_review.get("review_status") != "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_TOKENIZER_PREFLIGHT_874778":
        errors.append("same-family tokenizer review 874778 must pass")
    if int(tokenizer_review.get("total_checked_rows", -1)) != 786432:
        errors.append("same-family tokenizer review must check 786432 rows")
    if row_bank_rows.exists() and sum(1 for line in row_bank_rows.open("r", encoding="utf-8") if line.strip()) != 262144:
        errors.append("row bank must contain 262144 rows")

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

    wrapper = root_path(cfg.get("compute_policy", {}).get("wrapper", ""), errors, "wrapper")
    compute = cfg.get("compute_policy", {})
    for key, expected in {
        "partition": "pomplun",
        "qos": "pomplun",
        "account": "cs_yinxin.wan",
        "gres": "gpu:h200:1",
        "max_time": "30-00:00:00",
        "array": "0-191%6",
        "allowlist_entry": EXPECTED_ENTRY,
        "command_pattern": EXPECTED_COMMAND,
    }.items():
        if compute.get(key) != expected:
            errors.append(f"compute_policy.{key} mismatch")
    if wrapper.exists():
        text = wrapper.read_text(encoding="utf-8")
        for fragment in ("#SBATCH --array=0-191%6", "MODEL_GROUP_INDEX", "LOCAL_SHARD_INDEX", "Qwen/Qwen2.5-14B-Instruct", "r4_after_868016_controller_generation_h200.sbatch"):
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

    status = "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_ROUTE_PLAN_ONLY_NO_SUBMIT" if not errors else "FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_ROUTE_PLAN_ONLY_NO_SUBMIT"
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_870987_same_family_raw_null_generation_route_validation_v1",
        "status": status,
        "errors": errors,
        "allowlist_entry": EXPECTED_ENTRY,
        "enabled_entries": enabled,
        "expected_models": expected_models,
        "expected_total_shards": 192,
        "expected_generated_rows": 196608,
        "config_sha256": sha256_file(config_path) if config_path.exists() else "",
        "wrapper_sha256": sha256_file(ROOT / EXPECTED_WRAPPER) if (ROOT / EXPECTED_WRAPPER).exists() else "",
        "slurm_allowed": False,
        "generation_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": "Run local/remote hash preflight before exactly-one H200 same-family raw-null submission.",
    }
    if args.output_dir is not None:
        out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        write_json_new(out / "route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
