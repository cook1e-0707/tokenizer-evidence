from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_positive_selectivity_dev_diagnostic_route.yaml"
REQUIRED_CONDITIONS = {"protected", "raw", "task_only", "wrong_key", "wrong_payload"}
LOCKED_FALSE_FIELDS = (
    "slurm_allowed",
    "allowlist_enablement_allowed",
    "generation_allowed",
    "training_allowed",
    "tokenizer_model_scoring_allowed",
    "qwen_e2e_allowed",
    "llama_allowed",
    "same_family_null_allowed",
    "sanitizer_allowed",
    "far_aggregation_allowed",
    "payload_diversity_allowed",
    "paper_claim_allowed",
)


def _mapping(value: Any, field: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be a mapping")
        return {}
    return value


def _read_json(path: Path, errors: list[str]) -> Mapping[str, Any]:
    if not path.exists():
        errors.append(f"missing JSON artifact: {path}")
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        errors.append(f"JSON artifact is not an object: {path}")
        return {}
    return payload


def validate_route(config: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_positive_selectivity_dev_diagnostic_route_v1":
        errors.append("schema_name mismatch")
    if config.get("route_id") != "r4_positive_selectivity_dev_diagnostic_v1":
        errors.append("route_id mismatch")
    if config.get("payload_id") != "a55e":
        errors.append("route must remain same-contract a55e at this stage")

    permissions = _mapping(config.get("current_permissions"), "current_permissions", errors)
    for field in LOCKED_FALSE_FIELDS:
        if permissions.get(field) is not False:
            errors.append(f"current_permissions.{field} must be false in route-scope review")

    package_dir = root / str(config.get("source_selectivity_package", ""))
    package_summary = _read_json(package_dir / "package_summary.json", errors)
    if package_summary.get("status") != "PASS_SELECTIVITY_REPAIR_PACKAGE_STATIC_VALIDATION_NO_COMPUTE":
        errors.append("source selectivity package has not passed static validation")
    expected_bank_hash = str(config.get("expected_event_window_bank_hash", ""))
    if package_summary.get("artifact_hashes", {}).get("event_window_bank.json") != expected_bank_hash:
        errors.append("expected_event_window_bank_hash does not match source package")
    if package_summary.get("generic_raw_task_accept") is not False:
        errors.append("source selectivity package generic raw/task fixture must reject")
    if package_summary.get("wrong_key_accept") is not False or package_summary.get("wrong_payload_accept") is not False:
        errors.append("source selectivity package wrong-key/wrong-payload fixtures must reject")

    prompt_dir = root / str(config.get("source_prompt_policy_package", ""))
    prompt_manifest = _read_json(prompt_dir / "prompt_policy_manifest.json", errors)
    if prompt_manifest.get("status") != "PASS_SELECTIVITY_PROMPT_POLICY_STATIC_VALIDATION_NO_COMPUTE":
        errors.append("source prompt policy package has not passed static validation")
    expected_prompt_hash = str(config.get("expected_prompt_bank_hash", ""))
    if prompt_manifest.get("prompt_bank_sha256") != expected_prompt_hash:
        errors.append("expected_prompt_bank_hash does not match prompt policy package")
    if int(prompt_manifest.get("prompt_count", 0)) < 2048:
        errors.append("prompt policy package must include at least 2048 prompts")
    if int(prompt_manifest.get("forbidden_prompt_violation_count", 1)) != 0:
        errors.append("prompt policy package has forbidden prompt violations")

    scope = _mapping(config.get("future_route_scope"), "future_route_scope", errors)
    if scope.get("model_family") != "qwen_only":
        errors.append("future route must remain qwen_only")
    if scope.get("partition") != "pomplun" or scope.get("qos") != "pomplun":
        errors.append("future route must use H200 pomplun policy")
    if scope.get("account") != "cs_yinxin.wan":
        errors.append("future route account must be cs_yinxin.wan")
    if str(scope.get("gpu", "")).lower() != "h200":
        errors.append("future route GPU must be H200")
    if scope.get("max_time") != "30-00:00:00":
        errors.append("future route must use max H200 time limit")
    if scope.get("primary_scrub_mode") != "all":
        errors.append("primary scrub mode must be all")
    if set(scope.get("conditions", [])) != REQUIRED_CONDITIONS:
        errors.append("condition set mismatch")
    if int(scope.get("block_count", 0)) != 32:
        errors.append("dev diagnostic must use 32 blocks")
    if int(scope.get("prompts_per_block", 0)) != 64:
        errors.append("dev diagnostic must use 64 prompts per block")
    if int(scope.get("shards", 0)) != 4:
        errors.append("dev diagnostic must use four H200 shards")
    if int(scope.get("prompts_per_shard", 0)) != 512:
        errors.append("dev diagnostic must use 512 prompts per shard")
    if scope.get("same_contract_only") is not True:
        errors.append("future_route_scope.same_contract_only must be true")
    for field in ("payload_diversity_tested", "llama_tested", "paper_facing"):
        if scope.get(field) is not False:
            errors.append(f"future_route_scope.{field} must be false")

    gate = _mapping(config.get("future_dev_gate"), "future_dev_gate", errors)
    if int(gate.get("protected_accepts_min", 0)) < 26:
        errors.append("future_dev_gate.protected_accepts_min must be >= 26")
    if int(gate.get("protected_blocks", 0)) != 32:
        errors.append("future_dev_gate.protected_blocks must be 32")
    if int(gate.get("control_accepts_max_per_condition", -1)) != 0:
        errors.append("controls must require zero accepts")
    if float(gate.get("min_specificity_margin", 0.0)) < 4.0:
        errors.append("future_dev_gate.min_specificity_margin must be >= 4.0")
    if float(gate.get("min_keyed_score", 0.0)) < 8.0:
        errors.append("future_dev_gate.min_keyed_score must be >= 8.0")

    prerequisites = _mapping(config.get("required_before_any_submission"), "required_before_any_submission", errors)
    for field, value in prerequisites.items():
        if value is not True:
            errors.append(f"required_before_any_submission.{field} must be true")

    locked = _mapping(config.get("not_unlocked_by_this_route_scope"), "not_unlocked_by_this_route_scope", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_scope.{field} must be true")

    status = (
        "PASS_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_ROUTE_SCOPE_REVIEW_NO_SUBMIT"
        if not errors
        else "FAIL_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_ROUTE_SCOPE_REVIEW_NO_SUBMIT"
    )
    return {
        "status": status,
        "errors": errors,
        "route_id": config.get("route_id"),
        "source_selectivity_package": config.get("source_selectivity_package"),
        "source_prompt_policy_package": config.get("source_prompt_policy_package"),
        "current_compute_unlocked": False,
        "slurm_job_submitted": False,
        "allowlist_enabled": False,
        "generation_started": False,
        "training_started": False,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 selectivity dev diagnostic route scope.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_route(load_yaml(args.config))
    if args.output_dir is not None:
        write_json(args.output_dir / "route_scope_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
