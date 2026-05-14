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

DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_positive_dev_diagnostic_route.yaml"
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


def validate_route(config: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_positive_dev_diagnostic_route_v1":
        errors.append("schema_name mismatch")
    if config.get("route_id") != "r4_positive_event_bank_dev_diagnostic_v1":
        errors.append("route_id mismatch")
    if config.get("payload_id") != "a55e":
        errors.append("route must remain same-contract a55e at this stage")

    permissions = _mapping(config.get("current_permissions"), "current_permissions", errors)
    for field in LOCKED_FALSE_FIELDS:
        if permissions.get(field) is not False:
            errors.append(f"current_permissions.{field} must be false in route-scope review")

    package_dir = root / str(config.get("source_precommit_package", ""))
    summary_path = package_dir / "package_summary.json"
    manifest_path = package_dir / "precommit_manifest.json"
    if not summary_path.exists():
        errors.append("source precommit package_summary.json missing")
        summary: Mapping[str, Any] = {}
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not manifest_path.exists():
        errors.append("source precommit precommit_manifest.json missing")
    expected_hash = str(config.get("expected_precommit_hash", ""))
    if summary.get("status") != "PASS_R4_POSITIVE_EVENT_BANK_PRECOMMIT_PACKAGE":
        errors.append("source precommit package has not passed")
    if summary.get("precommit_hash") != expected_hash:
        errors.append("expected_precommit_hash does not match source package")
    if summary.get("key_material_exposed") is not False:
        errors.append("source precommit package must not expose key material")
    if int(summary.get("distinct_coordinate_count", 0)) < 20:
        errors.append("source precommit package coordinate coverage below route minimum")

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
    for field in ("same_contract_only",):
        if scope.get(field) is not True:
            errors.append(f"future_route_scope.{field} must be true")
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
    if float(gate.get("shallow_auc_max", 1.0)) > 0.60:
        errors.append("future_dev_gate.shallow_auc_max must be <= 0.60")
    if float(gate.get("min_specificity_margin", 0.0)) < 3.0:
        errors.append("future_dev_gate.min_specificity_margin must be >= 3.0")

    prerequisites = _mapping(config.get("required_before_any_submission"), "required_before_any_submission", errors)
    for field, value in prerequisites.items():
        if value is not True:
            errors.append(f"required_before_any_submission.{field} must be true")

    locked = _mapping(config.get("not_unlocked_by_this_route_scope"), "not_unlocked_by_this_route_scope", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_scope.{field} must be true")

    status = (
        "PASS_R4_POSITIVE_DEV_DIAGNOSTIC_ROUTE_SCOPE_REVIEW_NO_SUBMIT"
        if not errors
        else "FAIL_R4_POSITIVE_DEV_DIAGNOSTIC_ROUTE_SCOPE_REVIEW_NO_SUBMIT"
    )
    return {
        "status": status,
        "errors": errors,
        "route_id": config.get("route_id"),
        "source_precommit_package": config.get("source_precommit_package"),
        "precommit_hash": expected_hash,
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
    parser = argparse.ArgumentParser(description="Validate the R4 positive dev diagnostic route scope.")
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
