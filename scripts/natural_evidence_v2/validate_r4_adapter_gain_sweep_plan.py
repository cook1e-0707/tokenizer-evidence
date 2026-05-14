from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_candidate_v3_adapter_gain_sweep.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_plan(plan: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if plan.get("schema_name") != "natural_evidence_v2_r4_candidate_v3_adapter_gain_sweep_plan_v1":
        errors.append("schema_name mismatch")
    gains = plan.get("adapter_gain_values")
    if not isinstance(gains, list) or not gains:
        errors.append("adapter_gain_values must be a non-empty list")
        gains = []
    numeric_gains = [float(item) for item in gains]
    if sorted(numeric_gains) != numeric_gains:
        errors.append("adapter_gain_values must be sorted ascending")
    if len(set(numeric_gains)) != len(numeric_gains):
        errors.append("adapter_gain_values must be unique")
    if 0.0 not in numeric_gains or 1.0 not in numeric_gains:
        errors.append("adapter_gain_values must include 0.0 and 1.0")

    conditions = plan.get("conditions")
    if not isinstance(conditions, list):
        errors.append("conditions must be a list")
        conditions = []
    condition_set = {str(item) for item in conditions}
    if "base" not in condition_set:
        errors.append("base condition missing")
    if "task_only" not in condition_set:
        errors.append("task_only condition missing")
    protected_conditions = sorted(item for item in condition_set if item.startswith("protected_gain_"))
    expected_protected = sorted(
        f"protected_gain_{('%g' % float(gain)).replace('.', '_')}" for gain in numeric_gains
    )
    if protected_conditions != expected_protected:
        errors.append(f"protected gain conditions mismatch: expected {expected_protected}, observed {protected_conditions}")
    if any(item.startswith("base_gain_") or item.startswith("task_only_gain_") for item in condition_set):
        errors.append("base/task_only gain scaling conditions are not allowed")

    for field in (
        "scoring_only",
        "future_compute_requires_reviewed_route",
    ):
        if plan.get(field) is not True:
            errors.append(f"{field} must be true")
    for field in (
        "generation_allowed",
        "training_allowed",
        "qwen_e2e_allowed",
        "llama_allowed",
        "same_family_null_allowed",
        "sanitizer_allowed",
        "far_aggregation_allowed",
        "paper_claim_allowed",
        "allowlist_enablement_allowed",
    ):
        if plan.get(field) is not False:
            errors.append(f"{field} must be false")

    candidate_rows_value = plan.get("candidate_rows")
    candidate_rows = root / str(candidate_rows_value) if candidate_rows_value else None
    candidate_hash = None
    row_count = 0
    if candidate_rows is None or not candidate_rows.exists():
        errors.append("candidate_rows missing or does not exist")
    else:
        candidate_hash = sha256_file(candidate_rows)
        expected_hash = str(plan.get("candidate_rows_sha256", ""))
        if expected_hash and candidate_hash != expected_hash:
            errors.append("candidate_rows_sha256 mismatch")
        with candidate_rows.open("r", encoding="utf-8") as handle:
            row_count = sum(1 for line in handle if line.strip())
        if row_count != 8192:
            errors.append(f"candidate row count must remain 8192, observed {row_count}")

    status = "PASS_R4_ADAPTER_GAIN_SWEEP_PLAN_VALIDATION" if not errors else "FAIL_R4_ADAPTER_GAIN_SWEEP_PLAN_VALIDATION"
    return {
        "candidate_rows_sha256_observed": candidate_hash,
        "condition_count": len(condition_set),
        "errors": errors,
        "generation_started": False,
        "model_scoring_started": False,
        "row_count": row_count,
        "status": status,
        "training_started": False,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 candidate v3 adapter-gain sweep plan without scoring.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_plan(load_yaml(args.config))
    if args.output_dir is not None:
        write_json(args.output_dir / "adapter_gain_sweep_plan_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
