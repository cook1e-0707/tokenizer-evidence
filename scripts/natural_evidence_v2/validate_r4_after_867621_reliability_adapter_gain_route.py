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

from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_867621_reliability_adapter_gain_sweep.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_867621_reliability_adapter_gain_sweep_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_867621_reliability_adapter_gain_sweep_h200.sbatch"
EXPECTED_ROWS = 4096
EXPECTED_GAINS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def _mapping(value: Any, field: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be a mapping")
        return {}
    return value


def _path(value: Any, field: str, errors: list[str]) -> Path:
    path = ROOT / str(value)
    if not path.exists():
        errors.append(f"{field} missing: {path}")
    return path


def _read_json(path: Path, field: str, errors: list[str]) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        errors.append(f"{field} unreadable JSON: {exc}")
        return {}
    if not isinstance(payload, Mapping):
        errors.append(f"{field} must be a JSON object")
        return {}
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_allowlist_entry(allowlist: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        entries = allowlist.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("name") == name:
                return entry
    return None


def _expected_conditions(gains: list[float]) -> list[str]:
    return ["base", "task_only"] + [f"protected_gain_{('%g' % gain).replace('.', '_')}" for gain in gains]


def validate_route(config: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_after_867621_reliability_adapter_gain_sweep_plan_v1":
        errors.append("schema_name mismatch")
    if config.get("source_failure_job") != 867849:
        errors.append("source_failure_job must be 867849")
    if config.get("source_failure_status") != "FAIL_R4_AFTER_867621_RELIABILITY_SURFACE_MASS_GATE":
        errors.append("source_failure_status mismatch")

    failure_path = _path(config.get("source_failure_analysis", ""), "source_failure_analysis", errors)
    failure = _read_json(failure_path, "source_failure_analysis", errors) if failure_path.exists() else {}
    if failure.get("teacher_forced_surface_gate_status") != "FAIL":
        errors.append("source failure analysis must record teacher-forced gate FAIL")
    if failure.get("tokenizer_boundary_valid") is not True:
        errors.append("source failure analysis must record tokenizer_boundary_valid=true")
    if failure.get("clean_slurm_completion") is not True:
        errors.append("source failure analysis must record clean_slurm_completion=true")
    if failure.get("generation_unlocked") is not False:
        errors.append("source failure analysis must record generation_unlocked=false")

    rows_path = _path(config.get("candidate_rows", ""), "candidate_rows", errors)
    row_count = 0
    observed_hash = ""
    if rows_path.exists():
        observed_hash = _sha256(rows_path)
        row_count = sum(1 for line in rows_path.open("r", encoding="utf-8") if line.strip())
    if observed_hash != str(config.get("candidate_rows_sha256", "")):
        errors.append("candidate_rows_sha256 mismatch")
    if row_count != EXPECTED_ROWS:
        errors.append(f"candidate_rows line count must be {EXPECTED_ROWS}, observed {row_count}")

    gains = config.get("adapter_gain_values")
    if gains != EXPECTED_GAINS:
        errors.append(f"adapter_gain_values must be {EXPECTED_GAINS}")
    conditions = config.get("conditions")
    if conditions != _expected_conditions(EXPECTED_GAINS):
        errors.append("conditions do not match gain grid")
    if any(str(item).startswith(("base_gain_", "task_only_gain_")) for item in (conditions or [])):
        errors.append("base/task_only gain scaling conditions are forbidden")

    for field in ("scoring_only", "future_compute_requires_reviewed_route"):
        if config.get(field) is not True:
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
        if config.get(field) is not False:
            errors.append(f"{field} must be false")

    route = _mapping(config.get("route"), "route", errors)
    if route.get("allowlist_entry") != EXPECTED_ENTRY:
        errors.append("route.allowlist_entry mismatch")
    if route.get("wrapper") != EXPECTED_WRAPPER:
        errors.append("route.wrapper mismatch")
    if route.get("command_pattern") != f"sbatch {EXPECTED_WRAPPER}":
        errors.append("route.command_pattern mismatch")
    if int(route.get("max_rows", 0)) != EXPECTED_ROWS:
        errors.append("route.max_rows must be 4096")

    wrapper_path = _path(route.get("wrapper", ""), "route.wrapper", errors)
    wrapper_text = wrapper_path.read_text(encoding="utf-8") if wrapper_path.exists() else ""
    required_fragments = (
        "#SBATCH --partition=pomplun",
        "#SBATCH --account=cs_yinxin.wan",
        "#SBATCH --qos=pomplun",
        "#SBATCH --gres=gpu:h200:1",
        "#SBATCH --time=30-00:00:00",
        "r4_after_867621_reliability_surface_mass_rows_20260516/reliability_surface_mass_rows.jsonl",
        "r4_candidate_v3_micro_overfit_864761/protected_micro_overfit_train/adapter",
        "wp5_r2_teacher_forced_train_and_score_851481/task_only_train/adapter",
        "PROTECTED_ADAPTER_GAINS",
        "--protected-adapter-gains",
        "0.0,0.5,1.0,1.5,2.0,3.0,4.0",
        "generation_started=false",
        "training_started=false",
        "--require-cuda",
    )
    for fragment in required_fragments:
        if fragment not in wrapper_text:
            errors.append(f"wrapper missing fragment: {fragment}")

    gate = _mapping(config.get("teacher_forced_gate"), "teacher_forced_gate", errors)
    if float(gate.get("protected_lift_vs_base_min", 0.0)) < 0.15:
        errors.append("teacher_forced_gate lift vs base too low")
    if float(gate.get("protected_lift_vs_task_only_min", 0.0)) < 0.10:
        errors.append("teacher_forced_gate lift vs task_only too low")
    if float(gate.get("protected_rank1_min", 0.0)) < 0.75:
        errors.append("teacher_forced_gate rank1 too low")
    if float(gate.get("target_other_overlap_rate_max", 1.0)) != 0.0:
        errors.append("teacher_forced_gate target_other_overlap_rate_max must be 0")
    if int(gate.get("boundary_failures_max", -1)) != 0:
        errors.append("teacher_forced_gate boundary_failures_max must be 0")

    compute = _mapping(config.get("compute_policy"), "compute_policy", errors)
    if compute.get("partition") != "pomplun" or compute.get("qos") != "pomplun":
        errors.append("compute policy must use pomplun")
    if compute.get("account") != "cs_yinxin.wan":
        errors.append("compute account mismatch")
    if compute.get("gres") != "gpu:h200:1":
        errors.append("compute gres mismatch")
    if compute.get("max_time") != "30-00:00:00":
        errors.append("compute max_time mismatch")
    if compute.get("allowlist_enabled_now") is not False:
        errors.append("compute allowlist_enabled_now must be false")

    locked = _mapping(config.get("not_unlocked_by_this_route_package"), "not_unlocked_by_this_route_package", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_package.{field} must be true")

    allowlist = load_yaml(ALLOWLIST)
    entry = _find_allowlist_entry(allowlist, EXPECTED_ENTRY)
    if entry is None:
        errors.append("allowlist entry missing")
    else:
        if entry.get("enabled") is not False:
            errors.append("allowlist entry must be disabled")
        if entry.get("command_pattern") != f"sbatch {EXPECTED_WRAPPER}":
            errors.append("allowlist command_pattern mismatch")

    status = (
        "PASS_R4_AFTER_867621_RELIABILITY_ADAPTER_GAIN_ROUTE_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_867621_RELIABILITY_ADAPTER_GAIN_ROUTE_VALIDATION_NO_SUBMIT"
    )
    return {
        "status": status,
        "errors": errors,
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
        "score_rows": config.get("candidate_rows"),
        "score_rows_sha256_observed": observed_hash,
        "expected_rows": EXPECTED_ROWS,
        "row_count": row_count,
        "adapter_gain_values": EXPECTED_GAINS,
        "slurm_job_submitted": False,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R4 after-867621 reliability adapter-gain sweep route.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_route(load_yaml(args.config))
    if args.output_dir is not None:
        write_json_new(args.output_dir / "reliability_adapter_gain_route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
