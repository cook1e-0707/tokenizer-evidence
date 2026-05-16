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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_867621_reliability_surface_mass_score_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_867621_reliability_surface_mass_score_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_867621_reliability_surface_mass_score_h200.sbatch"
EXPECTED_ROWS = 4096


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


def _find_allowlist_entry(allowlist: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        entries = allowlist.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("name") == name:
                return entry
    return None


def validate_route(config: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_after_867621_reliability_surface_mass_score_route_v1":
        errors.append("schema_name mismatch")

    tok = _mapping(config.get("source_tokenizer_preflight"), "source_tokenizer_preflight", errors)
    tok_summary_path = _path(tok.get("summary", ""), "source_tokenizer_preflight.summary", errors)
    tok_review_path = _path(tok.get("review_summary", ""), "source_tokenizer_preflight.review_summary", errors)
    tok_summary = _read_json(tok_summary_path, "source_tokenizer_preflight.summary", errors) if tok_summary_path.exists() else {}
    tok_review = _read_json(tok_review_path, "source_tokenizer_preflight.review_summary", errors) if tok_review_path.exists() else {}
    if tok.get("job_id") != "867828":
        errors.append("source_tokenizer_preflight.job_id must be 867828")
    if tok_summary.get("status") != tok.get("expected_status"):
        errors.append("tokenizer preflight status mismatch")
    if tok_review.get("review_status") != tok.get("expected_review_status"):
        errors.append("tokenizer preflight review_status mismatch")
    summary_field_for = {
        "checked_rows": "checked_row_count",
        "failed_rows": "failed_row_count",
        "empty_target_id_row_count": "empty_target_id_row_count",
        "empty_other_id_row_count": "empty_other_id_row_count",
        "target_other_overlap_row_count": "target_other_overlap_row_count",
    }
    for field, summary_field in summary_field_for.items():
        expected = int(tok.get(field, -999))
        observed = int(tok_summary.get(summary_field, -999))
        if observed != expected:
            errors.append(f"tokenizer preflight {field} mismatch: expected {expected}, observed {observed}")

    rows = _mapping(config.get("source_rows"), "source_rows", errors)
    rows_path = _path(rows.get("rows", ""), "source_rows.rows", errors)
    rows_summary_path = _path(rows.get("summary", ""), "source_rows.summary", errors)
    rows_summary = _read_json(rows_summary_path, "source_rows.summary", errors) if rows_summary_path.exists() else {}
    if rows_summary.get("status") != rows.get("expected_status"):
        errors.append("source rows status mismatch")
    if int(rows_summary.get("row_count", -1)) != EXPECTED_ROWS:
        errors.append("source rows summary count must be 4096")
    if int(rows.get("expected_rows", -1)) != EXPECTED_ROWS:
        errors.append("source_rows.expected_rows must be 4096")
    if rows_summary.get("current_two_way_scorer_compatible") is not True:
        errors.append("source rows must be current_two_way_scorer_compatible=true")
    if rows_path.exists() and sum(1 for _ in rows_path.open("r", encoding="utf-8")) != EXPECTED_ROWS:
        errors.append("source rows JSONL line count mismatch")

    route = _mapping(config.get("route"), "route", errors)
    if route.get("allowlist_entry") != EXPECTED_ENTRY:
        errors.append("route.allowlist_entry mismatch")
    if route.get("wrapper") != EXPECTED_WRAPPER:
        errors.append("route.wrapper mismatch")
    if route.get("command_pattern") != f"sbatch {EXPECTED_WRAPPER}":
        errors.append("route.command_pattern mismatch")
    if route.get("conditions") != ["base", "protected", "task_only"]:
        errors.append("route.conditions must be base/protected/task_only")
    if int(route.get("max_rows", 0)) != EXPECTED_ROWS:
        errors.append("route.max_rows must be 4096")
    for field in ("generation_allowed", "training_allowed", "llama_allowed"):
        if route.get(field) is not False:
            errors.append(f"route.{field} must be false")

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
        "generation_started=false",
        "training_started=false",
        "--require-cuda",
    )
    for fragment in required_fragments:
        if fragment not in wrapper_text:
            errors.append(f"wrapper missing fragment: {fragment}")

    gate = _mapping(config.get("future_teacher_forced_gate"), "future_teacher_forced_gate", errors)
    if float(gate.get("protected_lift_vs_base_min", 0.0)) < 0.15:
        errors.append("future gate lift vs base too low")
    if float(gate.get("protected_lift_vs_task_only_min", 0.0)) < 0.10:
        errors.append("future gate lift vs task_only too low")
    if float(gate.get("protected_rank1_min", 0.0)) < 0.75:
        errors.append("future gate rank1 too low")
    if int(gate.get("boundary_failures_max", -1)) != 0:
        errors.append("future gate boundary_failures_max must be 0")
    if float(gate.get("target_other_overlap_rate_max", 1.0)) != 0.0:
        errors.append("future gate target_other_overlap_rate_max must be 0")

    compute = _mapping(config.get("compute_policy"), "compute_policy", errors)
    if compute.get("partition") != "pomplun" or compute.get("qos") != "pomplun":
        errors.append("compute policy must use pomplun")
    if compute.get("account") != "cs_yinxin.wan":
        errors.append("compute account mismatch")
    if compute.get("gres") != "gpu:h200:1":
        errors.append("compute gres mismatch")
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
        "PASS_R4_AFTER_867621_RELIABILITY_SURFACE_MASS_ROUTE_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_867621_RELIABILITY_SURFACE_MASS_ROUTE_VALIDATION_NO_SUBMIT"
    )
    return {
        "status": status,
        "errors": errors,
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
        "score_rows": rows.get("rows"),
        "expected_rows": EXPECTED_ROWS,
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
    parser = argparse.ArgumentParser(description="Validate R4 after-867621 reliability surface-mass scoring route.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_route(load_yaml(args.config))
    if args.output_dir is not None:
        write_json_new(args.output_dir / "reliability_surface_mass_route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
