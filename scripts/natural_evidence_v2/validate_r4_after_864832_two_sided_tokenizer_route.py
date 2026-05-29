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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_864832_two_sided_tokenizer_preflight_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_864832_two_sided_qwen_tokenizer_boundary_preflight_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_864832_two_sided_qwen_tokenizer_boundary_preflight_h200.sbatch"


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


def _expect_false(mapping: Mapping[str, Any], field: str, errors: list[str], prefix: str) -> None:
    if mapping.get(field) is not False:
        errors.append(f"{prefix}.{field} must be false")


def _expect_true(mapping: Mapping[str, Any], field: str, errors: list[str], prefix: str) -> None:
    if mapping.get(field) is not True:
        errors.append(f"{prefix}.{field} must be true")


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
    if config.get("schema_name") != "natural_evidence_v2_r4_after_864832_two_sided_tokenizer_preflight_route_v1":
        errors.append("schema_name mismatch")
    if config.get("package_id") != "r4_after_864832_two_sided_tokenizer_preflight_route_v1":
        errors.append("package_id mismatch")

    source_bank = _mapping(config.get("source_bank"), "source_bank", errors)
    bank_summary_path = _path(source_bank.get("summary", ""), "source_bank.summary", errors)
    _path(source_bank.get("surface_bank", ""), "source_bank.surface_bank", errors)
    _path(source_bank.get("codebook", ""), "source_bank.codebook", errors)
    bank_summary = _read_json(bank_summary_path, "source_bank.summary", errors) if bank_summary_path.exists() else {}
    if bank_summary.get("status") != source_bank.get("expected_status"):
        errors.append("source_bank summary status mismatch")
    if int(bank_summary.get("entry_count", 0)) != 256:
        errors.append("source_bank entry_count must be 256")
    if bank_summary.get("protected_codeword_missing_coordinates") not in ([], None):
        errors.append("source_bank protected_codeword_missing_coordinates must be empty")
    if bank_summary.get("forbidden_literal_hits") not in ([], None):
        errors.append("source_bank forbidden_literal_hits must be empty")

    source_rows = _mapping(config.get("source_rows"), "source_rows", errors)
    rows_path = _path(source_rows.get("rows", ""), "source_rows.rows", errors)
    rows_summary_path = _path(source_rows.get("summary", ""), "source_rows.summary", errors)
    rows_summary = _read_json(rows_summary_path, "source_rows.summary", errors) if rows_summary_path.exists() else {}
    if rows_summary.get("status") != source_rows.get("expected_status"):
        errors.append("source_rows summary status mismatch")
    if int(rows_summary.get("row_count", 0)) != int(source_rows.get("expected_rows", -1)):
        errors.append("source_rows row_count mismatch")
    if rows_summary.get("current_two_way_scorer_compatible") is not True:
        errors.append("source_rows must be current_two_way_scorer_compatible=true")
    if rows_path.exists():
        row_count = sum(1 for _ in rows_path.open("r", encoding="utf-8"))
        if row_count != int(source_rows.get("expected_rows", -1)):
            errors.append("source_rows JSONL line count mismatch")

    route = _mapping(config.get("route"), "route", errors)
    if route.get("allowlist_entry") != EXPECTED_ENTRY:
        errors.append("route.allowlist_entry mismatch")
    if route.get("wrapper") != EXPECTED_WRAPPER:
        errors.append("route.wrapper mismatch")
    wrapper_path = _path(route.get("wrapper", ""), "route.wrapper", errors)
    if route.get("command_pattern") != f"sbatch {EXPECTED_WRAPPER}":
        errors.append("route.command_pattern mismatch")
    for field in ("model_forward_allowed", "scoring_allowed", "generation_allowed", "training_allowed"):
        _expect_false(route, field, errors, "route")
    if route.get("run_qwen_tokenizer") is not True:
        errors.append("route.run_qwen_tokenizer must be true")
    if int(route.get("max_rows", 0)) != 8192:
        errors.append("route.max_rows must be 8192")

    wrapper_text = wrapper_path.read_text(encoding="utf-8") if wrapper_path.exists() else ""
    required_wrapper_fragments = (
        "#SBATCH --partition=pomplun",
        "#SBATCH --account=cs_yinxin.wan",
        "#SBATCH --qos=pomplun",
        "#SBATCH --gres=gpu:h200:1",
        "#SBATCH --time=30-00:00:00",
        "r4_after_864832_two_sided_cover_bank_rows_20260516/cover_bank_aligned_target_only_rows.jsonl",
        "model_forward_started=false",
        "scoring_started=false",
        "generation_started=false",
        "training_started=false",
        "--run-qwen-tokenizer",
    )
    for fragment in required_wrapper_fragments:
        if fragment not in wrapper_text:
            errors.append(f"wrapper missing fragment: {fragment}")

    compute = _mapping(config.get("compute_policy"), "compute_policy", errors)
    if compute.get("partition") != "pomplun" or compute.get("qos") != "pomplun":
        errors.append("compute_policy must use pomplun")
    if compute.get("account") != "cs_yinxin.wan":
        errors.append("compute_policy.account mismatch")
    if compute.get("gres") != "gpu:h200:1":
        errors.append("compute_policy.gres mismatch")
    if compute.get("max_time") != "30-00:00:00":
        errors.append("compute_policy.max_time mismatch")
    _expect_false(compute, "allowlist_enabled_now", errors, "compute_policy")
    for field in ("exactly_one_submission_when_unlocked", "remote_hash_preflight_required", "hermes_notification_required", "post_submit_allowlist_shutdown_required"):
        _expect_true(compute, field, errors, "compute_policy")

    gate = _mapping(config.get("future_tokenizer_gate"), "future_tokenizer_gate", errors)
    if int(gate.get("checked_rows", 0)) != 8192:
        errors.append("future_tokenizer_gate.checked_rows must be 8192")
    for field in ("failed_rows_max", "empty_target_id_row_count_max", "empty_other_id_row_count_max", "target_other_overlap_row_count_max"):
        if int(gate.get(field, -1)) != 0:
            errors.append(f"future_tokenizer_gate.{field} must be 0")

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
        "PASS_R4_AFTER_864832_TWO_SIDED_TOKENIZER_PREFLIGHT_ROUTE_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_864832_TWO_SIDED_TOKENIZER_PREFLIGHT_ROUTE_VALIDATION_NO_SUBMIT"
    )
    return {
        "status": status,
        "errors": errors,
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
        "score_rows": source_rows.get("rows"),
        "expected_rows": source_rows.get("expected_rows"),
        "current_compute_unlocked": False,
        "allowlist_enabled": False,
        "slurm_job_submitted": False,
        "tokenizer_validation_started": False,
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
    parser = argparse.ArgumentParser(description="Validate after-864832 two-sided tokenizer preflight route.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_route(load_yaml(args.config))
    if args.output_dir is not None:
        write_json_new(args.output_dir / "two_sided_tokenizer_route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
