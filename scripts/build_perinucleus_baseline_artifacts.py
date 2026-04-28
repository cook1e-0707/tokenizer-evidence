from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from scripts.prepare_perinucleus_baseline import _cases, _resolve_output_root_base
from src.infrastructure.paths import current_timestamp, discover_repo_root


FIELDS = [
    "case_id",
    "payload",
    "seed",
    "query_budget",
    "matched_budget_status",
    "status",
    "valid_completed",
    "success",
    "method_failure",
    "invalid_excluded",
    "pending",
    "accepted",
    "verifier_success",
    "decoded_payload",
    "ownership_score",
    "exact_response_match_ratio",
    "top_k_response_match_ratio",
    "exact_response_match_count",
    "top_k_response_match_count",
    "threshold",
    "target_far",
    "utility_acceptance_rate",
    "utility_status",
    "expected_response_probability_mean",
    "expected_response_probability_min",
    "training_compute_seconds",
    "embedding_compute_seconds",
    "model_forward_count",
    "contract_hash_status",
    "contract_hash_missing_fields",
    "contract_hash_mismatch_fields",
    "baseline_contract_hash",
    "failure_reason",
    "run_dir",
    "case_root",
    "eval_summary_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Perinucleus-style baseline artifacts.")
    parser.add_argument(
        "--package-config",
        default="configs/experiment/baselines/perinucleus/package__baseline_perinucleus_qwen_v1.yaml",
    )
    parser.add_argument("--output-dir", default="results/processed/paper_stats")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument(
        "--case-root-base",
        help="Optional base directory. Defaults to EXP_SCRATCH/baselines/perinucleus_qwen.",
    )
    return parser.parse_args()


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(os.path.expandvars(raw))
    return path if path.is_absolute() else repo_root / path


def _latest_eval_summary(case_root: Path) -> Path | None:
    matches = sorted(
        case_root.glob("runs/exp_eval/*/eval_summary.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
    )
    return matches[-1] if matches else None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _contract_status(diagnostics: dict[str, Any]) -> tuple[str, str, str]:
    contract = diagnostics.get("baseline_contract")
    observed_hash = diagnostics.get("baseline_contract_hash")
    missing: list[str] = []
    mismatch: list[str] = []
    if not isinstance(contract, dict):
        missing.append("baseline_contract")
    if not observed_hash:
        missing.append("baseline_contract_hash")
    if isinstance(contract, dict) and observed_hash and observed_hash != _stable_hash(contract):
        mismatch.append("baseline_contract_hash")
    if missing:
        return "missing_hash", ";".join(sorted(missing)), ""
    if mismatch:
        return "mismatch", "", ";".join(sorted(mismatch))
    return "match", "", ""


def _pending_row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        **case,
        "status": "pending",
        "valid_completed": False,
        "success": False,
        "method_failure": False,
        "invalid_excluded": False,
        "pending": True,
        "accepted": False,
        "verifier_success": False,
        "decoded_payload": "",
        "ownership_score": "",
        "exact_response_match_ratio": "",
        "top_k_response_match_ratio": "",
        "exact_response_match_count": "",
        "top_k_response_match_count": "",
        "threshold": "",
        "target_far": 0.01,
        "utility_acceptance_rate": "",
        "utility_status": "pending",
        "expected_response_probability_mean": "",
        "expected_response_probability_min": "",
        "training_compute_seconds": "",
        "embedding_compute_seconds": "",
        "model_forward_count": "",
        "contract_hash_status": "pending",
        "contract_hash_missing_fields": "",
        "contract_hash_mismatch_fields": "",
        "baseline_contract_hash": "",
        "failure_reason": "eval_summary_missing",
        "run_dir": "",
        "case_root": case["case_root"],
        "eval_summary_path": "",
    }


def _row_from_summary(case: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    diagnostics = summary.get("diagnostics", {})
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    contract_status, missing_fields, mismatch_fields = _contract_status(diagnostics)
    completed = str(summary.get("status", "")) in {"completed", "failed"}
    contract_ok = contract_status == "match"
    valid_completed = completed and contract_ok
    accepted = bool(summary.get("accepted", False))
    verifier_success = bool(summary.get("verifier_success", False))
    success = valid_completed and accepted and verifier_success
    method_failure = valid_completed and not success
    invalid = completed and not contract_ok
    failure_reason = ""
    if method_failure:
        failure_reason = "response_match_failed_under_frozen_threshold"
    elif invalid:
        failure_reason = contract_status
    elif not completed:
        failure_reason = f"eval_status={summary.get('status', '')}"
    return {
        **case,
        "status": str(summary.get("status", "")),
        "valid_completed": valid_completed,
        "success": success,
        "method_failure": method_failure,
        "invalid_excluded": invalid,
        "pending": False,
        "accepted": accepted,
        "verifier_success": verifier_success,
        "decoded_payload": summary.get("decoded_payload") or "",
        "ownership_score": diagnostics.get("ownership_score", summary.get("match_ratio", "")),
        "exact_response_match_ratio": diagnostics.get("exact_response_match_ratio", ""),
        "top_k_response_match_ratio": diagnostics.get("top_k_response_match_ratio", ""),
        "exact_response_match_count": diagnostics.get("exact_response_match_count", ""),
        "top_k_response_match_count": diagnostics.get("top_k_response_match_count", ""),
        "threshold": summary.get("threshold", diagnostics.get("threshold", "")),
        "target_far": diagnostics.get("target_far", 0.01),
        "utility_acceptance_rate": summary.get("utility_acceptance_rate", ""),
        "utility_status": diagnostics.get("utility_status", ""),
        "expected_response_probability_mean": diagnostics.get("expected_response_probability_mean", ""),
        "expected_response_probability_min": diagnostics.get("expected_response_probability_min", ""),
        "training_compute_seconds": diagnostics.get("training_compute_seconds", ""),
        "embedding_compute_seconds": diagnostics.get("embedding_compute_seconds", ""),
        "model_forward_count": diagnostics.get("model_forward_count", ""),
        "contract_hash_status": contract_status,
        "contract_hash_missing_fields": missing_fields,
        "contract_hash_mismatch_fields": mismatch_fields,
        "baseline_contract_hash": diagnostics.get("baseline_contract_hash", ""),
        "failure_reason": failure_reason,
        "run_dir": summary.get("run_dir", str(summary_path.parent)),
        "case_root": case["case_root"],
        "eval_summary_path": str(summary_path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _budget_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for budget in sorted({int(row["query_budget"]) for row in rows}):
        subset = [row for row in rows if int(row["query_budget"]) == budget]
        valid = [row for row in subset if row["valid_completed"]]
        success = [row for row in subset if row["success"]]
        top_k_success = [
            row
            for row in valid
            if row["top_k_response_match_ratio"] != ""
            and float(row["top_k_response_match_ratio"]) >= float(row["threshold"])
        ]
        result.append(
            {
                "query_budget": budget,
                "target_count": len(subset),
                "valid_completed_count": len(valid),
                "success_count": len(success),
                "top_k_success_count": len(top_k_success),
                "pending_count": sum(1 for row in subset if row["pending"]),
                "invalid_excluded_count": sum(1 for row in subset if row["invalid_excluded"]),
                "exact_success_rate": len(success) / len(valid) if valid else 0.0,
                "top_k_success_rate": len(top_k_success) / len(valid) if valid else 0.0,
            }
        )
    return result


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    root_base = _resolve_output_root_base(package_config, args.case_root_base)
    cases = _cases(package_config, root_base)
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_root = Path(str(case["case_root"]))
        if not case_root.is_absolute():
            case_root = repo_root / case_root
        summary_path = _latest_eval_summary(case_root)
        rows.append(_pending_row(case) if summary_path is None else _row_from_summary(case, summary_path))
    valid = [row for row in rows if row["valid_completed"]]
    successes = [row for row in rows if row["success"]]
    method_failures = [row for row in rows if row["method_failure"]]
    invalid = [row for row in rows if row["invalid_excluded"]]
    pending = [row for row in rows if row["pending"]]
    top_k_success = [
        row
        for row in valid
        if row["top_k_response_match_ratio"] != ""
        and float(row["top_k_response_match_ratio"]) >= float(row["threshold"])
    ]
    contract_status_counts = {
        status: sum(1 for row in rows if row["contract_hash_status"] == status)
        for status in sorted({str(row["contract_hash_status"]) for row in rows})
    }
    thresholds_frozen = False
    utility_suite_completed = all(
        row.get("utility_status") not in {"", "pending", "not_evaluated_requires_shared_organic_utility_suite"}
        for row in rows
    )
    summary = {
        "schema_name": "baseline_perinucleus_summary",
        "schema_version": 1,
        "generated_at": current_timestamp(),
        "package_config_path": str(package_config_path.relative_to(repo_root)),
        "new_case_root_base": root_base,
        "target_count": len(rows),
        "completed_count": len([row for row in rows if not row["pending"]]),
        "valid_completed_count": len(valid),
        "success_count": len(successes),
        "method_failure_count": len(method_failures),
        "invalid_excluded_count": len(invalid),
        "pending_count": len(pending),
        "exact_gate_success_rate": len(successes) / len(valid) if valid else 0.0,
        "top_k_success_count": len(top_k_success),
        "top_k_success_rate": len(top_k_success) / len(valid) if valid else 0.0,
        "thresholds_frozen": thresholds_frozen,
        "utility_suite_completed": utility_suite_completed,
        "paper_ready": bool(
            thresholds_frozen
            and utility_suite_completed
            and not pending
            and not invalid
            and len(valid) == len(rows)
        ),
        "contract_hash_status_counts": contract_status_counts,
        "query_budget_rows": _budget_rows(rows),
        "compare_against": package_config.get("compare_against", []),
        "notes": [
            "Perinucleus-style adapted baseline; not an exact reproduction of Scalable Fingerprinting.",
            "q1 and q3 are under-budget diagnostics relative to B0 M=4; q5 and q10 are over-budget diagnostics.",
            "paper_ready remains false until FAR calibration thresholds are frozen before final claims.",
        ],
    }
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = _resolve_path(repo_root, args.tables_dir)
    _write_json(output_dir / "baseline_perinucleus_summary.json", summary)
    _write_csv(tables_dir / "baseline_perinucleus.csv", rows)
    print(f"wrote Perinucleus baseline summary to {output_dir / 'baseline_perinucleus_summary.json'}")
    print(f"wrote Perinucleus baseline table to {tables_dir / 'baseline_perinucleus.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
