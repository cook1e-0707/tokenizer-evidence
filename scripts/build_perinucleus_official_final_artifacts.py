from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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
    "pending",
    "accepted",
    "verifier_success",
    "decoded_payload",
    "exact_response_match_ratio",
    "top_k_response_match_ratio",
    "exact_response_match_count",
    "top_k_response_match_count",
    "threshold",
    "expected_response_probability_mean",
    "expected_response_probability_min",
    "base_response_probability_mean",
    "eval_compute_seconds",
    "model_forward_count",
    "contract_hash_status",
    "baseline_contract_hash",
    "failure_reason",
    "case_root",
    "eval_summary_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build artifacts for frozen official Perinucleus Qwen final runs.")
    parser.add_argument(
        "--package-dry-run",
        default="results/processed/paper_stats/baseline_perinucleus_official_qwen_final_package_dry_run.json",
    )
    parser.add_argument(
        "--summary-out",
        default="results/processed/paper_stats/baseline_perinucleus_official_qwen_final_summary.json",
    )
    parser.add_argument(
        "--compute-out",
        default="results/processed/paper_stats/baseline_perinucleus_official_qwen_final_compute.json",
    )
    parser.add_argument("--table-out", default="results/tables/baseline_perinucleus_official_qwen_final.csv")
    return parser.parse_args()


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _contract_status(diagnostics: dict[str, Any]) -> tuple[str, str]:
    contract = diagnostics.get("baseline_contract")
    observed = diagnostics.get("baseline_contract_hash")
    if not isinstance(contract, dict) or not observed:
        return "missing_hash", ""
    expected = contract.get("contract_hash")
    if expected and str(expected) != str(observed):
        return "mismatch", str(observed)
    return "match", str(observed)


def _pending_row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        **case,
        "status": "pending",
        "valid_completed": False,
        "success": False,
        "method_failure": False,
        "pending": True,
        "accepted": False,
        "verifier_success": False,
        "decoded_payload": "",
        "exact_response_match_ratio": "",
        "top_k_response_match_ratio": "",
        "exact_response_match_count": "",
        "top_k_response_match_count": "",
        "threshold": 1.0,
        "expected_response_probability_mean": "",
        "expected_response_probability_min": "",
        "base_response_probability_mean": "",
        "eval_compute_seconds": "",
        "model_forward_count": "",
        "contract_hash_status": "pending",
        "baseline_contract_hash": "",
        "failure_reason": "eval_summary_missing",
    }


def _row_from_summary(case: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    diagnostics = summary.get("diagnostics", {})
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    contract_status, contract_hash = _contract_status(diagnostics)
    completed = str(summary.get("status")) == "completed"
    valid_completed = completed and contract_status == "match"
    accepted = bool(summary.get("accepted"))
    verifier_success = bool(summary.get("verifier_success"))
    success = bool(valid_completed and accepted and verifier_success)
    failure_reason = ""
    if not completed:
        failure_reason = f"status={summary.get('status')}"
    elif not valid_completed:
        failure_reason = contract_status
    elif not success:
        failure_reason = "fingerprint_exact_gate_failed"
    return {
        **case,
        "status": str(summary.get("status")),
        "valid_completed": valid_completed,
        "success": success,
        "method_failure": bool(valid_completed and not success),
        "pending": False,
        "accepted": accepted,
        "verifier_success": verifier_success,
        "decoded_payload": summary.get("decoded_payload", ""),
        "exact_response_match_ratio": diagnostics.get("exact_response_match_ratio", ""),
        "top_k_response_match_ratio": diagnostics.get("top_k_response_match_ratio", ""),
        "exact_response_match_count": diagnostics.get("exact_response_match_count", ""),
        "top_k_response_match_count": diagnostics.get("top_k_response_match_count", ""),
        "threshold": summary.get("threshold", 1.0),
        "expected_response_probability_mean": diagnostics.get("expected_response_probability_mean", ""),
        "expected_response_probability_min": diagnostics.get("expected_response_probability_min", ""),
        "base_response_probability_mean": diagnostics.get("base_response_probability_mean", ""),
        "eval_compute_seconds": diagnostics.get("eval_compute_seconds", ""),
        "model_forward_count": diagnostics.get("model_forward_count", ""),
        "contract_hash_status": contract_status,
        "baseline_contract_hash": contract_hash,
        "failure_reason": failure_reason,
        "eval_summary_path": str(summary_path),
    }


def _latest_eval_summary(case: dict[str, Any], repo_root: Path) -> Path | None:
    exact = str(case.get("eval_summary_path", "")).strip()
    if exact:
        exact_path = Path(exact)
        if not exact_path.is_absolute():
            exact_path = repo_root / exact_path
        if exact_path.exists():
            return exact_path
    pattern = str(case.get("eval_summary_glob", "")).strip()
    if not pattern:
        return None
    glob_path = Path(pattern)
    if glob_path.is_absolute():
        matches = sorted(glob_path.parent.parent.glob(f"*/{glob_path.name}"), key=lambda path: path.stat().st_mtime)
    else:
        matches = sorted((repo_root / glob_path.parent.parent).glob(f"*/{glob_path.name}"), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _budget_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for budget in sorted({int(row["query_budget"]) for row in rows}):
        subset = [row for row in rows if int(row["query_budget"]) == budget]
        valid = [row for row in subset if row["valid_completed"]]
        success = [row for row in subset if row["success"]]
        out.append(
            {
                "query_budget": budget,
                "target_count": len(subset),
                "valid_completed_count": len(valid),
                "success_count": len(success),
                "pending_count": sum(1 for row in subset if row["pending"]),
                "method_failure_count": sum(1 for row in subset if row["method_failure"]),
                "success_rate": len(success) / len(valid) if valid else 0.0,
            }
        )
    return out


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    dry_run_path = _resolve(repo_root, args.package_dry_run)
    dry_run = _read_json(dry_run_path)
    rows = []
    for case in dry_run["cases"]:
        summary_path = _latest_eval_summary(case, repo_root)
        rows.append(_row_from_summary(case, summary_path) if summary_path else _pending_row(case))
    valid = [row for row in rows if row["valid_completed"]]
    success = [row for row in rows if row["success"]]
    pending = [row for row in rows if row["pending"]]
    summary = {
        "schema_name": "baseline_perinucleus_official_qwen_final_summary",
        "schema_version": 1,
        "generated_at": current_timestamp(),
        "package_dry_run": str(dry_run_path.relative_to(repo_root)),
        "frozen_candidate": dry_run.get("frozen_candidate", {}).get("selected_candidate", {}),
        "target_count": len(rows),
        "valid_completed_count": len(valid),
        "success_count": len(success),
        "pending_count": len(pending),
        "method_failure_count": sum(1 for row in rows if row["method_failure"]),
        "success_rate": len(success) / len(valid) if valid else 0.0,
        "paper_ready": bool(len(valid) == len(rows) and not pending),
        "query_budget_rows": _budget_rows(rows),
    }
    compute = {
        "schema_name": "baseline_perinucleus_official_qwen_final_compute",
        "schema_version": 1,
        "generated_at": summary["generated_at"],
        "eval_compute_seconds_total": sum(
            float(row["eval_compute_seconds"]) for row in rows if row["eval_compute_seconds"] != ""
        ),
        "model_forward_count_total": sum(
            int(row["model_forward_count"]) for row in rows if row["model_forward_count"] != ""
        ),
    }
    summary_path = _resolve(repo_root, args.summary_out)
    compute_path = _resolve(repo_root, args.compute_out)
    table_path = _resolve(repo_root, args.table_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    compute_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compute_path.write_text(json.dumps(compute, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(table_path, rows)
    print(f"wrote official Perinucleus Qwen final summary to {summary_path}")
    print(f"wrote official Perinucleus Qwen final table to {table_path}")
    print(f"wrote official Perinucleus Qwen final compute to {compute_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
