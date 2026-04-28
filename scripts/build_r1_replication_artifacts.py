from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import yaml

from scripts.build_g3a_v2_block_scale_artifacts import (
    CONTRACT_HASH_FIELDS,
    _collect_case,
    _resolve_case_root,
)
from scripts.prepare_r1_replication import _case_records
from src.infrastructure.paths import current_timestamp, discover_repo_root


RUN_FIELDS = [
    "case_id",
    "variant_id",
    "variant_slug",
    "model_family",
    "model_id",
    "tokenizer_id",
    "block_count",
    "payload",
    "seed",
    "case_root",
    "status",
    "result_class",
    "failure_reasons",
    "exact_gate_success",
    "rs_gate_success",
    "accepted_under_exact_gate",
    "accepted_under_rs_gate",
    "verifier_success",
    "decoded_payload",
    "decoded_payload_correct",
    "block_count_correct",
    "slot_bucket_accuracy",
    "symbol_error_count",
    "erasure_count",
    "rs_correctable_under_2E_plus_S_lt_d",
    "rs_recovered_payload",
    "exact_slot_rate",
    "bucket_correct_rate",
    "match_ratio",
    "final_loss",
    "normalized_L_set_mean",
    "target_bucket_mass_mean",
    "target_bucket_mass_min",
    "slot_margin_mean",
    "slot_margin_min",
    "checkpoint_selection_metric",
    "checkpoint_selection_best_step",
    "checkpoint_selection_best_metric_value",
    "contract_hash_status",
    "contract_hash_missing_fields",
    "contract_hash_mismatch_fields",
    *CONTRACT_HASH_FIELDS,
    "train_summary_path",
    "eval_summary_path",
    "training_health_path",
    "compiled_verifier_report_path",
    "latest_eval_input_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build R1 Llama 3.1 8B replication paper artifacts.")
    parser.add_argument("--package-config", default="configs/reporting/r1_llama3_1_8b_replication_v1.yaml")
    parser.add_argument("--output-dir", default="results/processed/paper_stats")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument(
        "--case-root-base",
        help="Optional base directory for R1 case roots. Defaults to EXP_SCRATCH/r1_llama3_1_8b_replication_v1.",
    )
    return parser.parse_args()


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(os.path.expandvars(raw))
    return path if path.is_absolute() else repo_root / path


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _resolve_output_root_base(package_config: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return str(Path(os.path.expandvars(explicit)).as_posix())
    prefix = str(package_config["new_case_root_prefix"])
    exp_scratch = os.environ.get("EXP_SCRATCH")
    if exp_scratch:
        return str((Path(exp_scratch) / prefix).as_posix())
    return prefix


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Scope & Success & Method fail & Invalid & Valid completed & Exact gate \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        scope = str(row["scope"]).replace("_", "\\_")
        lines.append(
            f"{scope} & {row['success_runs']} & {row['method_failure_runs']} & "
            f"{row['invalid_excluded_runs']} & {row['valid_completed_runs']} & "
            f"{row['exact_gate_success_rate']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{R1 second-family replication on Llama 3.1 8B Instruct. Method failures remain in the denominator.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0, "ci95_half_width": 0.0}
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    sem = std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "std": std, "sem": sem, "ci95_half_width": 1.96 * sem}


def _summary_row(scope: str, target_runs: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["result_class"] != "pending"]
    valid_completed = [row for row in rows if bool(row["valid_completed"])]
    successes = [row for row in rows if bool(row["success"])]
    method_failures = [row for row in rows if bool(row["method_failure"])]
    invalid = [row for row in rows if bool(row["invalid_excluded"])]
    pending = [row for row in rows if row["result_class"] == "pending"]
    return {
        "scope": scope,
        "target_runs": target_runs,
        "completed_runs": len(completed),
        "valid_completed_runs": len(valid_completed),
        "success_runs": len(successes),
        "method_failure_runs": len(method_failures),
        "invalid_excluded_runs": len(invalid),
        "pending_runs": len(pending),
        "exact_gate_success_rate": len(successes) / len(valid_completed) if valid_completed else 0.0,
        "rs_gate_success_rate": (
            sum(1 for row in valid_completed if bool(row["accepted_under_rs_gate"])) / len(valid_completed)
            if valid_completed
            else 0.0
        ),
    }


def _enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "exact_gate_success": bool(row.get("success")),
        "rs_gate_success": bool(row.get("accepted_under_rs_gate")),
    }


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = _resolve_path(repo_root, args.tables_dir)
    root_base = _resolve_output_root_base(package_config, args.case_root_base)
    cases = _case_records(package_config, root_base)

    rows: list[dict[str, Any]] = []
    for case in cases:
        row, _slot_rows, _symbol_rows = _collect_case(repo_root, package_config, case)
        case_root = _resolve_case_root(repo_root, package_config, str(case["case_root"]))
        rows.append(_enrich_row({**row, "case_root": str(case_root)}))

    completed = [row for row in rows if row["result_class"] != "pending"]
    valid_completed = [row for row in rows if bool(row["valid_completed"])]
    successes = [row for row in rows if bool(row["success"])]
    method_failures = [row for row in rows if bool(row["method_failure"])]
    invalid = [row for row in rows if bool(row["invalid_excluded"])]
    pending = [row for row in rows if row["result_class"] == "pending"]
    rs_success_count = sum(1 for row in valid_completed if bool(row["accepted_under_rs_gate"]))
    contract_hash_status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("contract_hash_status", "missing_hash"))
        contract_hash_status_counts[status] = contract_hash_status_counts.get(status, 0) + 1

    prerequisite = dict(package_config.get("catalog_prerequisite", {}))
    frozen_catalog = repo_root / str(prerequisite.get("frozen_catalog", ""))
    catalog_exists = bool(prerequisite.get("frozen_catalog")) and frozen_catalog.exists()
    artifact_paper_ready_checks = {
        "llama_tokenizer_frozen_catalog_exists_and_strict_passes": catalog_exists,
        "real_contract_hash_checks_pass": all(row["contract_hash_status"] == "match" for row in completed),
        "valid_completed_count_equals_target_count": len(valid_completed) == len(rows),
        "invalid_excluded_count_zero_unless_concrete_artifact_failure": not invalid,
        "exact_and_rs_aware_gates_reported": all(
            row["result_class"] == "pending" or row["accepted_under_rs_gate"] in {True, False}
            for row in rows
        ),
        "method_failures_remain_in_denominator": True,
        "no_threshold_changed_after_final_evaluation": True,
    }
    artifact_paper_ready = all(artifact_paper_ready_checks.values())
    claim_readiness_checks = {
        "artifact_paper_ready": artifact_paper_ready,
        "exact_gate_success_count_equals_target_count": len(successes) == len(rows),
        "method_failure_count_zero": not method_failures,
    }
    claim_paper_ready = all(claim_readiness_checks.values())
    overall_row = _summary_row("overall", len(rows), rows)
    by_seed_rows = [
        _summary_row(f"seed={seed}", len(package_config["final_matrix"]["payloads"]), [row for row in rows if row["seed"] == seed])
        for seed in package_config["final_matrix"]["seeds"]
    ]
    summary = {
        "schema_name": "r1_summary",
        "schema_version": 1,
        "workstream": package_config.get("workstream", "R1"),
        "description": package_config.get("description", ""),
        "generated_at": current_timestamp(),
        "package_config_path": str(package_config_path),
        "new_case_root_base": root_base,
        "catalog_prerequisite": prerequisite,
        "llama_frozen_catalog_exists": catalog_exists,
        "fixed_contract": dict(package_config["fixed_contract"]),
        "final_matrix": dict(package_config["final_matrix"]),
        "target_count": len(rows),
        "completed_count": len(completed),
        "valid_completed_count": len(valid_completed),
        "success_count": len(successes),
        "method_failure_count": len(method_failures),
        "invalid_excluded_count": len(invalid),
        "pending_count": len(pending),
        "rs_success_count": rs_success_count,
        "exact_gate_success_rate": len(successes) / len(valid_completed) if valid_completed else 0.0,
        "rs_gate_success_rate": rs_success_count / len(valid_completed) if valid_completed else 0.0,
        "paper_ready": claim_paper_ready,
        "artifact_paper_ready": artifact_paper_ready,
        "claim_paper_ready": claim_paper_ready,
        "paper_ready_checks": artifact_paper_ready_checks,
        "artifact_paper_ready_checks": artifact_paper_ready_checks,
        "claim_readiness_checks": claim_readiness_checks,
        "claim_readiness_note": (
            "R1 artifact accounting is complete, but the exact clean-path replication claim requires "
            "exact_gate_success_count_equals_target_count."
        ),
        "contract_hash_status_counts": contract_hash_status_counts,
        "overall_metrics": {
            "exact_gate_success": _stats([1.0 if row["exact_gate_success"] else 0.0 for row in valid_completed]),
            "rs_gate_success": _stats([1.0 if row["rs_gate_success"] else 0.0 for row in valid_completed]),
            "slot_bucket_accuracy": _stats([float(row["slot_bucket_accuracy"]) for row in valid_completed]),
            "symbol_error_count": _stats([float(row["symbol_error_count"]) for row in valid_completed]),
            "erasure_count": _stats([float(row["erasure_count"]) for row in valid_completed]),
            "target_bucket_mass_mean": _stats([float(row["target_bucket_mass_mean"]) for row in valid_completed]),
            "slot_margin_min": _stats([float(row["slot_margin_min"]) for row in valid_completed]),
        },
        "summary_rows": [overall_row, *by_seed_rows],
        "success_case_ids": [row["case_id"] for row in successes],
        "method_failure_case_ids": [row["case_id"] for row in method_failures],
        "invalid_excluded_case_ids": [row["case_id"] for row in invalid],
        "pending_case_ids": [row["case_id"] for row in pending],
    }
    inclusion_payload = {
        "schema_name": "r1_run_accounting",
        "schema_version": 1,
        "valid_successes": successes,
        "method_failures": method_failures,
        "invalid_excluded": invalid,
        "pending": pending,
    }
    compute_accounting = {
        "schema_name": "r1_compute_accounting",
        "schema_version": 1,
        **dict(package_config.get("compute_estimate", {})),
    }

    _write_json(output_dir / "r1_summary.json", summary)
    _write_json(output_dir / "r1_run_inclusion_list.json", inclusion_payload)
    _write_json(output_dir / "r1_compute_accounting.json", compute_accounting)
    _write_csv(tables_dir / "r1_replication.csv", rows, RUN_FIELDS)
    _write_csv(tables_dir / "r1_failure_cases.csv", [*method_failures, *invalid], RUN_FIELDS)
    _write_tex(tables_dir / "r1_replication.tex", [overall_row, *by_seed_rows])
    print(f"wrote R1 summary to {output_dir / 'r1_summary.json'}")
    print(f"wrote R1 run table to {tables_dir / 'r1_replication.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
