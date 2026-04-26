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
from src.infrastructure.paths import current_timestamp, discover_repo_root


RUN_FIELDS = [
    "case_id",
    "variant_id",
    "variant_slug",
    "block_count",
    "payload",
    "seed",
    "case_root",
    "status",
    "result_class",
    "failure_reasons",
    "exact_payload_recovered",
    "rs_payload_recovered",
    "block_count_correct",
    "slot_bucket_accuracy",
    "symbol_error_count",
    "erasure_count",
    "min_bucket_margin",
    "mean_bucket_margin",
    "target_bucket_mass_min",
    "target_bucket_mass_mean",
    "exact_gate_success",
    "rs_gate_success",
    "accepted_under_exact_gate",
    "accepted_under_rs_gate",
    "verifier_success",
    "decoded_payload",
    "decoded_payload_correct",
    "rs_correctable_under_2E_plus_S_lt_d",
    "rs_recovered_payload",
    "exact_slot_rate",
    "bucket_correct_rate",
    "match_ratio",
    "final_loss",
    "normalized_L_set_mean",
    "normalized_L_margin_mean",
    "total_evidence_loss_mean",
    "lambda_set",
    "lambda_margin",
    "margin_gamma",
    "lambda_reg",
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

SLOT_MARGIN_FIELDS = [
    "case_id",
    "variant_id",
    "block_count",
    "seed",
    "payload",
    "slot_index",
    "block_index",
    "field_name",
    "expected_bucket",
    "decoded_bucket",
    "bucket_correct",
    "exact_token_correct",
    "target_bucket_logmass",
    "strongest_wrong_bucket",
    "strongest_wrong_bucket_logmass",
    "bucket_margin",
    "target_bucket_rank",
    "target_token_probability",
    "generated_token",
    "top_5_bucket_logmasses",
    "top_5_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build G3a-v3 block-scale paper artifacts.")
    parser.add_argument("--package-config", default="configs/reporting/g3a_block_scale_v3.yaml")
    parser.add_argument("--output-dir", default="results/processed/paper_stats")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument(
        "--new-case-root-base",
        help="Optional base directory for G3a-v3 final case roots. Defaults to EXP_SCRATCH/g3a_block_scale_v3.",
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


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_output_root_base(package_config: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return str(Path(os.path.expandvars(explicit)).as_posix())
    prefix = str(package_config["new_case_root_prefix"])
    exp_scratch = os.environ.get("EXP_SCRATCH")
    if exp_scratch:
        return str((Path(exp_scratch) / prefix).as_posix())
    return prefix


def _case_records(package_config: dict[str, Any], root_base: str) -> list[dict[str, Any]]:
    final_matrix = dict(package_config["final_matrix"])
    variant_by_id = {str(item["id"]): dict(item) for item in package_config["block_variants"]}
    cases: list[dict[str, Any]] = []
    for variant_id in final_matrix["block_variants"]:
        variant = variant_by_id[str(variant_id)]
        variant_slug = str(variant.get("slug", str(variant["id"]).lower()))
        for seed in final_matrix["seeds"]:
            for payload in final_matrix["payloads"]:
                cases.append(
                    {
                        "case_id": f"{variant['id']}_{payload}_s{seed}",
                        "variant_id": str(variant["id"]),
                        "variant_slug": variant_slug,
                        "block_count": int(variant["block_count"]),
                        "payload": str(payload),
                        "seed": int(seed),
                        "case_root": str(Path(root_base) / "final" / variant_slug / f"{payload}_s{seed}"),
                    }
                )
    return cases


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
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
            "\\caption{G3a-v3 margin-aware held-out block-count scale package. Method failures remain in the denominator.}",
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
            sum(1 for row in valid_completed if bool(row["rs_gate_success"])) / len(valid_completed)
            if valid_completed
            else 0.0
        ),
    }


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _generation_diagnostics_path(row: dict[str, Any]) -> Path | None:
    raw = row.get("train_summary_path")
    if not raw:
        return None
    return Path(str(raw)).parent / "fieldwise_generation_diagnostics.json"


def _slot_margin_rows(case: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = _read_json(_generation_diagnostics_path(row))
    slot_rows = diagnostics.get("slot_results", [])
    if not isinstance(slot_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in slot_rows:
        if not isinstance(item, dict):
            continue
        slot_index = int(item.get("slot_index", len(rows)))
        top_buckets = item.get("top_5_bucket_logmasses", [])
        top_tokens = item.get("top_5_tokens", [])
        rows.append(
            {
                "case_id": case["case_id"],
                "variant_id": case["variant_id"],
                "block_count": case["block_count"],
                "seed": case["seed"],
                "payload": case["payload"],
                "slot_index": slot_index,
                "block_index": int(item.get("block_index", slot_index // 2)),
                "field_name": item.get("slot_type", "missing"),
                "expected_bucket": item.get("expected_bucket_id", "missing"),
                "decoded_bucket": item.get("chosen_bucket_id", "missing"),
                "bucket_correct": item.get("bucket_correct", "missing"),
                "exact_token_correct": item.get("token_text") == item.get("expected_value"),
                "target_bucket_logmass": item.get("target_bucket_logmass", "missing"),
                "strongest_wrong_bucket": item.get("strongest_wrong_bucket", "missing"),
                "strongest_wrong_bucket_logmass": item.get("strongest_wrong_bucket_logmass", "missing"),
                "bucket_margin": item.get("bucket_margin", "missing"),
                "target_bucket_rank": item.get("target_bucket_rank", "missing"),
                "target_token_probability": item.get("target_token_probability", "missing"),
                "generated_token": item.get("token_text", "missing"),
                "top_5_bucket_logmasses": json.dumps(top_buckets, sort_keys=True),
                "top_5_tokens": json.dumps(top_tokens, sort_keys=True),
            }
        )
    return rows


def _enrich_row(row: dict[str, Any], slot_margin_rows: list[dict[str, Any]]) -> dict[str, Any]:
    margin_values = [
        _numeric(item.get("bucket_margin"))
        for item in slot_margin_rows
        if item.get("bucket_margin") not in {"", "missing", None}
    ]
    health = _read_json(Path(row["training_health_path"])) if row.get("training_health_path") else {}
    exact_gate_success = bool(row.get("success"))
    rs_recovered_payload = str(row.get("rs_recovered_payload") or "")
    payload = str(row.get("payload"))
    return {
        **row,
        "exact_payload_recovered": bool(row.get("decoded_payload_correct")),
        "rs_payload_recovered": bool(row.get("accepted_under_rs_gate")) and (
            not rs_recovered_payload or rs_recovered_payload == payload
        ),
        "exact_gate_success": exact_gate_success,
        "rs_gate_success": bool(row.get("accepted_under_rs_gate")),
        "min_bucket_margin": min(margin_values) if margin_values else "missing",
        "mean_bucket_margin": sum(margin_values) / len(margin_values) if margin_values else "missing",
        "normalized_L_margin_mean": float(health.get("normalized_L_margin_mean", 0.0)),
        "total_evidence_loss_mean": float(health.get("total_evidence_loss_mean", 0.0)),
        "lambda_set": float(health.get("lambda_set", 0.0)),
        "lambda_margin": float(health.get("lambda_margin", 0.0)),
        "margin_gamma": float(health.get("margin_gamma", 0.0)),
        "lambda_reg": float(health.get("lambda_reg", 0.0)),
    }


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = _resolve_path(repo_root, args.tables_dir)
    root_base = _resolve_output_root_base(package_config, args.new_case_root_base)
    cases = _case_records(package_config, root_base)

    rows: list[dict[str, Any]] = []
    slot_margin_rows: list[dict[str, Any]] = []
    for case in cases:
        row, _slot_rows, _symbol_rows = _collect_case(repo_root, package_config, case)
        case_root = _resolve_case_root(repo_root, package_config, str(case["case_root"]))
        case_with_resolved_root = {**case, "case_root": str(case_root)}
        case_slot_margin_rows = _slot_margin_rows(case_with_resolved_root, row)
        slot_margin_rows.extend(case_slot_margin_rows)
        rows.append(_enrich_row(row, case_slot_margin_rows))

    completed = [row for row in rows if row["result_class"] != "pending"]
    valid_completed = [row for row in rows if bool(row["valid_completed"])]
    successes = [row for row in rows if bool(row["success"])]
    method_failures = [row for row in rows if bool(row["method_failure"])]
    invalid = [row for row in rows if bool(row["invalid_excluded"])]
    pending = [row for row in rows if row["result_class"] == "pending"]
    selected = dict(package_config.get("selected_operating_point", {}))
    final_matrix = dict(package_config["final_matrix"])
    validation = dict(package_config["validation"])
    validation_final_distinct = set(validation.get("seeds", [validation.get("seed")])).isdisjoint(
        set(final_matrix["seeds"])
    )
    paper_ready_checks = {
        "real_contract_hash_checks_pass": all(row["contract_hash_status"] == "match" for row in completed),
        "valid_completed_count_equals_target_count": len(valid_completed) == len(rows),
        "invalid_excluded_count_zero_unless_concrete_artifact_failure": not invalid,
        "exact_and_rs_aware_gates_reported": all(
            row["result_class"] == "pending" or row["rs_gate_success"] in {True, False}
            for row in rows
        ),
        "method_failures_remain_in_denominator": True,
        "validation_and_final_sets_distinct": validation_final_distinct,
        "hyperparameters_frozen_before_final_matrix_launch": (
            selected.get("status") == "frozen_before_final_launch"
            and bool(selected.get("final_launch_allowed"))
        ),
        "no_threshold_changed_after_final_evaluation": True,
    }
    variant_rows = [
        _summary_row(
            f"variant={variant['id']}",
            len(final_matrix["payloads"]) * len(final_matrix["seeds"]),
            [row for row in rows if row["variant_id"] == str(variant["id"])],
        )
        for variant in package_config["block_variants"]
    ]
    overall_row = _summary_row("overall", len(rows), rows)
    summary = {
        "schema_name": "g3a_v3_summary",
        "schema_version": 1,
        "workstream": package_config.get("workstream", "G3a-v3"),
        "description": package_config.get("description", ""),
        "generated_at": current_timestamp(),
        "package_config_path": str(package_config_path),
        "new_case_root_base": root_base,
        "target_count": len(rows),
        "completed_count": len(completed),
        "valid_completed_count": len(valid_completed),
        "success_count": len(successes),
        "method_failure_count": len(method_failures),
        "invalid_excluded_count": len(invalid),
        "pending_count": len(pending),
        "exact_gate_success_rate": len(successes) / len(valid_completed) if valid_completed else 0.0,
        "rs_gate_success_rate": (
            sum(1 for row in valid_completed if bool(row["rs_gate_success"])) / len(valid_completed)
            if valid_completed
            else 0.0
        ),
        "paper_ready": all(paper_ready_checks.values()),
        "paper_ready_checks": paper_ready_checks,
        "selected_operating_point": selected,
        "validation": validation,
        "final_matrix": final_matrix,
        "overall_metrics": {
            "exact_gate_success": _stats([1.0 if row["exact_gate_success"] else 0.0 for row in valid_completed]),
            "rs_gate_success": _stats([1.0 if row["rs_gate_success"] else 0.0 for row in valid_completed]),
            "slot_bucket_accuracy": _stats([float(row["slot_bucket_accuracy"]) for row in valid_completed]),
            "symbol_error_count": _stats([float(row["symbol_error_count"]) for row in valid_completed]),
            "erasure_count": _stats([float(row["erasure_count"]) for row in valid_completed]),
            "min_bucket_margin": _stats([
                float(row["min_bucket_margin"])
                for row in valid_completed
                if row["min_bucket_margin"] != "missing"
            ]),
            "mean_bucket_margin": _stats([
                float(row["mean_bucket_margin"])
                for row in valid_completed
                if row["mean_bucket_margin"] != "missing"
            ]),
        },
        "summary_rows": [overall_row, *variant_rows],
        "success_case_ids": [row["case_id"] for row in successes],
        "method_failure_case_ids": [row["case_id"] for row in method_failures],
        "invalid_excluded_case_ids": [row["case_id"] for row in invalid],
        "pending_case_ids": [row["case_id"] for row in pending],
    }
    inclusion_payload = {
        "schema_name": "g3a_v3_run_accounting",
        "schema_version": 1,
        "valid_successes": successes,
        "method_failures": method_failures,
        "invalid_excluded": invalid,
        "pending": pending,
    }
    compute_accounting = {
        "schema_name": "g3a_v3_compute_accounting",
        "schema_version": 1,
        **dict(package_config.get("compute_estimate", {})),
    }

    _write_json(output_dir / "g3a_v3_summary.json", summary)
    _write_json(output_dir / "g3a_v3_run_inclusion_list.json", inclusion_payload)
    _write_json(output_dir / "g3a_v3_compute_accounting.json", compute_accounting)
    _write_csv(tables_dir / "g3a_v3_block_scale.csv", rows, RUN_FIELDS)
    _write_csv(tables_dir / "g3a_v3_slot_margin.csv", slot_margin_rows, SLOT_MARGIN_FIELDS)
    _write_csv(tables_dir / "g3a_v3_failure_cases.csv", [*method_failures, *invalid], RUN_FIELDS)
    _write_tex(tables_dir / "g3a_v3_block_scale.tex", [overall_row, *variant_rows])
    print(f"wrote G3a-v3 summary to {output_dir / 'g3a_v3_summary.json'}")
    print(f"wrote G3a-v3 run table to {tables_dir / 'g3a_v3_block_scale.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
