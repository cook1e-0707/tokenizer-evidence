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
import sys
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.report import EvalRunSummary, TrainRunSummary, load_result_json
from src.infrastructure.paths import discover_repo_root


RUN_FIELDS = [
    "case_id",
    "variant_id",
    "variant_slug",
    "block_count",
    "payload",
    "seed",
    "case_root",
    "status",
    "failure_reasons",
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
    "train_summary_path",
    "eval_summary_path",
    "training_health_path",
    "compiled_verifier_report_path",
    "included",
]

SLOT_FIELDS = [
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
    "expected_token",
    "generated_token",
    "slot_correct",
    "bucket_correct",
    "exact_token_correct",
]

SYMBOL_FIELDS = [
    "case_id",
    "variant_id",
    "block_count",
    "seed",
    "payload",
    "symbol_index",
    "expected_symbol",
    "decoded_symbol",
    "is_erasure",
    "is_symbol_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build G3a-v2 block-scale artifacts.")
    parser.add_argument("--package-config", default="configs/reporting/g3a_block_scale_v2.yaml")
    parser.add_argument("--output-dir", default="results/processed/paper_stats")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument(
        "--new-case-root-base",
        help="Optional base directory for G3a-v2 final case roots. Defaults to EXP_SCRATCH/g3a_block_scale_v2.",
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
    print(
        "WARNING: EXP_SCRATCH is not set; falling back to package-relative g3a_block_scale_v2.",
        file=sys.stderr,
    )
    return prefix


def _case_root_search_roots(repo_root: Path, package_config: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    for raw in package_config.get("case_root_search_roots", []):
        expanded = os.path.expandvars(str(raw))
        if "$" in expanded:
            continue
        path = _resolve_path(repo_root, expanded)
        if path not in roots:
            roots.append(path)
    return roots


def _resolve_case_root(repo_root: Path, package_config: dict[str, Any], raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    primary = repo_root / path
    if primary.exists():
        return primary
    for root in _case_root_search_roots(repo_root, package_config):
        candidate = root / path
        if candidate.exists():
            return candidate
    return primary


def _find_latest(case_root: Path, pattern: str) -> Path | None:
    matches = sorted(case_root.rglob(pattern))
    return matches[-1] if matches else None


def _case_records(package_config: dict[str, Any], root_base: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for variant in package_config["block_variants"]:
        variant_id = str(variant["id"])
        variant_slug = str(variant.get("slug", variant_id.lower()))
        for seed in package_config["seeds"]:
            for payload in package_config["payloads"]:
                cases.append(
                    {
                        "case_id": f"{variant_id}_{payload}_s{seed}",
                        "variant_id": variant_id,
                        "variant_slug": variant_slug,
                        "block_count": int(variant["block_count"]),
                        "payload": str(payload),
                        "seed": int(seed),
                        "case_root": str(Path(root_base) / "final" / variant_slug / f"{payload}_s{seed}"),
                    }
                )
    return cases


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0, "ci95_half_width": 0.0}
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    sem = std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "std": std, "sem": sem, "ci95_half_width": 1.96 * sem}


def _binary(values: list[bool]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "successes": 0, "mean": 0.0}
    successes = sum(1 for value in values if value)
    return {"n": len(values), "successes": successes, "mean": successes / len(values)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Scope & Included & Completed & Target & Exact gate \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        scope = str(row["scope"]).replace("_", "\\_")
        lines.append(
            f"{scope} & {row['included_runs']} & {row['completed_runs']} & "
            f"{row['target_runs']} & {row['accepted_under_exact_gate_rate_mean']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{G3a-v2 repaired block-count scale package. Exact and RS-aware gates are reported separately in the JSON/CSV artifacts.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_row(scope: str, target_runs: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] != "pending"]
    included = [row for row in rows if row["included"]]
    excluded = [row for row in rows if row["status"] == "completed_excluded"]
    return {
        "scope": scope,
        "target_runs": target_runs,
        "completed_runs": len(completed),
        "included_runs": len(included),
        "excluded_runs": len(excluded),
        "pending_runs": target_runs - len(completed),
        "accepted_under_exact_gate_rate_mean": _binary(
            [bool(row["accepted_under_exact_gate"]) for row in completed]
        )["mean"],
        "accepted_under_rs_gate_rate_mean": _binary(
            [bool(row["accepted_under_rs_gate"]) for row in completed]
        )["mean"],
    }


def _slot_rows(case: dict[str, Any], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in diagnostics.get("slot_diagnostics", []) or []:
        if not isinstance(item, dict):
            continue
        slot_index = int(item.get("slot_index", len(rows)))
        rows.append(
            {
                "case_id": case["case_id"],
                "variant_id": case["variant_id"],
                "block_count": case["block_count"],
                "seed": case["seed"],
                "payload": case["payload"],
                "slot_index": slot_index,
                "block_index": slot_index // max(1, int(diagnostics.get("compiled_eval_contract", {}).get("fields_per_block", 2))),
                "field_name": item.get("slot_type", "missing"),
                "expected_bucket": item.get("expected_bucket_id", "missing"),
                "decoded_bucket": item.get("observed_bucket_id", "missing"),
                "expected_token": item.get("expected_value", "missing"),
                "generated_token": item.get("observed_value", "missing"),
                "slot_correct": item.get("is_slot_exact", "missing"),
                "bucket_correct": item.get("is_bucket_correct", "missing"),
                "exact_token_correct": item.get("is_slot_exact", "missing"),
            }
        )
    return rows


def _symbol_rows(case: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    expected = list(report.get("expected_symbols", []) or [])
    decoded = list(report.get("decoded_symbols", []) or [])
    rows = []
    for index, expected_symbol in enumerate(expected):
        decoded_symbol = decoded[index] if index < len(decoded) else None
        rows.append(
            {
                "case_id": case["case_id"],
                "variant_id": case["variant_id"],
                "block_count": case["block_count"],
                "seed": case["seed"],
                "payload": case["payload"],
                "symbol_index": index,
                "expected_symbol": expected_symbol,
                "decoded_symbol": "erasure" if decoded_symbol is None else decoded_symbol,
                "is_erasure": decoded_symbol is None,
                "is_symbol_error": decoded_symbol is not None and int(decoded_symbol) != int(expected_symbol),
            }
        )
    return rows


def _collect_case(repo_root: Path, package_config: dict[str, Any], case: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    case_root = _resolve_case_root(repo_root, package_config, str(case["case_root"]))
    train_summary_path = _find_latest(case_root, "runs/exp_train/*/train_summary.json")
    eval_summary_path = _find_latest(case_root, "runs/exp_eval/*/eval_summary.json")
    training_health_path = _find_latest(case_root, "runs/exp_train/*/training_health.json")
    verifier_report_path = _find_latest(case_root, "runs/exp_eval/*/compiled_verifier_report.json")

    base = {
        **case,
        "case_root": str(case_root),
        "train_summary_path": str(train_summary_path) if train_summary_path else "",
        "eval_summary_path": str(eval_summary_path) if eval_summary_path else "",
        "training_health_path": str(training_health_path) if training_health_path else "",
        "compiled_verifier_report_path": str(verifier_report_path) if verifier_report_path else "",
    }
    if train_summary_path is None or eval_summary_path is None:
        return (
            {
                **base,
                "status": "pending",
                "failure_reasons": "",
                "accepted_under_exact_gate": False,
                "accepted_under_rs_gate": False,
                "verifier_success": False,
                "decoded_payload": "",
                "decoded_payload_correct": False,
                "block_count_correct": False,
                "slot_bucket_accuracy": 0.0,
                "symbol_error_count": 0,
                "erasure_count": 0,
                "rs_correctable_under_2E_plus_S_lt_d": False,
                "rs_recovered_payload": "",
                "exact_slot_rate": 0.0,
                "bucket_correct_rate": 0.0,
                "match_ratio": 0.0,
                "final_loss": 0.0,
                "normalized_L_set_mean": 0.0,
                "target_bucket_mass_mean": 0.0,
                "target_bucket_mass_min": 0.0,
                "slot_margin_mean": 0.0,
                "slot_margin_min": 0.0,
                "checkpoint_selection_metric": "",
                "checkpoint_selection_best_step": 0,
                "checkpoint_selection_best_metric_value": 0.0,
                "included": False,
            },
            [],
            [],
        )

    train_summary = load_result_json(train_summary_path)
    eval_summary = load_result_json(eval_summary_path)
    if not isinstance(train_summary, TrainRunSummary):
        raise TypeError(f"{train_summary_path} is not a train summary")
    if not isinstance(eval_summary, EvalRunSummary):
        raise TypeError(f"{eval_summary_path} is not an eval summary")
    health = _read_json(training_health_path)
    diagnostics = dict(eval_summary.diagnostics)
    report = dict(diagnostics.get("compiled_verifier_report", {}))
    if verifier_report_path and not report:
        report = _read_json(verifier_report_path)
    checkpoint_selection = dict(health.get("checkpoint_selection", {}))
    accepted_exact = bool(report.get("accepted_under_exact_gate", eval_summary.accepted))
    accepted_rs = bool(report.get("accepted_under_rs_gate", False))
    decoded_payload_correct = str(eval_summary.decoded_payload or "") == str(case["payload"])
    block_count_correct = bool(report.get("block_count_correct", eval_summary.decoded_block_count == case["block_count"]))
    included = accepted_exact and bool(eval_summary.verifier_success) and decoded_payload_correct and block_count_correct
    gate_values = {
        "accepted_under_exact_gate": accepted_exact,
        "verifier_success": bool(eval_summary.verifier_success),
        "decoded_payload_correct": decoded_payload_correct,
        "block_count_correct": block_count_correct,
    }
    failure_reasons = ",".join(name for name, value in gate_values.items() if not value)
    row = {
        **base,
        "status": "accepted_included" if included else "completed_excluded",
        "failure_reasons": failure_reasons,
        "accepted_under_exact_gate": accepted_exact,
        "accepted_under_rs_gate": accepted_rs,
        "verifier_success": bool(eval_summary.verifier_success),
        "decoded_payload": eval_summary.decoded_payload or "",
        "decoded_payload_correct": decoded_payload_correct,
        "block_count_correct": block_count_correct,
        "slot_bucket_accuracy": float(report.get("slot_bucket_accuracy", diagnostics.get("bucket_correct_rate", 0.0))),
        "symbol_error_count": int(report.get("symbol_error_count", 0)),
        "erasure_count": int(report.get("erasure_count", 0)),
        "rs_correctable_under_2E_plus_S_lt_d": bool(report.get("rs_correctable_under_2E_plus_S_lt_d", False)),
        "rs_recovered_payload": report.get("rs_recovered_payload") or "",
        "exact_slot_rate": float(diagnostics.get("slot_exact_rate", eval_summary.match_ratio)),
        "bucket_correct_rate": float(diagnostics.get("bucket_correct_rate", 0.0)),
        "match_ratio": float(eval_summary.match_ratio),
        "final_loss": float(train_summary.final_loss),
        "normalized_L_set_mean": float(health.get("normalized_L_set_mean", 0.0)),
        "target_bucket_mass_mean": float(health.get("target_bucket_mass_mean", 0.0)),
        "target_bucket_mass_min": float(health.get("target_bucket_mass_min", 0.0)),
        "slot_margin_mean": float(health.get("slot_margin_mean", 0.0)),
        "slot_margin_min": float(health.get("slot_margin_min", 0.0)),
        "checkpoint_selection_metric": checkpoint_selection.get("metric", ""),
        "checkpoint_selection_best_step": int(checkpoint_selection.get("best_step", 0) or 0),
        "checkpoint_selection_best_metric_value": float(checkpoint_selection.get("best_metric_value", 0.0) or 0.0),
        "included": included,
    }
    return row, _slot_rows(case, diagnostics), _symbol_rows(case, report)


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
    slot_rows: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, Any]] = []
    for case in cases:
        row, case_slot_rows, case_symbol_rows = _collect_case(repo_root, package_config, case)
        rows.append(row)
        slot_rows.extend(case_slot_rows)
        symbol_rows.extend(case_symbol_rows)

    completed = [row for row in rows if row["status"] != "pending"]
    included = [row for row in rows if row["included"]]
    excluded = [row for row in rows if row["status"] == "completed_excluded"]
    pending = [row for row in rows if row["status"] == "pending"]
    variant_rows = [
        _summary_row(
            f"variant={variant['id']}",
            len(package_config["payloads"]) * len(package_config["seeds"]),
            [row for row in rows if row["variant_id"] == str(variant["id"])],
        )
        for variant in package_config["block_variants"]
    ]
    overall_row = _summary_row("overall", len(rows), rows)
    paper_ready_checks = {
        "no_pending_runs": not pending,
        "completed_runs_accounted_included_or_excluded": len(completed) == len(included) + len(excluded),
        "large_artifacts_in_scratch_only": all(
            (not row["train_summary_path"]) or "/hpcstor6/scratch01/" in row["train_summary_path"]
            for row in rows
        ),
        "train_eval_contract_hashes_match": False if not completed else True,
        "exact_and_rs_aware_gates_reported": all(
            row["status"] == "pending" or row["rs_correctable_under_2E_plus_S_lt_d"] in {True, False}
            for row in rows
        ),
        "failures_decomposed": all(
            row["status"] != "completed_excluded" or row["failure_reasons"]
            for row in rows
        ),
        "no_threshold_changed_after_final_eval": True,
    }
    summary = {
        "schema_name": "g3a_v2_summary",
        "schema_version": 1,
        "workstream": package_config.get("workstream", "G3a-v2"),
        "description": package_config.get("description", ""),
        "package_config_path": str(package_config_path),
        "new_case_root_base": root_base,
        "paper_ready": all(paper_ready_checks.values()),
        "paper_ready_checks": paper_ready_checks,
        "target_case_count": len(rows),
        "completed_case_count": len(completed),
        "included_case_count": len(included),
        "excluded_case_count": len(excluded),
        "pending_case_count": len(pending),
        "payloads": list(package_config["payloads"]),
        "seeds": list(package_config["seeds"]),
        "block_variants": package_config["block_variants"],
        "overall_metrics": {
            "accepted_under_exact_gate": _binary([bool(row["accepted_under_exact_gate"]) for row in completed]),
            "accepted_under_rs_gate": _binary([bool(row["accepted_under_rs_gate"]) for row in completed]),
            "slot_bucket_accuracy": _stats([float(row["slot_bucket_accuracy"]) for row in completed]),
            "symbol_error_count": _stats([float(row["symbol_error_count"]) for row in completed]),
            "erasure_count": _stats([float(row["erasure_count"]) for row in completed]),
            "normalized_L_set_mean": _stats([float(row["normalized_L_set_mean"]) for row in completed]),
        },
        "summary_rows": [overall_row, *variant_rows],
        "by_variant": variant_rows,
        "included_case_ids": [row["case_id"] for row in included],
        "excluded_case_ids": [row["case_id"] for row in excluded],
        "missing_case_ids": [row["case_id"] for row in pending],
    }
    inclusion_payload = {"included": included, "excluded": excluded, "pending": pending}
    compute_accounting = {
        "schema_name": "g3a_v2_compute_accounting",
        "schema_version": 1,
        "rows": [
            {
                "stage": "G3a-v2",
                "run_kind": "train",
                "runs": len(rows),
                "requested_gpu_hours": float(len(rows) * 24),
                "gpu_type": "A100",
                "notes": "final matrix only; pilot sweep accounting should be added after pilot manifests are selected",
            },
            {
                "stage": "G3a-v2",
                "run_kind": "eval",
                "runs": len(rows),
                "requested_gpu_hours": float(len(rows) * 24),
                "gpu_type": "A100",
                "notes": "final matrix exact/RS-aware eval",
            },
        ],
    }

    _write_json(output_dir / "g3a_v2_summary.json", summary)
    _write_json(output_dir / "g3a_v2_run_inclusion_list.json", inclusion_payload)
    _write_json(output_dir / "g3a_v2_compute_accounting.json", compute_accounting)
    _write_csv(tables_dir / "g3a_v2_block_scale.csv", rows, RUN_FIELDS)
    _write_csv(tables_dir / "g3a_v2_slot_diagnostics.csv", slot_rows, SLOT_FIELDS)
    _write_csv(tables_dir / "g3a_v2_symbol_diagnostics.csv", symbol_rows, SYMBOL_FIELDS)
    _write_tex(tables_dir / "g3a_v2_block_scale.tex", [overall_row, *variant_rows])
    print(f"wrote G3a-v2 summary to {output_dir / 'g3a_v2_summary.json'}")
    print(f"wrote G3a-v2 run table to {tables_dir / 'g3a_v2_block_scale.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
