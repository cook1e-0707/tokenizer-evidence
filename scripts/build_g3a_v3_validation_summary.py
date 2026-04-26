from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.build_g3a_v3_block_scale_artifacts import _numeric, _slot_margin_rows
from scripts.build_g3a_v2_block_scale_artifacts import _collect_case
from scripts.prepare_g3a_v3_block_scale import _load_yaml, _validation_cases
from src.infrastructure.paths import current_timestamp, discover_repo_root


FIELDS = [
    "hp_id",
    "margin_gamma",
    "lambda_margin",
    "checkpoint_selection_metric",
    "checkpoint_selection_mode",
    "target_count",
    "completed_count",
    "valid_completed_count",
    "success_count",
    "method_failure_count",
    "invalid_excluded_count",
    "pending_count",
    "exact_gate_success_rate",
    "rs_gate_success_rate",
    "mean_bucket_margin",
    "mean_total_evidence_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build G3a-v3 validation selection summary.")
    parser.add_argument("--package-config", default="configs/reporting/g3a_block_scale_v3.yaml")
    parser.add_argument("--output-dir", default="results/processed/paper_stats")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument(
        "--validation-root-base",
        help="Optional base directory for G3a-v3 validation case roots. Defaults to EXP_SCRATCH/g3a_block_scale_v3.",
    )
    return parser.parse_args()


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(os.path.expandvars(raw))
    return path if path.is_absolute() else repo_root / path


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _case_total_evidence_loss(row: dict[str, Any]) -> float | None:
    health_path = row.get("training_health_path")
    if not health_path:
        return None
    path = Path(str(health_path))
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    if "total_evidence_loss_mean" in payload:
        return _numeric(payload["total_evidence_loss_mean"])
    if "normalized_L_set_mean" in payload:
        return _numeric(payload["normalized_L_set_mean"])
    return None


def _hp_summary(hp_id: str, hp_cases: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for _case, row in hp_cases]
    valid_completed = [row for row in rows if bool(row["valid_completed"])]
    successes = [row for row in rows if bool(row["success"])]
    method_failures = [row for row in rows if bool(row["method_failure"])]
    invalid = [row for row in rows if bool(row["invalid_excluded"])]
    pending = [row for row in rows if row["result_class"] == "pending"]
    margin_values: list[float] = []
    total_losses: list[float] = []
    for case, row in hp_cases:
        for slot_row in _slot_margin_rows(case, row):
            if slot_row.get("bucket_margin") not in {"", "missing", None}:
                margin_values.append(_numeric(slot_row["bucket_margin"]))
        total_loss = _case_total_evidence_loss(row)
        if total_loss is not None:
            total_losses.append(total_loss)
    hp = dict(hp_cases[0][0].get("hyperparameters", {})) if hp_cases else {}
    return {
        "hp_id": hp_id,
        "margin_gamma": hp.get("margin_gamma", ""),
        "lambda_margin": hp.get("lambda_margin", ""),
        "checkpoint_selection_metric": hp.get("checkpoint_selection_metric", ""),
        "checkpoint_selection_mode": hp.get("checkpoint_selection_mode", ""),
        "target_count": len(rows),
        "completed_count": len([row for row in rows if row["result_class"] != "pending"]),
        "valid_completed_count": len(valid_completed),
        "success_count": len(successes),
        "method_failure_count": len(method_failures),
        "invalid_excluded_count": len(invalid),
        "pending_count": len(pending),
        "exact_gate_success_rate": len(successes) / len(valid_completed) if valid_completed else 0.0,
        "rs_gate_success_rate": (
            sum(1 for row in valid_completed if bool(row["accepted_under_rs_gate"])) / len(valid_completed)
            if valid_completed
            else 0.0
        ),
        "mean_bucket_margin": sum(margin_values) / len(margin_values) if margin_values else 0.0,
        "mean_total_evidence_loss": sum(total_losses) / len(total_losses) if total_losses else 0.0,
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["exact_gate_success_rate"]),
        float(row["rs_gate_success_rate"]),
        float(row["mean_bucket_margin"]),
        -float(row["mean_total_evidence_loss"]),
    )


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = _resolve_path(repo_root, args.tables_dir)
    root_base = _resolve_output_root_base(package_config, args.validation_root_base)
    cases = _validation_cases(package_config, root_base)

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []
    for case in cases:
        row, _slot_rows, _symbol_rows = _collect_case(repo_root, package_config, case)
        grouped[str(case["hp_id"])].append((case, row))
        all_rows.append(row)
    summary_rows = [_hp_summary(hp_id, grouped[hp_id]) for hp_id in sorted(grouped)]
    selection_ready = bool(summary_rows) and all(int(row["pending_count"]) == 0 for row in summary_rows)
    selection_ready = selection_ready and all(int(row["invalid_excluded_count"]) == 0 for row in summary_rows)
    selected = max(summary_rows, key=_selection_key) if selection_ready else None
    payload = {
        "schema_name": "g3a_v3_validation_summary",
        "schema_version": 1,
        "generated_at": current_timestamp(),
        "package_config_path": str(package_config_path),
        "validation_root_base": root_base,
        "selection_ready": selection_ready,
        "selection_rule": package_config["validation"]["selection_rule"],
        "selected_operating_point": selected,
        "hp_rows": summary_rows,
        "target_count": len(all_rows),
        "completed_count": len([row for row in all_rows if row["result_class"] != "pending"]),
        "pending_count": len([row for row in all_rows if row["result_class"] == "pending"]),
    }
    _write_json(output_dir / "g3a_v3_validation_summary.json", payload)
    _write_csv(tables_dir / "g3a_v3_validation_summary.csv", summary_rows)
    print(f"wrote G3a-v3 validation summary to {output_dir / 'g3a_v3_validation_summary.json'}")
    print(f"wrote G3a-v3 validation table to {tables_dir / 'g3a_v3_validation_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
