from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import itertools
import json
import os
import statistics
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.report import EvalRunSummary, TrainRunSummary, load_result_json
from src.infrastructure.paths import current_timestamp, discover_repo_root


ROW_FIELDS = [
    "hp",
    "variant",
    "block_count",
    "payload",
    "seed",
    "case_root",
    "status",
    "accepted",
    "verifier_success",
    "decoded_payload",
    "decoded_payload_correct",
    "match_ratio",
    "accepted_under_exact_gate",
    "accepted_under_rs_gate",
    "slot_bucket_accuracy",
    "symbol_error_count",
    "erasure_count",
    "final_loss",
    "train_summary_path",
    "eval_summary_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build G3a-v2 pilot operating-point summary.")
    parser.add_argument("--package-config", default="configs/reporting/g3a_block_scale_v2.yaml")
    parser.add_argument(
        "--pilot-root",
        help="Pilot root. Defaults to EXP_SCRATCH/g3a_block_scale_v2/pilot.",
    )
    parser.add_argument(
        "--output",
        default="results/processed/paper_stats/g3a_v2_pilot_selection_summary.json",
    )
    parser.add_argument(
        "--table-out",
        default="results/tables/g3a_v2_pilot_selection_summary.csv",
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


def _resolve_pilot_root(package_config: dict[str, Any], explicit: str | None) -> Path:
    if explicit:
        return Path(os.path.expandvars(explicit))
    exp_scratch = os.environ.get("EXP_SCRATCH")
    if not exp_scratch:
        raise ValueError("EXP_SCRATCH is required unless --pilot-root is provided")
    return Path(exp_scratch) / str(package_config["new_case_root_prefix"]) / "pilot"


def _latest(case_root: Path, pattern: str) -> Path | None:
    matches = sorted(case_root.rglob(pattern))
    return matches[-1] if matches else None


def _pilot_cases(package_config: dict[str, Any], pilot_root: Path) -> list[dict[str, Any]]:
    pilot = dict(package_config["pilot_validation"])
    variant_by_id = {str(item["id"]): dict(item) for item in package_config["block_variants"]}
    sweep = dict(pilot["sweep"])
    keys = ["lora_r", "learning_rate", "epochs", "lambda_set"]
    cases: list[dict[str, Any]] = []
    for hp_index, values in enumerate(itertools.product(*(sweep[key] for key in keys)), start=1):
        hp = dict(zip(keys, values, strict=True))
        hp_id = f"hp{hp_index:02d}"
        for variant_id in pilot["block_variants"]:
            variant = variant_by_id[str(variant_id)]
            variant_slug = str(variant.get("slug", str(variant["id"]).lower()))
            for payload in pilot["payloads"]:
                seed = int(pilot["seed"])
                cases.append(
                    {
                        "hp": hp_id,
                        "hyperparameters": hp,
                        "variant": variant_slug,
                        "variant_id": str(variant["id"]),
                        "block_count": int(variant["block_count"]),
                        "payload": str(payload),
                        "seed": seed,
                        "case_root": pilot_root / hp_id / variant_slug / f"{payload}_s{seed}",
                    }
                )
    return cases


def _load_summary(path: Path | None, expected_type: type[TrainRunSummary] | type[EvalRunSummary]) -> Any | None:
    if path is None:
        return None
    summary = load_result_json(path)
    if not isinstance(summary, expected_type):
        raise TypeError(f"{path} is not a {expected_type.__name__}")
    return summary


def _float_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _collect_case(case: dict[str, Any]) -> dict[str, Any]:
    case_root = Path(case["case_root"])
    train_summary_path = _latest(case_root, "runs/exp_train/*/train_summary.json")
    eval_summary_path = _latest(case_root, "runs/exp_eval/*/eval_summary.json")
    train_summary = _load_summary(train_summary_path, TrainRunSummary)
    eval_summary = _load_summary(eval_summary_path, EvalRunSummary)

    status = "completed"
    if train_summary is None and eval_summary is None:
        status = "missing_train_and_eval"
    elif train_summary is None:
        status = "missing_train"
    elif eval_summary is None:
        status = "missing_eval"

    diagnostics = dict(eval_summary.diagnostics) if eval_summary is not None else {}
    report = dict(diagnostics.get("compiled_verifier_report", {}))
    accepted = bool(eval_summary.accepted) if eval_summary is not None else False
    verifier_success = bool(eval_summary.verifier_success) if eval_summary is not None else False
    decoded_payload = eval_summary.decoded_payload or "" if eval_summary is not None else ""
    accepted_exact = bool(report.get("accepted_under_exact_gate", accepted))
    accepted_rs = bool(report.get("accepted_under_rs_gate", False))

    return {
        "hp": case["hp"],
        "hyperparameters": case["hyperparameters"],
        "variant": case["variant"],
        "variant_id": case["variant_id"],
        "block_count": case["block_count"],
        "payload": case["payload"],
        "seed": case["seed"],
        "case_root": str(case_root),
        "status": status,
        "accepted": accepted,
        "verifier_success": verifier_success,
        "decoded_payload": decoded_payload,
        "decoded_payload_correct": decoded_payload == case["payload"] if eval_summary is not None else False,
        "match_ratio": float(eval_summary.match_ratio) if eval_summary is not None else 0.0,
        "accepted_under_exact_gate": accepted_exact,
        "accepted_under_rs_gate": accepted_rs,
        "slot_bucket_accuracy": float(report.get("slot_bucket_accuracy", diagnostics.get("bucket_correct_rate", 0.0))),
        "symbol_error_count": int(report.get("symbol_error_count", 0)),
        "erasure_count": int(report.get("erasure_count", 0)),
        "final_loss": float(train_summary.final_loss) if train_summary is not None else 0.0,
        "train_summary_path": str(train_summary_path) if train_summary_path else "",
        "eval_summary_path": str(eval_summary_path) if eval_summary_path else "",
    }


def _aggregate(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    output: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        completed = [row for row in group_rows if row["status"] == "completed"]
        train_completed = [row for row in group_rows if row["train_summary_path"]]
        eval_completed = [row for row in group_rows if row["eval_summary_path"]]
        output.append(
            {
                key: group_key,
                "target": len(group_rows),
                "completed": len(completed),
                "train_completed": len(train_completed),
                "eval_completed": len(eval_completed),
                "accepted": sum(1 for row in eval_completed if row["accepted"]),
                "verifier_success": sum(1 for row in eval_completed if row["verifier_success"]),
                "decoded_payload_correct": sum(1 for row in eval_completed if row["decoded_payload_correct"]),
                "accepted_under_exact_gate": sum(1 for row in eval_completed if row["accepted_under_exact_gate"]),
                "accepted_under_rs_gate": sum(1 for row in eval_completed if row["accepted_under_rs_gate"]),
                "mean_match_ratio": _float_mean([float(row["match_ratio"]) for row in eval_completed]),
                "mean_slot_bucket_accuracy": _float_mean(
                    [float(row["slot_bucket_accuracy"]) for row in eval_completed]
                ),
                "mean_final_loss": _float_mean([float(row["final_loss"]) for row in train_completed]),
                "status_counts": {
                    status: sum(1 for row in group_rows if row["status"] == status)
                    for status in sorted({str(row["status"]) for row in group_rows})
                },
            }
        )
    return output


def _rank_hps(by_hp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        by_hp,
        key=lambda row: (
            int(row["completed"]),
            int(row["accepted_under_exact_gate"]),
            int(row["verifier_success"]),
            float(row["mean_slot_bucket_accuracy"]),
            float(row["mean_match_ratio"]),
            -float(row["mean_final_loss"]),
        ),
        reverse=True,
    )
    return [{**row, "rank": index} for index, row in enumerate(ranked, start=1)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    pilot_root = _resolve_pilot_root(package_config, args.pilot_root)
    output_path = _resolve_path(repo_root, args.output)
    table_path = _resolve_path(repo_root, args.table_out)

    rows = [_collect_case(case) for case in _pilot_cases(package_config, pilot_root)]
    by_hp = _aggregate(rows, "hp")
    payload = {
        "schema_name": "g3a_v2_pilot_selection_summary",
        "schema_version": 2,
        "workstream": "G3a-v2",
        "generated_at": current_timestamp(),
        "package_config_path": str(package_config_path),
        "pilot_root": str(pilot_root),
        "target_case_count": len(rows),
        "train_completed_count": sum(1 for row in rows if row["train_summary_path"]),
        "eval_completed_count": sum(1 for row in rows if row["eval_summary_path"]),
        "completed_count": sum(1 for row in rows if row["status"] == "completed"),
        "accepted_count": sum(1 for row in rows if row["accepted"]),
        "verifier_success_count": sum(1 for row in rows if row["verifier_success"]),
        "by_hp": by_hp,
        "by_variant": _aggregate(rows, "variant"),
        "ranked_hps": _rank_hps(by_hp),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(table_path, rows)

    print(f"wrote G3a-v2 pilot selection summary to {output_path}")
    print(f"wrote G3a-v2 pilot selection table to {table_path}")
    print(f"completed={payload['completed_count']} eval_completed={payload['eval_completed_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
