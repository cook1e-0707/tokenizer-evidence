from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.report import EvalRunSummary, TrainRunSummary, load_result_json
from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build G1 payload x seed scale aggregation artifacts.")
    parser.add_argument(
        "--package-config",
        default="configs/reporting/g1_payload_seed_scale_v1.yaml",
        help="YAML package config for the G1 payload x seed scale package.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/processed/paper_stats",
        help="Directory for JSON aggregation outputs.",
    )
    parser.add_argument(
        "--tables-dir",
        default="results/tables",
        help="Directory for CSV/TeX table outputs.",
    )
    parser.add_argument(
        "--new-case-root-base",
        help=(
            "Optional base directory for new G1 case roots. "
            "Defaults to the package config new_case_root_prefix."
        ),
    )
    return parser.parse_args()


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _case_id(payload: str, seed: int) -> str:
    return f"{payload}_s{seed}"


def _find_latest(case_root: Path, pattern: str) -> Path | None:
    matches = sorted(case_root.rglob(pattern))
    return matches[-1] if matches else None


def _join_case_root(prefix: str, case_id: str) -> str:
    return str((Path(prefix) / case_id).as_posix())


def _build_case_records(
    *,
    package_config: dict[str, Any],
    new_case_root_base: str | None,
) -> list[dict[str, Any]]:
    payloads = [str(item) for item in package_config["payloads"]]
    seeds = [int(item) for item in package_config["seeds"]]
    existing_by_id = {str(item["id"]): dict(item) for item in package_config.get("existing_cases", [])}
    new_case_root_prefix = str(new_case_root_base or package_config["new_case_root_prefix"])

    cases: list[dict[str, Any]] = []
    for seed in seeds:
        for payload in payloads:
            case_id = _case_id(payload, seed)
            existing = existing_by_id.get(case_id)
            if existing is not None:
                cases.append(
                    {
                        "id": case_id,
                        "payload": payload,
                        "seed": seed,
                        "case_root": str(existing["case_root"]),
                        "source_stage": str(existing.get("stage", "")),
                        "source_kind": "reused",
                    }
                )
            else:
                cases.append(
                    {
                        "id": case_id,
                        "payload": payload,
                        "seed": seed,
                        "case_root": _join_case_root(new_case_root_prefix, case_id),
                        "source_stage": "G1",
                        "source_kind": "new",
                    }
                )
    return cases


def _continuous_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "ci95_half_width": 0.0,
        }
    mean = float(sum(values) / len(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    sem = float(std / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_half_width": float(1.96 * sem) if len(values) > 1 else 0.0,
    }


def _binary_stats(values: list[bool]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "successes": 0,
            "mean": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "ci95_half_width": 0.0,
        }
    successes = sum(1 for value in values if value)
    mean = successes / n
    z = 1.96
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (mean + (z2 / (2.0 * n))) / denom
    margin = (z / denom) * math.sqrt((mean * (1.0 - mean) / n) + (z2 / (4.0 * n * n)))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return {
        "n": n,
        "successes": successes,
        "mean": float(mean),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci95_half_width": float((high - low) / 2.0),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _latex_escape(raw: Any) -> str:
    return str(raw).replace("_", "\\_")


def _write_tex(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Scope & Included runs & Target runs & Accepted rate & Verifier rate \\\\",
        "\\midrule",
    ]
    for row in rows:
        accepted = f"{row['accepted_rate_mean']:.3f} [{row['accepted_rate_ci95_low']:.3f}, {row['accepted_rate_ci95_high']:.3f}]"
        verifier = f"{row['verifier_success_rate_mean']:.3f} [{row['verifier_success_rate_ci95_low']:.3f}, {row['verifier_success_rate_ci95_high']:.3f}]"
        lines.append(
            f"{_latex_escape(row['scope'])} & {row['included_runs']} & {row['target_runs']} & "
            f"{_latex_escape(accepted)} & {_latex_escape(verifier)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Current landing state for the G1 payload x seed scale package on the compiled Qwen/Qwen2.5-7B-Instruct path. Wilson 95\\% intervals are reported over the currently included runs.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _collect_case_row(repo_root: Path, case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    case_root = _resolve_path(repo_root, str(case["case_root"]))
    train_summary_path = _find_latest(case_root, "runs/exp_train/*/train_summary.json")
    eval_summary_path = _find_latest(case_root, "runs/exp_eval/*/eval_summary.json")
    training_health_path = _find_latest(case_root, "runs/exp_train/*/training_health.json")

    base_row = {
        "case_id": str(case["id"]),
        "payload": str(case["payload"]),
        "seed": int(case["seed"]),
        "case_root": str(case_root),
        "source_stage": str(case["source_stage"]),
        "source_kind": str(case["source_kind"]),
        "train_summary_path": str(train_summary_path) if train_summary_path is not None else "",
        "eval_summary_path": str(eval_summary_path) if eval_summary_path is not None else "",
        "training_health_path": str(training_health_path) if training_health_path is not None else "",
    }
    if train_summary_path is None or eval_summary_path is None:
        return (
            {
                **base_row,
                "status": "pending",
                "accepted": False,
                "verifier_success": False,
                "decoded_payload": "",
                "decoded_payload_correct": False,
                "numerically_healthy": False,
                "match_ratio": 0.0,
                "train_run_id": "",
                "eval_run_id": "",
                "final_loss": 0.0,
                "included": False,
            },
            None,
        )

    train_summary = load_result_json(train_summary_path)
    eval_summary = load_result_json(eval_summary_path)
    if not isinstance(train_summary, TrainRunSummary):
        raise TypeError(f"{train_summary_path} is not a train summary")
    if not isinstance(eval_summary, EvalRunSummary):
        raise TypeError(f"{eval_summary_path} is not an eval summary")

    training_health = {}
    if training_health_path is not None:
        training_health = json.loads(training_health_path.read_text(encoding="utf-8"))
    numerically_healthy = training_health.get("first_nan_step") is None
    decoded_payload_correct = str(eval_summary.decoded_payload or "") == str(case["payload"])
    included = bool(eval_summary.accepted) and bool(eval_summary.verifier_success) and numerically_healthy and decoded_payload_correct
    status = "accepted_included" if included else "completed_excluded"

    row = {
        **base_row,
        "status": status,
        "accepted": bool(eval_summary.accepted),
        "verifier_success": bool(eval_summary.verifier_success),
        "decoded_payload": str(eval_summary.decoded_payload or ""),
        "decoded_payload_correct": decoded_payload_correct,
        "numerically_healthy": numerically_healthy,
        "match_ratio": float(eval_summary.match_ratio),
        "train_run_id": str(train_summary.run_id),
        "eval_run_id": str(eval_summary.run_id),
        "final_loss": float(train_summary.final_loss),
        "included": included,
    }
    inclusion = None
    if included:
        inclusion = {
            "case_id": str(case["id"]),
            "payload": str(case["payload"]),
            "seed": int(case["seed"]),
            "case_root": str(case_root),
            "source_stage": str(case["source_stage"]),
            "source_kind": str(case["source_kind"]),
            "train_summary_path": str(train_summary_path),
            "eval_summary_path": str(eval_summary_path),
            "training_health_path": str(training_health_path) if training_health_path is not None else "",
            "train_run_id": str(train_summary.run_id),
            "eval_run_id": str(eval_summary.run_id),
        }
    return row, inclusion


def _summary_row(scope: str, target_runs: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_stats = _binary_stats([bool(row["accepted"]) for row in rows])
    verifier_stats = _binary_stats([bool(row["verifier_success"]) for row in rows])
    decoded_stats = _binary_stats([bool(row["decoded_payload_correct"]) for row in rows])
    return {
        "scope": scope,
        "included_runs": len(rows),
        "target_runs": target_runs,
        "accepted_rate_mean": accepted_stats["mean"],
        "accepted_rate_ci95_low": accepted_stats["ci95_low"],
        "accepted_rate_ci95_high": accepted_stats["ci95_high"],
        "verifier_success_rate_mean": verifier_stats["mean"],
        "verifier_success_rate_ci95_low": verifier_stats["ci95_low"],
        "verifier_success_rate_ci95_high": verifier_stats["ci95_high"],
        "decoded_payload_correct_rate_mean": decoded_stats["mean"],
        "decoded_payload_correct_rate_ci95_low": decoded_stats["ci95_low"],
        "decoded_payload_correct_rate_ci95_high": decoded_stats["ci95_high"],
    }


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = _resolve_path(repo_root, args.tables_dir)

    cases = _build_case_records(package_config=package_config, new_case_root_base=args.new_case_root_base)
    rows: list[dict[str, Any]] = []
    inclusion_rows: list[dict[str, Any]] = []
    for case in cases:
        row, inclusion = _collect_case_row(repo_root, case)
        rows.append(row)
        if inclusion is not None:
            inclusion_rows.append(inclusion)

    included_rows = [row for row in rows if bool(row["included"])]
    payloads = [str(item) for item in package_config["payloads"]]
    seeds = [int(item) for item in package_config["seeds"]]
    by_seed = [
        _summary_row(f"seed={seed}", len(payloads), [row for row in included_rows if int(row["seed"]) == seed])
        for seed in seeds
    ]
    by_payload = [
        _summary_row(
            f"payload={payload}",
            len(seeds),
            [row for row in included_rows if str(row["payload"]) == payload],
        )
        for payload in payloads
    ]
    overall_row = _summary_row("overall", len(payloads) * len(seeds), included_rows)

    summary = {
        "workstream": str(package_config.get("workstream", "G1")),
        "description": str(package_config.get("description", "")),
        "package_config_path": str(package_config_path),
        "paper_ready": len(included_rows) == len(rows),
        "target_case_count": len(rows),
        "included_case_count": len(included_rows),
        "pending_case_count": sum(1 for row in rows if row["status"] == "pending"),
        "excluded_case_count": sum(1 for row in rows if row["status"] == "completed_excluded"),
        "payloads": payloads,
        "seeds": seeds,
        "overall_metrics": {
            "accepted_rate": _binary_stats([bool(row["accepted"]) for row in included_rows]),
            "verifier_success_rate": _binary_stats([bool(row["verifier_success"]) for row in included_rows]),
            "decoded_payload_correct_rate": _binary_stats(
                [bool(row["decoded_payload_correct"]) for row in included_rows]
            ),
            "match_ratio": _continuous_stats([float(row["match_ratio"]) for row in included_rows]),
            "final_loss": _continuous_stats([float(row["final_loss"]) for row in included_rows]),
        },
        "summary_rows": [overall_row, *by_seed],
        "by_seed": by_seed,
        "by_payload": by_payload,
        "missing_case_ids": [str(row["case_id"]) for row in rows if row["status"] == "pending"],
        "included_case_ids": [str(row["case_id"]) for row in included_rows],
    }

    _write_json(output_dir / "g1_summary.json", summary)
    _write_json(
        output_dir / "g1_run_inclusion_list.json",
        {
            "included": inclusion_rows,
            "pending": [row for row in rows if row["status"] == "pending"],
            "excluded": [row for row in rows if row["status"] == "completed_excluded"],
        },
    )
    _write_csv(tables_dir / "g1_payload_seed_scale.csv", rows)
    _write_tex(tables_dir / "g1_payload_seed_scale.tex", [overall_row, *by_seed])

    print(f"wrote G1 summary to {output_dir / 'g1_summary.json'}")
    print(f"wrote G1 inclusion list to {output_dir / 'g1_run_inclusion_list.json'}")
    print(f"wrote G1 table to {tables_dir / 'g1_payload_seed_scale.csv'}")
    print(f"wrote G1 TeX table to {tables_dir / 'g1_payload_seed_scale.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
