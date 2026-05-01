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

from scripts.run_full_far_payload_claim_benchmark import (
    _build_execution_summary,
    _build_metrics_tex,
    _load_yaml,
    _read_csv,
    _resolve,
    _write_csv,
    _write_json,
    _write_text,
    build_plan_rows,
    execute_plan_rows,
)
from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate sharded full-FAR outputs into the final table and summary."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--fresh-null-mode", default="organic-prompts")
    parser.add_argument("--expected-shard-count", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _read_csv_any(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _completed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("case_id", ""))
        and str(row.get("claim_accept", "")) != ""
        and str(row.get("status", "")).startswith("completed_")
    ]


def _shard_paths(shard_dir: Path, fresh_null_mode: str) -> list[Path]:
    mode = str(fresh_null_mode).replace("-", "_")
    return sorted(shard_dir.glob(f"full_far_payload_claim_{mode}_shard_*_of_*.csv"))


def _base_rows(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    final_table_path: Path,
    plan_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if final_table_path.exists():
        return _read_csv(final_table_path)
    return execute_plan_rows(repo_root, cfg, plan_rows, fresh_null_mode="off")


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    cfg = _load_yaml(config_path)
    outputs = cfg.get("outputs") or {}
    final_summary_path = _resolve(repo_root, outputs.get("final_summary"))
    final_table_path = _resolve(repo_root, outputs.get("final_table"))
    final_tex_path = _resolve(repo_root, outputs.get("final_tex"))
    if final_summary_path is None or final_table_path is None or final_tex_path is None:
        raise ValueError("outputs.final_summary, outputs.final_table, and outputs.final_tex are required.")

    shard_dir = _resolve(repo_root, args.shard_dir)
    if shard_dir is None or not shard_dir.exists():
        raise FileNotFoundError(f"Shard directory does not exist: {args.shard_dir}")
    shard_paths = _shard_paths(shard_dir, args.fresh_null_mode)
    if args.expected_shard_count is not None and len(shard_paths) != args.expected_shard_count:
        raise RuntimeError(
            f"Expected {args.expected_shard_count} shard CSVs, found {len(shard_paths)} in {shard_dir}."
        )
    if not shard_paths:
        raise FileNotFoundError(f"No shard CSVs found in {shard_dir} for mode={args.fresh_null_mode}.")

    plan_rows = build_plan_rows(cfg)
    combined_by_case_id = {
        str(row["case_id"]): row
        for row in _base_rows(
            repo_root=repo_root,
            cfg=cfg,
            final_table_path=final_table_path,
            plan_rows=plan_rows,
        )
        if str(row.get("case_id", ""))
    }
    duplicate_completed_case_ids: set[str] = set()
    shard_completed_count = 0
    for path in shard_paths:
        rows = _read_csv_any(path)
        for row in _completed_rows(rows):
            case_id = str(row["case_id"])
            if (
                case_id in combined_by_case_id
                and str(combined_by_case_id[case_id].get("claim_accept", "")) != ""
                and str(combined_by_case_id[case_id].get("execution_backend", ""))
                == str(row.get("execution_backend", ""))
            ):
                duplicate_completed_case_ids.add(case_id)
            combined_by_case_id[case_id] = row
            shard_completed_count += 1

    combined_rows: list[dict[str, Any]] = []
    missing_case_ids: list[str] = []
    for plan_row in plan_rows:
        case_id = str(plan_row["case_id"])
        row = combined_by_case_id.get(case_id)
        if row is None:
            missing_case_ids.append(case_id)
            combined_rows.append(plan_row)
        else:
            combined_rows.append(row)
    if missing_case_ids:
        raise RuntimeError(f"Missing {len(missing_case_ids)} planned case ids after aggregation.")

    execution_summary = _build_execution_summary(
        repo_root,
        cfg,
        config_path,
        plan_rows,
        combined_rows,
    )
    execution_summary["shard_aggregation"] = {
        "fresh_null_mode": args.fresh_null_mode,
        "shard_dir": str(shard_dir),
        "shard_csv_count": len(shard_paths),
        "expected_shard_count": args.expected_shard_count,
        "shard_completed_row_count": shard_completed_count,
        "duplicate_completed_case_id_count": len(duplicate_completed_case_ids),
        "shard_csvs": [str(path) for path in shard_paths],
    }
    _write_json(final_summary_path, execution_summary, force=args.force)
    _write_csv(final_table_path, combined_rows, force=args.force)
    _write_text(final_tex_path, _build_metrics_tex(execution_summary["metrics"]), force=args.force)
    print(
        json.dumps(
            {
                "status": "aggregated",
                "execution_status": execution_summary["status"],
                "summary": str(final_summary_path),
                "table": str(final_table_path),
                "tex": str(final_tex_path),
                "shard_csv_count": len(shard_paths),
                "shard_completed_row_count": shard_completed_count,
                "full_far_complete": execution_summary["full_far_complete"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
