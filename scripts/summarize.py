from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from src.evaluation.report import (
    AggregatedComparisonRow,
    AttackRunSummary,
    CalibrationSummary,
    EvalRunSummary,
    TrainRunSummary,
    maybe_load_result_json,
)
from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate structured run outputs.")
    parser.add_argument("--results", default="results/raw", help="Raw results root.")
    parser.add_argument("--output-dir", default="results/processed", help="Processed results output dir.")
    return parser.parse_args()


def collect_summary_objects(raw_root: Path) -> list[object]:
    payloads: list[object] = []
    for path in sorted(raw_root.rglob("*.json")):
        result = maybe_load_result_json(path)
        if result is not None:
            payloads.append(result)
    return payloads


def build_comparison_rows(results: list[object]) -> list[AggregatedComparisonRow]:
    rows: list[AggregatedComparisonRow] = []
    for result in results:
        if isinstance(result, TrainRunSummary):
            metric_name = "final_loss"
            metric_value = result.final_loss
        elif isinstance(result, EvalRunSummary):
            metric_name = "match_ratio"
            metric_value = result.match_ratio
        elif isinstance(result, CalibrationSummary):
            metric_name = "observed_far"
            metric_value = result.observed_far
        elif isinstance(result, AttackRunSummary):
            metric_name = "accepted_after"
            metric_value = 1.0 if result.accepted_after else 0.0
        else:
            continue

        rows.append(
            AggregatedComparisonRow(
                run_id=result.run_id,
                experiment_name=result.experiment_name,
                method_name=result.method_name,
                model_name=result.model_name,
                seed=result.seed,
                git_commit=result.git_commit,
                timestamp=result.timestamp,
                hostname=result.hostname,
                slurm_job_id=result.slurm_job_id,
                status=result.status,
                metric_name=metric_name,
                metric_value=float(metric_value),
                source_schema=result.schema_name,
                notes="aggregated from raw run artifacts",
            )
        )
    return rows


def write_jsonl(path: Path, payloads: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(payload, sort_keys=True) for payload in payloads]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root()
    raw_root = Path(args.results)
    output_dir = Path(args.output_dir)
    if not raw_root.is_absolute():
        raw_root = repo_root / raw_root
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    results = collect_summary_objects(raw_root)
    summary_payloads = [result.to_json_dict() for result in results]
    comparison_rows = [row.to_json_dict() for row in build_comparison_rows(results)]
    write_jsonl(output_dir / "run_summaries.jsonl", summary_payloads)
    write_jsonl(output_dir / "comparison_rows.jsonl", comparison_rows)
    print(f"aggregated {len(summary_payloads)} run summaries into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
