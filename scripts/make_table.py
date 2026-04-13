from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a publication-friendly markdown table.")
    parser.add_argument(
        "--input",
        default="results/processed/comparison_rows.jsonl",
        help="JSONL file of comparison rows.",
    )
    parser.add_argument(
        "--output",
        default="results/tables/summary.md",
        help="Output markdown table path.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render_markdown_table(rows: list[dict[str, object]]) -> str:
    header = "| experiment | method | model | metric | value | status |\n|---|---|---|---|---:|---|"
    body = [
        "| {experiment_name} | {method_name} | {model_name} | {metric_name} | {metric_value:.4f} | {status} |".format(
            experiment_name=row["experiment_name"],
            method_name=row.get("method_name", row.get("method", "unknown_method")),
            model_name=row["model_name"],
            metric_name=row["metric_name"],
            metric_value=float(row["metric_value"]),
            status=row["status"],
        )
        for row in rows
    ]
    return "\n".join([header, *body, ""])


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = repo_root / input_path
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    output_path.write_text(render_markdown_table(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
