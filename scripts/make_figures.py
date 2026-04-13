from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a lightweight SVG figure from comparison rows.")
    parser.add_argument(
        "--input",
        default="results/processed/comparison_rows.jsonl",
        help="JSONL file of comparison rows.",
    )
    parser.add_argument(
        "--output",
        default="results/figures/summary.svg",
        help="Output SVG path.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render_svg(rows: list[dict[str, object]]) -> str:
    selected_rows = rows[:8]
    width = 720
    bar_height = 28
    gap = 14
    left_margin = 220
    top_margin = 40
    chart_height = top_margin + len(selected_rows) * (bar_height + gap) + 40
    max_value = max((float(row["metric_value"]) for row in selected_rows), default=1.0)
    scale = 360 / max(max_value, 1.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{chart_height}" viewBox="0 0 {width} {chart_height}">',
        '<rect width="100%" height="100%" fill="#f7f4ec" />',
        '<text x="24" y="28" font-size="20" font-family="Georgia, serif" fill="#1d2a38">Experiment Summary</text>',
    ]

    for index, row in enumerate(selected_rows):
        y = top_margin + index * (bar_height + gap)
        value = float(row["metric_value"])
        bar_width = max(2.0, value * scale)
        method_name = row.get("method_name", row.get("method", "unknown_method"))
        label = f'{row["experiment_name"]} | {method_name} | {row["metric_name"]}'
        parts.extend(
            [
                f'<text x="24" y="{y + 18}" font-size="12" font-family="Menlo, monospace" fill="#36454f">{label}</text>',
                f'<rect x="{left_margin}" y="{y}" width="{bar_width}" height="{bar_height}" rx="6" fill="#2c7a7b" />',
                f'<text x="{left_margin + bar_width + 8}" y="{y + 18}" font-size="12" font-family="Menlo, monospace" fill="#1d2a38">{value:.4f}</text>',
            ]
        )

    parts.append("</svg>")
    return "\n".join(parts)


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
    output_path.write_text(render_svg(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
