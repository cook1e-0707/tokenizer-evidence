from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from src.infrastructure.paths import current_timestamp, discover_repo_root


FIELDS = [
    "row_id",
    "display_name",
    "method_type",
    "paper_status",
    "target_count",
    "success_count",
    "method_failure_count",
    "pending_count",
    "success_rate",
    "ci95_low",
    "ci95_high",
    "query_budget_scope",
    "required_label",
    "source_summary",
    "source_table",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the paper-facing baseline comparison table.")
    parser.add_argument(
        "--summary-out",
        default="results/processed/paper_stats/paper_baseline_comparison_summary.json",
    )
    parser.add_argument("--csv-out", default="results/tables/paper_baseline_comparison.csv")
    parser.add_argument("--tex-out", default="results/tables/paper_baseline_comparison.tex")
    return parser.parse_args()


def _read_json(repo_root: Path, path: str) -> dict[str, Any]:
    payload = json.loads((repo_root / path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _summary_row(summary: dict[str, Any], method_slug: str) -> dict[str, Any]:
    for row in summary.get("summary_rows", []):
        if row.get("method_slug") == method_slug:
            return row if isinstance(row, dict) else {}
    return {}


def _int(row: dict[str, Any], key: str) -> int:
    return int(row.get(key) or 0)


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _row(
    *,
    row_id: str,
    display_name: str,
    method_type: str,
    paper_status: str,
    target_count: int,
    success_count: int,
    method_failure_count: int,
    pending_count: int,
    query_budget_scope: str,
    required_label: str,
    source_summary: str,
    source_table: str,
    notes: str,
) -> dict[str, Any]:
    low, high = _wilson(success_count, target_count)
    return {
        "row_id": row_id,
        "display_name": display_name,
        "method_type": method_type,
        "paper_status": paper_status,
        "target_count": target_count,
        "success_count": success_count,
        "method_failure_count": method_failure_count,
        "pending_count": pending_count,
        "success_rate": success_count / target_count if target_count else 0.0,
        "ci95_low": low,
        "ci95_high": high,
        "query_budget_scope": query_budget_scope,
        "required_label": required_label,
        "source_summary": source_summary,
        "source_table": source_table,
        "notes": notes,
    }


def _format_rate(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Method & Status & Success & Fail & Rate (95\% CI) \\",
        r"\midrule",
    ]
    for row in rows:
        rate = _format_rate(float(row["success_rate"]))
        low = _format_rate(float(row["ci95_low"]))
        high = _format_rate(float(row["ci95_high"]))
        lines.append(
            "{} & {} & {}/{} & {} & {} [{}, {}] \\\\".format(
                _escape_latex(str(row["display_name"])),
                _escape_latex(str(row["paper_status"])),
                row["success_count"],
                row["target_count"],
                row["method_failure_count"],
                rate,
                low,
                high,
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Paper-facing baseline comparison. The Scalable/Perinucleus row is an "
                r"official-code Qwen LoRA adaptation and must be labeled as such.}"
            ),
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    g1 = _read_json(repo_root, "results/processed/paper_stats/g1_summary.json")
    matched = _read_json(repo_root, "results/processed/paper_stats/baseline_summary.json")
    official = _read_json(repo_root, "results/processed/paper_stats/baseline_perinucleus_official_qwen_final_summary.json")
    rows = []
    g1_success = int(g1.get("overall_metrics", {}).get("verifier_success_rate", {}).get("successes") or 0)
    g1_target = int(g1.get("overall_metrics", {}).get("verifier_success_rate", {}).get("n") or g1.get("target_case_count") or 0)
    rows.append(
        _row(
            row_id="ours_g1_payload_seed_scale",
            display_name="Ours (compiled ownership)",
            method_type="proposed_method",
            paper_status="main_method",
            target_count=g1_target,
            success_count=g1_success,
            method_failure_count=max(0, g1_target - g1_success),
            pending_count=int(g1.get("pending_case_count") or 0),
            query_budget_scope="frozen paper protocol",
            required_label="proposed method",
            source_summary="results/processed/paper_stats/g1_summary.json",
            source_table="results/tables/g1_payload_seed_scale.csv",
            notes="Publication-scale Qwen payload x seed package.",
        )
    )
    for slug, display, status, notes in [
        ("fixed_representative", "Fixed representative", "internal_ablation", "Internal ablation, not an external baseline."),
        ("uniform_bucket", "Uniform bucket", "internal_ablation", "Internal ablation, not an external baseline."),
        (
            "english_random_active_fingerprint",
            "English-random active fingerprint",
            "negative_diagnostic",
            "Weak proxy baseline; valid failures remain in denominator.",
        ),
    ]:
        item = _summary_row(matched, slug)
        rows.append(
            _row(
                row_id=slug,
                display_name=display,
                method_type="baseline_or_ablation",
                paper_status=status,
                target_count=_int(item, "target_count"),
                success_count=_int(item, "success_count"),
                method_failure_count=_int(item, "method_failure_count"),
                pending_count=_int(item, "pending_count"),
                query_budget_scope="B0 matched budget",
                required_label=status,
                source_summary="results/processed/paper_stats/baseline_summary.json",
                source_table="results/tables/matched_budget_baselines.csv",
                notes=notes,
            )
        )
    rows.append(
        _row(
            row_id="scalable_fingerprinting_perinucleus_official_qwen_final",
            display_name="Scalable/Perinucleus (Qwen-adapted official)",
            method_type="external_active_ownership_baseline",
            paper_status="main_external_baseline_with_adaptation_label",
            target_count=_int(official, "target_count"),
            success_count=_int(official, "success_count"),
            method_failure_count=_int(official, "method_failure_count"),
            pending_count=_int(official, "pending_count"),
            query_budget_scope="q=1/3/5/10 diagnostic grid",
            required_label="Qwen-adapted official Scalable/Perinucleus baseline",
            source_summary="results/processed/paper_stats/baseline_perinucleus_official_qwen_final_summary.json",
            source_table="results/tables/baseline_perinucleus_official_qwen_final.csv",
            notes="Official-code Qwen LoRA adaptation; do not conflate with legacy adapted baseline_perinucleus artifacts.",
        )
    )
    summary = {
        "schema_name": "paper_baseline_comparison_summary",
        "schema_version": 1,
        "generated_at": current_timestamp(),
        "row_count": len(rows),
        "main_external_baseline": "scalable_fingerprinting_perinucleus_official_qwen_final",
        "main_external_baseline_label": "Qwen-adapted official Scalable/Perinucleus baseline",
        "legacy_perinucleus_excluded": True,
        "comparison_rows": rows,
    }
    _write_json(repo_root / args.summary_out, summary)
    _write_csv(repo_root / args.csv_out, rows)
    _write_tex(repo_root / args.tex_out, rows)
    print(f"wrote paper baseline comparison summary to {repo_root / args.summary_out}")
    print(f"wrote paper baseline comparison csv to {repo_root / args.csv_out}")
    print(f"wrote paper baseline comparison tex to {repo_root / args.tex_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
