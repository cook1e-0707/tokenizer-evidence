from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ROWS = Path(
    "results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_v3_20260513/"
    "r4_prefix_native_surface_probe_rows_v3.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("results/natural_evidence_v2/status/r4_candidate_v3_prefix_template_leakage_audit_20260513")

TECHNICAL_TERMS = (
    "bucket",
    "fingerprint",
    "watermark",
    "payload",
    "secret key",
    "coordinate",
    "decoder",
    "hidden signal",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    prefix_counts = Counter(str(row.get("assistant_prefix_before_surface", "")).strip() for row in rows)
    surface_counts = Counter(str(row.get("target_surface", "")) for row in rows)
    prompt_count = len({str(row.get("prompt_id")) for row in rows})
    row_count = len(rows)
    technical_hits: list[dict[str, Any]] = []
    step_label_hits = 0
    for row in rows:
        haystack = " ".join(
            str(row.get(key, ""))
            for key in (
                "prompt_text",
                "assistant_prefix_before_surface",
                "target_response_text",
                "target_surface",
            )
        ).lower()
        if "step " in haystack or "step:" in haystack:
            step_label_hits += 1
        for term in TECHNICAL_TERMS:
            if term in haystack:
                technical_hits.append(
                    {
                        "prompt_id": row.get("prompt_id"),
                        "coordinate_id": row.get("coordinate_id"),
                        "term": term,
                    }
                )
    max_prefix_fraction = max(prefix_counts.values()) / row_count if row_count else 0.0
    max_surface_fraction = max(surface_counts.values()) / row_count if row_count else 0.0
    risk_flags = []
    if max_prefix_fraction > 0.50:
        risk_flags.append("prefix_family_concentration_gt_0_50")
    if max_surface_fraction > 0.35:
        risk_flags.append("surface_family_concentration_gt_0_35")
    if technical_hits:
        risk_flags.append("technical_literal_hits")
    if step_label_hits:
        risk_flags.append("step_label_hits")
    return {
        "max_prefix_fraction": max_prefix_fraction,
        "max_surface_fraction": max_surface_fraction,
        "prompt_count": prompt_count,
        "risk_flags": risk_flags,
        "row_count": row_count,
        "status": "ARTIFACT_ONLY_R4_PREFIX_TEMPLATE_LEAKAGE_AUDIT_RECORDED_NO_RUN",
        "step_label_hit_rows": step_label_hits,
        "technical_literal_hit_rows": len(technical_hits),
        "technical_hits": technical_hits[:25],
        "top_prefixes": prefix_counts.most_common(20),
        "top_surfaces": surface_counts.most_common(20),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R4 candidate rows for visible prefix-template leakage without generation.")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {args.output_dir}")
    rows = read_jsonl(args.rows)
    summary = audit(rows)
    write_json(args.output_dir / "prefix_template_leakage_summary.json", summary)
    write_csv(
        args.output_dir / "prefix_counts.csv",
        [{"assistant_prefix_before_surface": key, "row_count": value} for key, value in summary["top_prefixes"]],
    )
    write_csv(
        args.output_dir / "surface_counts.csv",
        [{"target_surface": key, "row_count": value} for key, value in summary["top_surfaces"]],
    )
    report = [
        "# R4 Prefix-Template Leakage Audit",
        "",
        "Artifact-only audit over candidate rows. No generation or model scoring was started.",
        "",
        f"- rows: `{summary['row_count']}`",
        f"- prompts: `{summary['prompt_count']}`",
        f"- max prefix fraction: `{summary['max_prefix_fraction']}`",
        f"- max surface fraction: `{summary['max_surface_fraction']}`",
        f"- technical literal hit rows: `{summary['technical_literal_hit_rows']}`",
        f"- step-label hit rows: `{summary['step_label_hit_rows']}`",
        f"- risk flags: `{summary['risk_flags']}`",
    ]
    (args.output_dir / "prefix_template_leakage_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
