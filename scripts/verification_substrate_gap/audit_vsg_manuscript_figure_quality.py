#!/usr/bin/env python3
"""Audit active VSG manuscript figure quality and traceability.

The audit is artifact-only. It reads the active manuscript figures, LaTeX
sources, render manifest, and frozen figure-data CSVs. It does not render new
figures, start compute, run generation, score models, or create new claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "manuscripts" / "69db2644566dcc36c9da320e"
FIGURE_DIR = MANUSCRIPT / "figures"
DATA_DIR = ROOT / "results" / "verification_substrate_gap" / "paper_figure_data_20260530"
RENDER_DIR = ROOT / "results" / "verification_substrate_gap" / "paper_manuscript_figures_20260531"
RENDER_MANIFEST = RENDER_DIR / "manuscript_figure_render_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "verification_substrate_gap" / "manuscript_figure_quality_audit_20260601"

EXPECTED_FIGURES = {
    "figure_1_verification_substrate_map.png": {
        "min_width": 1400,
        "min_height": 700,
        "max_aspect": 2.4,
        "min_nonwhite_ratio": 0.03,
        "tex_file": "section_03_problem_setup.tex",
        "caption_terms": ["taxonomy", "ownership proof"],
    },
    "figure_2_first_divergence_diagnostic.png": {
        "min_width": 1400,
        "min_height": 600,
        "max_aspect": 2.7,
        "min_nonwhite_ratio": 0.03,
        "tex_file": "section_04_tokenizer_alignment.tex",
        "caption_terms": ["trace-bound", "public final-text"],
    },
    "figure_3_controllability_vs_observability.png": {
        "min_width": 1500,
        "min_height": 800,
        "max_aspect": 2.4,
        "min_nonwhite_ratio": 0.03,
        "tex_file": "section_07_experiments.tex",
        "caption_terms": ["Trace-bound", "public final-text"],
    },
    "figure_4_public_predicate_attack_ladder.png": {
        "min_width": 1500,
        "min_height": 800,
        "max_aspect": 2.4,
        "min_nonwhite_ratio": 0.03,
        "tex_file": "section_07_experiments.tex",
        "caption_terms": ["source-mismatch", "not protected success"],
    },
    "figure_5_ownership_scenario_heatmap.png": {
        "min_width": 1400,
        "min_height": 900,
        "max_aspect": 1.8,
        "min_nonwhite_ratio": 0.03,
        "tex_file": "section_07_experiments.tex",
        "caption_terms": ["trace-bundle", "ownership proof"],
    },
}

INTERNAL_VISUAL_TERMS = [
    "artifact-only placeholder",
    "placeholder draft",
    "visual draft",
    "claim-scope lint",
    "not paper-facing",
]


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_stats(path: Path) -> dict[str, Any]:
    width, height, color_type, decompressed = read_png_payload(path)
    channels = channels_for_color_type(color_type)
    if channels not in {1, 3, 4}:
        raise ValueError(f"unsupported PNG color type {color_type} in {path}")
    bytes_per_pixel = channels
    row_len = width * channels
    step_y = max(1, height // 200)
    step_x = max(1, width // 200)
    idx = 0
    prev = bytearray(row_len)
    visible_count = 0
    nonwhite_count = 0
    unique_sampled: set[tuple[int, int, int]] = set()
    for y in range(height):
        filter_type = decompressed[idx]
        idx += 1
        raw = bytearray(decompressed[idx : idx + row_len])
        idx += row_len
        row = unfilter_scanline(raw, prev, filter_type, bytes_per_pixel)
        for x in range(width):
            offset = x * channels
            if channels == 1:
                r = g = b = row[offset]
                a = 255
            elif channels == 3:
                r, g, b = row[offset : offset + 3]
                a = 255
            else:
                r, g, b, a = row[offset : offset + 4]
            if a > 2:
                visible_count += 1
                if r < 247 or g < 247 or b < 247:
                    nonwhite_count += 1
            if y % step_y == 0 and x % step_x == 0:
                unique_sampled.add((int(r), int(g), int(b)))
        prev = row
    nonwhite_ratio = nonwhite_count / max(1, visible_count)
    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "nonwhite_ratio": nonwhite_ratio,
        "unique_sampled_colors": len(unique_sampled),
    }


def read_png_payload(path: Path) -> tuple[int, int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG file: {path}")
    offset = 8
    width = height = color_type = None
    idat = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, png_filter, interlace = struct.unpack(">IIBBBBB", chunk)
            if bit_depth != 8 or compression != 0 or png_filter != 0 or interlace != 0:
                raise ValueError(f"unsupported PNG encoding in {path}")
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or color_type is None:
        raise ValueError(f"missing IHDR in {path}")
    return width, height, color_type, zlib.decompress(bytes(idat))


def channels_for_color_type(color_type: int) -> int:
    if color_type == 0:
        return 1
    if color_type == 2:
        return 3
    if color_type == 6:
        return 4
    raise ValueError(f"unsupported PNG color type {color_type}")


def unfilter_scanline(raw: bytearray, prev: bytearray, filter_type: int, bpp: int) -> bytearray:
    row = bytearray(len(raw))
    for i, value in enumerate(raw):
        left = row[i - bpp] if i >= bpp else 0
        up = prev[i] if prev else 0
        up_left = prev[i - bpp] if prev and i >= bpp else 0
        if filter_type == 0:
            recon = value
        elif filter_type == 1:
            recon = value + left
        elif filter_type == 2:
            recon = value + up
        elif filter_type == 3:
            recon = value + ((left + up) // 2)
        elif filter_type == 4:
            recon = value + paeth(left, up, up_left)
        else:
            raise ValueError(f"unsupported PNG filter type {filter_type}")
        row[i] = recon & 0xFF
    return row


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def render_manifest_hashes(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {Path(row["path"]).name: row["sha256"] for row in manifest.get("figures", [])}


def active_tex_text() -> str:
    paths = [
        MANUSCRIPT / "section_01_introduction.tex",
        MANUSCRIPT / "section_02_related_work.tex",
        MANUSCRIPT / "section_03_problem_setup.tex",
        MANUSCRIPT / "section_04_tokenizer_alignment.tex",
        MANUSCRIPT / "section_05_bucket_level_injection.tex",
        MANUSCRIPT / "section_06_deterministic_verification.tex",
        MANUSCRIPT / "section_07_experiments.tex",
        MANUSCRIPT / "section_08_discussion_limitations.tex",
        MANUSCRIPT / "section_09_conclusion.tex",
        MANUSCRIPT / "appendix" / "extended_related_work.tex",
        MANUSCRIPT / "appendix" / "asset_licenses.tex",
    ]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths if path.is_file())


def audit_figure(name: str, spec: dict[str, Any], manifest_hashes: dict[str, str], tex_text: str) -> dict[str, Any]:
    manuscript_path = FIGURE_DIR / name
    rendered_path = RENDER_DIR / name
    failures: list[str] = []
    exists = manuscript_path.is_file()
    rendered_exists = rendered_path.is_file()
    stats: dict[str, Any] = {}
    sha = ""
    rendered_sha = ""
    if not exists:
        failures.append("missing manuscript figure")
    else:
        sha = sha256_file(manuscript_path)
        stats = image_stats(manuscript_path)
        if stats["width"] < spec["min_width"]:
            failures.append("width below threshold")
        if stats["height"] < spec["min_height"]:
            failures.append("height below threshold")
        if stats["aspect_ratio"] > spec["max_aspect"]:
            failures.append("aspect ratio above threshold")
        if stats["nonwhite_ratio"] < spec["min_nonwhite_ratio"]:
            failures.append("image appears too blank")
        if stats["unique_sampled_colors"] < 12:
            failures.append("too few sampled colors")
    if not rendered_exists:
        failures.append("missing rendered result figure")
    else:
        rendered_sha = sha256_file(rendered_path)
        if exists and rendered_sha != sha:
            failures.append("manuscript figure differs from rendered result copy")
    if manifest_hashes.get(name) != rendered_sha:
        failures.append("render manifest hash mismatch")
    tex_path = MANUSCRIPT / spec["tex_file"]
    tex = tex_path.read_text(encoding="utf-8") if tex_path.is_file() else ""
    if f"figures/{name}" not in tex:
        failures.append("figure not referenced from expected tex file")
    for term in spec["caption_terms"]:
        if term not in tex:
            failures.append(f"missing caption/scope term: {term}")
    lower_tex = tex_text.lower()
    for term in INTERNAL_VISUAL_TERMS:
        if term in lower_tex:
            failures.append(f"internal visual term remains in active tex: {term}")
    return {
        "figure": name,
        "status": "PASS" if not failures else "FAIL",
        "failures": "; ".join(failures),
        "manuscript_path": repo_rel(manuscript_path),
        "rendered_path": repo_rel(rendered_path),
        "bytes": manuscript_path.stat().st_size if exists else "",
        "sha256": sha,
        "rendered_sha256": rendered_sha,
        **stats,
    }


def data_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    trace = read_csv(DATA_DIR / "trace_bound_accepts.csv")
    public = read_csv(DATA_DIR / "public_text_verifier_baselines.csv")
    attacks = read_csv(DATA_DIR / "attack_ladder_summary.csv")
    ownership = read_csv(DATA_DIR / "ownership_scenario_heatmap.csv")
    checks.append(
        check_row(
            "figure_3_trace_bound_counts",
            any(r["model_family"] == "qwen" and r["protected_accepts"] == "94" and r["protected_blocks"] == "96" for r in trace)
            and any(r["model_family"] == "llama" and r["protected_accepts"] == "96" and r["protected_blocks"] == "96" for r in trace),
            "trace_bound_accepts.csv contains Qwen 94/96 and Llama 96/96",
        )
    )
    checks.append(
        check_row(
            "figure_3_public_codeword_zero",
            sum(int(r["codeword_recovered_blocks"]) for r in public) == 0,
            "public_text_verifier_baselines.csv has zero recovered codeword blocks",
        )
    )
    guided_top100 = [
        r
        for r in attacks
        if r["attack_family"] == "public_predicate_guided_rewrite_graft" and r["budget_label"] == "100"
    ]
    checks.append(
        check_row(
            "figure_4_guided_attack_top100",
            len(guided_top100) >= 3 and all(int(r["accepted_count"]) == 100 for r in guided_top100),
            "guided rewrite/graft top-100 source-mismatch accepts are 100/100 for all plotted groups",
        )
    )
    checks.append(
        check_row(
            "figure_5_ownership_matrix_shape",
            len(ownership) == 63
            and len({r["scenario_id"] for r in ownership}) == 7
            and len({r["method_family"] for r in ownership}) == 9,
            "ownership_scenario_heatmap.csv contains 7 scenarios x 9 method families",
        )
    )
    checks.append(
        check_row(
            "figure_5_supported_public_text_zero",
            not any(
                r["method_family"] == "public_deterministic_text_predicate"
                and r["current_assessment"].startswith("SUPPORTED")
                for r in ownership
            ),
            "ownership matrix has zero supported public final-text rows",
        )
    )
    return checks


def check_row(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any], figure_rows: list[dict[str, Any]], data_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# VSG Manuscript Figure Quality Audit",
        "",
        "This artifact-only audit checks active manuscript PNG figures for",
        "dimensions, nonblank rendered content, render-manifest consistency,",
        "LaTeX references, scope terms, and core data traceability.",
        "",
        f"Status: `{summary['status']}`",
        f"Figures checked: `{summary['figure_count']}`",
        f"Failed figures: `{summary['failed_figure_count']}`",
        f"Data checks: `{summary['data_check_count']}`",
        f"Failed data checks: `{summary['failed_data_check_count']}`",
        "",
        "## Figure Checks",
        "",
        "| Figure | Status | Size | Nonwhite ratio | Failures |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in figure_rows:
        size = f"{row.get('width', '')}x{row.get('height', '')}"
        lines.append(
            f"| `{row['figure']}` | `{row['status']}` | {size} | {row.get('nonwhite_ratio', ''):.4f} | {row.get('failures', '')} |"
        )
    lines.extend(["", "## Data Traceability Checks", "", "| Check | Status | Detail |", "| --- | --- | --- |"])
    for row in data_rows:
        lines.append(f"| `{row['check']}` | `{row['status']}` | {row['detail']} |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- This audit does not render new figures.",
            "- This audit does not start Slurm, generation, model scoring, or training.",
            "- This audit does not create public text-only verification success or ownership-proof claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hashes = render_manifest_hashes(RENDER_MANIFEST)
    tex_text = active_tex_text()
    figure_rows = [audit_figure(name, spec, manifest_hashes, tex_text) for name, spec in EXPECTED_FIGURES.items()]
    data_rows = data_checks()
    failed_figures = [row for row in figure_rows if row["status"] != "PASS"]
    failed_data = [row for row in data_rows if row["status"] != "PASS"]

    figure_csv = output_dir / "figure_quality_checks.csv"
    data_csv = output_dir / "figure_data_traceability_checks.csv"
    summary_json = output_dir / "figure_quality_summary.json"
    report_md = output_dir / "figure_quality_report.md"
    manifest_json = output_dir / "figure_quality_manifest.json"
    write_csv(figure_csv, figure_rows)
    write_csv(data_csv, data_rows)
    summary = {
        "status": "PASS" if not failed_figures and not failed_data else "FAIL",
        "schema_name": "verification_substrate_gap_manuscript_figure_quality_audit_v1",
        "output_dir": repo_rel(output_dir),
        "figure_count": len(figure_rows),
        "failed_figure_count": len(failed_figures),
        "data_check_count": len(data_rows),
        "failed_data_check_count": len(failed_data),
        "failed_figures": [row["figure"] for row in failed_figures],
        "failed_data_checks": [row["check"] for row in failed_data],
        "render_manifest": repo_rel(RENDER_MANIFEST),
        "new_slurm_started": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "public_text_only_verification_claimed": False,
        "ownership_proof_claimed": False,
    }
    write_json(summary_json, summary)
    write_report(report_md, summary, figure_rows, data_rows)
    manifest = {
        "status": summary["status"],
        "schema_name": "verification_substrate_gap_manuscript_figure_quality_manifest_v1",
        "files": [
            {"path": repo_rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in [figure_csv, data_csv, summary_json, report_md]
        ],
    }
    write_json(manifest_json, manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = build(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
