#!/usr/bin/env python3
"""Render publication-readable VSG manuscript figures from frozen CSV data.

This script is artifact-only.  It reads existing figure-data CSV files and
writes static PNG figures plus a manifest.  It does not run model inference,
generation, Slurm, scoring, or training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STATUS = "PASS_VSG_MANUSCRIPT_FIGURES_RENDERED_ARTIFACT_ONLY_NO_NEW_CLAIMS"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig1(output_dir: Path) -> Path:
    substrates = ["Metadata", "Protocol", "Proof", "Trace", "Model state", "Final text"]
    props = ["Public", "Det.", "Portable", "Natural", "Non-coop", "Spoof-hard"]
    values = np.array(
        [
            [0.5, 1, 0.25, 0, 0.5, 0.5],
            [0.5, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 0.5, 0, 0, 0, 0.5],
            [1, 0, 1, 1, 1, 0],
        ]
    )
    labels = np.array(
        [
            ["cond.", "yes", "fragile", "no", "cond.", "cond."],
            ["cond.", "yes", "no", "no", "no", "yes"],
            ["yes", "yes", "no", "no", "no", "yes"],
            ["no", "yes", "no", "no", "no", "yes"],
            ["no", "cond.", "no", "no", "no", "cond."],
            ["yes", "not shown", "yes", "yes", "yes", "no"],
        ]
    )
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    im = ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(props)), props, fontsize=10)
    ax.set_yticks(range(len(substrates)), substrates, fontsize=10)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Verification substrate map", fontsize=14, fontweight="bold", pad=12)
    ax.text(
        0,
        -0.9,
        "Scope: taxonomy only; trace-bound diagnostics are provider-side, not copied-text verification.",
        fontsize=9,
        transform=ax.transData,
    )
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, ticks=[0, 0.5, 1], label="property support")
    path = output_dir / "figure_1_verification_substrate_map.png"
    save(fig, path)
    return path


def fig2(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.axis("off")
    ax.set_title("First-divergence token event diagnostic", fontsize=14, fontweight="bold", pad=10)
    ax.text(0.03, 0.88, "Any later text difference has an earliest tokenizer-native branch.", fontsize=10)
    nodes = {
        "prefix": (0.18, 0.50),
        "a": (0.42, 0.72),
        "b": (0.42, 0.28),
        "ta": (0.72, 0.72),
        "tb": (0.72, 0.28),
    }
    ax.annotate("", xy=nodes["a"], xytext=nodes["prefix"], arrowprops={"arrowstyle": "->", "lw": 2, "color": "#2563eb"})
    ax.annotate("", xy=nodes["b"], xytext=nodes["prefix"], arrowprops={"arrowstyle": "->", "lw": 2, "color": "#dc2626"})
    for key, text, color in [
        ("prefix", "shared\nprefix", "#dbeafe"),
        ("a", "next token\nbranch A", "#dbeafe"),
        ("b", "next token\nbranch B", "#fee2e2"),
        ("ta", "later phrase / style /\nsemantic continuation", "#f8fafc"),
        ("tb", "later phrase / style /\nsemantic continuation", "#f8fafc"),
    ]:
        x, y = nodes[key]
        box = dict(boxstyle="round,pad=0.5", fc=color, ec="#334155", lw=1.2)
        ax.text(x, y, text, ha="center", va="center", fontsize=10, fontweight="bold", bbox=box)
    ax.annotate("", xy=nodes["ta"], xytext=nodes["a"], arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#334155"})
    ax.annotate("", xy=nodes["tb"], xytext=nodes["b"], arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#334155"})
    ax.text(0.03, 0.08, "Scope: event access is an upper-bound diagnostic, not a public final-text verifier.", fontsize=9, fontweight="bold")
    path = output_dir / "figure_2_first_divergence_diagnostic.png"
    save(fig, path)
    return path


def fig3(data_dir: Path, output_dir: Path) -> Path:
    trace = read_csv(data_dir / "trace_bound_accepts.csv")
    public_rows = read_csv(data_dir / "public_text_verifier_baselines.csv")
    trace_labels = [r["model_family"].capitalize() for r in trace]
    trace_vals = [int(r["protected_accepts"]) / int(r["protected_blocks"]) for r in trace]
    trace_text = [f"{r['protected_accepts']}/{r['protected_blocks']}" for r in trace]
    public_zero = sum(int(r["codeword_recovered_blocks"]) for r in public_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    bars = ax1.bar(trace_labels, trace_vals, color=["#2563eb", "#16a34a"])
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("protected trace-bound accept rate")
    ax1.set_title("Trace-bound controllability")
    for bar, label in zip(bars, trace_text):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, label, ha="center", fontweight="bold")
    ax1.text(0.5, -0.22, "controls: raw/task-only/wrong-key/wrong-payload = 0/96", transform=ax1.transAxes, ha="center", fontsize=9)

    ax2.bar(["Public\nfinal text"], [public_zero], color="#dc2626")
    ax2.set_ylim(0, 1)
    ax2.set_title("Codeword recovery")
    ax2.set_ylabel("recovered blocks")
    ax2.text(0, 0.05, "0 blocks", ha="center", fontweight="bold")
    fig.suptitle("Controllability versus public observability", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "Scope: trace-bound provider-side diagnostics; no public text-only verification success claim.", ha="center", fontsize=9)
    path = output_dir / "figure_3_controllability_vs_observability.png"
    save(fig, path)
    return path


def fig4(data_dir: Path, output_dir: Path) -> Path:
    rows = read_csv(data_dir / "attack_ladder_summary.csv")
    sources = [("qwen", "raw"), ("qwen", "task_only"), ("llama", "raw")]
    families = [
        ("rejection_sampling", "Rejection"),
        ("distillation_lite", "Surrogate rank"),
        ("public_predicate_guided_rewrite_graft", "Guided rewrite"),
    ]
    matrix = []
    for model, arm in sources:
        row_vals = []
        for family, _ in families:
            matches = [
                r
                for r in rows
                if r["model_family"] == model
                and r["candidate_arm"] == arm
                and r["attack_family"] == family
                and r["budget_label"] == "100"
            ]
            row_vals.append(int(matches[0]["accepted_count"]) if matches else 0)
        matrix.append(row_vals)

    x = np.arange(len(sources))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    colors = ["#94a3b8", "#7c3aed", "#dc2626"]
    for i, (_, label) in enumerate(families):
        vals = [row[i] for row in matrix]
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=colors[i])
        for j, val in enumerate(vals):
            ax.text(x[j] + (i - 1) * width, val + 2, f"{val}/100", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, ["Qwen raw", "Qwen task-only", "Llama raw"])
    ax.set_ylim(0, 112)
    ax.set_ylabel("accepted source-mismatch rows in top 100")
    ax.set_title("Public-predicate attack ladder", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.5, -0.20, "Scope: accepted source-mismatch rows are spoofing evidence, not protected success.", transform=ax.transAxes, ha="center", fontsize=9)
    path = output_dir / "figure_4_public_predicate_attack_ladder.png"
    save(fig, path)
    return path


def fig5(data_dir: Path, output_dir: Path) -> Path:
    rows = read_csv(data_dir / "ownership_scenario_heatmap.csv")
    scenarios = []
    methods = []
    for row in rows:
        if row["scenario_id"] not in scenarios:
            scenarios.append(row["scenario_id"])
        if row["method_family"] not in methods:
            methods.append(row["method_family"])
    methods_short = {
        "statistical_watermark": "stat.\nwm",
        "publicly_detectable_watermark": "public\nwm",
        "tee_or_2pc_public_watermark_protocol": "TEE/\n2PC",
        "zk_inference_proof": "ZK\nproof",
        "signed_metadata": "signed\nmeta",
        "model_fingerprint_or_trigger": "model\nfp",
        "provider_side_trace": "provider\ntrace",
        "first_divergence_diagnostic": "first\ndiv.",
        "public_deterministic_text_predicate": "public\ntext",
    }
    scenario_short = {
        "S1_cooperative_provider_with_signed_metadata": "S1 signed metadata",
        "S2_cooperative_provider_with_trace_bundle": "S2 trace bundle",
        "S3_non_cooperative_api_only_suspect_model": "S3 API-only suspect",
        "S4_copy_paste_text_without_metadata": "S4 copied text",
        "S5_post_processed_or_rewritten_output": "S5 rewritten text",
        "S6_wrapper_model_proxying_another_model": "S6 wrapper/proxy",
        "S7_distilled_or_fine_tuned_descendant_model": "S7 distilled/fine-tuned",
    }
    score = {
        "SUPPORTED_TRACE_BOUND_DIAGNOSTIC": 2,
        "UNTESTED_AND_AT_RISK_UNDER_CURRENT_SPOOFING_RECORD": 1,
        "UNTESTED_NEEDS_SIGNATURE_PROTOCOL": 1,
        "UNTESTED_API_DEPENDENT": 1,
        "FAILS_NO_PROTOCOL_SUBSTRATE": 0,
        "FAILS_NO_TRACE_SUBSTRATE": 0,
        "FAILS_PUBLIC_PREDICATE_SPOOFABLE": 0,
        "FAILS_METADATA_STRIPPED": 0,
        "FAILS_NO_API_OR_MODEL_ACCESS": 0,
        "FAILS_NO_FINAL_TEXT_SUBSTRATE": 0,
    }
    by_key = {(r["scenario_id"], r["method_family"]): r for r in rows}
    mat = np.array([[score.get(by_key[(s, m)]["current_assessment"], 0) for m in methods] for s in scenarios])
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    cmap = plt.matplotlib.colors.ListedColormap(["#fee2e2", "#fde68a", "#bbf7d0"])
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(range(len(methods)), [methods_short.get(m, m[:8]) for m in methods], fontsize=8)
    ax.set_yticks(range(len(scenarios)), [scenario_short.get(s, s[:14]) for s in scenarios], fontsize=8)
    ax.set_title("Ownership scenario stress test", fontsize=14, fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            label = "support" if mat[i, j] == 2 else ("untested" if mat[i, j] == 1 else "fail")
            ax.text(j, i, label, ha="center", va="center", fontsize=6.5, fontweight="bold")
    ax.text(0.5, -0.18, "Scope: current evidence supports cooperative trace-bundle diagnostics only.", transform=ax.transAxes, ha="center", fontsize=9)
    path = output_dir / "figure_5_ownership_scenario_heatmap.png"
    save(fig, path)
    return path


def render(data_dir: Path, manuscript_figures: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = [
        fig1(manuscript_figures),
        fig2(manuscript_figures),
        fig3(data_dir, manuscript_figures),
        fig4(data_dir, manuscript_figures),
        fig5(data_dir, manuscript_figures),
    ]
    copied = []
    for path in rendered:
        target = output_dir / path.name
        target.write_bytes(path.read_bytes())
        copied.append(target)
    rows = [
        {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in copied
    ]
    manifest = {
        "schema_name": "vsg_manuscript_figure_render_manifest_v1",
        "status": STATUS,
        "data_dir": str(data_dir),
        "manuscript_figures": str(manuscript_figures),
        "output_dir": str(output_dir),
        "figure_count": len(rows),
        "figures": rows,
        "new_compute_started": False,
        "slurm_submitted": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
        "public_text_only_verification_claim_allowed": False,
    }
    (output_dir / "manuscript_figure_render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("results/verification_substrate_gap/paper_figure_data_20260530"))
    parser.add_argument("--manuscript-figures", type=Path, default=Path("manuscripts/69db2644566dcc36c9da320e/figures"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/verification_substrate_gap/paper_manuscript_figures_20260531"))
    args = parser.parse_args()
    print(json.dumps(render(args.data_dir, args.manuscript_figures, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
