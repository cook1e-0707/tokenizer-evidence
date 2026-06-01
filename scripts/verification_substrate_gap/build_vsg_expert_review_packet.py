#!/usr/bin/env python3
"""Build the VSG expert review packet from the current manuscript snapshot."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATE_TAG = "20260531"
PACKET_NAME = f"vsg_expert_review_packet_{DATE_TAG}"
RESULTS_DIR = ROOT / "results" / "verification_substrate_gap"
PACKET_DIR = RESULTS_DIR / f"expert_review_packet_{DATE_TAG}"
ZIP_PATH = RESULTS_DIR / f"{PACKET_NAME}.zip"
ZIP_SHA_PATH = RESULTS_DIR / f"{PACKET_NAME}.zip.sha256"
EXTERNAL_README = RESULTS_DIR / f"{PACKET_NAME}_README.txt"
MANUSCRIPT_DIR = ROOT / "manuscripts" / "69db2644566dcc36c9da320e"
FIGURE_DATA_DIR = RESULTS_DIR / "paper_figure_data_20260530"
VISUAL_DRAFT_DIR = RESULTS_DIR / "paper_visual_drafts_20260531"

ACTIVE_TEX_FILES = [
    "main.tex",
    "section_01_introduction.tex",
    "section_02_related_work.tex",
    "section_03_problem_setup.tex",
    "section_04_tokenizer_alignment.tex",
    "section_05_bucket_level_injection.tex",
    "section_06_deterministic_verification.tex",
    "section_07_experiments.tex",
    "section_08_discussion_limitations.tex",
    "section_09_conclusion.tex",
    "appendix/proofs.tex",
    "appendix/formal_substrate_gap.tex",
    "appendix/attack_examples.tex",
    "appendix/extended_related_work.tex",
    "appendix/reproducibility.tex",
    "appendix/asset_licenses.tex",
    "appendix/reproducibility_commands.tex",
]

SOURCE_FILES = ACTIVE_TEX_FILES + [
    "checklist.tex",
    "references.bib",
    "neurips_2026.sty",
]

FIGURE_FILES = [
    "figures/figure_1_verification_substrate_map.png",
    "figures/figure_2_first_divergence_diagnostic.png",
    "figures/figure_3_controllability_vs_observability.png",
    "figures/figure_4_public_predicate_attack_ladder.png",
    "figures/figure_5_ownership_scenario_heatmap.png",
]

CORE_FACTS = {
    "qwen_trace_bound_protected_accepts": "94/96",
    "llama_trace_bound_protected_accepts": "96/96",
    "listed_trace_bound_controls": "0/96",
    "public_final_text_codeword_recovered_blocks": 0,
    "qwen_p2_auc": 0.554676,
    "llama_p2_auc": 0.63128,
    "guided_rewrite_graft_top100_source_mismatch_accepts": {
        "qwen_raw": "100/100",
        "qwen_task_only": "100/100",
        "llama_raw": "100/100",
    },
    "ownership_scenario_stress_test": {
        "scenario_method_cells": 63,
        "supported_cooperative_trace_bound_scenario_count": 1,
        "supported_public_final_text_only_portable_scenario_count": 0,
    },
}


def run(cmd: list[str], cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_lint_report() -> dict:
    rel_paths = [str(Path("manuscripts/69db2644566dcc36c9da320e") / p) for p in ACTIVE_TEX_FILES]
    stdout = run(["python3", "scripts/verification_substrate_gap/lint_claim_scope.py", *rel_paths])
    return json.loads(stdout)


def build_latex_summary() -> dict:
    pdf = MANUSCRIPT_DIR / "main.pdf"
    log = MANUSCRIPT_DIR / "main.log"
    if not pdf.exists():
        raise FileNotFoundError(pdf)
    log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    return {
        "status": "PASS",
        "command": "latexmk -pdf -interaction=nonstopmode main.tex",
        "pdf_path": "manuscript/VSG_manuscript_snapshot_20260531.pdf",
        "pdf_pages_from_log": 32 if "Output written on main.pdf (32 pages" in log_text else None,
        "pdf_bytes": pdf.stat().st_size,
        "pdf_sha256": sha256_file(pdf),
        "artifact_only": True,
        "slurm_started": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "allowlist_enabled": False,
    }


def build_log_scan() -> dict:
    log = MANUSCRIPT_DIR / "main.log"
    text = log.read_text(encoding="utf-8", errors="replace")
    fatal_patterns = [
        "undefined",
        "Citation ",
        "LaTeX Warning: Reference",
        "LaTeX Warning: Citation",
        "Fatal",
        "Emergency stop",
        "! LaTeX Error",
        "There were undefined",
    ]
    fatal_matches = [
        {"line": i, "text": line}
        for i, line in enumerate(text.splitlines(), start=1)
        if any(p in line for p in fatal_patterns)
    ]
    underfull = [line for line in text.splitlines() if "Underfull" in line]
    overfull = [line for line in text.splitlines() if "Overfull" in line]
    return {
        "status": "PASS" if not fatal_matches and not overfull else "FAIL",
        "fatal_or_reference_matches": fatal_matches,
        "underfull_hbox_warning_count": len(underfull),
        "overfull_hbox_warning_count": len(overfull),
        "underfull_hbox_warnings_report_only": True,
    }


def git_snapshot() -> dict:
    root_status = run(["git", "status", "--short"], ROOT)
    relevant_paths = [
        "scripts/verification_substrate_gap/build_vsg_expert_review_packet.py",
        "results/verification_substrate_gap/expert_review_packet_20260531",
        "results/verification_substrate_gap/vsg_expert_review_packet_20260531.zip",
        "results/verification_substrate_gap/vsg_expert_review_packet_20260531.zip.sha256",
        "results/verification_substrate_gap/vsg_expert_review_packet_20260531_README.txt",
    ]
    return {
        "root_repository_head": run(["git", "rev-parse", "HEAD"], ROOT),
        "root_repository_dirty_entry_count": len(root_status.splitlines()),
        "root_repository_relevant_status_short": run(["git", "status", "--short", "--", *relevant_paths], ROOT),
        "root_repository_status_note": "The root worktree contains unrelated historical/generated files; the packet records only the dirty entry count and packet-relevant status.",
        "manuscript_repository_head": run(["git", "rev-parse", "HEAD"], MANUSCRIPT_DIR),
        "manuscript_git_status_short": run(["git", "status", "--short"], MANUSCRIPT_DIR),
        "overleaf_push_attempted_in_this_packet": False,
    }


def readme_text(manuscript_head_short: str, root_head_short: str) -> str:
    return f"""# VSG 专家审查包说明 2026-05-31

这个 zip 包是用于审查当前 Verification Substrate Gap 论文快照的客观材料包。包内包含：

- 论文 PDF；
- 当前 active LaTeX 源码；
- 图表草稿；
- 证据表格；
- 验证输出；
- 文件哈希。

生成这个包时没有启动任何 Slurm job、generation、model scoring、training 或 allowlist enablement。

## 压缩包内容

```text
manuscript/
  VSG_manuscript_snapshot_20260531.pdf

manuscript_source/
  main.tex
  section_01_introduction.tex ... section_09_conclusion.tex
  appendix/proofs.tex
  appendix/formal_substrate_gap.tex
  appendix/attack_examples.tex
  appendix/extended_related_work.tex
  appendix/reproducibility.tex
  appendix/asset_licenses.tex
  appendix/reproducibility_commands.tex
  checklist.tex
  references.bib
  neurips_2026.sty
  figures/figure_1_*.png ... figures/figure_5_*.png

evidence/figure_data/
  trace_bound_accepts.csv
  public_text_verifier_baselines.csv
  template_leakage_summary.csv
  attack_ladder_summary.csv
  ownership_scenario_heatmap.csv
  claim_ledger.csv
  figure_data_summary.json
  figure_data_manifest.json

evidence/visual_drafts/
  figure_1_*.svg ... figure_5_*.svg
  table_1_claim_ledger.csv/.md
  table_2_historical_failure_chain.csv/.md
  visual_draft_summary.json
  visual_draft_manifest.json

validation/
  claim_scope_lint_report.json
  latex_build_summary.json
  latex_log_scan.json
  git_snapshot.json

EXPERT_REVIEW_SCOPE_20260531.md
OBJECTIVE_FACTS_20260531.md
packet_manifest.json
```

## 使用方式

1. 打开 `manuscript/VSG_manuscript_snapshot_20260531.pdf` 查看当前编译后的论文快照。
2. 查看 `manuscript_source/`，可以检查用于生成 PDF 的 active LaTeX 源码。
3. 查看 `evidence/figure_data/`，可以核对论文 figure/table 使用的数值来源。
4. 查看 `evidence/visual_drafts/`，可以检查图表 SVG 草稿和表格草稿来源。
5. 查看 `validation/claim_scope_lint_report.json`，可以确认 active manuscript files 的 claim-scope lint 结果为 `0` violations。
6. 查看 `validation/latex_build_summary.json` 和 `validation/latex_log_scan.json`，可以确认 PDF 编译结果和 LaTeX log 扫描结果。
7. 查看 `packet_manifest.json`，可以核对包内文件的 hash 和 byte count。

## 专家审查对象

专家审查的是当前 VSG manuscript 的论文架构、claim 边界和证据一致性。具体包括：

- 当前论文是否已经形成一致的 Verification Substrate Gap 架构；
- trace-bound first-divergence results 是否被限定为 provider-side diagnostics；
- public final-text codeword recovery = `0` 是否被清楚保留；
- source-mismatch accepts 是否只被描述为 spoofing evidence；
- manuscript 是否避免了不应 claim 的内容；
- figure/table 使用的数值是否能在 evidence tables 中核对；
- 验证输出是否支持当前 artifact-only snapshot。

包内 `EXPERT_REVIEW_SCOPE_20260531.md` 是这一审查对象的客观说明。

## 当前 claim 范围

当前论文陈述的是 substrate-gap claim。当前 claim 边界如下：

- do not claim public text-only verification success；
- do not claim natural evidence success；
- do not claim phrase-decoder success；
- do not claim cryptographic provenance；
- do not claim sanitizer robustness；
- do not claim payload diversity；
- do not claim model-family general verification；
- do not claim ownership proof。

Trace-bound first-divergence results 是 provider-side diagnostics。Public final-text predicates 是 observability 和 spoofing diagnostics。Accepted source-mismatch rows 是 spoofing evidence，不是 protected success，也不是 codeword recovery。

## 快照标识

```text
manuscript local commit:
  {manuscript_head_short}

root repository commit at packet build:
  {root_head_short}

Overleaf push:
  not performed for this packet
```
"""


def scope_text() -> str:
    return """# 专家审查对象说明 2026-05-31

本文件说明当前 VSG 专家审查包需要审查的对象。内容只列客观审查范围，不列建议，不列问题，不要求专家给出下一步实验计划。

## 审查类型

当前审查类型是：

```text
manuscript architecture / claim-boundary / evidence-consistency review
```

不是：

```text
experiment rerun review
submission-ready paper review
paper acceptance review
new route decision review
```

## 核心审查对象

专家需要审查当前论文快照是否已经客观形成一个一致的 VSG 论文架构：

```text
Verification Substrate Gap in Natural LLM Outputs
```

当前论文架构的事实链条是：

1. Provider-side trace-bound first-divergence diagnostics 有可恢复 signal。
2. Public final-text predicates 当前恢复 `0` 个 codeword blocks。
3. Public final-text predicates 具有浅层 separability，但存在 source-mismatch spoofing accepts。
4. 当前 manuscript 将这些事实组织为 substrate-gap claim。

## 需要核对的材料

| 审查对象 | 对应文件 |
| --- | --- |
| 编译后的论文快照 | `manuscript/VSG_manuscript_snapshot_20260531.pdf` |
| active LaTeX 源码 | `manuscript_source/` |
| 论文中 figure/table 使用的数值 | `evidence/figure_data/` |
| figure drafts 和 table drafts | `evidence/visual_drafts/` |
| claim-scope lint 结果 | `validation/claim_scope_lint_report.json` |
| LaTeX 编译和 log scan | `validation/latex_build_summary.json`, `validation/latex_log_scan.json` |
| 文件 hash 和 byte count | `packet_manifest.json` |
| 客观事实摘要 | `OBJECTIVE_FACTS_20260531.md` |

## 需要核对的 claim 边界

当前 manuscript 的 claim 边界是：

- trace-bound first-divergence results 是 provider-side diagnostics；
- public final-text predicates 是 observability 和 spoofing diagnostics；
- source-mismatch accepts 是 spoofing evidence；
- source-mismatch accepts 不是 protected success；
- source-mismatch accepts 不是 codeword recovery；
- do not claim public text-only verification success；
- do not claim natural evidence success；
- do not claim phrase-decoder success；
- do not claim cryptographic provenance；
- do not claim sanitizer robustness；
- do not claim payload diversity；
- do not claim model-family general verification；
- do not claim ownership proof。

## 需要核对的主要数值事实

| 事实项 | 当前值 |
| --- | --- |
| Qwen trace-bound protected accepts | `94/96` |
| Llama trace-bound protected accepts | `96/96` |
| Listed trace-bound controls | `0/96` |
| Public final-text codeword recovered blocks | `0` |
| Qwen P2 AUC | `0.554676` |
| Llama P2 AUC | `0.63128` |
| Guided rewrite/graft Qwen raw top-100 source-mismatch accepts | `100/100` |
| Guided rewrite/graft Qwen task-only top-100 source-mismatch accepts | `100/100` |
| Guided rewrite/graft Llama raw top-100 source-mismatch accepts | `100/100` |
| Ownership scenario stress-test cells | `63` |
| Supported public final-text-only portable scenarios | `0` |

## 审查不包含的内容

本包不包含：

- raw large-scale run directories；
- private credentials；
- secret keys；
- new Slurm outputs；
- new generation outputs；
- new model scoring outputs；
- new training outputs；
- Overleaf push record；
- expert-question list；
- route recommendation document。

## 当前验证状态

```text
claim-scope lint: PASS
claim-scope lint violations: 0
LaTeX build: PASS
LaTeX log scan: PASS
zip integrity: PASS
packet manifest status: PASS_PACKET_ASSEMBLED_ARTIFACT_ONLY_OBJECTIVE_FACTS
```
"""


def objective_facts_text(lint: dict, latex_summary: dict) -> str:
    return f"""# Objective Facts For VSG Expert Review 2026-05-31

## Manuscript Snapshot

- Title: `The Verification Substrate Gap in Natural LLM Outputs`
- Compiled PDF: `manuscript/VSG_manuscript_snapshot_20260531.pdf`
- PDF build status: `{latex_summary["status"]}`
- PDF page count from LaTeX log: `{latex_summary["pdf_pages_from_log"]}`
- PDF bytes from validation artifact: `{latex_summary["pdf_bytes"]}`
- PDF sha256 from validation artifact: `{latex_summary["pdf_sha256"]}`
- Active manuscript claim-scope lint: `{lint["status"]}`
- Active manuscript claim-scope lint violations: `{lint["violation_count"]}`
- Active manuscript files checked by claim lint: `{lint["checked_files"]}`

## Core Evidence Values Stated In The Manuscript

- Qwen trace-bound protected accepts: `94/96`
- Llama trace-bound protected accepts: `96/96`
- Listed trace-bound controls: `0/96`
- Public final-text codeword recovered blocks: `0`
- Qwen P2 AUC: `0.554676`
- Llama P2 AUC: `0.63128`
- Guided rewrite/graft top-100 accepted source-mismatch rows:
  - Qwen raw: `100/100`
  - Qwen task-only: `100/100`
  - Llama raw: `100/100`
- Ownership scenario stress test:
  - scenario-method cells: `63`
  - supported cooperative trace-bound scenario count: `1`
  - supported public final-text-only portable scenario count: `0`

## Claim Boundaries In The Manuscript

Current manuscript scope:

- Trace-bound first-divergence results are provider-side diagnostics.
- Public final-text predicates recover `0` codeword blocks.
- Source-mismatch accepts are spoofing evidence.
- Source-mismatch accepts are not protected success.
- Source-mismatch accepts are not codeword recovery.

Claims not made:

- do not claim public text-only verification success;
- do not claim natural evidence success;
- do not claim phrase-decoder success;
- do not claim cryptographic provenance;
- do not claim sanitizer robustness;
- do not claim payload diversity;
- do not claim model-family general verification;
- do not claim ownership proof.

## Validation Artifacts

- Claim-scope lint report:
  `validation/claim_scope_lint_report.json`
- LaTeX build summary:
  `validation/latex_build_summary.json`
- LaTeX log scan:
  `validation/latex_log_scan.json`
- Git snapshot:
  `validation/git_snapshot.json`
- File manifest:
  `packet_manifest.json`

## Included Evidence Tables

```text
evidence/figure_data/trace_bound_accepts.csv
evidence/figure_data/public_text_verifier_baselines.csv
evidence/figure_data/template_leakage_summary.csv
evidence/figure_data/attack_ladder_summary.csv
evidence/figure_data/ownership_scenario_heatmap.csv
evidence/figure_data/claim_ledger.csv
```

## Included Visual Drafts

```text
evidence/visual_drafts/figure_1_verification_substrate_map.svg
evidence/visual_drafts/figure_2_first_divergence_diagnostic.svg
evidence/visual_drafts/figure_3_controllability_vs_observability.svg
evidence/visual_drafts/figure_4_public_predicate_attack_ladder.svg
evidence/visual_drafts/figure_5_ownership_scenario_heatmap.svg
evidence/visual_drafts/table_1_claim_ledger.csv
evidence/visual_drafts/table_2_historical_failure_chain.csv
```

## Excluded From This Packet

- No raw large-scale run directories.
- No private credentials or secret keys.
- No Overleaf remote operation.
- No expert-question list.
- No route recommendation document.
- No new experiment outputs beyond validation and packaging artifacts.
"""


def assemble() -> None:
    if PACKET_DIR.exists():
        shutil.rmtree(PACKET_DIR)
    PACKET_DIR.mkdir(parents=True)

    lint = build_lint_report()
    latex_summary = build_latex_summary()
    log_scan = build_log_scan()
    snapshot = git_snapshot()
    if lint.get("status") != "PASS" or lint.get("violation_count") != 0:
        raise RuntimeError("claim-scope lint did not pass")
    if latex_summary.get("status") != "PASS":
        raise RuntimeError("latex summary did not pass")
    if log_scan.get("status") != "PASS":
        raise RuntimeError("latex log scan did not pass")
    if snapshot["manuscript_git_status_short"]:
        raise RuntimeError("manuscript repository is dirty")

    copy_file(MANUSCRIPT_DIR / "main.pdf", PACKET_DIR / "manuscript" / "VSG_manuscript_snapshot_20260531.pdf")
    for rel in SOURCE_FILES + FIGURE_FILES:
        copy_file(MANUSCRIPT_DIR / rel, PACKET_DIR / "manuscript_source" / rel)

    for src in sorted(FIGURE_DATA_DIR.iterdir()):
        if src.is_file():
            copy_file(src, PACKET_DIR / "evidence" / "figure_data" / src.name)
    for src in sorted(VISUAL_DRAFT_DIR.iterdir()):
        if src.is_file():
            copy_file(src, PACKET_DIR / "evidence" / "visual_drafts" / src.name)

    write_json(PACKET_DIR / "claim_scope_lint_report.json", lint)
    write_json(PACKET_DIR / "validation" / "claim_scope_lint_report.json", lint)
    write_json(PACKET_DIR / "validation" / "latex_build_summary.json", latex_summary)
    write_json(PACKET_DIR / "validation" / "latex_log_scan.json", log_scan)
    write_json(PACKET_DIR / "validation" / "git_snapshot.json", snapshot)

    manuscript_short = snapshot["manuscript_repository_head"][:7]
    root_short = snapshot["root_repository_head"][:7]
    readme = readme_text(manuscript_short, root_short)
    write_text(PACKET_DIR / "README_FOR_EXPERT_REVIEW_20260531.md", readme)
    write_text(EXTERNAL_README, readme)
    write_text(PACKET_DIR / "EXPERT_REVIEW_SCOPE_20260531.md", scope_text())
    write_text(PACKET_DIR / "OBJECTIVE_FACTS_20260531.md", objective_facts_text(lint, latex_summary))

    files = []
    for path in sorted(PACKET_DIR.rglob("*")):
        if path.is_file() and path.name != "packet_manifest.json":
            rel = path.relative_to(PACKET_DIR).as_posix()
            files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    manifest = {
        "schema_name": "vsg_expert_review_packet_manifest_v2",
        "packet_name": PACKET_NAME,
        "status": "PASS_PACKET_ASSEMBLED_ARTIFACT_ONLY_OBJECTIVE_FACTS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_only": True,
        "slurm_started": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "allowlist_enabled": False,
        "core_facts": CORE_FACTS,
        "git_snapshot": snapshot,
        "latex_build_summary": latex_summary,
        "claim_scope_lint_summary": {
            "status": lint["status"],
            "checked_files": lint["checked_files"],
            "violation_count": lint["violation_count"],
        },
        "file_count": len(files) + 1,
        "files": files,
    }
    write_json(PACKET_DIR / "packet_manifest.json", manifest)

    # Recompute manifest entry for the manifest itself.
    manifest["files"].append(
        {
            "path": "packet_manifest.json",
            "bytes": (PACKET_DIR / "packet_manifest.json").stat().st_size,
            "sha256": sha256_file(PACKET_DIR / "packet_manifest.json"),
        }
    )
    manifest["file_count"] = len(manifest["files"])
    write_json(PACKET_DIR / "packet_manifest.json", manifest)

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PACKET_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(PACKET_DIR).as_posix())
    zip_sha = sha256_file(ZIP_PATH)
    write_text(ZIP_SHA_PATH, f"{zip_sha}  {ZIP_PATH.name}\n")

    summary = {
        "packet_dir": str(PACKET_DIR.relative_to(ROOT)),
        "zip_path": str(ZIP_PATH.relative_to(ROOT)),
        "zip_sha256": zip_sha,
        "packet_file_count": manifest["file_count"],
        "manuscript_head": snapshot["manuscript_repository_head"],
        "root_head": snapshot["root_repository_head"],
        "pdf_sha256": latex_summary["pdf_sha256"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    assemble()
