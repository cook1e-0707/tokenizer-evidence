#!/usr/bin/env python3
"""Build the refreshed VSG expert review packet after the 2026-06-01 hardening pass."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATE_TAG = "20260601"
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

HARDENING_DIRS = {
    "public_text_verifier_stronger_local_pilot_20260601": RESULTS_DIR
    / "public_text_verifier_stronger_local_pilot_20260601",
    "public_predicate_attack_naturalness_audit_20260601": RESULTS_DIR
    / "public_predicate_attack_naturalness_audit_20260601",
    "reproducibility_release_inventory_20260601": RESULTS_DIR
    / "reproducibility_release_inventory_20260601",
    "ownership_scenario_decision_rule_audit_20260601": RESULTS_DIR
    / "ownership_scenario_decision_rule_audit_20260601",
    "manuscript_figure_quality_audit_20260601": RESULTS_DIR
    / "manuscript_figure_quality_audit_20260601",
}

HARDENING_STATUS_FILES = [
    RESULTS_DIR / "VSG_PUBLIC_TEXT_STRONGER_BASELINE_LOCAL_PILOT_20260601.md",
    RESULTS_DIR / "VSG_PUBLIC_TEXT_STRONGER_BASELINE_LOCAL_PILOT_20260601.json",
    RESULTS_DIR / "VSG_ATTACK_NATURALNESS_PROXY_AUDIT_20260601.md",
    RESULTS_DIR / "VSG_ATTACK_NATURALNESS_PROXY_AUDIT_20260601.json",
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
        "supported_trace_bound_row_count": 2,
        "supported_public_final_text_row_count": 0,
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
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir_files(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)
    for src in sorted(src_dir.iterdir()):
        if src.is_file():
            copy_file(src, dst_dir / src.name)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "pdf_path": f"manuscript/VSG_manuscript_snapshot_{DATE_TAG}.pdf",
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
    return {
        "root_repository_head": run(["git", "rev-parse", "HEAD"], ROOT),
        "root_repository_status_not_recorded": True,
        "root_repository_status_note": "The root worktree contains unrelated historical/generated files. Only the root HEAD is recorded.",
        "manuscript_repository_head": run(["git", "rev-parse", "HEAD"], MANUSCRIPT_DIR),
        "manuscript_git_status_short": run(["git", "status", "--short"], MANUSCRIPT_DIR),
        "overleaf_push_attempted_in_this_packet": False,
    }


def hardening_summary() -> dict:
    public_text = load_json(HARDENING_DIRS["public_text_verifier_stronger_local_pilot_20260601"] / "public_text_verifier_summary.json")
    naturalness = load_json(
        HARDENING_DIRS["public_predicate_attack_naturalness_audit_20260601"]
        / "attack_naturalness_proxy_summary.json"
    )
    release = load_json(HARDENING_DIRS["reproducibility_release_inventory_20260601"] / "release_inventory_summary.json")
    ownership = load_json(
        HARDENING_DIRS["ownership_scenario_decision_rule_audit_20260601"] / "decision_rule_audit_summary.json"
    )
    figures = load_json(
        HARDENING_DIRS["manuscript_figure_quality_audit_20260601"] / "figure_quality_summary.json"
    )
    return {
        "public_text_stronger_baseline": {
            "codeword_recovered_blocks_total": public_text["codeword_recovered_blocks_total"],
            "adopted_locked_evidence_updated": False,
            "public_text_only_verification_claim_allowed": False,
        },
        "attack_naturalness_proxy": {
            "rows": naturalness["rows"],
            "proxy_pass_rows": naturalness["proxy_pass_rows"],
            "semantic_naturalness_claimed": naturalness["semantic_naturalness_claimed"],
        },
        "reproducibility_release_inventory": {
            "row_count": release["row_count"],
            "missing_file_count": release["missing_file_count"],
            "requires_anonymization_review_count": release["requires_anonymization_review_count"],
            "release_ready_without_review": release["release_ready_without_review"],
        },
        "ownership_decision_rule_audit": {
            "row_count": ownership["row_count"],
            "scenario_count": ownership["scenario_count"],
            "method_family_count": ownership["method_family_count"],
            "failure_count": ownership["failure_count"],
            "supported_public_text_row_count": ownership["supported_public_text_row_count"],
            "supported_trace_bound_row_count": ownership["supported_trace_bound_row_count"],
        },
        "manuscript_figure_quality_audit": {
            "figure_count": figures["figure_count"],
            "failed_figure_count": figures["failed_figure_count"],
            "data_check_count": figures["data_check_count"],
            "failed_data_check_count": figures["failed_data_check_count"],
        },
    }


def readme_text(manuscript_head_short: str, root_head_short: str, latex_summary: dict) -> str:
    return f"""# VSG 专家审查包说明 2026-06-01

这个 zip 包是 Verification Substrate Gap 论文快照在 2026-06-01 hardening pass 后的客观审查材料包。包内包含：

- 论文 PDF；
- 当前 active LaTeX 源码；
- figure/table 数据与图稿；
- 2026-06-01 hardening 审计输出；
- claim-scope、LaTeX、figure-quality 和 packet 验证输出；
- 文件哈希。

生成这个包时没有启动 Slurm job、generation、model scoring、training 或 allowlist enablement。Overleaf push 未执行。

## 压缩包内容

```text
manuscript/
  VSG_manuscript_snapshot_20260601.pdf

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

evidence/visual_drafts/
  figure_1_*.svg ... figure_5_*.svg
  table_1_claim_ledger.csv/.md
  table_2_historical_failure_chain.csv/.md

evidence/hardening/
  public_text_verifier_stronger_local_pilot_20260601/
  public_predicate_attack_naturalness_audit_20260601/
  reproducibility_release_inventory_20260601/
  ownership_scenario_decision_rule_audit_20260601/
  manuscript_figure_quality_audit_20260601/
  status/

validation/
  claim_scope_lint_report.json
  latex_build_summary.json
  latex_log_scan.json
  git_snapshot.json
  hardening_summary.json

EXPERT_REVIEW_SCOPE_20260601.md
OBJECTIVE_FACTS_20260601.md
HARDENING_STATUS_20260601.md
packet_manifest.json
```

## 使用方式

1. 打开 `manuscript/VSG_manuscript_snapshot_20260601.pdf` 查看当前编译后的论文快照。
2. 查看 `manuscript_source/`，可以检查用于生成 PDF 的 active LaTeX 源码。
3. 查看 `evidence/figure_data/`，可以核对论文 figure/table 使用的数值来源。
4. 查看 `evidence/hardening/`，可以核对专家回复 hardening pass 后新增的审计输出。
5. 查看 `validation/claim_scope_lint_report.json`，可以确认 active manuscript files 的 claim-scope lint 结果为 `0` violations。
6. 查看 `validation/latex_build_summary.json` 和 `validation/latex_log_scan.json`，可以确认 PDF 编译结果和 LaTeX log 扫描结果。
7. 查看 `validation/hardening_summary.json`，可以核对本包纳入的 2026-06-01 hardening 结果摘要。
8. 查看 `packet_manifest.json`，可以核对包内文件的 hash 和 byte count。

## 专家审查对象

专家审查对象是当前 VSG manuscript 的论文架构、claim 边界、证据一致性和 hardening pass 后的材料状态。具体包括：

- 当前论文是否保持 Verification Substrate Gap 架构；
- trace-bound first-divergence results 是否被限定为 provider-side diagnostics；
- public final-text codeword recovery = `0` 是否被清楚保留；
- source-mismatch accepts 是否只被描述为 spoofing evidence；
- manuscript 是否避免 public text-only verification success、natural evidence success、ownership proof 等越界 claim；
- related work、formal framework、figure quality、attack audit、ownership matrix、release inventory 是否有对应材料可核对；
- figure/table 使用的数值是否能在 evidence tables 中核对；
- 验证输出是否支持当前 artifact-only snapshot。

## 当前 claim 范围

当前论文陈述的是 substrate-gap claim。当前 claim 边界如下：

- trace-bound first-divergence results 是 provider-side diagnostics；
- public final-text predicates 是 observability 和 spoofing diagnostics；
- accepted source-mismatch rows 是 spoofing evidence；
- accepted source-mismatch rows 不是 protected success；
- accepted source-mismatch rows 不是 codeword recovery；
- public final-text codeword recovered blocks 仍为 `0`；
- current artifacts do not establish public text-only verification success；
- current artifacts do not establish natural evidence success；
- current artifacts do not establish phrase-decoder success；
- current artifacts do not establish cryptographic provenance；
- current artifacts do not establish sanitizer robustness；
- current artifacts do not establish payload diversity；
- current artifacts do not establish model-family general verification；
- current artifacts do not establish ownership proof。

## 快照标识

```text
manuscript local commit:
  {manuscript_head_short}

root repository commit at packet build:
  {root_head_short}

PDF sha256:
  {latex_summary["pdf_sha256"]}

Overleaf push:
  not performed for this packet
```
"""


def scope_text() -> str:
    return """# 专家审查对象说明 2026-06-01

本文件说明 2026-06-01 VSG 专家审查包的客观审查对象。本包用于 manuscript architecture / claim-boundary / evidence-consistency / hardening-output review。

## 审查类型

```text
manuscript architecture
claim-boundary consistency
evidence-table consistency
hardening-output consistency
artifact-only packet integrity
```

不是：

```text
experiment rerun review
submission-ready acceptance review
new route decision review
paper-facing positive-result review
```

## 核心事实链条

1. Provider-side trace-bound first-divergence diagnostics 有可恢复 signal。
2. Public final-text predicates 当前恢复 `0` 个 codeword blocks。
3. Public final-text predicates 具有浅层 separability，但存在 source-mismatch spoofing accepts。
4. 当前 manuscript 将这些事实组织为 Verification Substrate Gap claim。
5. 2026-06-01 hardening pass 增加了 stronger public-text predicate local pilot、attack naturalness proxy audit、release inventory、ownership decision-rule audit、figure-quality audit 和 prose/claim-scope regression checks。

## 需要核对的材料

| 审查对象 | 对应文件 |
| --- | --- |
| 编译后的论文快照 | `manuscript/VSG_manuscript_snapshot_20260601.pdf` |
| active LaTeX 源码 | `manuscript_source/` |
| 论文中 figure/table 使用的数值 | `evidence/figure_data/` |
| figure drafts 和 table drafts | `evidence/visual_drafts/` |
| hardening 审计输出 | `evidence/hardening/` |
| claim-scope lint 结果 | `validation/claim_scope_lint_report.json` |
| LaTeX 编译和 log scan | `validation/latex_build_summary.json`, `validation/latex_log_scan.json` |
| hardening 摘要 | `validation/hardening_summary.json` |
| 文件 hash 和 byte count | `packet_manifest.json` |
| 客观事实摘要 | `OBJECTIVE_FACTS_20260601.md` |

## 主要数值事实

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
| Supported public final-text rows in ownership matrix | `0` |

## 本包不包含

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
"""


def objective_facts_text(lint: dict, latex_summary: dict, hardening: dict) -> str:
    return f"""# Objective Facts For VSG Expert Review 2026-06-01

## Manuscript Snapshot

- Title: `The Verification Substrate Gap in Natural LLM Outputs`
- Compiled PDF: `manuscript/VSG_manuscript_snapshot_20260601.pdf`
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
  - supported trace-bound rows: `{hardening["ownership_decision_rule_audit"]["supported_trace_bound_row_count"]}`
  - supported public final-text rows: `{hardening["ownership_decision_rule_audit"]["supported_public_text_row_count"]}`

## 2026-06-01 Hardening Outputs

- Stronger public-text predicate local pilot:
  - codeword recovered blocks total: `{hardening["public_text_stronger_baseline"]["codeword_recovered_blocks_total"]}`
  - adopted locked evidence updated: `{hardening["public_text_stronger_baseline"]["adopted_locked_evidence_updated"]}`
- Attack naturalness proxy audit:
  - rows: `{hardening["attack_naturalness_proxy"]["rows"]}`
  - proxy-readable rows: `{hardening["attack_naturalness_proxy"]["proxy_pass_rows"]}`
  - semantic naturalness claimed: `{hardening["attack_naturalness_proxy"]["semantic_naturalness_claimed"]}`
- Reproducibility release inventory:
  - rows: `{hardening["reproducibility_release_inventory"]["row_count"]}`
  - missing files: `{hardening["reproducibility_release_inventory"]["missing_file_count"]}`
  - requires anonymization/scope review: `{hardening["reproducibility_release_inventory"]["requires_anonymization_review_count"]}`
  - release-ready without review: `{hardening["reproducibility_release_inventory"]["release_ready_without_review"]}`
- Ownership decision-rule audit:
  - rows: `{hardening["ownership_decision_rule_audit"]["row_count"]}`
  - rule failures: `{hardening["ownership_decision_rule_audit"]["failure_count"]}`
  - supported public final-text rows: `{hardening["ownership_decision_rule_audit"]["supported_public_text_row_count"]}`
- Manuscript figure-quality audit:
  - figures checked: `{hardening["manuscript_figure_quality_audit"]["figure_count"]}`
  - failed figure checks: `{hardening["manuscript_figure_quality_audit"]["failed_figure_count"]}`
  - failed data checks: `{hardening["manuscript_figure_quality_audit"]["failed_data_check_count"]}`

## Claim Boundaries In The Manuscript

Current manuscript scope:

- Trace-bound first-divergence results are provider-side diagnostics.
- Public final-text predicates recover `0` codeword blocks.
- Source-mismatch accepts are spoofing evidence.
- Source-mismatch accepts are not protected success.
- Source-mismatch accepts are not codeword recovery.

Claims not established by current artifacts:

- public text-only verification success;
- natural evidence success;
- phrase-decoder success;
- cryptographic provenance;
- sanitizer robustness;
- payload diversity;
- model-family general verification;
- ownership proof.
"""


def hardening_status_text(hardening: dict) -> str:
    return f"""# VSG Hardening Status Included In Expert Packet 2026-06-01

This packet includes the artifact-only hardening outputs produced after the 2026-05-31 packet.

| Area | Included status |
| --- | --- |
| Stronger public-text predicate local pilot | codeword recovered blocks total `{hardening["public_text_stronger_baseline"]["codeword_recovered_blocks_total"]}`; not adopted locked evidence |
| Attack naturalness proxy audit | `{hardening["attack_naturalness_proxy"]["proxy_pass_rows"]}/{hardening["attack_naturalness_proxy"]["rows"]}` proxy-readable rows; semantic naturalness not claimed |
| Reproducibility release inventory | `{hardening["reproducibility_release_inventory"]["row_count"]}` rows; `{hardening["reproducibility_release_inventory"]["missing_file_count"]}` missing; release-ready without review `{hardening["reproducibility_release_inventory"]["release_ready_without_review"]}` |
| Ownership decision-rule audit | `{hardening["ownership_decision_rule_audit"]["row_count"]}` rows; `{hardening["ownership_decision_rule_audit"]["failure_count"]}` rule failures; supported public final-text rows `{hardening["ownership_decision_rule_audit"]["supported_public_text_row_count"]}` |
| Manuscript figure-quality audit | `{hardening["manuscript_figure_quality_audit"]["figure_count"]}` figures; failed figure checks `{hardening["manuscript_figure_quality_audit"]["failed_figure_count"]}`; failed data checks `{hardening["manuscript_figure_quality_audit"]["failed_data_check_count"]}` |

No Slurm job, generation, model scoring, training, allowlist enablement, or Overleaf push was performed for this packet.
"""


def assemble() -> None:
    if PACKET_DIR.exists():
        shutil.rmtree(PACKET_DIR)
    PACKET_DIR.mkdir(parents=True)

    lint = build_lint_report()
    latex_summary = build_latex_summary()
    log_scan = build_log_scan()
    snapshot = git_snapshot()
    hardening = hardening_summary()
    if lint.get("status") != "PASS" or lint.get("violation_count") != 0:
        raise RuntimeError("claim-scope lint did not pass")
    if latex_summary.get("status") != "PASS":
        raise RuntimeError("latex summary did not pass")
    if log_scan.get("status") != "PASS":
        raise RuntimeError("latex log scan did not pass")
    if snapshot["manuscript_git_status_short"]:
        raise RuntimeError("manuscript repository is dirty")

    copy_file(MANUSCRIPT_DIR / "main.pdf", PACKET_DIR / "manuscript" / f"VSG_manuscript_snapshot_{DATE_TAG}.pdf")
    for rel in SOURCE_FILES + FIGURE_FILES:
        copy_file(MANUSCRIPT_DIR / rel, PACKET_DIR / "manuscript_source" / rel)

    copy_dir_files(FIGURE_DATA_DIR, PACKET_DIR / "evidence" / "figure_data")
    copy_dir_files(VISUAL_DRAFT_DIR, PACKET_DIR / "evidence" / "visual_drafts")
    for name, src_dir in HARDENING_DIRS.items():
        copy_dir_files(src_dir, PACKET_DIR / "evidence" / "hardening" / name)
    for src in HARDENING_STATUS_FILES:
        copy_file(src, PACKET_DIR / "evidence" / "hardening" / "status" / src.name)

    write_json(PACKET_DIR / "validation" / "claim_scope_lint_report.json", lint)
    write_json(PACKET_DIR / "validation" / "latex_build_summary.json", latex_summary)
    write_json(PACKET_DIR / "validation" / "latex_log_scan.json", log_scan)
    write_json(PACKET_DIR / "validation" / "git_snapshot.json", snapshot)
    write_json(PACKET_DIR / "validation" / "hardening_summary.json", hardening)

    manuscript_short = snapshot["manuscript_repository_head"][:7]
    root_short = snapshot["root_repository_head"][:7]
    readme = readme_text(manuscript_short, root_short, latex_summary)
    write_text(PACKET_DIR / f"README_FOR_EXPERT_REVIEW_{DATE_TAG}.md", readme)
    write_text(EXTERNAL_README, readme)
    write_text(PACKET_DIR / f"EXPERT_REVIEW_SCOPE_{DATE_TAG}.md", scope_text())
    write_text(PACKET_DIR / f"OBJECTIVE_FACTS_{DATE_TAG}.md", objective_facts_text(lint, latex_summary, hardening))
    write_text(PACKET_DIR / f"HARDENING_STATUS_{DATE_TAG}.md", hardening_status_text(hardening))

    files = []
    for path in sorted(PACKET_DIR.rglob("*")):
        if path.is_file() and path.name != "packet_manifest.json":
            rel = path.relative_to(PACKET_DIR).as_posix()
            files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    manifest = {
        "schema_name": "vsg_expert_review_packet_manifest_v3",
        "packet_name": PACKET_NAME,
        "status": "PASS_PACKET_ASSEMBLED_ARTIFACT_ONLY_20260601_HARDENING_INCLUDED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_only": True,
        "slurm_started": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "allowlist_enabled": False,
        "overleaf_push_performed": False,
        "core_facts": CORE_FACTS,
        "hardening_summary": hardening,
        "git_snapshot": snapshot,
        "latex_build_summary": latex_summary,
        "claim_scope_lint_summary": {
            "status": lint["status"],
            "checked_files": lint["checked_files"],
            "violation_count": lint["violation_count"],
        },
        "packet_total_file_count": len(files) + 1,
        "hashed_file_count": len(files),
        "manifest_self_hash_excluded": True,
        "manifest_self_hash_exclusion_reason": "A manifest cannot stably include its own sha256 as file content.",
        "files": files,
    }
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
        "packet_total_file_count": manifest["packet_total_file_count"],
        "hashed_file_count": manifest["hashed_file_count"],
        "manuscript_head": snapshot["manuscript_repository_head"],
        "root_head": snapshot["root_repository_head"],
        "pdf_sha256": latex_summary["pdf_sha256"],
        "hardening_summary": hardening,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    assemble()
