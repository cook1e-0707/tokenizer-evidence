# VSG 专家审查包说明 2026-05-31

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
  64510b9

root repository commit at packet build:
  9dae989

Overleaf push:
  not performed for this packet
```
