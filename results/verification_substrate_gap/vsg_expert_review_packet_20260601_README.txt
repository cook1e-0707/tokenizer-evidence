# VSG 专家审查包说明 2026-06-01

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
  c10b3f1

root repository commit at packet build:
  54772ac

PDF sha256:
  a64c984fac6503b20138805c8a9a323799f6feb1acfdcc1f7bb7310237f5a0fa

Overleaf push:
  not performed for this packet
```
