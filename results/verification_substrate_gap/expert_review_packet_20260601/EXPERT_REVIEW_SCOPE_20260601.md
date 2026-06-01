# 专家审查对象说明 2026-06-01

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
