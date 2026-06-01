# 专家审查对象说明 2026-05-31

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
