# Master Revision Plan

## 1. Current Diagnosis

当前论文已经有一个有潜力冲 NeurIPS Spotlight 的核心主线：

> Ownership evidence for autoregressive LMs should be designed as next-token-measurable evidence; tokenizer-aligned carrier buckets make the verifier-relevant event exactly trainable, bucket-mass training aligns the objective with the verifier, and public RS decoding gives a conditional deterministic recovery rule.

但是当前稿件仍有几个会影响命运的问题：

1. **叙事仍像项目状态报告，而不是定稿论文。**
   - 主文中仍可能出现 `current artifacts`、`paper-facing artifacts`、`partial calibration evidence`、run-name 等内部项目语言。
   - NeurIPS reviewer 需要看到的是 problem → principle → method → evidence → scope，而不是项目流水账。

2. **claim 与 evidence 没有完全压紧。**
   - Full FAR 尚未完成，不能写 audit-grade FAR claim。
   - Perinucleus baseline clean success 是 parity，不是 superiority。
   - R1 Llama 是 diagnostic，不是 exact cross-family replication。
   - robustness 是 conditional / bounded，不是 broad robustness。

3. **Introduction 的 problem framing 仍然偏大。**
   - “stolen LLM ownership verification” 是大问题。
   - 当前方法解决的是 declared owner-probe audit protocol 下的 tokenizer-aligned evidence injection and public decoding。
   - 必须在前 1–2 页提前说明 scope，否则会被批评为 “big problem, narrow solution”。

4. **formalization 已有雏形，但需要更像论文而非实现说明。**
   - 需要 notation table。
   - 需要 assumptions / threat model box。
   - 需要 protocol box。
   - 需要 method algorithm box。
   - 不需要堆砌新的空洞 theorem。

5. **Related Work 需要从“列文献”改为“定位贡献边界”。**
   - 最关键的是正面对照 Scalable/Perinucleus。
   - 不能把本文写成 clean success 更强。
   - 应该强调 evidence object、next-token measurability、public deterministic decoder、claim-calibrated evaluation。

6. **Evaluation 必须按 research question 组织。**
   - 每个实验要回答一个明确 claim。
   - 每个结果后必须写 “what this supports / what this does not support”。
   - 未完成实验必须标记为 `NEEDS_RESULTS` 或 `TODO_AFTER_RESULTS`。

## 2. Revision Objective

目标不是语法润色，而是把论文重写成一个 NeurIPS Spotlight 风格的结构：

1. **Problem-first framing**
   - 先解释为什么 ownership evidence 的难点是 next-token interface。
   - 用 running example 展示 measurable vs non-measurable evidence 的差别。

2. **Minimal but precise formalization**
   - 用 first-token measurability theorem 支撑核心主张。
   - 用 bucket-mass KL projection 支撑 training object。
   - 用 conditional RS decoder proposition 支撑 bounded recovery。

3. **Claim-evidence alignment**
   - 每个 claim 必须有 theory、experiment、citation 或明确 `NEEDS_RESULTS`。
   - 未完成的 FAR / utility / robustness / generalization 不能写成结论。

4. **Conservative but sharp contribution statements**
   - 保守：不写 unsupported superiority。
   - 锋利：明确本文的新意是 tractability criterion + verifier-aligned objective + public conditional decoder。

5. **Evaluation as audit protocol**
   - 不仅看 clean recovery。
   - 还要按 recovery、acceptance、FAR、utility、baseline、robustness boundary、failure case 组织。

## 3. Immediate Revisions Before Full Results

以下内容现在就可以修改，不需要等待新实验：

### 3.1 Title / Abstract

修改目标：
- 把论文从 broad stolen-model claim 收缩到 next-token-measurable ownership evidence。
- 删除项目状态语言。
- 不写未完成结果。

必须加入：
- `first-token measurability`
- `tokenizer-audited carrier buckets`
- `bucket-mass objective`
- `public bucket/RS verifier`
- `conditional / bounded scope`

不得写：
- `we outperform`
- `state-of-the-art`
- `robust ownership verification`
- `full FAR-calibrated`
- `generalizes across model families`

### 3.2 Introduction

现在可改：
- 加 running example。
- 前移 scope / threat model。
- 重写 contribution bullets。
- 明确 “not text provenance, not generic fingerprinting, not cryptographic theft proof”。

不能提前写死：
- Full FAR 结论。
- Utility-preserving 结论。
- Perinucleus superiority。
- Broad robustness。
- Llama exact replication。

### 3.3 Problem Setup / Threat Model

现在可改：
- 加 notation table。
- 加 assumptions box。
- 加 public audit protocol。
- 把 utility-constrained scrubbing 从 “target setting” 改为 “motivating adversary unless experiments complete”。

### 3.4 Formal Definitions

现在可改：
- 统一 definitions。
- 保留 Theorem 4.1、Corollary 4.2、Theorem 5.1、Proposition 6.3。
- 强调它们证明什么、不证明什么。

不建议：
- 为了显得高级添加新 theorem。
- 把 standard RS decoding 包装成 coding contribution。

### 3.5 Method Overview

现在可改：
- 在 Method section 开头加入 overview。
- 加 Algorithm 1 and Algorithm 2。
- 为每个 module 写 input/output。
- 把 implementation detail 移到 appendix 或 implementation subsection。

### 3.6 Related Work

现在可改：
- 按 contribution boundary 重组。
- 单独正面对照 Scalable/Perinucleus。
- 把 KGW/PostMark 放在 text provenance/control 类，不作为 primary ownership baseline。

### 3.7 Evaluation Plan

现在可改：
- 用 research questions 重写 section opening。
- 建立 claim-evidence map。
- 用 `NEEDS_RESULTS` 标记未完成结论。
- 把 artifact ledger 移到 appendix 或压缩为 main-text summary。

### 3.8 Limitations

现在可改：
- 把 limitations 写成 scope boundaries。
- 不要隐藏 carrier deletion、delimiter destruction、probe-free distillation。
- 明确这些不是 implementation bugs，而是 declared protocol boundaries。

## 4. Revisions After Experimental Results

以下内容必须等实验结果或最终 artifact 验证后才能定稿：

1. **FAR claim**
   - 需要 Full FAR，包括 registered null、organic prompt-bank null、non-owner probes。
   - 没有完成前只能写 `NEEDS_RESULTS: full FAR calibration`.

2. **Utility claim**
   - 需要 matched utility degradation。
   - 没有完成前不能写 utility-preserving 或 low-distortion empirical conclusion。

3. **Baseline superiority**
   - Perinucleus clean success 是 parity。
   - 只有在 FAR / utility / compute / query budget 某个维度有完整证据时，才能写有条件优势。

4. **Robustness claim**
   - 当前只能写 bounded / conditional recovery。
   - Broad robustness 需要 systematic attack grid 或 scrubbing experiments。

5. **Cross-family generalization**
   - R1 Llama exact gate 失败。
   - 只能写 diagnostic result，不能写 exact replication。

6. **Bucket-mass empirical superiority**
   - 需要 bucket_mass vs fixed_representative vs uniform_bucket 的 recovery-utility-distortion Pareto。
   - 当前 exact-slot diagnostic 不能支撑 superiority。

7. **Scalability claim**
   - Qwen compiled-path scaling 可以写，但要限定在 Qwen compiled path。
   - Universal scalability 必须等更多 model/tokenizer/task 实验。

## 5. Spotlight-Level Strengthening

如果目标是 Spotlight，除基本投稿修改外，建议增加：

1. **Main thesis figure**
   - non-measurable evidence → no exact one-step mass
   - tokenizer-aligned buckets → exact mass
   - bucket-mass training → verifier-aligned objective
   - RS decoder → conditional public recovery

2. **Non-measurable control experiment**
   - 最能支撑 Theorem 4.1 的实验。
   - 需要显示 multi-token / non-measurable carrier 在同预算下更难训练或 recovery/utility tradeoff 更差。

3. **Full-response experiment**
   - 验证 `y_org || y_car`。
   - 解决 reviewer 对 exact-slot compiled setting 的 toy concern。

4. **Perinucleus parity positioning**
   - 主动承认 clean parity。
   - 把本文区别定位为 structural mechanism and audit semantics。

5. **Reproducibility appendix**
   - 当前 checklist 中 reproducibility / code / experimental details / stats / compute / broader impact / licenses 仍需补强。
   - 复杂 pipeline 论文如果 reproducibility 不清楚，会严重伤 credibility。

## 6. Execution Order for Codex

Codex 应按以下阶段执行，不要一次性大规模改动所有文件：

### Phase 0: Repository inspection

1. 定位主 LaTeX 文件。
2. 搜索 all `\section`、`\subsection`、`\paragraph`、`\label`。
3. 搜索当前 Abstract、Introduction、Related Work、Problem Setup、Experiments、Limitations。
4. 定位 table / figure / algorithm packages。
5. 运行一次 LaTeX build，记录现有 warnings。

### Phase 1: Claim-control pass

1. 全文搜索 risky wording：
   - `we show`
   - `we demonstrate`
   - `outperform`
   - `state-of-the-art`
   - `robust`
   - `secure`
   - `general`
   - `guarantee`
   - `superior`
2. 对每个 risky claim 判断：
   - theory supported?
   - experiment supported?
   - citation supported?
   - otherwise mark `NEEDS_RESULTS` or weaken.

### Phase 2: Front-matter rewrite

1. Rewrite abstract.
2. Rewrite Introduction first 1–2 pages.
3. Rewrite contribution bullets.
4. Add scope box if space allows.

### Phase 3: Formalization and method

1. Add notation table.
2. Add assumptions / threat model box.
3. Add protocol box.
4. Add method overview.
5. Add Algorithm 1/2 if package exists or can be safely added.

### Phase 4: Related work positioning

1. Reorganize section around contribution boundaries.
2. Add closest-work comparison paragraph.
3. Avoid unsupported citation keys.
4. Mark missing references as `TODO_ADD_CITATION`.

### Phase 5: Evaluation rewrite

1. Convert experiments to RQ structure.
2. Add claim-evidence map.
3. Replace incomplete result text with `NEEDS_RESULTS`.
4. Move internal run ledger to appendix if necessary.

### Phase 6: Limitations and checklist

1. Rewrite limitations as scope boundaries.
2. Add coverage table.
3. Add reproducibility appendix material.
4. Update checklist only where supported.

### Phase 7: Compile and verify

1. Compile after each major phase.
2. Check undefined references.
3. Check bibliography.
4. Check overfull/underfull hboxes.
5. Check page limit if applicable.
6. Produce summary of changed files.

## 7. Global Constraints

1. Do not fabricate experimental results.
2. Do not infer missing numbers from partial artifacts.
3. Do not aggregate deprecated or partial FAR outputs.
4. All unsupported empirical claims must be marked:
   - `NEEDS_RESULTS`
   - `PLACEHOLDER`
   - `TODO_AFTER_RESULTS`
5. Use conservative wording unless the paper already has theory or artifact-backed evidence.
6. Maintain LaTeX compilability after each phase.
7. Preserve existing labels and citations unless there is a clear reason to change.
8. Do not break anonymity.
9. Do not add new packages unless necessary and compile-verified.
10. Do not delete limitations that define the paper’s legitimate scope.

## 8. Do-Not-Do List

- Do not fabricate experimental results.
- Do not write `we show`, `we demonstrate`, or `we outperform` unless the claim is already supported by completed results or formal proof.
- Do not turn a design goal into an experimental conclusion.
- Do not claim broad robustness from whitespace-only robustness.
- Do not claim robustness to carrier deletion, delimiter destruction, or probe-free distillation.
- Do not claim clean-success superiority over Perinucleus; current status is parity.
- Do not claim exact Llama replication from R1; current status is diagnostic only.
- Do not overclaim generalization across tokenizer families.
- Do not present KGW/PostMark-style provenance controls as primary ownership baselines.
- Do not package standard RS decoding as a novel coding contribution.
- Do not hide Full FAR incompleteness.
- Do not remove necessary limitations to make the paper sound stronger.
- Do not add empty theorem/proposition statements for appearance.
- Do not add broad security language such as `secure`, `unforgeable`, or `cryptographic proof` unless formally justified.
- Do not massively restructure LaTeX without compiling after each step.
- Do not assume file names; search first.
