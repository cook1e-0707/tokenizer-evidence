# Abstract and Introduction Revision Guide

## 1. Main Problems to Fix

当前 Abstract 和 Introduction 已经包含正确核心：first-token measurability、tokenizer-aligned carriers、bucket-mass objective、public RS verifier、explicit limitations。

但仍需改成更像 NeurIPS Spotlight 论文：

1. **Abstract 过像项目状态摘要。**
   - 避免 `completed artifacts`、`paper-facing artifacts`、`partial calibration evidence` 等内部项目语言。
   - Abstract 应讲 scientific claim and scoped evidence，而不是 repo status。

2. **Introduction 需要更早建立核心瓶颈。**
   - 现在的开头容易先把问题拉到 broad stolen-model verification。
   - 应该更早解释：为什么 evidence event 是否 next-token measurable 是 tractability bottleneck。

3. **缺少强 running example。**
   - Reviewer 需要在前 1 页理解：
     - multi-token / semantic evidence 为什么不是 exact one-step object；
     - tokenizer-aligned bucket 为什么是 exact current-logit object；
     - bucket-mass 为什么是 verifier-aligned objective。

4. **Contribution bullets 需要更 sharp and verifiable。**
   - 当前 bullets 容易像模块清单。
   - 应该写成 criterion / objective / protocol / evaluation-boundary。

5. **Claim 控制必须更严格。**
   - 不能暗示 Full FAR 已完成。
   - 不能暗示 clean superiority over Perinucleus。
   - 不能暗示 broad robustness。
   - 不能暗示 cross-family exact replication。

## 2. What a NeurIPS Reviewer Must See in the First 1–2 Pages

Reviewer 在前两页必须能回答：

1. **Problem**
   - What is the exact problem?
   - Not generic text watermarking.
   - Not arbitrary forensic proof of theft.
   - It is owner-probe black-box recovery of a committed payload from an autoregressive suspect service under a declared public verifier.

2. **Gap**
   - Existing fingerprinting asks which trigger/response behavior to implant.
   - This paper asks the earlier structural question: when is the evidence event exactly visible at the next-token interface?

3. **Key insight**
   - Evidence is exact one-step controllable iff it is first-token measurable.
   - Tokenizer-aligned single-token carrier buckets are a sufficient auditable construction.

4. **Method**
   - Compile payload into bucket IDs.
   - Train bucket mass, not canonical token.
   - Decode public bucket identities with RS error/erasure recovery.

5. **Scope**
   - Conditional recovery only.
   - No carrier-block deletion robustness.
   - No delimiter destruction robustness.
   - No probe-free distillation claim.
   - FAR / utility / baseline superiority only where completed.

6. **Contribution**
   - Criterion.
   - Objective.
   - Public conditional decoder.
   - Claim-controlled evaluation.

## 3. Recommended Abstract Structure

Use four paragraphs or four long sentences:

### Sentence 1: Problem

State the black-box audit problem and the hidden structural issue.

### Sentence 2: Insight / Formalization

Introduce first-token measurability.

### Sentence 3: Method

Tokenizer-audited buckets + bucket-mass objective + public RS verifier.

### Sentence 4: Evidence / Scope

State completed evidence conservatively. Use `NEEDS_RESULTS` for incomplete claims.

## 4. Conservative Abstract Skeleton

Replace the current abstract with a version structurally similar to the following. Codex must adjust length to fit the conference format and must not insert unsupported numerical results.

```latex
\begin{abstract}
Black-box ownership auditing for autoregressive language models requires evidence that can be trained and verified through the same interface: the next-token distribution. We study a declared owner-probe audit protocol in which a suspect service is queried for a committed payload and a public verifier attempts to recover that payload from generated text.

We formalize the corresponding tractability condition as \emph{first-token measurability}: an evidence event is exactly computable from current next-token logits only when it is a union of first-token cylinders. This criterion turns tokenizer alignment from an implementation detail into a design requirement. We instantiate it with tokenizer-audited carrier buckets, train the verifier-relevant bucket mass rather than a canonical lexical representative, and decode recovered bucket identities using a public self-synchronizing Reed--Solomon verifier.

The resulting guarantee is conditional. Bucket-preserving variation can be absorbed, and residual errors and erasures are recoverable only within the stated decoding bound; carrier-block deletion, delimiter destruction, and probe-free distillation are outside the claimed regime. \textbf{NEEDS_RESULTS:} Insert only completed empirical evidence here, e.g., Qwen compiled-path clean recovery and explicit boundary results. Do not claim baseline superiority, full FAR calibration, utility preservation, broad robustness, or cross-family exact replication unless the corresponding final experiments are complete.
\end{abstract}
```

If the final results are complete, replace the last sentence with a supported result sentence. Otherwise keep the `NEEDS_RESULTS` marker or use a conservative version:

```latex
Our current evaluation reports completed clean-recovery and boundary evidence separately from false-accept, utility, and baseline-superiority claims, which require full calibration before being asserted.
```

## 5. Recommended Introduction Structure

Use the following structure:

### Paragraph 1: Black-box ownership audit problem

Goal:
- Establish relevance.
- Avoid overbroad claim.

Suggested text:

```latex
Large language models are increasingly deployed through opaque services, where an owner may observe only query responses from a suspect interface. In this setting, ownership verification is not a question of inspecting weights or logits; it is an audit protocol over observable text. We study the following concrete version: the owner commits to a payload, queries a suspect autoregressive service with hidden probes, and a public verifier attempts to recover that payload from the returned text.
```

### Paragraph 2: Distinguish from text provenance and generic fingerprinting

Suggested text:

```latex
This problem is adjacent to text watermarking and LLM fingerprinting, but the evidence object is different. Text watermarking asks whether a generated sample carries a provenance signal. Trigger-based fingerprinting asks whether a model exhibits owner-associated behavior on selected prompts. We ask an earlier structural question: when is the ownership evidence event itself exactly controllable through the autoregressive next-token interface?
```

### Paragraph 3: Running example

Add this early.

```latex
Consider two ways to encode the next field of an owner payload. One design asks the model to emit a multi-token phrase or a semantic predicate, such as a short natural-language label. Whether the event has occurred may depend on future continuation tokens, so its probability cannot be computed from the current logits alone. A second design uses a public bucket of stable single-token carriers. In that case, the verifier-relevant event is a sum of next-token probabilities over the bucket. The two designs differ not merely in engineering convenience, but in whether the training objective can exactly optimize the event that the verifier later decodes.
```

### Paragraph 4: Thesis

```latex
This paper's thesis is that ownership evidence for autoregressive LMs should be designed as next-token-measurable evidence. First-token measurability is the structural condition for exact one-step control; tokenizer-aligned carrier buckets are a conservative auditable construction that satisfies it.
```

### Paragraph 5: Method overview

```latex
The criterion determines the rest of the system. We encode a committed payload into public bucket identities, supervise the total probability mass of each target bucket, and verify recovered bucket identities using a public Reed--Solomon decoder. Training therefore targets the object used by the verifier, rather than a fixed lexical representative or an artificial uniform distribution inside a bucket.
```

### Paragraph 6: Scope boundary

```latex
The claim is intentionally bounded. The verifier is deterministic under the declared protocol, but it is not a cryptographic proof of theft and it is not robust to arbitrary serving-side rewriting. If the service deletes the carrier block, destroys the parsing structure, or distills a model only from organic non-probe queries, recovery may fail. These are scope boundaries of the audit protocol, not hidden implementation assumptions.
```

### Paragraph 7: Contributions

Use revised bullets below.

## 6. Contribution Bullets Rewrite

Replace vague module-style bullets with verifiable contribution bullets.

```latex
Our contributions are:

\begin{itemize}
    \item \textbf{A tractability criterion for autoregressive ownership evidence.}
    We formalize first-token measurability as the condition under which a verifier-relevant evidence event can be computed exactly from current next-token logits. Tokenizer-audited single-token carrier buckets give a conservative sufficient construction.

    \item \textbf{A verifier-aligned training objective.}
    Given measurable carrier buckets, we optimize bucket mass rather than a canonical representative. A KL projection characterizes the corresponding least-distorting distributional update under an idealized bucket-mass constraint.

    \item \textbf{A public conditional recovery protocol.}
    We map decoded bucket tuples to self-synchronizing Reed--Solomon symbols, yielding a deterministic accept/reject rule with an explicit error/erasure recovery condition.

    \item \textbf{A claim-controlled empirical protocol.}
    We separate clean recovery, false-accept calibration, utility, baseline comparison, robustness boundaries, and cross-family diagnostics. \textbf{NEEDS_RESULTS:} Insert final result-supported summary only after FAR, utility, and baseline calibration are complete.
\end{itemize}
```

## 7. Sentences to Avoid

Codex must remove or weaken these sentence types unless the paper has support:

| Avoid | Why | Safer Replacement |
|---|---|---|
| `We solve stolen-model ownership verification.` | Too broad. | `We study a declared owner-probe audit protocol for ownership evidence recovery.` |
| `Our method is robust to output transformations.` | Current robustness is bounded. | `The decoder is conditional on recoverable carrier structure and bounded error/erasure corruption.` |
| `We outperform Perinucleus.` | Current clean result is parity. | `The current clean comparison is parity; the distinction is structural and verifier-semantic.` |
| `We provide full false-accept calibration.` | Full FAR incomplete unless final artifacts exist. | `False-accept calibration is reported where complete; incomplete axes are marked NEEDS_RESULTS.` |
| `The method generalizes across tokenizers.` | R1 exact gate currently fails. | `Cross-family results are diagnostic unless exact verifier replication is established.` |
| `The evidence is secure.` | No cryptographic proof. | `The verifier is public and deterministic under the declared protocol.` |

## 8. Codex Search Instructions

Codex must run these searches before editing:

```bash
rg -n "\\begin\\{abstract\\}|\\end\\{abstract\\}" .
rg -n "\\section\\{Introduction\\}|\\section\\*\\{Introduction\\}" .
rg -n "Our contributions|contributions are|This yields four contributions" .
rg -n "we show|we demonstrate|outperform|state-of-the-art|robust|secure|generalize|guarantee|superior" .
rg -n "completed|paper-facing|artifact|partial calibration|current artifacts|standing" .
```

For each risky phrase:
1. Check whether it is supported by:
   - theorem/proposition;
   - completed result table;
   - citation;
   - project status artifact.
2. If unsupported, rewrite or mark `NEEDS_RESULTS`.

## 9. Codex Modification Instructions

1. Locate the abstract environment.
2. Replace current abstract with the conservative skeleton.
3. Locate Introduction.
4. Insert running example before detailed related-work discussion.
5. Add or rewrite scope paragraph before contributions.
6. Replace contribution bullets with the revised version.
7. Preserve existing citations where valid.
8. Do not invent new citations.
9. Compile.
10. If page length exceeds limit, compress wording rather than deleting scope boundaries.
