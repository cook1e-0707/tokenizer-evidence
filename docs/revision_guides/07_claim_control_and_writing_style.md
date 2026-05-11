# Claim Control and Writing Style Guide

## 1. Purpose

This document controls the entire paper’s writing style and claim strength.

The target style is:
- strict;
- precise;
- NeurIPS-level;
- problem-first;
- formal where needed;
- conservative on unsupported results;
- sharp on supported contributions.

## 2. Global Claim Rule

Every claim must be supported by at least one of:

1. Formal theorem / proposition.
2. Completed experiment.
3. Citation to prior work.
4. Explicit design goal labeled as such.
5. Explicit placeholder:
   - `NEEDS_RESULTS`
   - `PLACEHOLDER`
   - `TODO_AFTER_RESULTS`

If none applies, weaken or delete the claim.

## 3. Risky Words

Codex must search for:

```bash
rg -n "prove|guarantee|demonstrate|show|outperform|state-of-the-art|general|robust|secure|optimal|superior|solve|fully|universal|cryptographic" .
```

These words are not banned, but each occurrence must be justified.

## 4. Safe Wording Before Full Results

Use:

- `we design`
- `we formalize`
- `we instantiate`
- `we evaluate`
- `we aim to test`
- `our evaluation is designed to assess`
- `under the stated assumptions`
- `within the declared protocol`
- `conditional on`
- `in the completed Qwen compiled-path setting`
- `where calibration is complete`
- `requires empirical validation`
- `NEEDS_RESULTS`
- `TODO_AFTER_RESULTS`

## 5. Claim Rewrite Table

| Risky Wording | Problem | Safer Replacement | When Stronger Wording Is Allowed |
|---|---|---|---|
| `We show robust ownership verification.` | Overbroad; current robustness is bounded. | `We define a conditional recovery protocol under stated parser and error/erasure assumptions.` | Only after broad robustness experiments and scope-matched attacks. |
| `We outperform Perinucleus.` | Current clean success is parity. | `The clean comparison is parity; our distinction is evidence object and verifier semantics.` | Only if matched FAR/utility/query-budget results support superiority. |
| `We demonstrate full FAR control.` | Full FAR incomplete unless final nulls complete. | `FAR calibration is reported where complete; remaining axes are NEEDS_RESULTS.` | After registered, organic, and non-owner nulls are complete with CIs. |
| `The method preserves utility.` | Utility not proven by KL theorem. | `Utility preservation is an empirical axis and requires matched evaluation.` | After matched utility results. |
| `The method generalizes across tokenizers.` | R1 exact gate currently diagnostic. | `Cross-family results are diagnostic unless exact verifier gate succeeds.` | After second tokenizer exact-path success. |
| `The decoder is robust to transformations.` | Only some transformations preserve parser/buckets. | `The decoder absorbs bucket-preserving variation and bounded code-layer corruption.` | After systematic robustness grid. |
| `We provide a secure proof of ownership.` | No cryptographic proof. | `We provide a public deterministic audit rule under a declared protocol.` | Only with formal cryptographic model, which is not current paper. |
| `Bucket-mass is optimal.` | Theorem is idealized KL projection. | `Bucket-mass is KL-minimal under an idealized distributional constraint.` | Only for theorem’s exact assumptions. |
| `RS decoding guarantees recovery.` | Only under E/S condition. | `RS decoding recovers when residual errors and erasures satisfy the stated bound.` | Only under Proposition assumptions. |
| `This solves stolen-model verification.` | Too broad. | `This studies owner-probe evidence recovery for autoregressive LLM audit protocols.` | Not appropriate for current scope. |

## 6. Writing Style Principles

### 6.1 Problem-first framing

Bad:
```latex
We propose a pipeline with tokenizer buckets, bucket loss, and RS decoding.
```

Good:
```latex
The design follows from a structural requirement: the evidence event must be exactly visible at the next-token interface. Tokenizer-aligned buckets satisfy this requirement, bucket-mass training controls the verifier-relevant object, and RS decoding makes the final decision public and deterministic under bounded corruption.
```

### 6.2 Formalization should be minimal but decisive

Do:
- Define first-token measurability.
- Define one-step decomposability.
- State theorem and proof sketch.
- State assumptions.
- State conditional decoder proposition.

Do not:
- Add decorative theorem.
- Overclaim theorem implications.
- Use theory to cover missing experiments.

### 6.3 Evaluation must answer claims

Bad:
```latex
Table X reports runs G1, G2, G3, and G4.
```

Good:
```latex
RQ4 asks whether the completed Qwen compiled path remains stable across payloads, seeds, prompt families, block counts, and training-signal sizes. Table X reports the corresponding completed sweeps. The result supports only the Qwen compiled-path scope, not tokenizer-family universality.
```

### 6.4 Distinguish design goal from result

Bad:
```latex
Our method controls false accepts.
```

Good:
```latex
The audit protocol requires false-accept calibration under a declared null family. \textbf{NEEDS_RESULTS:} The empirical FAR estimate must be inserted after the full null-family evaluation is complete.
```

### 6.5 Avoid project-report language

Replace:
- `current artifacts`
- `paper-facing artifacts`
- `standing package`
- `landing state`
- `artifact-backed`
- `partial calibration evidence`

With:
- `completed evaluation`
- `reported setting`
- `verified result table`
- `declared calibration protocol`
- `incomplete calibration axis`

## 7. Section-Specific Claim Control

### Abstract

Allowed:
- `we formalize`
- `we instantiate`
- `we train`
- `we verify`
- `conditional`

Disallowed unless completed:
- `we outperform`
- `robust`
- `full FAR`
- `general`

### Introduction

Allowed:
- `we ask`
- `we study`
- `we argue`
- `we define`

Disallowed:
- claiming broad stolen-model solution;
- claiming clean superiority;
- claiming utility preservation.

### Method

Allowed:
- theorem-supported statements;
- assumptions;
- conditional guarantees.

Disallowed:
- empirical claims not in Evaluation.

### Related Work

Allowed:
- structural difference;
- evidence object difference;
- verifier mechanism difference.

Disallowed:
- claiming prior work cannot do something unless verified.

### Evaluation

Allowed:
- result statements with actual numbers from verified tables.
- `NEEDS_RESULTS` for incomplete axes.

Disallowed:
- inference beyond scope.

### Limitations

Allowed:
- clear scope boundaries.

Disallowed:
- dismissive or defensive wording.

## 8. Codex Global Search Instructions

Run after each major rewrite:

```bash
rg -n "we show|we demonstrate|outperform|state-of-the-art|robust|secure|generalize|generalization|guarantee|superior|optimal|fully|universal|solve" .
rg -n "NEEDS_RESULTS|PLACEHOLDER|TODO_AFTER_RESULTS" .
rg -n "current artifacts|paper-facing|standing|landing|partial calibration|artifact-backed" .
```

For each risky phrase, Codex must produce a note:

```markdown
File:
Line:
Phrase:
Support:
Action taken:
```

## 9. Reviewer Concern Served

This pass serves:
- Quality: claims supported by evidence.
- Clarity: no ambiguity about scope.
- Significance: contribution is sharp, not overhyped.
- Originality: emphasizes structural criterion.
- Reproducibility: separates protocol from artifact status.

## 10. Final Claim-Control Checklist

- [ ] No unsupported `we show`.
- [ ] No unsupported `we demonstrate`.
- [ ] No `outperform` without matched result.
- [ ] No broad `robust`.
- [ ] No broad `secure`.
- [ ] No broad `general`.
- [ ] All missing results marked.
- [ ] Perinucleus parity preserved.
- [ ] R1 diagnostic status preserved.
- [ ] FAR incomplete status preserved if not completed.
- [ ] Utility incomplete status preserved if not completed.
- [ ] Conclusion does not exceed Abstract.
