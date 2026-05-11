# Figures, Tables, and Algorithms Revision Guide

## 1. Purpose

Figures, tables, and algorithms should support the main story:

> measurability criterion → tokenizer-aligned construction → bucket-mass training → public conditional decoding → claim-calibrated evaluation.

They should not merely display run logs.

## 2. Codex Search Instructions

```bash
rg -n "\\begin\\{figure\\}|\\begin\\{table\\}|\\caption\\{|\\includegraphics|Algorithm|algorithm" .
rg -n "Figure 1|Table 1|Claim-to-artifact|baseline|FAR|utility|robustness|Perinucleus" .
rg -n "usepackage\\{graphicx|usepackage\\{booktabs|usepackage\\{algorithm|usepackage\\{algpseudocode|usepackage\\{tikz" .
```

Do not add new packages unless compile-verified.

## 3. Figures That Can Be Added Before Results

| Figure | Purpose | Where to Place | Required Content | Notes |
|---|---|---|---|---|
| Proof-to-protocol overview | Show how theorem implies system design | Introduction or Method Overview | first-token measurability → tokenizer buckets → bucket-mass loss → public RS verifier | Can revise existing Figure 1 caption rather than redraw |
| Running example diagram | Explain measurable vs non-measurable evidence | Introduction | multi-token phrase depends on future continuation; bucket of single tokens sums current logits | Can be a small schematic; no results needed |
| Threat model / scope diagram | Prevent overbroad interpretation | Problem Setup | covered, bounded, outside scope | Optional if table used |
| Public audit protocol flow | Make verifier reproducible | Problem Setup or Method | commit payload, train, query, parse, decode, accept, FAR null | No results needed |
| Method module diagram | Reduce pipeline perception | Method | inputs/outputs of compile, train, verify | Could merge with existing Figure 1 |

## 4. Caption Drafts for Pre-Result Figures

### Revised Figure 1 Caption

```latex
\caption{
Proof-to-protocol overview for tokenizer-aligned ownership evidence. First-token measurability identifies which evidence events are exact next-token objects. The owner instantiates this criterion with tokenizer-audited carrier buckets, trains bucket mass at carrier slots, and publishes a verifier that maps decoded bucket identities to self-synchronizing Reed--Solomon symbols. The recovery statement is conditional: bucket-preserving variation is absorbed, while residual parseable corruption must satisfy the stated error/erasure bound.
}
```

### Running Example Figure Caption

```latex
\caption{
Measurable versus non-measurable evidence events. A multi-token or continuation-dependent phrase cannot be assigned an exact event probability from current logits alone. A bucket of stable single-token carriers is a union of first-token cylinders, so its probability is the sum of next-token probabilities over the bucket.
}
```

### Threat Model / Scope Figure Caption

```latex
\caption{
Scope of the audit protocol. The verifier covers owner-probe elicitation, tokenizer-audited carrier buckets, public parsing, and bounded error/erasure recovery. Carrier-block deletion, delimiter destruction that prevents parsing, and probe-free distillation from organic queries are outside the claimed regime.
}
```

## 5. Tables That Can Be Added Before Results

| Table | Purpose | Where | Required Content | Status |
|---|---|---|---|---|
| Notation table | Make formal sections readable | Problem Setup | symbols and meanings | READY_TO_WRITE |
| Assumptions / scope table | Control claim scope | Problem Setup or Limitations | covered / partially evaluated / outside scope | READY_TO_WRITE |
| Related work positioning table | Clarify contribution boundary | Related Work | evidence object, payload, NT-measurability, verifier | READY_TO_WRITE |
| Claim-evidence map | Prevent unsupported claims | Evaluation opening | claim, required evidence, status, scope | READY_TO_WRITE |
| Algorithm boxes | Make method reproducible | Method | compile/train and verify | READY_TO_WRITE |

## 6. Figures/Tables That Need Results

| Figure/Table | Required Data | Claim Supported | Status |
|---|---|---|---|
| Main clean recovery table | verified Qwen compiled-path results | clean recovery within Qwen setting | READY_TO_WRITE only if table verified |
| Perinucleus comparison table | matched clean, FAR, utility, query budget | baseline comparison | NEEDS_RESULTS for FAR/utility; clean parity if verified |
| FAR by query budget | full null-family results | audit-grade false accept calibration | NEEDS_RESULTS |
| Utility/distortion table | task utility and distortion metrics | utility preservation / low distortion | NEEDS_RESULTS |
| Bucket objective Pareto | bucket_mass vs fixed vs uniform across utility/recovery | objective advantage | NEEDS_RESULTS |
| E/S phase diagram | controlled errors and erasures | RS boundary behavior | NEEDS_RESULTS |
| Robustness attack-family table | whitespace, truncation, delimiter, candidate injection, etc. | failure boundary | READY_TO_WRITE only for completed attacks; NEEDS_RESULTS for full grid |
| Cross-family table | exact verifier and RS-aware results on second family | generalization | RISKY_CLAIM; diagnostic only unless exact gate succeeds |
| Full-response table | y_org + y_car experiment | non-toy end-to-end setting | NEEDS_RESULTS |

## 7. Algorithm Boxes

Algorithm boxes are recommended because they turn the method from “pipeline” into reproducible framework.

### Algorithm 1

Title:
```latex
Tokenizer-aligned evidence compilation and training
```

Caption:
```latex
The owner compiles a committed payload into bucket IDs, instantiates tokenizer-audited carrier examples, and trains bucket mass at carrier slots. Representative choice is used only to instantiate teacher-forced text; the loss targets the full bucket mass.
```

### Algorithm 2

Title:
```latex
Public bucket/RS verification
```

Caption:
```latex
The auditor queries a suspect service, parses candidate carriers, decodes bucket tuples into coordinate-symbol pairs, and accepts only if Reed--Solomon decoding recovers the committed payload under the public decision rule.
```

## 8. Claim-Evidence Map Table

Main text version should be compact. Full artifact ledger can go appendix.

Suggested caption:

```latex
\caption{
Claim-evidence map for the evaluation. The table separates completed evidence from axes requiring additional calibration. It is included to prevent unsupported superiority, robustness, or generalization claims from entering the main narrative.
}
```

## 9. Related Work Table

Current related work table is useful. Strengthen with:
- “not an empirical ranking”
- “Payload” definition
- “NT-meas.” definition
- Perinucleus footnote or separate table if included.

Caption:

```latex
\caption{
Positioning by evidence object and verification rule. The table compares the question each method family is designed to answer, not empirical superiority. Our distinction is the combination of owner-payload recovery, explicit next-token measurability, and public bucket/RS verification.
}
```

## 10. FAR and Utility Tables

Must not be filled until results are complete.

Use placeholder table if needed:

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{lllll}
\toprule
Method & Query budget & Clean recovery & FAR & Utility delta \\
\midrule
Ours & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} & \textbf{NEEDS\_RESULTS} & \textbf{NEEDS\_RESULTS} \\
Qwen-adapted Scalable/Perinucleus & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} & \textbf{NEEDS\_RESULTS} & \textbf{NEEDS\_RESULTS} \\
\bottomrule
\end{tabular}
\caption{
Matched-budget comparison template. Do not report superiority until clean recovery, FAR, utility, and query budget are all populated from final artifacts.
}
\label{tab:matched-comparison-template}
\end{table}
```

## 11. Appendix Tables

Move detailed run ledgers to appendix:
- G1/G2/G3/G4 details.
- Batch attack-family details.
- T1/T2 theorem package details.
- R1 diagnostic table.
- Compute accounting.
- Reproducibility details.

Main text should contain:
- summarized claim-supporting tables only;
- no internal run names unless necessary.

## 12. Codex Checklist

- [ ] Existing Figure 1 caption revised to proof-to-protocol story.
- [ ] Notation table added.
- [ ] Assumptions/scope table added or paragraphs added.
- [ ] Algorithm 1 added or box equivalent added.
- [ ] Algorithm 2 added or box equivalent added.
- [ ] Claim-evidence map added.
- [ ] Related work table checked for correctness.
- [ ] FAR table has `NEEDS_RESULTS` if incomplete.
- [ ] Utility table has `NEEDS_RESULTS` if incomplete.
- [ ] Perinucleus table says parity if clean parity.
- [ ] Detailed artifact/run tables moved to appendix where appropriate.
- [ ] All captions state scope and avoid overclaiming.
- [ ] LaTeX compiles.
