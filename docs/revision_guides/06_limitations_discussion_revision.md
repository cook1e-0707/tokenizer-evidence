# Limitations and Discussion Revision Guide

## 1. Purpose

Limitations should be honest, precise, and scoped. They should not sound like apologies, and they should not hide critical boundaries.

High-level rule:

> A limitation is acceptable when it is part of the declared protocol; it becomes fatal when the paper claims beyond it.

## 2. Current Reviewer Attack Points

Reviewers may attack:

1. **Threat model is narrow.**
   - The method requires owner probes and recoverable carrier structure.

2. **Serving adversary can delete carrier block.**
   - This is outside guarantee and must be explicit.

3. **Delimiter destruction breaks parsing.**
   - Do not present robustness broadly.

4. **Probe-free distillation is not covered.**
   - Must be in threat model and limitations.

5. **FAR incomplete.**
   - No audit-grade ownership claim until completion.

6. **Utility incomplete.**
   - No utility-preserving claim until measured.

7. **Perinucleus clean parity.**
   - Do not claim clean superiority.

8. **R1 Llama exact replication fails.**
   - Must be diagnostic only.

9. **Tokenizer dependence.**
   - This is central, not incidental.

10. **Theory does not prove neural optimizer success.**
    - Theorem 5.1 is distributional, not empirical guarantee.

## 3. Codex Search Instructions

```bash
rg -n "\\section\\{Discussion|\\section\\{Limitations|\\section\\{Conclusion|limitations|outside scope|future work" .
rg -n "robust|guarantee|secure|distillation|carrier-block|delimiter|FAR|utility|Perinucleus|Llama|general" .
```

## 4. Recommended Limitations Structure

Use subsections or bold paragraphs:

1. Output transformations outside the guarantee.
2. Probe-free distillation.
3. FAR and audit calibration.
4. Utility and distortion calibration.
5. Model/tokenizer generality.
6. Baseline positioning.
7. No cryptographic adjudication.
8. Reproducibility and deployment assumptions.

## 5. High-Level Limitations Opening

Suggested LaTeX:

```latex
\section{Discussion and Limitations}
\label{sec:limitations}

The paper's contribution is a structural design principle for autoregressive ownership evidence: make the verifier-relevant event next-token measurable, train bucket mass, and decode recovered bucket identities with a public conditional decoder. This principle does not imply universal robustness or cryptographic proof of theft. The limitations below are therefore part of the declared audit protocol rather than post-hoc caveats.
```

## 6. Output Transformations Outside Guarantee

```latex
\paragraph{Output transformations outside the guarantee.}
The verifier assumes that enough of the carrier block and parse structure survives to produce candidate bucket tuples. Bucket-preserving variation and limited formatting changes can be absorbed when the parser still recovers the relevant fields. By contrast, complete carrier-block deletion, tail truncation before the block is complete, delimiter destruction that prevents parsing, or filters that suppress carrier-like strings can cause verification failure. These transformations violate the recoverable-carrier assumption and should not be described as covered robustness cases.
```

## 7. Probe-Free Distillation

```latex
\paragraph{Probe-free distillation is outside scope.}
The audit protocol assumes that the suspect service is the protected model, a derivative that retains the probe behavior, or a service that routes to such a model. A student trained only from organic non-probe queries may never observe the evidence behavior and therefore may not inherit it. Failure in that setting would not contradict the first-token measurability criterion or the bucket-mass objective; it would fall outside the protocol studied here.
```

## 8. FAR and Audit Calibration

```latex
\paragraph{False-accept calibration.}
Payload recovery on owner probes is not by itself an ownership audit. A deployable audit requires false-accept estimates under a declared null family, fixed query budget, parser cap, and accept rule. \textbf{NEEDS_RESULTS:} Until the full registered-probe, organic prompt-bank, and non-owner probe null experiments are complete, the empirical claim should be limited to the completed recovery and boundary artifacts rather than full FAR-calibrated ownership verification.
```

## 9. Utility and Distortion

```latex
\paragraph{Utility and distortion.}
The KL projection motivates bucket-mass control as a low-distortion distributional update, but it does not prove that finite-parameter training preserves ordinary task utility. \textbf{NEEDS_RESULTS:} Claims about utility preservation, matched utility degradation, or distortion advantages over baselines require explicit measurements and should not be asserted before those results are complete.
```

## 10. Model and Tokenizer Generality

```latex
\paragraph{Model and tokenizer generality.}
The construction depends on a public tokenizer audit and stable carrier vocabularies. Tokenizer changes, normalization changes, or parser changes can invalidate a carrier catalog. Evidence from one model/tokenizer family should therefore be reported with that scope. \textbf{TODO_AFTER_RESULTS:} Cross-family results should be described as exact replication only if the same verifier gate succeeds; otherwise they should be reported as diagnostic.
```

## 11. Baseline Positioning

```latex
\paragraph{Baseline positioning.}
The strongest active fingerprinting baseline should be reported under matched budgets and with its adaptation label. If the final clean comparison remains parity, the paper should not claim clean-success superiority. Instead, the distinction should be the evidence object, next-token measurability, bucket-mass training, public payload recovery, and calibrated operating point.
```

## 12. No Cryptographic Adjudication

```latex
\paragraph{No cryptographic adjudication.}
The verifier is public and deterministic once the audit protocol is fixed, but it is not a cryptographic proof of theft. Probe secrecy, payload commitment, false-accept calibration, and comparison against null services remain necessary for an ownership claim. The method should therefore be described as an audit protocol rather than a cryptographic ownership certificate.
```

## 13. Coverage Table

Add if space allows:

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{llll}
\toprule
Transformation & Covered? & Reason & Evidence status \\
\midrule
Same-bucket substitution & Conditional & Decodes to same bucket & Theoretical; NEEDS_RESULTS for full grid \\
Whitespace formatting & Conditional & Parser may still recover fields & READY_TO_WRITE if final table verified \\
Cross-bucket substitution & Bounded & Becomes symbol error & NEEDS_RESULTS for E/S grid \\
Missing carrier & Bounded & Becomes erasure & NEEDS_RESULTS for E/S grid \\
Tail truncation & No, if block removed & Violates recoverable-carrier assumption & Boundary failure \\
Delimiter destruction & No, if parser fails & Violates parse assumption & Boundary failure \\
Carrier-block deletion & No & Removes evidence object & Outside scope \\
Probe-free distillation & No & Student may never observe probes & Outside scope \\
\bottomrule
\end{tabular}
\caption{Scope boundaries for the public verifier. Covered cases require the stated parser and bounded-corruption assumptions; outside-scope cases should not be described as robustness successes.}
\label{tab:scope-boundaries}
\end{table}
```

## 14. Discussion Closing

Replace defensive sentences such as:

```latex
These limitations narrow the claim but do not weaken the main thesis.
```

with:

```latex
These boundaries define the protocol studied in this paper. Within that protocol, the central claim is structural: the evidence event must be next-token measurable to be exactly controlled at the autoregressive interface, and verification should decode the same bucket identities that training optimizes.
```

## 15. What Must Be in Threat Model vs Limitations

### Put in Threat Model

- owner-probe audit protocol;
- public tokenizer and decoder;
- carrier block must be recoverable;
- query budget and accept rule fixed;
- FAR null family declared;
- probe-free distillation outside scope.

### Put in Limitations

- Full FAR may be incomplete.
- Utility may be incomplete.
- Model/tokenizer generality limited.
- Broad robustness not claimed.
- No cryptographic adjudication.
- Baseline clean superiority not claimed if parity.

### Must Be Fixed by Experiments

- FAR calibration.
- Utility degradation.
- Perinucleus matched comparison.
- Robustness E/S grid.
- Non-measurable control.
- Full-response setting.
- Cross-family exact path if claiming generality.

## 16. Codex Checklist

- [ ] Limitations section exists.
- [ ] Limitations are not hidden only in appendix.
- [ ] No defensive language.
- [ ] No broad robustness wording.
- [ ] FAR incompleteness marked if unresolved.
- [ ] Utility incompleteness marked if unresolved.
- [ ] Perinucleus parity acknowledged if included.
- [ ] R1 diagnostic status preserved.
- [ ] Scope table added if space allows.
- [ ] Conclusion matches limitations.
- [ ] LaTeX compiles.
