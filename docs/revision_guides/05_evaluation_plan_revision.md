# Evaluation Plan Revision Guide

## 1. Purpose

The Evaluation section must be organized around claims, not around run names or artifact status.

Current risk:
- The section can read like an internal artifact report.
- Some results are complete; others are still partial.
- If incomplete calibration is written as a final result, reviewers will reject the audit claim.

Goal:
- Convert Evaluation into a research-question-driven section.
- Use `NEEDS_RESULTS`, `PLACEHOLDER`, and `TODO_AFTER_RESULTS` wherever final results are unavailable.
- Separate completed clean evidence from incomplete FAR / utility / broad robustness / cross-family generalization.

## 2. Codex Search Instructions

```bash
rg -n "\\section\\{Experiments\\}|\\section\\{Evaluation\\}|Claim A|Claim B|Claim C|Claim D|Claim E|Claim F" .
rg -n "current artifact|paper-facing|partial|standing|G1|G2|G3|G4|Batch|R1|Perinucleus|FAR|utility" .
rg -n "outperform|superior|robust|generalize|full FAR|calibrated|state-of-the-art" .
```

Do not assume file names.

## 3. Recommended Evaluation Section Structure

Use this structure:

```latex
\section{Evaluation}
\label{sec:evaluation}

\paragraph{Research questions.}
We evaluate the method by the claims required for an ownership-audit protocol:
\begin{description}
    \item[RQ1:] Does tokenizer-aligned measurability make the carrier event trainable in the compiled setting?
    \item[RQ2:] Does bucket-mass training align success with bucket-level verifier semantics rather than canonical lexical identity?
    \item[RQ3:] Does the public decoder behave according to the stated error/erasure boundary?
    \item[RQ4:] How far does the completed Qwen compiled path scale across payloads, seeds, prompts, block counts, and training-signal sizes?
    \item[RQ5:] How does the method compare to ownership and provenance baselines under matched budgets?
    \item[RQ6:] What are the explicit failure modes and out-of-scope transformations?
\end{description}

\paragraph{Claim control.}
Results that require unfinished FAR, utility, or cross-family calibration are marked \textbf{NEEDS_RESULTS} and are not used to support superiority or deployment claims.
```

Then use subsections:

```latex
\subsection{Experimental Setup}
\subsection{RQ1: Tokenizer-Aligned Measurability}
\subsection{RQ2: Bucket-Mass Semantics}
\subsection{RQ3: Conditional Decoding and Robustness Boundary}
\subsection{RQ4: Qwen Compiled-Path Scaling}
\subsection{RQ5: Baselines, FAR, and Utility Calibration}
\subsection{RQ6: Failure Cases and Scope Boundaries}
```

## 4. Claim-Evidence Map

Add a table early in Evaluation. Use exact statuses.

```markdown
## Claim-Evidence Map
| Claim | Required Experiment | Required Baseline | Required Figure/Table | Status |
|---|---|---|---|---|
| First-token measurable carriers are exact one-step trainable in the compiled setting | aligned carrier Qwen compiled path; ideally non-measurable control | non-measurable or multi-token carrier control | contextual-alignment table; non-measurable control table | READY_TO_WRITE for aligned construction; NEEDS_RESULTS for negative control |
| Bucket-level verifier semantics differ from canonical exact-slot identity | bucket_mass vs fixed_representative vs uniform_bucket | fixed representative; uniform bucket | bucket-correct vs exact-slot table; utility/distortion Pareto | READY_TO_WRITE for semantic distinction; NEEDS_RESULTS for empirical superiority |
| Public decoder recovers under bounded error/erasure corruption | controlled corruption grid over E/S | no-code or no-RS ablation | E/S phase diagram | NEEDS_RESULTS unless grid complete |
| Qwen compiled path clean recovery scales within current setting | payload/seed/prompt/block/training-signal sweeps | base/unprotected null for FAR separately | scaling tables | READY_TO_WRITE within Qwen compiled-path scope |
| Clean success is not superior to Perinucleus | matched clean Qwen matrix | Qwen-adapted official Scalable/Perinucleus | baseline clean table | READY_TO_WRITE as parity if table verified |
| Audit-grade FAR is controlled | full FAR: registered null, organic prompt-bank null, non-owner probes | base Qwen, organic prompt-bank, non-owner probes | FAR by query budget table with CI | NEEDS_RESULTS |
| Utility is preserved or comparable | matched utility benchmark | Perinucleus and base model | utility table / Pareto curve | NEEDS_RESULTS |
| Cross-family generalization | exact verifier on second tokenizer/model | Llama or other tokenizer family | cross-family table | RISKY_CLAIM; current R1 should be diagnostic only |
| Robustness to output transformations | attack grid with covered and outside-scope transformations | no attack; parser variants | attack-family table; E/S boundary | READY_TO_WRITE only for bounded observed behavior; RISKY_CLAIM for broad robustness |
```

In LaTeX, use compact version:

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{p{0.26\linewidth}p{0.25\linewidth}p{0.18\linewidth}p{0.18\linewidth}}
\toprule
Claim & Required evidence & Current status & Scope \\
\midrule
Measurable carriers are trainable & Aligned compiled carrier results; non-measurable control & \textbf{READY\_TO\_WRITE} for aligned construction; \textbf{NEEDS\_RESULTS} for negative control & Qwen compiled path unless extended \\
Bucket semantics & Bucket-mass vs canonical representative diagnostics & \textbf{READY\_TO\_WRITE} for semantic distinction; \textbf{NEEDS\_RESULTS} for utility superiority & Bucket-level verifier only \\
FAR calibration & Null-family FAR by query budget & \textbf{NEEDS\_RESULTS} & No audit-grade claim before completion \\
Baseline comparison & Matched budget, utility, FAR & \textbf{TODO\_AFTER\_RESULTS} & Clean parity if supported \\
\bottomrule
\end{tabular}
\caption{Claim-evidence map. Unsupported empirical conclusions are explicitly marked rather than folded into the main claims.}
\label{tab:claim-evidence}
\end{table}
```

## 5. Minimum Viable Evidence

Submission minimum:

1. **Clean recovery within Qwen compiled path**
   - Already appears available in current project status.
   - Codex must verify tables exist before citing numbers.

2. **Perinucleus clean parity**
   - Must be reported honestly if included.
   - Do not call this superiority.

3. **FAR status**
   - If Full FAR incomplete, say incomplete.
   - Do not aggregate deprecated partial organic outputs.

4. **Bucket semantics**
   - Report bucket-correct vs exact-slot diagnostic.
   - Do not conclude utility superiority unless Pareto is done.

5. **Robustness boundary**
   - Report whitespace preserve / truncation fail / delimiter fail as boundary.
   - Do not say broad robustness.

6. **Limitations and failure cases**
   - Include carrier deletion, delimiter destruction, probe-free distillation.

## 6. NeurIPS-Required Evidence Before Final Submission

These are necessary for a strong main-conference submission:

1. Full FAR calibration.
2. Matched utility degradation.
3. Official Qwen-adapted Perinucleus comparison under matched clean matrix.
4. Ablation of bucket_mass vs fixed_representative vs uniform_bucket.
5. Clear robustness/failure boundary.
6. Reproducibility details.
7. Confidence intervals or uncertainty for key rates.

## 7. Spotlight-Level Evidence

These are not all mandatory but strongly improve Spotlight potential:

1. Non-measurable / multi-token control.
2. Full-response `y_{\mathrm{org}}\Vert y_{\mathrm{car}}` experiment.
3. E/S phase diagram for RS recovery.
4. Cross-family exact verifier repair or rigorous diagnostic explanation.
5. Utility-distortion-recovery Pareto.
6. A main figure connecting theory to empirical axes.

## 8. Baseline Requirements

### Primary baseline

- Qwen-adapted official Scalable/Perinucleus baseline.
- Must be labelled exactly as adaptation, not original out-of-the-box result.
- If clean matrix is parity, write parity.

Suggested text:

```latex
The Qwen-adapted official Scalable/Perinucleus baseline reaches the same clean success matrix as our method in the matched clean setting. We therefore do not claim clean-success superiority. The remaining comparison axes are false-accept calibration, utility, query budget, compute, and verifier semantics. \textbf{TODO_AFTER_RESULTS:} update this paragraph after full FAR and matched utility are complete.
```

### Secondary baselines / controls

- fixed representative;
- uniform bucket;
- English-random active fingerprint;
- KGW/PostMark-style provenance controls only as task-mismatched controls;
- no-evidence null;
- base Qwen null;
- non-owner probes.

Do not present provenance controls as primary ownership baselines.

## 9. Ablation Requirements

Required ablations:

1. bucket_mass vs fixed_representative vs uniform_bucket.
2. tokenizer-aligned single-token vs non-measurable or multi-token carrier.
3. with RS vs without RS.
4. bucket size / number of carriers.
5. query budget.
6. parser cap / candidate cap.
7. prompt family.
8. block count.
9. training-signal size.

For each ablation, write:
- what it tests;
- what result would support the claim;
- what result would falsify or weaken the claim.

## 10. Robustness / Failure Analysis

Use categories:

1. **Covered / absorbed**
   - same-bucket substitution;
   - whitespace formatting if parser survives.

2. **Bounded code-layer corruption**
   - cross-bucket substitution;
   - candidate injection;
   - missing carriers up to RS budget.

3. **Outside guarantee**
   - carrier-block deletion;
   - tail truncation removing evidence;
   - delimiter destruction preventing parse;
   - probe-free distillation;
   - strong filter suppressing carriers.

Suggested text:

```latex
We treat failure under tail truncation and delimiter destruction as boundary evidence, not as a robustness failure of Proposition~\ref{prop:conditional-decoding}. These transformations violate the recoverable-carrier assumption required by the public verifier.
```

## 11. Main Result Text Placeholders

Codex must not fabricate values. Use placeholders:

```latex
\paragraph{Result.}
\textbf{PLACEHOLDER / NEEDS_RESULTS:} Insert final numbers from the verified artifact table. Report the numerator, denominator, confidence interval if applicable, model/tokenizer, query budget, and whether failures remain in the denominator.
```

For incomplete FAR:

```latex
\paragraph{False-accept calibration.}
\textbf{NEEDS_RESULTS:} Full FAR is not complete. This paragraph must not claim calibrated false-accept control until registered-probe null, organic prompt-bank null, and non-owner probe results have been aggregated under the declared protocol.
```

For incomplete utility:

```latex
\paragraph{Utility.}
\textbf{NEEDS_RESULTS:} Matched utility degradation is required before claiming utility preservation or utility-superior evidence injection.
```

## 12. Codex Checklist

- [ ] Evaluation section begins with research questions.
- [ ] Every result maps to a claim.
- [ ] Every incomplete claim is marked.
- [ ] No internal artifact status language remains in main text unless in appendix.
- [ ] Perinucleus clean comparison is parity, not superiority.
- [ ] FAR not claimed if incomplete.
- [ ] Utility not claimed if incomplete.
- [ ] Robustness not overclaimed.
- [ ] R1 cross-family result written as diagnostic if included.
- [ ] All tables have captions explaining scope.
- [ ] All rates include denominator.
- [ ] Confidence intervals reported where available.
- [ ] LaTeX compiles.
