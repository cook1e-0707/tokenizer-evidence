# Problem Formalization Revision Guide

## 1. Purpose

This guide instructs Codex to revise the problem statement, notation, assumptions, threat model, definitions, and theoretical framing.

The goal is to make the paper read like a principled NeurIPS submission rather than an implementation report.

## 2. Diagnosis

The current paper already contains a strong Problem Setup section with:
- black-box service;
- owner / suspect / auditor;
- hidden probe distribution;
- payload `m`;
- structured carrier block `y_car`;
- public verifier;
- FAR null family;
- desired properties;
- first-token measurability theorem;
- bucket-mass KL projection;
- conditional RS recovery.

However, the structure can be improved:

1. **Notation is scattered.**
   - Add a notation table.

2. **Threat model should be protocol-first.**
   - It should clearly say what the verifier assumes and what is outside scope.

3. **Utility-constrained scrubbing needs careful wording.**
   - Unless corresponding experiments are complete, treat it as a motivating adversary or future evaluation axis, not an already supported empirical claim.

4. **Assumptions should be explicit.**
   - Many limitations are actually assumptions of the audit protocol.

5. **Definitions should not be inflated.**
   - Keep first-token measurability and one-step decomposability.
   - Do not add decorative theorems.

## 3. Codex Search Instructions

```bash
rg -n "\\section\\{Problem|Problem Setup|Setup|Preliminaries" .
rg -n "\\subsection\\{Threat Model|Evidence Representation|Public Protocol|Decision Rule|Desired Properties" .
rg -n "first-token measurability|one-step decomposable|Tokenizer-aligned|Reed--Solomon|false-accept|null family" .
rg -n "utility-constrained scrubbing|probe-free|distillation|carrier-block|delimiter" .
rg -n "\\begin\\{definition\\}|\\begin\\{theorem\\}|\\begin\\{lemma\\}|\\begin\\{proposition\\}" .
```

Do not assume file names. Search first.

## 4. Required Formal Components

### 4.1 Notation Table — REQUIRED

Add near the beginning of Problem Setup or immediately after the first paragraph of Section 3.

Purpose:
- Make the formal sections readable.
- Reduce reviewer perception of ad-hoc complexity.

Suggested LaTeX:

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{ll}
\toprule
Symbol & Meaning \\
\midrule
$P_\theta^\star$ & Owner's protected autoregressive model \\
$S$ & Suspect black-box service queried by the auditor \\
$x \sim X_{\mathrm{probe}}$ & Owner probe input sampled from a hidden probe distribution \\
$y = y_{\mathrm{org}} \Vert y_{\mathrm{car}}$ & Ordinary response followed by structured carrier evidence \\
$m^\star$ & Owner-committed payload to be recovered by the verifier \\
$V_f$ & Public carrier vocabulary for field type $f$ \\
$\mathrm{Bucket}_f(b)$ & Public bucket with bucket identity $b$ for field type $f$ \\
$b=(b_1,\ldots,b_M)$ & Target bucket-ID sequence for an evidence block \\
$\mathrm{Tok}(\cdot)$ & Public tokenizer used to audit carrier alignment \\
$C=(c_1,\ldots,c_n)$ & Reed--Solomon codeword over $\mathbb{F}_q$ \\
$E,S$ & Code-layer symbol errors and erasures \\
$N_{\mathrm{query}}$ & Maximum audit query budget \\
$H_0$ & Declared null family for false-accept calibration \\
\bottomrule
\end{tabular}
\caption{Notation used in the audit protocol.}
\label{tab:notation}
\end{table}
```

Codex must check whether `booktabs` is already loaded. If not loaded, either:
- add `\usepackage{booktabs}` only if safe and compile-verified; or
- replace `\toprule`, `\midrule`, `\bottomrule` with plain `\hline`.

### 4.2 Problem Setting Paragraph — REQUIRED

Place before or at the start of Problem Setup.

```latex
\paragraph{Problem setting.}
We study an owner-probe black-box audit protocol for autoregressive language models. The owner commits to a payload $m^\star$ and trains a protected model to emit structured evidence on hidden probe inputs. An auditor later queries a suspect service $S$ and applies a public verifier to the returned text. The verifier accepts only if it recovers the committed payload under the pre-specified parser, decoder, query budget, and false-accept calibration protocol.
```

Do not write:
```latex
We solve stolen model ownership verification.
```

Use:
```latex
We study a declared audit protocol for recovering owner-controlled evidence from black-box outputs.
```

### 4.3 Assumptions Box — REQUIRED

Add after Problem Setting or Threat Model.

Purpose:
- Prevent overbroad reviewer interpretation.
- Make limitations part of the formal scope.

Suggested LaTeX:

```latex
\paragraph{Assumptions.}
The audit protocol relies on the following assumptions.

\begin{description}
    \item[A1: Public tokenizer and carrier audit.]
    The tokenizer version, carrier vocabularies, bucket partitions, parser, and decoder are fixed before evaluation and available to the auditor.

    \item[A2: Stable single-token carriers.]
    Each valid carrier representative used in the main construction is a stable single tokenizer token after detokenization and parsing. Unstable strings, delimiter-like forms, control characters, and leading-whitespace variants are excluded.

    \item[A3: Probe-based elicitation.]
    The ownership behavior is evaluated on owner probes drawn from or selected according to $X_{\mathrm{probe}}$. Probe-free distillation from organic queries that never expose the evidence behavior is outside the claimed regime.

    \item[A4: Recoverable carrier structure.]
    The verifier assumes that enough of the carrier block and parse structure survives to produce candidate bucket tuples. Complete carrier-block deletion, severe delimiter destruction, or filtering of all carrier-like strings is outside the recovery guarantee.

    \item[A5: Pre-registered decision rule.]
    The query budget, parser cap, candidate-window budget, payload commitment, and accept threshold are fixed before evaluation.

    \item[A6: Declared null family.]
    False accepts are measured against a declared null family $H_0$ of non-owner models or services. \textbf{NEEDS_RESULTS:} full FAR claims require completed null-family experiments.
\end{description}
```

### 4.4 Threat Model Paragraph — REQUIRED

Revise current threat model to distinguish:
- covered;
- motivating but not fully evaluated;
- outside scope.

Suggested LaTeX:

```latex
\paragraph{Threat model.}
The serving adversary controls the black-box interface observed by the auditor and may apply lightweight output-side transformations. Our recovery claim applies only when these transformations preserve enough carrier structure to be parsed and when the remaining corruption can be represented as bounded symbol errors and erasures. A stronger parameter-scrubbing adversary, such as PEFT-style fine-tuning intended to erase probe behavior while preserving utility, is a motivating post-release threat; empirical claims against it should be made only for experiments reported in Section~\ref{sec:evaluation}. Pure probe-free distillation from organic non-probe queries is outside scope.
```

If the paper currently says:
```latex
Our target is deterministic ownership verification under these black-box and utility-constrained scrubbing settings.
```

Replace with:
```latex
Our target is deterministic ownership verification under the declared black-box probe-and-carrier protocol. Utility-constrained scrubbing is an important post-release adversary, but claims against it require explicit experiments and are not implied by the measurability or decoding results alone.
```

### 4.5 Public Protocol — REQUIRED

If there is already a Public Protocol subsection, rewrite to be concise and auditable.

```latex
\paragraph{Public audit protocol.}
Before evaluation, the owner fixes the tokenizer, public carrier vocabularies, bucket partitions, mixed-radix map, Reed--Solomon parameters, parser, query budget, payload commitment, and accept rule. The only private component is the probe selection mechanism. Given transcripts $\{(x_j,\hat{y}_j)\}_{j=1}^{N_{\mathrm{query}}}$ from a suspect service, the verifier scans each output, decodes candidate bucket tuples, attempts symbol recovery, and accepts only if a recovered payload matches $m^\star$ and passes the public metadata or checksum tests.
```

### 4.6 False-Accept Definition — REQUIRED

Keep or add:

```latex
\paragraph{False-accept calibration.}
Let $H_0$ denote the declared null family of non-owner models or services. A verifier operating point is meaningful only together with a query budget and false-accept estimate:
\[
\Pr_{S_0 \sim H_0}\left[\mathrm{Accept}(S_0)=1\right] \leq \alpha.
\]
\textbf{NEEDS_RESULTS:} This inequality is a calibration target, not an empirical conclusion, until the full null-family experiments are complete.
```

### 4.7 Definitions — KEEP AND CLEAN

The following definitions are necessary:

1. First-token measurability.
2. One-step decomposability.
3. Tokenizer-aligned carrier bucket.

Suggested Definition:

```latex
\begin{definition}[First-token measurability]
Let $\Omega = V^{\mathbb{N}}$ be the space of token continuations after a prefix $h$, and let $[v]=\{\omega \in \Omega : \omega_1=v\}$ be the first-token cylinder for token $v$. An evidence event $E \subseteq \Omega$ is \emph{first-token measurable} if there exists a token set $T_E \subseteq V$ such that
\[
E = \bigcup_{v \in T_E} [v].
\]
\end{definition}
```

Suggested Definition:

```latex
\begin{definition}[One-step decomposability]
An evidence event $E$ is one-step decomposable if there exists a function $\Psi_E$ such that, for every autoregressive model $P$ and prefix $h$,
\[
P_h(E) = \Psi_E(p_1^P(\cdot \mid h)),
\]
where $p_1^P(\cdot \mid h)$ is the current next-token distribution.
\end{definition}
```

## 5. Theorems / Propositions

### 5.1 Theorem: One-Step Decomposability iff First-Token Measurability

Keep this theorem.

It serves:
- Core claim: measurability is the exact trainability condition.

It proves:
- Event probability can be computed from current next-token distribution for all autoregressive models iff the event is a union of first-token cylinders.

It does not prove:
- Empirical recovery.
- Robustness.
- FAR.
- Utility preservation.
- Cross-model generalization.

Codex should add a sentence after the theorem:

```latex
The theorem is a design criterion rather than an empirical performance claim: it identifies which evidence events can be controlled exactly at the one-step interface, but it does not imply that a finite-parameter model trained by SGD will recover the payload without the experiments in Section~\ref{sec:evaluation}.
```

### 5.2 Corollary: Tokenizer-Aligned Carrier Buckets

Keep.

It serves:
- Construction claim.

It proves:
- Stable single-token buckets are measurable and exact bucket mass is a sum of next-token probabilities.

It does not prove:
- All measurable events must be single-token.
- Cross-tokenizer universality.

Add sentence:

```latex
This corollary is a sufficient construction, not a necessity theorem for all possible evidence schemes.
```

### 5.3 Theorem: Bucket-Mass KL Projection

Keep.

It serves:
- Training objective claim.

It proves:
- In an idealized distributional projection, increasing bucket mass while preserving within-bucket preferences is KL-minimal.

It does not prove:
- Neural optimizer convergence.
- Empirical lower distortion.
- Utility preservation.

Add sentence:

```latex
The result should be read as a distribution-level characterization of the control object; empirical utility and distortion must still be measured directly.
```

### 5.4 Proposition: Conditional End-to-End Decodability

Keep.

It serves:
- Verification claim.

It proves:
- If perturbations reduce to bounded errors/erasures satisfying RS condition, payload recovery follows.

It does not prove:
- Broad robustness.
- Survival under deletion or delimiter destruction.
- Security against adversarial serving policies.

Add sentence:

```latex
This is a conditional decoding statement, not a blanket robustness guarantee.
```

## 6. Formal Components Not to Add

Do not add:

1. A fake “security theorem” unless there is a cryptographic model.
2. A broad “impossibility theorem” beyond the first-token measurability theorem.
3. A theorem claiming robustness to arbitrary transformations.
4. A theorem claiming FAR control without null-family experiments.
5. A theorem claiming utility preservation from KL projection.
6. A theorem claiming tokenizer-family generality.

## 7. Codex Checklist

After editing formalization:

- [ ] Does the Problem Setup define owner, suspect, auditor?
- [ ] Is the query-only black-box setting explicit?
- [ ] Is probe-free distillation outside scope?
- [ ] Is utility-constrained scrubbing not overclaimed?
- [ ] Is FAR described as calibration target unless completed?
- [ ] Is tokenizer dependence explicit?
- [ ] Are all theorem assumptions stated?
- [ ] Are theorem conclusions not overextended?
- [ ] Does LaTeX compile?
- [ ] Are labels and refs resolved?
