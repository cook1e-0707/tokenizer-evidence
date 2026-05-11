# Method Revision Guide

## 1. Purpose

The Method section must read as a principled framework derived from the formalization, not as an ad-hoc engineering pipeline.

Current core method:
1. Commit payload.
2. RS encode.
3. Mixed-radix map to bucket IDs.
4. Use tokenizer-audited carrier buckets.
5. Train bucket mass.
6. Parse suspect output.
7. Decode bucket tuples.
8. Apply RS recovery.
9. Accept if recovered payload matches commitment.

This is strong if written as a theorem-to-protocol chain. It is weak if written as implementation plumbing.

## 2. Codex Search Instructions

```bash
rg -n "\\section\\{Method|\\section\\{Training|\\section\\{Verification|Bucket-Mass|Reed--Solomon|pipeline" .
rg -n "Algorithm|algorithm|compile|bucket-mass|mixed-radix|public verifier|parser|decoder" .
rg -n "guarantee|robust|ensures|always|secure|outperform|general" .
```

Locate the method-related sections. Do not assume names.

## 3. Required Edits

### Edit 1: Add Method Overview

At the beginning of the method section, add a concise overview that explains why each component exists.

Suggested LaTeX:

```latex
\paragraph{Method overview.}
The method follows directly from the measurability criterion. First, the owner compiles a committed payload into public bucket identities whose carrier events are first-token measurable. Second, training optimizes the mass of the target bucket at each carrier slot, aligning the loss with the object decoded by the verifier. Third, the public verifier maps recovered bucket tuples to self-synchronizing Reed--Solomon symbols and accepts only when the committed payload is recovered. Each component serves a specific role: tokenizer alignment makes the event one-step exact, bucket-mass training controls the verifier-relevant probability, and the coding layer provides conditional recovery under bounded symbol errors and erasures.
```

### Edit 2: Replace “pipeline” language with “framework” language

Risky:
```latex
Our pipeline guarantees robust ownership verification.
```

Safer:
```latex
Under the stated tokenizer, parser, and bounded-corruption assumptions, the framework gives a deterministic recovery rule for the declared audit protocol.
```

Risky:
```latex
The decoder is robust to text perturbations.
```

Safer:
```latex
The decoder absorbs bucket-preserving variation and reduces remaining parseable corruption to symbol errors and erasures.
```

### Edit 3: Add Module Input/Output Descriptions

For each module, add a short input/output paragraph.

#### Module A: Payload compilation

```latex
\paragraph{Payload compilation.}
Input: a committed payload $m^\star$, public metadata, and Reed--Solomon parameters $(n,k,q)$. Output: a codeword $C=(c_1,\ldots,c_n)$ and coordinate-symbol pairs $(i,c_i)$. This stage fixes the owner-controlled object to be recovered; it does not depend on suspect-service outputs.
```

#### Module B: Bucket mapping

```latex
\paragraph{Bucket mapping.}
Input: coordinate-symbol pairs and public bucket counts. Output: bucket-ID tuples produced by an injective mixed-radix map. This stage turns code symbols into verifier-decoded bucket identities; it is not a learned classifier.
```

#### Module C: Carrier realization

```latex
\paragraph{Carrier realization.}
Input: bucket-ID tuples and tokenizer-audited carrier vocabularies. Output: valid textual carrier representatives. The verifier treats all representatives in the same bucket as equivalent, so representative choice should not define ownership success.
```

#### Module D: Training objective

```latex
\paragraph{Training objective.}
Input: probe examples with carrier slots and target bucket IDs. Output: a model or adapter trained to assign mass to the target bucket at each slot. The optimization target is bucket mass, not exact lexical identity. \textbf{NEEDS_RESULTS:} empirical utility and distortion effects must be reported separately.
```

#### Module E: Verification

```latex
\paragraph{Verification.}
Input: suspect output text and the public verifier specification. Output: reject, or a recovered payload that matches the commitment. The verifier uses only observable text and public decoding rules; it does not inspect weights, logits, hidden states, or private judge models.
```

### Edit 4: Add Algorithm Boxes

Codex must check whether the repo already uses an algorithm package.

Search:

```bash
rg -n "usepackage\\{algorithm|usepackage\\{algorithmic|usepackage\\{algpseudocode|usepackage\\{algorithm2e" .
```

If algorithm packages are present, use them. If not, either:
- add a package only if compile-safe; or
- use a table/enumerate box instead.

#### Algorithm 1: Evidence Compilation and Training

Suggested LaTeX using `algorithm` + `algpseudocode`:

```latex
\begin{algorithm}[t]
\caption{Tokenizer-aligned evidence compilation and training}
\label{alg:compile-train}
\begin{algorithmic}[1]
\Require Payload $m^\star$, tokenizer $\mathrm{Tok}$, carrier vocabularies $\{V_f\}$, bucket partitions, RS parameters $(n,k,q)$, probe set $\mathcal{X}_{\mathrm{probe}}$
\Ensure Trained protected model or adapter
\State Audit carrier vocabularies and remove unstable or non-single-token representatives.
\State Encode $m^\star$ into an RS codeword $C=(c_1,\ldots,c_n)$.
\For{each coordinate-symbol pair $(i,c_i)$}
    \State Map $(i,c_i)$ to a bucket-ID tuple via the public mixed-radix map.
    \State Select carrier representatives only to instantiate teacher-forced text.
\EndFor
\State Train on probe examples with task loss on $y_{\mathrm{org}}$ and bucket-mass loss on carrier slots.
\State Return the trained model or adapter and the public verifier specification.
\end{algorithmic}
\end{algorithm}
```

If algorithmic package not available, convert to:

```latex
\begin{table}[t]
\centering
\small
\fbox{\begin{minipage}{0.95\linewidth}
\textbf{Algorithm 1: Tokenizer-aligned evidence compilation and training.}
\begin{enumerate}
    \item Audit carrier vocabularies under the public tokenizer.
    \item Encode payload $m^\star$ with RS parameters.
    \item Map coordinate-symbol pairs to bucket-ID tuples.
    \item Instantiate valid carrier text for teacher forcing.
    \item Train bucket mass at carrier slots while preserving ordinary task loss.
\end{enumerate}
\end{minipage}}
\caption{Compilation and training procedure.}
\label{alg:compile-train-box}
\end{table}
```

#### Algorithm 2: Public Verification

```latex
\begin{algorithm}[t]
\caption{Public bucket/RS verification}
\label{alg:verify}
\begin{algorithmic}[1]
\Require Suspect transcripts $\{(x_j,\hat{y}_j)\}_{j=1}^{N_{\mathrm{query}}}$, public parser, bucket map, RS parameters, committed payload $m^\star$
\Ensure Accept or reject
\For{each output $\hat{y}_j$}
    \State Extract carrier candidates using the public parser and candidate caps.
    \State Decode carrier representatives into bucket IDs.
    \State Map valid bucket tuples into coordinate-symbol pairs.
    \State Resolve duplicate coordinates using the public rule.
    \State Attempt RS error/erasure decoding.
    \If{decoded payload equals $m^\star$ and passes metadata/checksum tests}
        \State \Return Accept
    \EndIf
\EndFor
\State \Return Reject
\end{algorithmic}
\end{algorithm}
```

### Edit 5: Add Running Example in Method

Use the same example as Introduction. Keep it minimal.

```latex
\paragraph{Running example.}
For intuition, suppose a carrier field has a public bucket $\{\texttt{update},\texttt{review}\}$ whose members are stable single tokens and decode to the same bucket identity. Training a canonical representative such as \texttt{update} would over-specify the verifier's requirement; training the bucket mass allows either representative to carry the same symbol. By contrast, a multi-token phrase whose validity depends on future tokens would not provide an exact one-step bucket mass.
```

Codex must verify that the example tokens are consistent with the actual carrier catalog or mark the example as illustrative:
```latex
The following example is illustrative and not tied to a particular experimental catalog.
```

### Edit 6: Clarify What The Method Does Not Claim

Add a short method-scope paragraph:

```latex
\paragraph{Scope of the method.}
The method does not prevent a serving adversary from deleting the carrier block or destroying the parse structure. It also does not imply that a student model trained only on organic non-probe data will inherit the evidence behavior. These cases are excluded by the audit protocol and evaluated, when applicable, as failure boundaries rather than robustness successes.
```

## 4. Complexity / Overhead Discussion

Add a small paragraph if space allows.

Do not overclaim runtime. Use symbolic complexity unless actual measurements exist.

```latex
\paragraph{Verifier cost.}
The public verifier is lightweight relative to model inference. Its cost is dominated by parsing candidate carrier fields and attempting RS decoding over capped candidate windows. With a fixed query budget, parser cap, and window budget, verification cost is bounded by the number of extracted candidates rather than by the size of the suspect model. \textbf{NEEDS_RESULTS:} report wall-clock or compute overhead only if measured.
```

## 5. Method Claims That Cannot Be Written Yet

Do not write:

1. `The method preserves utility.`
   - Needs utility experiments.

2. `The method is robust to post-processing.`
   - Current status only supports conditional boundary.

3. `The method beats fingerprinting baselines.`
   - Clean is parity with Perinucleus; superiority needs calibrated axes.

4. `The method generalizes to all tokenizers.`
   - R1 exact gate fails; needs repair or diagnostic framing.

5. `The method gives cryptographic ownership proof.`
   - No cryptographic proof.

## 6. Codex Execution Checklist

- [ ] Method overview added.
- [ ] Each module has input/output.
- [ ] Algorithm 1 added or box equivalent added.
- [ ] Algorithm 2 added or box equivalent added.
- [ ] No broad robustness claim remains.
- [ ] Bucket-mass objective described as verifier-aligned, not just grouped CE.
- [ ] RS described as standard coding layer, not novel coding theorem.
- [ ] Implementation details moved to appendix if too long.
- [ ] All claims involving performance marked `NEEDS_RESULTS` unless supported.
- [ ] LaTeX compiles.
