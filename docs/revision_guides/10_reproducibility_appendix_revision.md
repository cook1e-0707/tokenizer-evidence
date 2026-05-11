# Reproducibility and Appendix Revision Guide

## 1. Purpose

The current manuscript checklist indicates that several reproducibility-related items may still be incomplete. This is dangerous for a complex pipeline paper.

This guide instructs Codex to add or prepare appendix material that improves credibility without inventing results.

## 2. Why This Matters

NeurIPS reviewers will care about:

1. Can another expert understand the audit protocol?
2. Are tokenizer, carriers, parser, and decoder specified?
3. Are training hyperparameters specified?
4. Are null families for FAR specified?
5. Are baseline adaptations clearly labelled?
6. Are compute resources reported?
7. Are licenses and assets documented?
8. Are limitations and broader impacts discussed?

A paper that depends on tokenizer audit, carrier vocabulary, parser caps, RS parameters, and baseline adaptation must be especially explicit.

## 3. Codex Search Instructions

```bash
rg -n "Checklist|Reproducibility|compute|license|asset|hyperparameter|LoRA|optimizer|seed|FAR|null|baseline_protocol|calibration_protocol" .
rg -n "Qwen|Llama|Perinucleus|Scalable|KGW|PostMark|TinyBench|LoRA|A100|H200" .
find . -maxdepth 4 -type f \( -name "*.csv" -o -name "*.json" -o -name "*.md" \) | sort | head -200
```

Do not copy private paths into main text if they break anonymity. Use appendix descriptions rather than identifying cluster/user paths unless anonymized.

## 4. Required Appendix Sections

Add or revise appendix sections:

```latex
\section{Reproducibility Details}
\label{app:reproducibility}

\subsection{Tokenizer and Carrier Audit}
\subsection{Payload, Bucket, and Code Parameters}
\subsection{Training Setup}
\subsection{Generation and Verification Setup}
\subsection{Baseline Adaptation Protocol}
\subsection{False-Accept Calibration Protocol}
\subsection{Compute Resources}
\subsection{Assets, Licenses, and Terms}
\subsection{Broader Impact and Misuse Considerations}
```

If page limit is tight, use supplementary material.

## 5. Tokenizer and Carrier Audit

Suggested LaTeX:

```latex
\subsection{Tokenizer and Carrier Audit}
The verifier depends on a fixed tokenizer version and public carrier vocabularies. For each carrier vocabulary, we audit candidate representatives by tokenization, detokenization stability, parser compatibility, and exclusion rules for delimiter-like strings, control characters, leading-whitespace variants, and visually unstable forms. \textbf{PLACEHOLDER:} Insert tokenizer version, audit script path if anonymized, and final carrier-vocabulary summary.
```

## 6. Payload, Bucket, and Code Parameters

```latex
\subsection{Payload, Bucket, and Code Parameters}
For each reported experiment, we specify the payload label set, number of carrier blocks, field types, bucket counts, mixed-radix map, Reed--Solomon parameters, parser cap, candidate-window budget, and accept rule. \textbf{PLACEHOLDER:} Insert a compact table of parameters for the main Qwen compiled-path experiments and any cross-family diagnostics.
```

Suggested table:

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{lllll}
\toprule
Setting & Payloads & Blocks/fields & Bucket geometry & Decoder parameters \\
\midrule
Qwen compiled path & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} \\
Cross-family diagnostic & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} \\
\bottomrule
\end{tabular}
\caption{Protocol parameters for reported experiments. Values must be filled only from verified artifacts.}
\label{tab:protocol-parameters}
\end{table}
```

## 7. Training Setup

```latex
\subsection{Training Setup}
\textbf{PLACEHOLDER:} Report model name, tokenizer, adapter recipe, optimizer, learning rate, batch size, epochs, sequence length, seeds, data size, and selection criteria. Do not report unverified values.
```

If current project values are verified in artifacts, insert them. Otherwise leave placeholder.

Suggested table:

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{ll}
\toprule
Item & Value \\
\midrule
Base model & \textbf{PLACEHOLDER} \\
Tokenizer & \textbf{PLACEHOLDER} \\
Adapter & \textbf{PLACEHOLDER} \\
Learning rate & \textbf{PLACEHOLDER} \\
Batch size & \textbf{PLACEHOLDER} \\
Epochs & \textbf{PLACEHOLDER} \\
Max sequence length & \textbf{PLACEHOLDER} \\
Seeds & \textbf{PLACEHOLDER} \\
\bottomrule
\end{tabular}
\caption{Training configuration for the main reported setting.}
\label{tab:training-config}
\end{table}
```

## 8. Generation and Verification Setup

```latex
\subsection{Generation and Verification Setup}
We report the prompt family, generation parameters, number of generated tokens at carrier slots, parser rules, candidate caps, query budget, and accept threshold. \textbf{PLACEHOLDER:} Insert exact generation and verification settings from the final artifact registry.
```

Important:
- If exact-slot compiled packages use one token per slot, say so.
- Do not hide this setting.
- If full-response experiments are not done, mark as `NEEDS_RESULTS`.

## 9. Baseline Adaptation Protocol

```latex
\subsection{Baseline Adaptation Protocol}
External baselines are reported only when their adaptation fidelity is audited. In particular, the Scalable/Perinucleus baseline must be labelled as Qwen-adapted if evaluated on the Qwen compiled setting. Task-mismatched provenance controls are reported separately from primary ownership baselines.
```

Add:

```latex
\textbf{TODO_AFTER_RESULTS:} Insert final matched-budget table including query budget, clean recovery, FAR, utility, and compute where complete.
```

## 10. False-Accept Calibration Protocol

```latex
\subsection{False-Accept Calibration Protocol}
False accepts are measured under declared null families. The protocol should specify registered-probe nulls, organic prompt-bank nulls, non-owner probes, query budgets, and confidence interval method. \textbf{NEEDS_RESULTS:} Do not report final FAR conclusions until all required null-family shards are complete and aggregated.
```

Add warning:

```latex
Deprecated partial null outputs must not be aggregated into final FAR estimates.
```

## 11. Compute Resources

```latex
\subsection{Compute Resources}
\textbf{PLACEHOLDER:} Report worker type, accelerator type, CPU count, memory, wall-clock request, approximate consumed time if available, and total tracked compute for each experiment group. Distinguish requested allocation from consumed wall-clock utilization.
```

Suggested table:

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{lllll}
\toprule
Experiment group & Accelerator & CPU/RAM & Wall-clock & Notes \\
\midrule
Clean compiled path & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} \\
Robustness attacks & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} \\
Baselines & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} & \textbf{PLACEHOLDER} \\
\bottomrule
\end{tabular}
\caption{Compute resources for reported experiments.}
\label{tab:compute}
\end{table}
```

## 12. Assets, Licenses, and Terms

```latex
\subsection{Assets, Licenses, and Terms}
\textbf{PLACEHOLDER:} List all base models, tokenizers, datasets, code packages, and baselines used in the experiments, with version, license, citation, and terms-of-use status where available. Do not include non-anonymized private repository paths in the submission.
```

## 13. Broader Impact

```latex
\subsection{Broader Impact and Misuse Considerations}
Ownership-audit tools can support accountability for unauthorized model reuse, but they also carry risks if used to make unsupported accusations. The audit protocol therefore requires declared null families, query budgets, and false-accept calibration. The method also introduces an owner-controlled evidence channel, so deployment should consider disclosure, access control, and safeguards against misuse of probe mechanisms.
```

## 14. Checklist Updates

Only update checklist answers from `No` to `Yes` if the manuscript actually contains the needed details.

Do not change checklist for appearance.

| Checklist item | When Yes is allowed |
|---|---|
| Experimental reproducibility | main or appendix has enough details to reproduce key results |
| Open access to code/data | anonymized code/data or sufficient instructions are provided |
| Experimental setting/details | hyperparameters, data, prompts, verifier settings included |
| Statistical significance | confidence intervals/error bars and variability unit included |
| Compute resources | worker type, memory, wall-clock/compute reported |
| Broader impacts | positive and negative impacts discussed |
| Licenses | models/datasets/code licenses documented |

## 15. Codex Checklist

- [ ] Reproducibility appendix exists.
- [ ] Tokenizer audit described.
- [ ] Carrier exclusion rules described.
- [ ] Protocol parameters described.
- [ ] Training hyperparameters described or marked `PLACEHOLDER`.
- [ ] Generation settings described.
- [ ] Verification settings described.
- [ ] Baseline adaptation labels included.
- [ ] FAR protocol described and incomplete axes marked.
- [ ] Compute resources described.
- [ ] Assets and licenses described.
- [ ] Broader impact included.
- [ ] Checklist answers updated only if justified.
- [ ] LaTeX compiles.
