# Expert Plan: Variable-Arity Natural Evidence Capacity

Date: 2026-05-06

This document records the expert decision after the Qwen actual-prefix
higher-cap suffix compatibility diagnostic. It is an internal execution plan,
not a manuscript claim update.

## Executive Decision

The current natural-output route is not close to passing the E2E gate. It has
shown that actual generated prefixes can be scored with the reference model and
that some prefixes can be bucketized, but it has not shown trainable,
recoverable, auditable natural-output evidence.

The main blocker is compatible diversity across buckets. Raising
`max_candidates_per_bucket` helps only modestly:

| Diagnostic | Value |
|---|---:|
| actual-prefix scoring rows | 57164 |
| observed token in top-k rate | 1.0 |
| strict 4-way bucketized entries | 4996 |
| direct cap=16 min1-compatible entries | 1235 |
| direct cap=16 configured-min entries | 224 |
| direct cap=16 probability-gated entries | 86 |
| comparison cap=16 min1-compatible entries | 1202 |
| comparison cap=16 configured-min entries | 220 |
| comparison cap=16 missing-compatible-bucket entries | 3617 |
| comparison cap=16 diagnostic min1 bits/response | 0.16768973214285715 |

This blocks training, E2E reruns, Llama, Qwen 8-way mainline runs, and full
matrix experiments.

## Claims Still Forbidden

- Natural-output payload recovery.
- Natural-output success.
- Full FAR.
- Cross-family generality.
- Robustness to all paraphrasing or sanitization.
- Stealth guarantee.
- Superiority over Scalable/Perinucleus.
- Treating bucket-bank entries as fingerprints.
- Treating old exact-slot or carrier-slot success as natural-output success.

## Current Interpretation

The failure is not top-k coverage. The observed generated token is covered by
the reference top-k diagnostic.

The failure is not simply that observed tokens cannot be bucketized. Among
accepted bucketized entries, observed-token bucketization is high.

The failure is that fixed 4-way evidence positions often lack enough
suffix-compatible alternatives in natural language contexts. Natural local
choice is often binary or ternary rather than uniformly 4-way.

## Required Method Pivot

The main method should pivot from:

```text
fixed 4-way natural buckets
```

to:

```text
compatibility-adjusted variable-arity natural evidence capacity
```

The evidence unit should be an actual-prefix, context-conditioned opportunity
whose arity is determined after compatibility filtering:

- 2 compatible buckets: contributes 1 bit.
- 3 compatible buckets: contributes `log2(3)` bits or enters a mixed-radix
  diagnostic.
- 4 compatible buckets: remains the aspirational fixed-4-way case.
- 8 compatible buckets: ablation only, not a mainline gate.

## Execution Plan

### 1. Reconcile High-Cap Accounting

Goal: explain the direct summary versus merged comparison discrepancy before
using high-cap numbers in any paper-facing table.

Finding:

- Direct cap=8 and cap=16 by-entry CSV files each contain 4962 rows but only
  4819 unique `bank_entry_id` values.
- There are 137 duplicated `bank_entry_id` values, with 143 duplicate extra
  rows and max duplicate count 4.
- `summarize_actual_prefix_suffix_sensitivity.py` currently loads by-entry CSVs
  into a dictionary keyed only by `bank_entry_id`, which drops duplicate rows.
- This explains why the merged comparison summary undercounts relative to the
  direct cap summary.

Action started:

- The reducer was patched to use an actual-prefix composite key:
  `bank_entry_id`, `prompt_id`, `prompt_split`, `model_condition`,
  `payload_id`, `seed`, `query_index`, `generated_row_index`, and
  `position_index`.
- A focused regression test now covers duplicate `bank_entry_id` rows at
  different actual generated prefixes.
- A local temp recomputation with the patched reducer matches the direct cap
  summaries:
  - cap=4: min1=853, configured-min=117, probability-gated=59
  - cap=8: min1=1082, configured-min=176, probability-gated=75
  - cap=16: min1=1235, configured-min=224, probability-gated=86

Next engineering action:

- Recompute and archive the high-cap sensitivity summary artifact with the
  patched reducer under a non-overwriting versioned output directory.
- Keep the old copied artifact as historical evidence from job 845358.

This accounting issue does not change the current gate decision: both direct
and merged cap=16 counts remain below the expert thresholds.

### 2. Implement Variable-Arity Diagnostic Constructor

Input should reuse existing actual-prefix suffix compatibility artifacts first.
No new model scoring is needed for the first implementation.

Proposed flow:

```text
actual-prefix bucketized candidates
-> suffix compatibility rows
-> compatible token surface filtering
-> compatible bucket availability
-> variable-arity evidence-position construction
-> capacity and density audit
```

Required outputs:

- `variable_arity_bank_entries.jsonl`
- `variable_arity_manifest.json`
- `arity_distribution.csv`
- `effective_bits_per_response.csv`
- `eligible_density_by_split.csv`
- `variable_arity_rejections.csv`

Minimum reported metrics:

- accepted variable-arity entries
- arity distribution
- effective bits per response
- eligible positions per 100 tokens
- held-out density
- organic density when available
- reconstructability proxy
- raw accidental observation risk
- wrong-key and wrong-payload pre-null readiness

### 3. Change Construction Order

Do not static-bucketize first and then filter compatibility as the main method.
That ordering overstates usable capacity.

The preferred construction order is:

```text
actual-prefix candidates
-> compatibility scoring
-> token functional class filtering
-> compatibility-aware clustering or bucket grouping
-> variable-arity bank construction
-> post-compatibility capacity audit
```

The first pass may reuse current fixed-bucket compatibility rows as a diagnostic
input, but the next mainline implementation should construct after
compatibility filtering.

### 4. Add Token Functional Class Filtering

Filter or tag candidates whose surface form makes them poor natural evidence
tokens:

- formatting-only artifacts
- extreme punctuation
- brittle whitespace-only alternatives
- obvious old-route surfaces
- low semantic substitutability candidates

The old-route forbidden surfaces remain:

- `FIELD=`
- `SECTION=`
- `TOPIC=`
- `PAYLOAD`
- `CERT`
- `EVIDENCE`
- carrier block
- structured evidence block

Every rejection should have a machine-readable reason.

### 5. Add Branch-Aware Compatibility Diagnostic

Suffix-preserving compatibility may be too conservative for free generation.

Add a second diagnostic:

```text
prefix + candidate
-> short reference continuation
-> local naturalness/coherence score
```

Required report:

- suffix-preserving pass rate
- branch-aware pass rate
- examples where suffix-preserving fails but branch-aware passes
- examples where both fail

This diagnostic should not launch E2E training. If it needs model continuation
on Chimera, it must be submitted through Slurm.

### 6. Finish Invalid Suffix Record Policy

Current evidence says invalid suffix rows are mostly boundary/no-suffix cases.
Before using any compatibility result in a manuscript-facing table, produce a
final reason table and examples:

- reason counts
- representative examples
- tokenizer offset bug suspected: true/false
- boundary/EOS exclusion policy

If an offset bug is found, fix and rescore. If records are genuine boundary
cases, exclude and document.

### 7. Strengthen Protocol Commitment

Before any new E2E run, precommit:

- key
- payload
- selector policy
- bucket or variable-arity policy
- query budget
- decode threshold
- allowed number of keys/payloads
- multiple-testing correction or held-out lockbox policy

No post-hoc key, payload, threshold, or opportunity search is allowed.

### 8. Qwen Proof-of-Life Gate

Only prepare a Qwen proof-of-life E2E after all of these pass:

| Gate | Minimum |
|---|---:|
| variable-arity compatible entries | >= 2000 |
| effective bits per response | >= 0.8 |
| held-out density | >= 0.5 / 100 tokens |
| high-quality configured subset | >= 500 |
| raw pre-null | no obvious accidental accept |
| task-only LoRA null plan | exists |
| wrong-key / wrong-payload null plan | exists |

If these pass, prepare but do not casually broaden the run. Qwen proof-of-life
minimum arms:

- Qwen protected natural bucket-mass LoRA
- Qwen raw
- Qwen task-only LoRA
- wrong key
- wrong payload

Recommended query budgets for this proof-of-life stage:

- `[64, 128, 256, 512]`

### 9. Llama, Same-Family Nulls, and Sanitizer Benchmark

Do not start Llama until Qwen shows payload recovery and null rejection.

After Qwen, plan same-family near-null checks:

- raw Qwen 3B
- raw Qwen 7B
- raw Qwen 14B or 8B if available
- raw Llama variants if available
- one unrelated raw family

Only after clean E2E recovery should sanitizer benchmarking begin:

- no attack
- generic paraphrase
- style normalization
- compression / summarization
- low-temperature regeneration
- suffix truncation
- public surface scrub
- oracle keyed sanitizer after key reveal

Do not claim unconditional robustness. Oracle keyed sanitizer is an upper bound,
not a standard threat model.

## Immediate Next Actions

1. Archive a corrected high-cap reducer output in a versioned local status
   directory.
2. Implement a CPU-local variable-arity diagnostic over existing actual-prefix
   compatibility artifacts.
3. Add focused tests for variable-arity arity/capacity accounting.
4. Run only local validation first. Submit Slurm only if a future diagnostic
   requires model scoring or continuation on Chimera.

## Current No-Go Decision

No training or E2E rerun is allowed from the current cap=16 result. The current
result is evidence for a mechanism bottleneck, not a natural-output success and
not a final impossibility result.
