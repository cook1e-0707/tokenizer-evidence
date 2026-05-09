# R1 selector-contract repair analysis

This document records the artifact-only interpretation of Phase R1
prefix-conditioned selector replay. It is not payload recovery, not FAR, and not
a paper-facing positive claim.

## Inputs

Primary R1 replay artifacts:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_prefix_conditioned_selector_replay/
```

R1 interpretation artifacts:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_r1_selector_contract_analysis/
```

Files:

```text
r1_selector_contract_repair_summary.json
r1_selector_contract_pairwise_lift.csv
r1_selector_contract_by_policy_budget.csv
r1_selector_contract_repair_analysis.md
```

## Decision

Status:

```text
COMPLETE_R1_ANALYSIS_NO_PROTECTED_LIFT_OVER_NULLS
```

The prefix-conditioned replay confirms that strict token-index anchoring was
discarding many observable prefix-conditioned events. It does not produce an
ownership-specific signal because protected rows do not beat raw or task-only
null behavior.

Across 64 protected-vs-null comparison slices:

| Comparison | Positive protected-lift rows |
|---|---:|
| protected vs raw | 0 / 64 |
| protected vs task-only | 0 / 64 |

Therefore direct replay-verifier use is blocked, new training is blocked, and
Qwen E2E rerun is blocked.

## Budget 512 Lift Summary

| Policy | mean protected target | mean raw target | mean task-only target | protected-raw delta | protected-task delta | positive vs raw | positive vs task-only | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| exact_full | 0.020089 | 0.384905 | 0.113979 | -0.364816 | -0.093890 | 0/4 | 0/4 | FAIL_NO_SLICE_HAS_PROTECTED_LIFT |
| suffix_16 | 0.021973 | 0.385882 | 0.118652 | -0.363909 | -0.096680 | 0/4 | 0/4 | FAIL_NO_SLICE_HAS_PROTECTED_LIFT |
| suffix_32 | 0.020438 | 0.384905 | 0.114397 | -0.364467 | -0.093959 | 0/4 | 0/4 | FAIL_NO_SLICE_HAS_PROTECTED_LIFT |
| suffix_8 | 0.030134 | 0.386440 | 0.130999 | -0.356306 | -0.100865 | 0/4 | 0/4 | FAIL_NO_SLICE_HAS_PROTECTED_LIFT |

## Interpretation

R1 separates two facts that were previously entangled:

1. Prefix-conditioned event rediscovery is possible.
   The replay can recover many actual-prefix coordinates, especially for raw
   outputs. This supports the critique of strict token-index anchoring.

2. The current protected training does not create ownership signal.
   Raw and task-only outputs produce higher coordinate-level target-hit rates
   than protected outputs. This means R1 target hits are not verifier accepts
   and cannot be used as evidence of ownership.

The highest protected-minus-task-only slice is still negative. This rules out
choosing a favorable match policy, budget, payload, or seed from R1 as a repair.
Any such choice would be post-hoc selection and would worsen multiple-testing
risk.

## Selector-Contract Requirements

Any future prefix-conditioned selector must be fixed before generation. The
commitment must include at least:

- selector id and match policy;
- reference tokenizer and reference model;
- bucket policy and candidate filtering policy;
- audit key id;
- payload id;
- query budget;
- prompt split and prompt sampling rule;
- thresholds and decode rule;
- allowed number of payloads, keys, policies, and thresholds.

R1 cannot be used to choose a successful policy post hoc. A new selector must be
evaluated on a locked replay or fresh lockbox.

## Repair Implications

- Do not use R1 target-hit counts as verifier accepts.
- Do not select match policy, payload, seed, or threshold post hoc from this
  replay.
- Repair training targets before any new Qwen run. Current protected outputs do
  not preserve or create the prefix-conditioned target event distribution better
  than nulls.
- Treat branch-aware or regenerated-suffix training data as a required preflight
  candidate, not an optional cleanup.
- Keep sparse coordinate-level coding as the likely decoder-side repair after
  coordinate survival shows owner-specific lift.

## Next Allowed Action

Artifact-only selector-contract precommit design and branch-aware/regenerated-
suffix training-target preflight.

Forbidden:

- no training;
- no generation;
- no E2E rerun;
- no Llama;
- no same-family null;
- no sanitizer benchmark;
- no manuscript positive claim.
