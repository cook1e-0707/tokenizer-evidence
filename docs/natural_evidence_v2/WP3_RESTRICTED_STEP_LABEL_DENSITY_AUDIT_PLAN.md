# WP3 Restricted Step-Label Density Audit Plan

## Scope

This document reviews the artifact-only density audit plan for the restricted
`Step N:` route. It is not model output, not model scoring, not training, not
E2E, not FAR, and not a paper-facing positive claim.

The plan is written here:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_density_audit_plan_20260508_2055/
```

Builder:

```text
scripts/natural_evidence_v2/build_wp3_restricted_step_label_density_audit_plan.py
```

## Decision

The immediate density route remains:

```text
A_16_step_checklist_step_label_only
```

This route uses only the two passing step-label banks from job `850398`:

```text
restricted_step_label_check_review_choose_make_v1
restricted_step_label_start_begin_create_set_v1
```

The alternative route remains blocked:

```text
B_8_step_plus_extra_slots
```

Reason: the 8-step route needs at least eight additional non-step slots, but
transition, hedge, and function-word banks have not yet passed the
context-specific model-mass gate.

## Audit Prompts

The artifact-only plan writes `256` prompt rows:

```text
restricted_step_label_density_audit_prompts.jsonl
```

Each prompt asks for a normal sixteen-step checklist using:

```text
Step 1:
...
Step 16:
```

The prompts do not name the restricted candidate surfaces and do not expose old
structured-carrier vocabulary. The audit is designed to test whether base Qwen
will naturally follow the requested structure, not whether raw Qwen already
emits the protected target surfaces.

## Gate Spec

The planned density gate is:

| Metric | Threshold | Meaning |
|---|---:|---|
| `complete_step_label_response_rate` | `>=0.95` | response contains `Step 1:` through `Step 16:` exactly once |
| `mean_detected_structural_slots_per_response` | `>=16.0` | detector can locate all planned anchors |
| `responses_with_at_least_16_structural_slots_rate` | `>=0.90` | density is not carried by a few unusually compliant outputs |
| `forbidden_public_surface_rate` | `==0.0` | no old structured-carrier vocabulary appears |
| `raw_bank_surface_hit_rate` | report only | accidental raw hits are a null-risk diagnostic |
| `naturalness_manual_review_examples` | `>=32` examples | sixteen-step control still reads as ordinary checklist prose |

Important distinction: raw candidate-bank hits are not a pass gate. A high raw
hit rate would increase accidental-accept risk; a low raw hit rate does not
block density if the structural anchors are stable.

## Slurm Requirement

The next actual model-output density audit must run through Chimera Slurm. Do
not run Qwen generation or tokenizer/model scoring on a Chimera login node.

The plan recommends a future wrapper name:

```text
scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch
```

No Slurm job was submitted by this artifact-only planning step.

## Still Blocked

Even if the density audit later passes, WP4 is not automatically allowed. WP4
still requires a reviewed prompt-local small payload contract and decoder-oracle
substitution.

Still forbidden:

- no training
- no protected E2E generation
- no Qwen proof-of-life E2E
- no Llama
- no same-family null
- no sanitizer benchmark
- no FAR aggregation
- no natural-output success claim
- no payload recovery claim
- no full FAR claim

## Next Allowed Action

Review this density audit plan. If approved, implement or review exactly one
Chimera Slurm wrapper that generates base-Qwen outputs for the 256 planned
prompts and runs the restricted detector. Do not start WP4 or training.
