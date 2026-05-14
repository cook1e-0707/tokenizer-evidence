# R4 Candidate v3 Rebalanced Micro-Overfit H200 Route

Timestamp UTC: 2026-05-14T04:25:00Z

## Decision

Prepare exactly one future H200/pomplun Slurm submission for an intermediate
pressure/concentration rebalance diagnostic.

This route is teacher-forced protected micro-overfit training/scoring only. It
is not generation, not Qwen E2E, not Llama, not same-family null, not sanitizer,
not FAR aggregation, not payload diversity, and not a paper-facing positive
claim.

## Motivation

The two completed bracket runs show opposite failures:

- job `857458`: main teacher-forced gate passed, but max surface mean target
  mass `0.643060527741909` exceeded the `0.50` concentration cap;
- job `857611`: concentration cap passed with max surface mean target mass
  `0.22298632306046784`, but protected lift vs base was
  `0.13019710095503`, missing the `+0.15` gate by
  `0.019802899044969985`.

The next diagnostic should relax the ceiling penalty without removing
concentration control.

## Route Parameters

Wrapper:

`scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch`

Submission environment:

```bash
TARGET_MASS_CEILING=0.50 \
TARGET_MASS_CEILING_LAMBDA=2.0 \
sbatch scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
```

Objective controls:

- `TARGET_MASS_FLOOR=0.20`;
- `TARGET_MASS_FLOOR_LAMBDA=5.0`;
- `TARGET_MASS_CEILING=0.50`;
- `TARGET_MASS_CEILING_LAMBDA=2.0`;
- `STRATUM_WEIGHTING_MODE=r4_candidate_v3_failed_surface`;
- `STRATUM_WEIGHT_MAX=3.0`.

## Preconditions Already Passed Locally

- wrapper supports the ceiling knobs;
- local plan-only wrapper mode passed with the rebalance values and exited
  before model/tokenizer loading, CUDA, adapter loading, training, scoring,
  remote sync, or Slurm submission;
- zero-enabled allowlist safety passed.

## Future Submission Gate

Before submission:

1. sync this route doc, state, and control-plane files to Chimera;
2. run remote wrapper plan-only validation with the rebalance values;
3. run remote zero-enabled allowlist safety;
4. confirm no active Chimera jobs;
5. send Hermes TG/email pre-submit notification;
6. enable exactly one allowlist entry:
   `v2_r4_candidate_v3_micro_overfit_h200`;
7. submit exactly one H200/pomplun Slurm job;
8. immediately disable the allowlist and re-run local/remote allowlist safety.

## Pass Gate

The rebalanced run must satisfy:

- protected lift vs base `>= +0.15`;
- protected lift vs task-only `>= +0.10`;
- protected rank1 `>= 0.75`;
- protected median margin `> 0`;
- task-only lift vs base not positive;
- max surface mean target mass `<= 0.50`;
- no target/other token-id overlap;
- no scorer boundary failures.

Only a pass on both the main teacher-forced gate and concentration cap can
unlock a separate generation diagnostic route.
