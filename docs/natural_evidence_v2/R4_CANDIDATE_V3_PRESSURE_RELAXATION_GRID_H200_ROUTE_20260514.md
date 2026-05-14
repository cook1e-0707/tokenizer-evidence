# R4 Candidate v3 Pressure-Relaxation Grid H200 Route

Timestamp UTC: 2026-05-14T05:00:00Z

## Decision

Prepare exactly one H200/pomplun Slurm array submission for the fixed two-arm
post-rebalance pressure-relaxation diagnostic recorded at `04:50Z`.

This route is teacher-forced protected micro-overfit training/scoring only. It
is not generation, not Qwen E2E, not Llama, not same-family null, not sanitizer,
not FAR aggregation, not payload diversity, and not a paper-facing positive
claim.

## Fixed Grid

The grid is locked to two arms:

| arm | target mass floor | floor lambda | target mass ceiling | ceiling lambda | stratum weighting | max stratum weight |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| A | `0.20` | `5.0` | `0.50` | `1.0` | `r4_candidate_v3_failed_surface` | `3.0` |
| B | `0.20` | `5.0` | `0.50` | `0.5` | `r4_candidate_v3_failed_surface` | `3.0` |

No arms may be added after seeing either result. The concentration cap remains
max surface mean target mass `<= 0.50`.

## Wrapper

Wrapper:

`scripts/natural_evidence_v2/slurm/r4_candidate_v3_pressure_relaxation_grid_h200.sbatch`

The wrapper submits as one Slurm array:

```text
#SBATCH --array=0-1%2
```

Each array task calls the reviewed single-arm wrapper:

`scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch`

with a distinct output directory:

```text
r4_candidate_v3_pressure_relaxation_grid_${SLURM_ARRAY_JOB_ID}/A_ceiling_lambda_1_0
r4_candidate_v3_pressure_relaxation_grid_${SLURM_ARRAY_JOB_ID}/B_ceiling_lambda_0_5
```

## Allowlist

Added disabled-by-default allowlist entry:

`v2_r4_candidate_v3_pressure_relaxation_grid_h200`

Command pattern:

```bash
sbatch scripts/natural_evidence_v2/slurm/r4_candidate_v3_pressure_relaxation_grid_h200.sbatch
```

## Local Validation

No model/tokenizer loading, CUDA initialization, adapter loading, training,
scoring, remote sync, Slurm submission, generation, or claims occurred.

Passed:

- wrapper syntax: `bash -n`;
- plan-only wrapper mode for both arms A and B;
- zero-enabled allowlist safety;
- design summary and gate-status JSON parse checks.

## Future Submission Gate

Before any submission:

1. sync the reviewed route, wrapper, allowlist, and state files to Chimera;
2. run remote wrapper plan-only validation for both arms;
3. run remote zero-enabled allowlist safety;
4. confirm no active Chimera jobs;
5. send Hermes TG/email pre-submit notification;
6. enable exactly one allowlist entry:
   `v2_r4_candidate_v3_pressure_relaxation_grid_h200`;
7. submit exactly one H200/pomplun Slurm array job;
8. immediately disable the allowlist and run local/remote post-submit
   allowlist safety.

## Pass Gate For Each Arm

Each terminal arm review must report:

- protected lift vs base `>= +0.15`;
- protected lift vs task-only `>= +0.10`;
- protected rank1 `>= 0.75`;
- protected median margin `> 0`;
- task-only lift vs base not positive;
- max surface mean target mass `<= 0.50`;
- no target/other token-id overlap;
- no scorer boundary failures.

Only a passing arm can permit a separate artifact-only generation-route review.
It does not directly unlock generation.
