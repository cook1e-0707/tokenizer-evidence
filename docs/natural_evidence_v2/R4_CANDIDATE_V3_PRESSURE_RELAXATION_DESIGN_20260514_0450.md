# R4 Candidate v3 Pressure-Relaxation Design

Timestamp UTC: 2026-05-14T04:50:00Z

## Decision

Record an artifact-only design for a bounded post-rebalance pressure-relaxation
diagnostic. This design does not authorize Slurm submission, remote preflight,
training, scoring, generation, Qwen E2E rerun, Llama, same-family null,
sanitizer benchmark, FAR aggregation, payload-diversity work, or paper-facing
positive claims.

The controlling spec context is the Stage 4 real-integration contract plus the
active `natural_evidence_v2` protocol and claim guardrails. The relevant current
state is the post-rebalance teacher-forced failure: job `857653` missed the
protected lift-vs-base gate by `0.007125643499983875` while keeping max surface
mean target mass at `0.190783511439804`, well below the `0.50` concentration
cap.

## Fixed Diagnostic Grid

If a later route review explicitly opens a launch path, the pressure-relaxation
diagnostic should use only this fixed two-arm grid:

| arm | target mass floor | floor lambda | target mass ceiling | ceiling lambda | stratum weighting | max stratum weight |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| A | `0.20` | `5.0` | `0.50` | `1.0` | `r4_candidate_v3_failed_surface` | `3.0` |
| B | `0.20` | `5.0` | `0.50` | `0.5` | `r4_candidate_v3_failed_surface` | `3.0` |

The grid relaxes only the ceiling penalty strength. It does not raise the
post-hoc concentration cap and does not change the training rows, heldout rows,
score rows, row mode, tokenizer boundary assumptions, or scorer gates.

## Bounds And Stop Rules

- No additional arms may be added after seeing either arm's result.
- No single-point follow-up may be inferred from this artifact.
- Both arms are diagnostic only unless a separate terminal review records the
  preregistered teacher-forced gate and concentration-cap results.
- The concentration cap remains max surface mean target mass `<= 0.50`.
- A passing arm would at most permit a separate artifact-only generation-route
  review. It would not directly start generation or Qwen E2E rerun.

## Required Later Review Before Any Compute

A later tick must still review and record, before any remote preflight or Slurm
submission:

- the exact wrapper or array route that would run the fixed grid;
- one disabled-by-default allowlist entry, or a stricter equivalent route
  guard;
- plan-only behavior that exits before model/tokenizer loading, CUDA, adapter
  loading, training, scoring, remote sync, or Slurm submission;
- local and remote zero-enabled allowlist safety;
- no active Chimera jobs;
- Hermes Telegram/email pre-submit notification;
- immediate allowlist disablement and local/remote post-submit safety if a
  later submission is explicitly allowed.

## Locked Actions

Still locked from this design:

- Slurm submission;
- generation and Qwen E2E rerun;
- Llama;
- same-family null;
- sanitizer benchmark;
- FAR aggregation;
- payload-diversity claim;
- paper-facing positive claims.

Any future Chimera tokenizer/model/CPU/GPU work must use Slurm and must not run
directly on the Chimera login node.

## Next Allowed Action

Artifact-only objective/route review for the fixed two-arm
pressure-relaxation diagnostic above. No Slurm submission, generation, Qwen
E2E rerun, Llama, null/FAR, sanitizer, payload diversity, or paper-claim work.
