# R4 Candidate v3 Post-Rebalance Route Decision

Timestamp UTC: 2026-05-14T04:37:00Z

## Decision

Record an artifact-only route decision after rebalanced H200 job `857653`.
Do not submit another Slurm job from this state.

The next project-advancing step, if continued, is artifact-only preparation or
review of a narrowly bounded pressure-relaxation route. That later artifact
must predeclare its objective knobs and safety checks before any remote
preflight or submission path is considered.

## Inputs Reviewed

- floor-only micro-overfit job `857458`: protected lift vs base
  `0.39194896617371455`, max surface mean target mass
  `0.643060527741909`, concentration cap `FAIL`;
- capped micro-overfit job `857611`: protected lift vs base
  `0.13019710095503`, gap to `+0.15` of `0.019802899044969985`,
  max surface mean target mass `0.22298632306046784`, concentration cap
  `PASS`;
- rebalanced micro-overfit job `857653`: protected lift vs base
  `0.14287435650001612`, gap to `+0.15` of
  `0.007125643499983875`, max surface mean target mass
  `0.190783511439804`, concentration cap `PASS`.

## Interpretation

The rebalanced run moved closer to the main teacher-forced lift gate while
keeping concentration far below the `0.50` cap. It still failed the
preregistered lift-vs-base requirement, so it cannot unlock generation, Qwen
E2E rerun, Llama, same-family null, sanitizer, FAR aggregation, payload
diversity, or paper-facing positive claims.

The safe route shape is a small, predeclared target-pressure relaxation, not an
open-ended sequence of single-point attempts. A candidate later route may
compare a tiny fixed grid of ceiling penalties, but only after a separate
artifact review defines:

- exact objective values;
- one reviewed Slurm wrapper path;
- one allowlist entry kept disabled by default;
- the same teacher-forced gate thresholds;
- the same concentration cap of max surface mean target mass `<= 0.50`;
- explicit no-generation and no-claim boundaries.

## Current Locks

Still locked from this route decision:

- Slurm submission;
- generation and Qwen E2E rerun;
- Llama;
- same-family null;
- sanitizer benchmark;
- FAR aggregation;
- payload-diversity claim;
- paper-facing positive claims.

Any future Chimera CPU/GPU work must use Slurm and must not run directly on the
login node.

## Next Allowed Action

Artifact-only objective/route design or review for a bounded post-rebalance
pressure-relaxation diagnostic. No Slurm submission, generation, Qwen E2E
rerun, Llama, null/FAR, sanitizer, payload diversity, or paper-claim work.
