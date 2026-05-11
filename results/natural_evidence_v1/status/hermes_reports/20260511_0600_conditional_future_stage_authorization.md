# Conditional future-stage authorization

Timestamp UTC: `2026-05-11T05:41:06Z`

## Decision

The user authorized later-stage training, Llama, FAR/null expansion, sanitizer,
and paper-claim work after their prerequisite gates pass.

## Important boundary

This is not an immediate unlock. The following booleans remain false now:

```text
training_allowed=false
llama_allowed=false
same_family_null_allowed=false
sanitizer_allowed=false
far_aggregation_allowed=false
paper_claim_allowed=false
```

Each class can proceed automatically only after its prerequisite evidence is
reviewed and the corresponding gate boolean is explicitly set true in
`gate_status.json`, with `next_allowed_action` naming that class.

## Current next action

Continue the current R3.2 blocker-clearing path: finish or upgrade the R3.2
wrapper from plan-only to a reviewed full locked-scale generation/eval wrapper,
validate locally, record review, enable one allowlist entry, notify configured
channels, then submit exactly one Chimera Slurm job.
