# R4 Pressure-Controller Repaired Single-Submission Route 20260515 0550

## Decision

Proceed to exactly one repaired H200/pomplun Slurm array submission for the R4
pressure-controller teacher-forced scoring route.

This route replaces failed job `859590`, which was reviewed as a wrapper
output-directory collision before model scoring began. The repaired wrapper has
passed local and remote plan-only validation.

## Bound Repair Evidence

- Failed submission record:
  `results/natural_evidence_v2/status/r4_pressure_controller_submission_20260515_0530/submission_record.json`
- Wrapper repair review:
  `docs/natural_evidence_v2/R4_PRESSURE_CONTROLLER_859590_WRAPPER_COLLISION_REPAIR_20260515_0540.md`
- Repaired remote preflight:
  `results/natural_evidence_v2/status/r4_pressure_controller_repaired_remote_preflight_20260515_0545/remote_preflight_summary.json`

## Authorized Submission

Enable exactly one allowlist entry:

```text
v2_r4_positive_selectivity_pressure_controller_score_h200
```

Submit exactly one H200/pomplun Slurm array job:

```text
sbatch --export=ALL,ALLOW_PRESSURE_CONTROLLER_SCORING=1 scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
```

Immediately disable the allowlist entry after `sbatch` returns.

## Scope

- Qwen only
- Same-contract `a55e`
- Teacher-forced scoring only
- Conditions:
  `base`, `task_only`, `controlled_protected`, `wrong_key_controlled`,
  `wrong_payload_controlled`
- Controller grid size: `72`
- No generation
- No training
- No Llama
- No same-family null
- No sanitizer
- No FAR aggregation
- No payload-diversity claim
- No paper-facing positive claim

## Stop Rules

Stop and record a blocker if submission fails, if more than one allowlist entry
is enabled, if the post-submit allowlist remains enabled, if any remote safety
check fails, or if the repaired job fails before model scoring for another
wrapper/control-plane reason.

