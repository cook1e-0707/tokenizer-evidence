# R4 Candidate v3 Pressure-Relaxation B Dev Generation Route

Timestamp UTC: 2026-05-14T05:22:00Z

## Decision

Prepare a small Qwen dev generation diagnostic using the passing
pressure-relaxation arm `B_ceiling_lambda_0_5` from Slurm array job `857764`.

This route is not a paper-facing positive claim. It is a diagnostic to test
whether the teacher-forced target-mass pass transfers to free generation under
the existing R4 cover-natural decoder and required null/format-scrub controls.

## Selected Adapter

- source job: `857764`;
- selected arm: `B_ceiling_lambda_0_5`;
- protected adapter:
  `status/r4_candidate_v3_pressure_relaxation_grid_857764/B_ceiling_lambda_0_5/protected_micro_overfit_train/adapter`;
- protected adapter gain: `1.0`;
- reason: arm B has the strongest reviewed lift vs base
  (`0.2692875063101212`) while still passing the max surface mean target-mass
  cap (`0.4625823280075565 <= 0.50`).

## Scope

Allowed after local and remote preflight:

- exactly one H200/pomplun Slurm array submission;
- Qwen dev prompts only;
- protected, raw, and task-only generation conditions;
- wrong-key and wrong-payload decoder controls;
- primary decode under `format_scrub=all`;
- secondary decode under `format_scrub=none`.

Not allowed:

- training;
- Qwen E2E rerun outside this dev diagnostic;
- Llama;
- same-family null;
- sanitizer benchmark;
- FAR aggregation;
- payload-diversity claim;
- paper-facing positive claim.

## Wrapper

Wrapper:

`scripts/natural_evidence_v2/slurm/r4_candidate_v3_pressure_relaxation_b_dev_diagnostic_h200.sbatch`

It wraps the reviewed R4 dev diagnostic wrapper and fixes:

```text
SOURCE_JOB_ID=857764
SELECTED_ARM=B_ceiling_lambda_0_5
PROTECTED_ADAPTER_GAIN=1.0
```

It submits as one four-task array:

```text
#SBATCH --array=0-3%4
```

## Submission Gate

Before any submission:

1. local wrapper syntax must pass;
2. local plan-only mode must validate all four shards;
3. zero-enabled allowlist safety must pass;
4. route, wrapper, allowlist, state, and gate-status files must sync to Chimera;
5. remote plan-only mode must validate all four shards;
6. remote zero-enabled allowlist safety must pass;
7. active-job preflight must be clean;
8. Hermes TG/email pre-submit notification must succeed;
9. exactly one allowlist entry may be enabled:
   `v2_r4_candidate_v3_pressure_relaxation_b_dev_diagnostic_h200`;
10. after `sbatch` returns, the allowlist must be disabled immediately and
    local/remote zero-enabled safety must pass.

## Diagnostic Gate

Review must report, at minimum:

- protected accepts under `format_scrub=all`;
- protected accepts under `format_scrub=none`;
- raw/task-only/wrong-key/wrong-payload accepts;
- forbidden public technical surface count;
- duplicate generated-output hashes;
- duplicate decode-row hashes;
- protected-vs-raw shallow structural/template leakage if available.

No generation result directly unlocks Llama, same-family null, sanitizer, FAR,
payload diversity, or paper claims.
