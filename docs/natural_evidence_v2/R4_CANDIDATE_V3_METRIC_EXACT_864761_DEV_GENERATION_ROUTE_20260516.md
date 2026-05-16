# R4 Candidate v3 Metric-Exact 864761 Dev Generation Route

Canonical status: `V2_R4_METRIC_EXACT_864761_DEV_GENERATION_ROUTE_ARTIFACT_ONLY`

This route prepares a small Qwen dev generation diagnostic using the reviewed
adapter from job `864761`. It is not a paper-facing positive claim and does not
unlock Llama, same-family null, sanitizer, FAR, or payload diversity.

## Source Gate

Job `864761` passed the teacher-forced surface-mass gate:

```text
protected lift vs base:      +0.151589
protected lift vs task-only: +0.154749
protected rank1 rate:        1.0
protected median margin:    +0.154256
```

Caveat: job `864761` trained by repeated-cycling a 512-row train split, while
scoring 8192 rows. This diagnostic must preserve that caveat.

## Scope

Allowed after local and remote preflight:

- exactly one H200/pomplun Slurm array submission;
- Qwen dev prompts only;
- protected, raw, and task-only generation conditions;
- wrong-key and wrong-payload decoder controls;
- primary decode under `format_scrub=all`;
- secondary decode under `format_scrub=none`;
- protected adapter: job `864761`, gain `1.0`.

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

```text
scripts/natural_evidence_v2/slurm/r4_candidate_v3_metric_exact_864761_dev_diagnostic_h200.sbatch
```

It wraps the reviewed R4 cover-natural dev diagnostic wrapper and fixes:

```text
SOURCE_JOB_ID=864761
PROTECTED_ADAPTER=$RUN_ROOT/status/r4_candidate_v3_micro_overfit_864761/protected_micro_overfit_train/adapter
PROTECTED_ADAPTER_GAIN=1.0
```

It submits as one four-task array:

```text
#SBATCH --array=0-3%4
```

## Submission Gate

Before submission:

1. local wrapper syntax must pass;
2. local plan-only mode must validate all four shards;
3. zero-enabled allowlist safety must pass;
4. route, wrapper, allowlist, state, and gate-status files must sync to Chimera;
5. remote plan-only mode must validate all four shards;
6. remote zero-enabled allowlist safety must pass;
7. active-job preflight must be clean;
8. Hermes TG/email pre-submit notification must succeed;
9. exactly one allowlist entry may be enabled:
   `v2_r4_candidate_v3_metric_exact_864761_dev_diagnostic_h200`;
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

Passing this diagnostic does not directly unlock Llama, same-family null,
sanitizer, FAR, payload diversity, or paper-facing claims.
