# R4 After 867621 Reliability Surface-Mass Score Route

Status: `ARTIFACT_ONLY_ROUTE_RECORDED_NO_SUBMIT`

## Context

Job `867621` showed that the reliability generation route did not emit the
coordinate-unique reliability surfaces in free generation. Codex then built
4096 teacher-forced score rows for that frozen coordinate-unique reliability
surface bank and ran actual Qwen tokenizer-boundary preflight in job `867828`.

Tokenizer preflight result:

```text
status: PASS_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT
job_id: 867828
checked rows: 4096
failed rows: 0
empty target id rows: 0
empty other id rows: 0
target/other overlap rows: 0
```

## Proposed Compute Scope

This route prepares a teacher-forced surface-mass scoring job for the exact
same 4096 rows:

```text
entry: v2_r4_after_867621_reliability_surface_mass_score_h200
wrapper: scripts/natural_evidence_v2/slurm/r4_after_867621_reliability_surface_mass_score_h200.sbatch
conditions: base, protected, task_only
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
time: 30-00:00:00
```

Not allowed in this route:

```text
generation
training
Llama
same-family null
sanitizer
FAR
payload-diversity claim
paper-facing positive claim
```

## Future Gate

The scoring result must meet:

```text
protected lift vs base >= +0.15
protected lift vs task-only >= +0.10
protected rank1 >= 0.75
protected median margin > 0
boundary failures = 0
target/other token overlap rate = 0
```

This route package does not submit Slurm by itself. A fresh remote hash
preflight, Hermes notification, exactly-one allowlist enablement, and immediate
post-submit allowlist disable are still required before compute.
