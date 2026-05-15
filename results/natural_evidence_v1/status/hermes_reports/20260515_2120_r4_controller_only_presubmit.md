# Hermes Pre-Submit: R4 Controller-Only Pressure Scoring

Phase:

```text
V2_R4_CONTROLLER_ONLY_SINGLE_SUBMISSION_READY
```

Allowed entry:

```text
v2_r4_positive_selectivity_controller_only_score_h200
```

Authorized command:

```text
sbatch --export=ALL,ALLOW_PRESSURE_CONTROLLER_SCORING=1,CONTROLLER_CONDITION_SET=controller_only_controls,ROUTE_CONFIG=configs/natural_evidence_v2/r4_positive_selectivity_controller_only_route.yaml scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
```

Preconditions already reviewed:

- `859672` reviewed as wrong-control selectivity failure.
- controller-only scorer patch validated.
- local route validator passed.
- local wrapper plan-only passed.
- reviewed files synced to Chimera.
- remote wrapper plan-only passed.
- remote zero-enabled allowlist safety passed.
- active job preflight observed no active Chimera jobs.

Submission rule:

```text
Enable exactly v2_r4_positive_selectivity_controller_only_score_h200,
submit exactly one H200/pomplun Slurm array,
then disable the entry immediately after sbatch returns.
```

Not included:

```text
generation
training
Qwen E2E
Llama
same-family null
sanitizer
FAR
payload-diversity claim
paper-facing positive claim
```

