# Hermes/Codex latest state sync

## Current phase

`V2_R3_2C_JOB_853070_FAILED_PROMPT_SPLIT_MISMATCH_NO_RESUBMIT`

## Latest progress

Job `853070` was submitted as the single allowed R3.2 Slurm job and failed
before model generation:

```text
State: FAILED
Elapsed: 00:00:00
ExitCode: 1:0
Failure: split 'wp3_r1_eval' has only 0 prompts; need 512
```

The failure was traced to a prompt split/window mismatch, not a model or
evidence-channel result.

The repair has now been implemented artifact-only:

- repair decision:
  `docs/natural_evidence_v2/R3_2_PROMPT_SPLIT_CONTRACT_REPAIR_20260511_1747.md`
- implementation review:
  `docs/natural_evidence_v2/R3_2_PROMPT_SPLIT_IMPLEMENTATION_20260511_1801.md`
- plan-only precommit:
  `results/natural_evidence_v2/status/r3_2_prompt_split_repair_precommit_20260511_1801`

Repaired allocation:

```text
selected_split = wp3_r1_eval
eval prompt rows = 512..2559
eval prompt count = 2048
prompt_window_policy = deterministic_4_eval_window_circular_reuse_by_replicate_group_index
selected_prompt_manifest_sha256 = 3e50a08773c4c7dca3be976a762840a8d8a960ac63f4cfce382af3051a2b82d1
```

## Control plane

- local allowlist enabled entries: `[]`
- remote allowlist enabled entries: `[]`
- active R3.2 Slurm jobs: none
- job `853070` remains `FAILED 1:0`

## Gate status

The following remain gate-controlled and not unlocked:

- training
- Llama
- same-family null
- sanitizer
- FAR aggregation
- paper-facing positive claims

## Next allowed action

Artifact-only next step: re-review `852426` replay compatibility or explicitly
supersede it under the repaired R3.2 prompt split contract. Do not submit
another R3.2 Slurm job until replay compatibility is re-reviewed or superseded,
allowlist safety is rechecked, and a new single-job submission route is
recorded.
