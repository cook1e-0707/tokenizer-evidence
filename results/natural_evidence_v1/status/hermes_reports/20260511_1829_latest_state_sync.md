# Hermes/Codex latest state sync

## Current phase

`V2_R3_2C_JOB_853070_FAILED_PROMPT_SPLIT_MISMATCH_NO_RESUBMIT`

## Latest Hermes progress

Hermes tick `20260511_1816` completed an artifact-only replay compatibility
re-review.

Result:

```text
PASS_R3_2_852426_REPLAY_COMPATIBILITY_REREVIEW_UNDER_REPAIRED_PROMPT_SPLIT_NO_SLURM
```

Evidence:

- `docs/natural_evidence_v2/R3_2_852426_REPLAY_COMPATIBILITY_REREVIEW_20260511_1817.md`
- `results/natural_evidence_v2/status/r3_2_852426_replay_compatibility_rereview_20260511_1817.json`
- `results/natural_evidence_v2/status/r3_2_same_contract_852426_replay_rereview_20260511_1817/r3_2_852426_replay_summary.json`

Key compatibility point:

```text
852426 used wp3_r1_eval rows 768..1279.
The repaired R3.2 contract uses wp3_r1_eval rows 512..2559.
Therefore the reviewed 852426 replay window is inside the repaired eval-only allocation.
```

Fresh replay exact-matched:

```text
protected accepts @64 = 7/8
raw/task-only/wrong-key/wrong-payload accepts @64 = 0/8 each
min accepted-block support = 26
min accepted-block majority margin = 5
forbidden_public_surface_count = 0
```

## Control plane

- local allowlist enabled entries: `[]`
- remote allowlist enabled entries before this sync were stale relative to
  local; this report triggers a local-to-remote sync.
- job `853070` remains `FAILED 1:0`
- no active R3.2 Slurm job

## Gate status

No gated class was unlocked:

- `training_allowed=false`
- `llama_allowed=false`
- `same_family_null_allowed=false`
- `sanitizer_allowed=false`
- `far_aggregation_allowed=false`
- `paper_claim_allowed=false`

## Next allowed action

Artifact-only next step: recheck R3.2 allowlist safety under the repaired prompt
split contract, then record a new single-job submission route before any Slurm
submission. Do not submit another R3.2 Slurm job in this tick.
