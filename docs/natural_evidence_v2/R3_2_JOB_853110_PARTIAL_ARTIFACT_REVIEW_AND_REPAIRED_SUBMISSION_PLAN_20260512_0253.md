# R3.2 Job 853110 Partial Artifact Review and Repaired Submission Plan: 2026-05-12 02:53Z

## Scope

This is an artifact-only review of partial R3.2 job `853110` outputs after the
job reached terminal state `PREEMPTED`. It does not resubmit Slurm, enable any
allowlist entry, start generation, rerun Qwen E2E, train, start Llama, run
same-family nulls, run sanitizer benchmarks, aggregate FAR, or make
paper-facing positive claims.

Machine-readable record:

```text
results/natural_evidence_v2/status/r3_2_job_853110_partial_artifact_review_and_repaired_submission_plan_20260512_0253.json
```

## Inputs Reviewed

- Hermes report:
  `results/natural_evidence_v1/status/hermes_reports/20260512_0252_scheduled_tick.md`
- Preemption review:
  `results/natural_evidence_v1/status/hermes_reports/20260512_0246_job_853110_preempted_review.md`
- Remote partial output directory, read-only:
  `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r3_2_qwen_locked_scale_eval_853110`
- Remote Slurm logs, read-only:
  `nat-ev-v2-r32qwen-853110.out` and `nat-ev-v2-r32qwen-853110.err`

## Findings

Job `853110` is not a valid R3.2 aggregate attempt. It completed only
`6/12` shard directories and did not produce the 96-block aggregate gate.
The Slurm stderr reports cancellation due to preemption on `chimera12`, so the
terminal failure mode remains scheduler preemption rather than a Python, CUDA,
model, or prompt-loader exception.

The partial artifacts are also not reusable as canonical R3.2 shard evidence.
The reviewed generation summaries show the submitted remote job used prompt
file-row windows beginning at `0..511`, and `shard_05` repeats the same
selected prompt sha256 as `shard_00`. That conflicts with the repaired R3.2
prompt split contract, which requires eval-only windows:

```text
shard_00, shard_04, shard_08 -> rows 512..1023
shard_01, shard_05, shard_09 -> rows 1024..1535
shard_02, shard_06, shard_10 -> rows 1536..2047
shard_03, shard_07, shard_11 -> rows 2048..2559
```

Therefore the completed `853110` shards must remain diagnostic-only. They must
not be merged with any later run, used for an aggregate R3.2 pass/fail gate, or
used for paper-facing positive claims.

## Runtime and Preemption Handling

The local wrapper now requests the DGXA100 partition maximum time:

```text
#SBATCH --time=30-00:00:00
```

This addresses the earlier underbudgeted `10:00:00` wall-clock limit, but it
does not prevent `scavenger_unlim` preemption. A later route review must treat
preemption as still possible and must require a clean new output directory
rather than resuming or overwriting `853110` artifacts.

## Repaired Submission Plan

A later submission tick is not authorized by this review. Before any new R3.2
Slurm submission, the next worker must re-review and record all of the
following:

1. The local and remote R3.2 wrapper/config/precommit-builder are synchronized
   to the repaired eval-only prompt split contract.
2. The wrapper's `EXPECTED_SELECTED_PROMPT_MANIFEST_SHA256` matches the current
   repaired precommit manifest.
3. A plan-only precommit on the exact remote code/artifacts confirms the
   repaired 4-window eval allocation without writing into an existing run
   directory.
4. The R3.2 allowlist safety check passes with zero enabled entries before the
   tick enables anything.
5. The single-job route is re-recorded after this partial-artifact review,
   including TG and email pre-notice requirements.
6. The later submission, if authorized, enables exactly
   `v2_r3_2_qwen_locked_scale_eval`, submits exactly one Slurm job, disables
   that entry immediately after `sbatch` returns, records the submission, and
   stops.

## Status

```text
REVIEWED_R3_2_JOB_853110_PARTIAL_ARTIFACTS_REPAIRED_PLAN_RECORDED_NO_RESUBMIT
```
