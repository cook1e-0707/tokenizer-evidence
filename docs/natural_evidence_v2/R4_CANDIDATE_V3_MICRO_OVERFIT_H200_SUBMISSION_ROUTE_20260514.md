# R4 Candidate v3 Micro-Overfit H200 Submission Route

Timestamp UTC: 2026-05-14T03:15:00Z

## Decision

Allow exactly one H200/pomplun Slurm submission for the R4 candidate-v3
metric-exact protected micro-overfit route.

This route is not generation, not Qwen E2E, not Llama, not same-family null,
not sanitizer, not FAR aggregation, not payload diversity, and not a
paper-facing positive claim.

## Preconditions

Passed:

- artifact implementation review:
  `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_artifact_review_20260514_0305/artifact_review.md`;
- local split artifacts:
  `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_split_20260514_codex/`;
- local/remote hash preflight:
  `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_remote_sync_hash_preflight_20260514_0310/remote_sync_hash_preflight_summary.json`;
- remote wrapper plan-only check:
  `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_remote_sync_hash_preflight_20260514_0310/remote_wrapper_plan_only.log`;
- remote allowlist zero-enabled safety check:
  `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_remote_sync_hash_preflight_20260514_0310/remote_allowlist_safety.log`.

## Submission Scope

Allowlist entry:

`v2_r4_candidate_v3_micro_overfit_h200`

Command:

```bash
sbatch scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
```

Wrapper:

`scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch`

Slurm requirements:

- partition: `pomplun`
- account: `cs_yinxin.wan`
- QoS: `pomplun`
- GRES: `gpu:h200:1`
- time limit: `30-00:00:00`

Inputs:

- train rows:
  `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_split_20260514_codex/train_rows.jsonl`
- heldout rows:
  `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_split_20260514_codex/heldout_rows.jsonl`
- score rows:
  `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_split_20260514_codex/score_rows.jsonl`

Execution:

- protected-only R4 prefix-native micro-overfit training;
- base/protected/task-only teacher-forced surface-mass scoring;
- no generation stage;
- no decoder/FAR/sanitizer/Llama stage.

## Required After Submission

Immediately after `sbatch` returns:

1. disable `v2_r4_candidate_v3_micro_overfit_h200` locally;
2. sync disabled allowlist to Chimera;
3. run local and remote zero-enabled allowlist safety checks;
4. record submission JSON and update `CURRENT_STATE.md`;
5. notify Hermes with job id.

## Claim Control

This job can only produce teacher-forced micro-overfit training/scoring
diagnostics. It cannot by itself unlock generation, Qwen E2E rerun, Llama,
same-family nulls, sanitizer, FAR, payload diversity, or paper-facing claims.
