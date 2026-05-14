# R4 Candidate v3 Adapter-Gain Sweep Launch Route

Timestamp UTC: 2026-05-13T23:37Z

## Decision

The next eligible compute route is a single H200 teacher-forced
protected-adapter gain sweep for R4 candidate v3. This route is diagnostic only:
it tests whether the failed candidate v3 surface-mass gate is caused by
insufficient protected-adapter logit pressure rather than tokenizer-boundary
failure, task-only leakage, or another surface-bank narrowing issue.

This route does not authorize training, free generation, Qwen E2E, Llama,
same-family nulls, sanitizer benchmark, FAR aggregation, payload-diversity
evaluation, or paper-facing positive claims.

## Fixed Inputs

- Candidate rows:
  `results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_v3_20260513/r4_prefix_native_surface_probe_rows_v3.jsonl`
- Candidate rows sha256:
  `d35e5483ce7f6d3d782ce17961b2c407909afc879a12917c5ccc27090f3c80b7`
- Contract: same-contract `a55e`; payload diversity is not tested.
- Conditions:
  `base`, `task_only`, `protected_gain_0`, `protected_gain_0_5`,
  `protected_gain_1`, `protected_gain_1_5`, `protected_gain_2`,
  `protected_gain_3`, `protected_gain_4`.
- Protected adapter gains:
  `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]`.

## Reviewed Command

Allowlist entry:

```text
v2_r4_candidate_v3_adapter_gain_sweep_h200
```

Slurm command:

```text
sbatch scripts/natural_evidence_v2/slurm/r4_candidate_v3_adapter_gain_sweep_h200.sbatch
```

The wrapper uses `pomplun` / `cs_yinxin.wan` / `gpu:h200:1` with time limit
`30-00:00:00`. It runs teacher-forced model scoring only. It must refuse to
overwrite existing scorer outputs.

## Validation Completed

- Plan validation:
  `results/natural_evidence_v2/status/r4_candidate_v3_adapter_gain_sweep_plan_validation_20260513_rerun/adapter_gain_sweep_plan_validation_summary.json`
- Scorer dry run:
  `results/natural_evidence_v2/status/r4_candidate_v3_gain_sweep_scorer_dry_run_20260513_rerun/r4_teacher_forced_surface_mass_summary.json`
- Allowlist zero-enabled safety:
  `results/natural_evidence_v2/status/r4_candidate_v3_gain_sweep_allowlist_zero_20260513_rerun.json`
- Focused tests:
  `uv run pytest tests/natural_evidence_v2/test_r4_adapter_gain_sweep_plan.py tests/natural_evidence_v2/test_r4_prefix_native_soft_logit_controller.py`
- Static checks:
  `uv run python -m py_compile scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py scripts/natural_evidence_v2/validate_r4_adapter_gain_sweep_plan.py`
  and `bash -n scripts/natural_evidence_v2/slurm/r4_candidate_v3_adapter_gain_sweep_h200.sbatch`.

## Submission Rules

Before submission:

1. Send Hermes TG/email notification.
2. Re-run local allowlist safety and confirm zero enabled entries.
3. Sync reviewed files to Chimera and confirm remote file hashes.
4. Enable exactly one allowlist entry:
   `v2_r4_candidate_v3_adapter_gain_sweep_h200`.
5. Submit exactly one H200 Slurm job.

Immediately after `sbatch` returns:

1. Disable the allowlist entry.
2. Re-run local and remote allowlist safety and require zero enabled entries.
3. Record a submission JSON/Markdown artifact with the job id.

## Review Gate After Completion

The gain sweep passes only if at least one protected-gain condition satisfies:

- protected lift vs base `>= +0.15`;
- protected lift vs task-only `>= +0.10`;
- target rank-1 rate `>= 0.75`;
- protected median margin `> 0`;
- no scorer boundary failures;
- no task-only lift anomaly;
- no collapse diagnostic such as one surface carrying the whole result without
  being recorded and reviewed.

If the gain sweep fails, do not train or generate. Review the gain response by
prefix, surface, coordinate, and target-token set.

If it passes, it still is not a natural-output positive result. It only supports
the next reviewed route: small Qwen generation diagnostic under the existing R4
format-scrub and null-control gates.

## Status

`READY_FOR_SINGLE_H200_ADAPTER_GAIN_SWEEP_SUBMISSION_AFTER_NOTIFICATION_AND_REMOTE_HASH_PREFLIGHT`
