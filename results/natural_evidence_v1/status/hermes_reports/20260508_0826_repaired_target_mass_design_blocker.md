# Repaired target-mass design blocker

Hermes notification succeeded before this Codex worker started:

```text
results/natural_evidence_v1/status/hermes_reports/20260508_0826_scheduled_tick_notification.json
status=SENT_ALL_REQUIRED_CHANNELS
```

Requested intended action:

```text
artifact-only repaired teacher-forced target-mass probe design over the primary branch-aware candidates
```

Blocker:

The requested design action is not safe as the next state-changing step. The
same repaired teacher-forced target-mass design over primary branch-aware
candidates already exists:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_repaired_teacher_forced_target_mass_probe_design/
```

That design was scored by Slurm job `848547`, and the completed score review
rejected progression from this repaired target-mass path:

```text
results/natural_evidence_v1/status/hermes_reports/20260508_0811_repaired_target_mass_score_review.md
decision=REJECT_REPAIRED_DATASET_OR_TRAINING_PREFLIGHT_FROM_JOB_848547
threshold_pass=false
protected_minus_base_target_candidate_mass=-0.007645810655699581
protected_minus_task_only_target_candidate_mass=-0.04776975171334799
protected_minus_task_only_target_rank1_rate=-0.03296703296703296
```

Conclusion:

- do not create another repaired target-mass design from this path;
- do not submit another repaired target-mass probe from job `848547`;
- do not start repaired dataset construction or training preflight from job
  `848547`;
- only a new explicit artifact-only negative-diagnosis/root-cause plan or
  user/expert review may proceed.

No training, generation, Qwen E2E rerun, Llama, same-family null, sanitizer,
FAR aggregation, or paper-facing positive claim was started.
