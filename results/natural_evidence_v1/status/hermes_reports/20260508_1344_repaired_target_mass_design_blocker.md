# Repaired target-mass design blocker

Hermes notification succeeded before this Codex worker started:

```text
results/natural_evidence_v1/status/hermes_reports/20260508_1344_scheduled_tick_notification.json
status=SENT_ALL_REQUIRED_CHANNELS
```

Requested intended action:

```text
artifact-only repaired teacher-forced target-mass probe design over the primary branch-aware candidates
```

Blocker:

The requested action is not safe or unambiguous for the current phase. The
current Hermes report permits only a new explicit artifact-only
negative-diagnosis/root-cause plan or user/expert review. A repeated repaired
teacher-forced target-mass design over the primary branch-aware candidates is
not such a plan.

The same repaired design already exists:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_repaired_teacher_forced_target_mass_probe_design/
```

That design produced a 257-row scoring plan, was scored by Slurm job `848547`,
and the completed score review rejected progression from this repaired
target-mass path:

```text
results/natural_evidence_v1/status/hermes_reports/20260508_0811_repaired_target_mass_score_review.md
decision=REJECT_REPAIRED_DATASET_OR_TRAINING_PREFLIGHT_FROM_JOB_848547
threshold_pass=false
protected_minus_base_target_candidate_mass=-0.007645810655699581
protected_minus_task_only_target_candidate_mass=-0.04776975171334799
protected_minus_task_only_target_rank1_rate=-0.03296703296703296
```

The previous Hermes tick also blocked this repeated design instruction:

```text
results/natural_evidence_v1/status/hermes_reports/20260508_1328_repaired_target_mass_design_blocker.md
```

Conclusion:

- do not create another repaired target-mass design from this path;
- do not submit another repaired target-mass probe from job `848547`;
- do not start repaired dataset construction or training preflight from job
  `848547`;
- only a new explicit artifact-only negative-diagnosis/root-cause plan or
  user/expert review may proceed.

No training, generation, Qwen E2E rerun, Llama, same-family null, sanitizer,
FAR aggregation, model scoring, Slurm job, or paper-facing positive claim was
started.
