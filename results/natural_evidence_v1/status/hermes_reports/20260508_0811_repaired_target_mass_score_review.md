# Repaired target-mass score review

Hermes notification succeeded before this Codex worker started:

```text
results/natural_evidence_v1/status/hermes_reports/20260508_0811_scheduled_tick_notification.json
status=SENT_ALL_REQUIRED_CHANNELS
```

Reviewed artifacts:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_repaired_teacher_forced_target_mass_probe_scored/repaired_teacher_forced_target_mass_probe_score_summary.json
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_repaired_teacher_forced_target_mass_probe_scored/repaired_teacher_forced_target_mass_probe_score_by_group.csv
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_repaired_teacher_forced_target_mass_probe_design/repaired_teacher_forced_target_mass_probe_scoring_plan.jsonl
```

Completeness:

```text
score_plan_rows=257
scored_rows=257
base_rows=75
protected_trained_rows=91
task_only_lora_rows=91
slurm_job_id=848547
slurm_status=COMPLETED 0:0
```

Decision metrics:

```text
threshold_pass=false
protected_minus_base_target_candidate_mass=-0.007645810655699581
protected_minus_task_only_target_candidate_mass=-0.04776975171334799
protected_minus_task_only_target_rank1_rate=-0.03296703296703296
required_lift_each=+0.05
```

Condition summary:

| Scoring arm | Rows | Mean target mass | Target rank-1 rate | Mean target margin |
|---|---:|---:|---:|---:|
| base | 75 | 0.10418856937661333 | 0.17333333333333334 | -0.7025922842857958 |
| protected_trained | 91 | 0.09654275872091375 | 0.10989010989010989 | -0.7084092110245943 |
| task_only_lora | 91 | 0.14431251043426174 | 0.14285714285714285 | -0.6531208019441337 |

Review conclusion:

- The scored probe is complete, not missing results.
- The protected arm is lower than base and task-only on mean target candidate
  mass.
- The protected arm is also lower than task-only on target rank-1 rate.
- The predeclared target-mass lift gate failed by the aggregate decision rule.
- This result rejects repaired dataset or training preflight from job `848547`.

Allowed/forbidden state:

- no training started;
- no generation started;
- no Qwen E2E rerun started;
- no Llama, same-family null, sanitizer benchmark, FAR aggregation, or
  paper-facing positive claim started;
- do not submit another repaired target-mass probe unless a new reviewed plan
  explicitly requires it.

Next allowed action:

Stop positive-E2E progression from this repaired target-mass path. Any further
work must be a new explicit artifact-only negative-diagnosis/root-cause plan or
user/expert review. Repaired dataset preflight and training remain forbidden.
