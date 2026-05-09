# Repaired target-mass probe submission

Hermes notification succeeded before this Codex worker started:

```text
results/natural_evidence_v1/status/hermes_reports/20260508_0741_scheduled_tick_notification.json
status=SENT_ALL_REQUIRED_CHANNELS
```

Action taken:

- added `scripts/natural_evidence_v1/score_repaired_teacher_forced_target_mass_probe.py`;
- added `scripts/natural_evidence_v1/slurm/qwen_repaired_teacher_forced_target_mass_probe.sbatch`;
- updated the single enabled GPU allowlist command to the repaired wrapper;
- synced only the required scorer/wrapper/allowlist and Option R design artifacts to Chimera;
- submitted exactly one Slurm job: `848547` (`nat-ev-qwen-rtfprob`).
- after job completion, disabled the repaired-probe allowlist entry locally and
  on Chimera to prevent duplicate submission.

Submission command:

```text
sbatch scripts/natural_evidence_v1/slurm/qwen_repaired_teacher_forced_target_mass_probe.sbatch
```

Submit-time queue state:

```text
848547 DGXA100 nat-ev-qwen-rtfprob guanjie.lin001 PD 0:00 1 (Resources)
```

Completion:

```text
848547|nat-ev-qwen-rtfprob|COMPLETED|00:01:35|0:0
```

Inputs:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_repaired_teacher_forced_target_mass_probe_design/repaired_teacher_forced_target_mass_probe_scoring_plan.jsonl
rows=257
base=75
protected_trained=91
task_only_lora=91
```

Output target:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_repaired_teacher_forced_target_mass_probe_scored/
```

Result:

```text
status=COMPLETE_REPAIRED_TEACHER_FORCED_TARGET_MASS_PROBE_SCORED_NOT_RECOVERY_NOT_FAR
scored_rows=257
threshold_pass=false
protected_minus_base_target_candidate_mass=-0.007645810655699581
protected_minus_task_only_target_candidate_mass=-0.04776975171334799
protected_minus_task_only_target_rank1_rate=-0.03296703296703296
```

Validation before submission:

```text
python3 -m py_compile scripts/natural_evidence_v1/score_repaired_teacher_forced_target_mass_probe.py scripts/natural_evidence_v1/validate_static.py
bash -n scripts/natural_evidence_v1/slurm/qwen_repaired_teacher_forced_target_mass_probe.sbatch
python3 scripts/natural_evidence_v1/validate_static.py --summary /tmp/natural_evidence_v1_static_validation_repaired_probe.json
pytest tests/test_natural_evidence_v1.py -q -k 'repaired_teacher_forced_target_mass_probe_design or repaired_teacher_forced_target_mass_score_stats'
```

Claim control:

- no training;
- no generation;
- no Qwen E2E rerun;
- no Llama;
- no same-family null;
- no sanitizer benchmark;
- no FAR aggregation;
- no paper-facing positive claim.

Next allowed action:

Decision review before any repaired dataset or training preflight. The scored
probe did not clear thresholds. Training, generation, E2E rerun, FAR
aggregation, and paper-facing claims remain forbidden.
