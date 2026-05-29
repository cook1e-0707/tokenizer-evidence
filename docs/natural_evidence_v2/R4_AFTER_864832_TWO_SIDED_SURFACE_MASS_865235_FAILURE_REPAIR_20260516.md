# R4 After 864832 Two-Sided Surface-Mass 865235 Failure Repair

Job `865235` failed before model scoring:

```text
state: FAILED
elapsed: 00:00:01
exit_code: 2:0
stderr: REQUIRED_ADAPTER_MISSING_OR_EMPTY
missing path: r4_candidate_v3_micro_overfit_864761/protected_train/adapter/adapter_config.json
```

This is a wrapper adapter-path bug, not a tokenizer, model, or surface-mass
gate failure. The route pointed at old `protected_train` and `task_only_train`
paths under job `864761`, but the available protected adapter is:

```text
r4_candidate_v3_micro_overfit_864761/protected_micro_overfit_train/adapter
```

The task-only control remains the reviewed WP5-R2 task-only adapter:

```text
wp5_r2_teacher_forced_train_and_score_851481/task_only_train/adapter
```

Patch:

```text
scripts/natural_evidence_v2/slurm/r4_after_864832_two_sided_surface_mass_score_h200.sbatch
scripts/natural_evidence_v2/validate_r4_after_864832_two_sided_surface_mass_route.py
```

No scoring started in `865235`; no generation/training/downstream action was
unlocked. A repaired scoring submission requires route revalidation, remote
hash preflight, Hermes notification, exactly-one allowlist enablement, and
post-submit allowlist shutdown.
