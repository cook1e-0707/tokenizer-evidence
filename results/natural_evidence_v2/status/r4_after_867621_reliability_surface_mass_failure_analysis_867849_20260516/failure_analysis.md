# R4 After 867621 Surface-Mass Failure Analysis

status: `FAILURE_ANALYSIS_RECORDED_NO_GENERATION`

Job `867849` completed cleanly, but the teacher-forced surface-mass gate failed.

```text
protected lift vs base: 0.006302
protected lift vs task_only: 0.011221
protected rank1 rate: 0.482666
protected median target margin: -0.00009988
task_only lift vs base: -0.004919
```

Gate targets were `+0.15` lift vs base, `+0.10` lift vs task-only, rank1
`>=0.75`, and median target margin `>0`. The protected adapter produced only a
small positive target-mass lift and had lower rank1 than both base and task-only.

Interpretation:

```text
clean_slurm_completion: true
tokenizer_boundary_valid: true
task_only_leakage: false
protected_pressure_sufficient: false
generation_unlocked: false
```

This failure should not trigger generation. The next allowed step is an
artifact-only repair/pivot decision using the per-coordinate, per-prefix, and
per-surface failure tables.
