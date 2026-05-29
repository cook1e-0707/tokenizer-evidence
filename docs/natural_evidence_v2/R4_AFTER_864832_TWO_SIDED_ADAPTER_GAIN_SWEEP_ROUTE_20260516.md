# R4 After-864832 Two-Sided Adapter-Gain Sweep Route

Status: `ROUTE_RECORDED_NO_SUBMIT`

Job `865252` completed cleanly but failed the teacher-forced surface-mass gate:

```text
protected lift vs base: +0.047113
protected lift vs task-only: +0.056045
protected rank1 rate: 0.437500
protected median margin: -0.012231
```

The tokenizer boundary, two-sided row construction, and Slurm wrapper path are now validated. The remaining question is whether the protected adapter direction is useful but too weak, or whether it is misaligned with the cover-natural two-sided surface bank.

## Route

Run one H200 teacher-forced scoring job over the same 8192 two-sided rows with protected adapter gains:

```text
0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0
```

The route is scoring-only. It does not train, generate, run Qwen E2E, run Llama, run null/FAR/sanitizer jobs, test payload diversity, or support paper-facing claims.

## Artifacts

```text
configs/natural_evidence_v2/r4_after_864832_two_sided_adapter_gain_sweep.yaml
scripts/natural_evidence_v2/slurm/r4_after_864832_two_sided_adapter_gain_sweep_h200.sbatch
```

## Pass Interpretation

A pass requires at least one protected gain condition to satisfy the teacher-forced surface-mass gate with no boundary failures and no target/other overlap. If no gain passes, the adapter direction is insufficient for this two-sided cover-natural bank and the next repair must pivot to objective/data/controller design rather than generation.

