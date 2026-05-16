# Hermes Sync: R4 floor-dominant job 864705 reviewed

Phase:
`V2_R4_METRIC_EXACT_FLOOR_DOMINANT_864705_FAILED_REVIEWED_NO_NEXT_COMPUTE`

Summary:

- H200/pomplun job `864705` completed cleanly on `chimera21` with exit code `0:0`.
- The run used the floor-dominant metric-exact micro-overfit route:
  `TASK_CE_WEIGHT=0.0`, `TARGET_MASS_FLOOR=0.20`,
  `TARGET_MASS_FLOOR_LAMBDA=50.0`, `MARGIN_LAMBDA=1.0`, `MAX_STEPS=128`.
- Teacher-forced surface-mass gate failed:
  - protected mean target mass: `0.0847697`
  - protected lift vs base: `+0.0799378` required `>= +0.15`
  - protected lift vs task-only: `+0.0830972` required `>= +0.10`
  - protected rank1 rate: `1.0`
  - protected median margin: `+0.0772580`
- This is a strong directional improvement over job `864332`, but not a positive result.
- Trainer audit patch added: future train summaries now record `task_ce_weight`.
- Local and Chimera allowlist safety are PASS with zero enabled entries.
- Chimera active jobs: none.

Reviewed artifacts:

- `docs/natural_evidence_v2/R4_METRIC_EXACT_FLOOR_DOMINANT_864705_REVIEW_20260516.md`
- `results/natural_evidence_v2/status/r4_metric_exact_floor_dominant_864705_review/`
- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_864705/`

Next allowed action:

Artifact-only route decision for coverage-scale or stronger-floor repair. Do not submit
new Slurm, run generation, start Llama, run same-family null, run sanitizer,
aggregate FAR, make payload-diversity claims, or make paper-facing positive
claims until a new route records prerequisites and control-plane checks.
