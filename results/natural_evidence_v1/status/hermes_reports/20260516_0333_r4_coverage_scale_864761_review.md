# Hermes Sync: R4 coverage-scale job 864761 reviewed

Phase:
`V2_R4_METRIC_EXACT_COVERAGE_SCALE_864761_PASSED_TF_GATE_WITH_CAVEAT`

Summary:

- H200/pomplun job `864761` completed cleanly on `chimera21`, exit code `0:0`.
- Teacher-forced surface-mass gate passed:
  - protected mean target mass: `0.156421`
  - protected lift vs base: `+0.151589`
  - protected lift vs task-only: `+0.154749`
  - protected rank1 rate: `1.0`
  - protected median margin: `+0.154256`
  - task-only lift vs base: `-0.003159`
- No generation, Qwen E2E, Llama, same-family null, sanitizer, FAR,
  payload-diversity, or paper-facing claim work was started.
- Local and remote allowlist safety: PASS with zero enabled entries.

Important caveat:

The route intended `MAX_TRAIN_ROWS=8192`, but the train artifact contains `512`
rows. The job therefore tested repeated-cycled 512-row training with stronger
floor pressure and scored on 8192 rows. Preserve this caveat; do not describe
job `864761` as an 8192 unique train-row coverage result.

Reviewed artifacts:

- `docs/natural_evidence_v2/R4_METRIC_EXACT_COVERAGE_SCALE_864761_REVIEW_20260516.md`
- `results/natural_evidence_v2/status/r4_metric_exact_coverage_scale_864761_review/`
- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_864761/`

Next allowed action:

Artifact-only route decision for a small Qwen dev generation diagnostic using
the reviewed adapter and same candidate-v3 surface contract. No generation may
start until the route records wrapper contract, allowlist entry, Hermes
notification, remote hash preflight, and post-submit allowlist shutdown policy.
