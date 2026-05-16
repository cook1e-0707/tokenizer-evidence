# R4 Controller-Only Safety-Bound Score 864117 Review
## Outcome
Status: `FAIL_R4_CONTROLLER_ONLY_SAFETY_BOUND_SCORE_864117_NO_SELECTIVE_GATE`
Job `864117` completed as a H200/pomplun teacher-forced scoring array. It did not run generation, training, Llama, FAR, sanitizer, payload-diversity, or paper-claim work.
## Gate Counts
| Metric | Count |
| --- | ---: |
| Slurm tasks completed 0:0 | 24/24 |
| summaries present | 24/24 |
| controlled basic gate pass | 0/24 |
| overall selective gate pass | 0/24 |
| wrong-key basic gate pass | 0/24 |
| wrong-payload basic gate pass | 0/24 |

Primary pass requires at least one `overall_selective_gate_pass` grid and zero wrong-key/wrong-payload basic-gate passes.
## Best Observed Grids
Best controlled lift vs base: grid `21` with bonus `2.0`, penalty `0.5`, max_target_mass `0.35`, max_kl_budget `0.2`, lift `0.026958`, rank1 `0.601562`, median margin `0.003388`.
Best controlled rank1: grid `20` with bonus `2.0`, penalty `0.5`, max_target_mass `0.35`, max_kl_budget `0.1`, lift `0.024285`, rank1 `0.601562`, median margin `0.003388`.
## Interpretation
No safety-bound controller grid satisfies the selective teacher-forced gate. Wrong controls remain clean, but positive controlled-base pressure still does not reach the required lift/rank/margin thresholds. This does not unlock generation.

## Artifacts
- Aggregate JSON: `results/natural_evidence_v2/status/r4_controller_only_safety_bound_score_864117_review/aggregate_summary.json`
- Grid CSV: `results/natural_evidence_v2/status/r4_controller_only_safety_bound_score_864117_review/grid_summary.csv`
- Slurm summary: `results/natural_evidence_v2/status/r4_controller_only_safety_bound_score_864117_review/slurm_completion_summary.json`
