# R4 Controller-Only Score 863274 Review
## Outcome
Status: `FAIL_R4_CONTROLLER_ONLY_SCORE_863274_NO_SELECTIVE_GATE`
Job `863274` completed as a Slurm array with all scoring summaries reviewed locally. This was a teacher-forced controller-only scoring job; it did not run generation, training, Llama, FAR, sanitizer, or paper-claim work.
## Gate Counts
| Metric | Count |
| --- | ---: |
| summaries present | 72/72 |
| controlled basic gate pass | 0/72 |
| overall selective gate pass | 0/72 |
| wrong-key basic gate pass | 0/72 |
| wrong-payload basic gate pass | 0/72 |

Primary pass requires at least one `overall_selective_gate_pass` grid and zero wrong-key/wrong-payload basic-gate passes.
## Best Observed Grids
Best controlled lift vs base: grid `67` with bonus `1.5`, penalty `0.25`, max_target_mass `0.25`, lift `0.015404`, rank1 `0.498047`, median margin `-0.000110`.
Best controlled rank1: grid `66` with bonus `1.5`, penalty `0.25`, max_target_mass `0.25`, lift `0.014224`, rank1 `0.498047`, median margin `-0.000110`.
## Interpretation
No controller-only grid satisfies the selective teacher-forced gate. The controller-only repair removed the protected-adapter leakage from wrong controls, but the controlled-base arm did not reach the required teacher-forced lift/rank/margin thresholds. This is not a generation-ready result.

## Artifacts
- Aggregate JSON: `results/natural_evidence_v2/status/r4_controller_only_score_863274_review/aggregate_summary.json`
- Grid CSV: `results/natural_evidence_v2/status/r4_controller_only_score_863274_review/grid_summary.csv`
