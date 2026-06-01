# VSG Manuscript Figure Quality Audit

This artifact-only audit checks active manuscript PNG figures for
dimensions, nonblank rendered content, render-manifest consistency,
LaTeX references, scope terms, and core data traceability.

Status: `PASS`
Figures checked: `5`
Failed figures: `0`
Data checks: `5`
Failed data checks: `0`

## Figure Checks

| Figure | Status | Size | Nonwhite ratio | Failures |
| --- | --- | ---: | ---: | --- |
| `figure_1_verification_substrate_map.png` | `PASS` | 1609x983 | 0.4674 |  |
| `figure_2_first_divergence_diagnostic.png` | `PASS` | 1610x785 | 0.0984 |  |
| `figure_3_controllability_vs_observability.png` | `PASS` | 1791x1010 | 0.3411 |  |
| `figure_4_public_predicate_attack_ladder.png` | `PASS` | 1767x1077 | 0.2563 |  |
| `figure_5_ownership_scenario_heatmap.png` | `PASS` | 1510x1180 | 0.6293 |  |

## Data Traceability Checks

| Check | Status | Detail |
| --- | --- | --- |
| `figure_3_trace_bound_counts` | `PASS` | trace_bound_accepts.csv contains Qwen 94/96 and Llama 96/96 |
| `figure_3_public_codeword_zero` | `PASS` | public_text_verifier_baselines.csv has zero recovered codeword blocks |
| `figure_4_guided_attack_top100` | `PASS` | guided rewrite/graft top-100 source-mismatch accepts are 100/100 for all plotted groups |
| `figure_5_ownership_matrix_shape` | `PASS` | ownership_scenario_heatmap.csv contains 7 scenarios x 9 method families |
| `figure_5_supported_public_text_zero` | `PASS` | ownership matrix has zero supported public final-text rows |

## Claim Boundary

- This audit does not render new figures.
- This audit does not start Slurm, generation, model scoring, or training.
- This audit does not create public text-only verification success or ownership-proof claims.
