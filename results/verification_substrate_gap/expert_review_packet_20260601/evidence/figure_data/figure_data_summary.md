# VSG Paper Figure Data 2026-05-30

Status: `VSG_PAPER_FIGURE_DATA_EXTRACTED_ARTIFACT_ONLY_NO_NEW_CLAIMS`

This artifact-only extraction builds plot-ready tables for the VSG paper
skeleton. It starts no generation, model calls, Slurm submission, or
training, and it does not create a paper-facing positive claim.

## Tables

| Table | Rows | Purpose |
| --- | ---: | --- |
| `trace_bound_accepts.csv` | 2 | Figure 3 trace-bound controllability |
| `public_text_verifier_baselines.csv` | 8 | Figure 3 public final-text observability |
| `template_leakage_summary.csv` | 2 | Template leakage audit table |
| `attack_ladder_summary.csv` | 30 | Figure 4 public-predicate attack ladder |
| `ownership_scenario_heatmap.csv` | 63 | Figure 5 ownership scenario heatmap |
| `claim_ledger.csv` | 4 | Table 1 claim ledger |

## Claim Boundary

The extracted tables preserve the existing claim boundary: trace-bound
diagnostics are provider-side only, public final-text predicates recover
zero codeword blocks, and source-mismatch accepts are spoofing evidence
rather than protected success.
