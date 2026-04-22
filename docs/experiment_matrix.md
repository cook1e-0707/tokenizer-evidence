# Experiment Matrix

This matrix is the strict next-step execution package for the current paper-facing stage.

Guardrails:
- do not expand `Batch 3`
- do not open baselines yet
- do not open `Meta-Llama-3.1-8B-Instruct` yet
- do not rerun standing successful grids unless aggregation requires a tiny missing artifact
- keep all new work centered on manuscript-facing consolidation, theorem-package preparation, and statistics

| ID | Paper claim / section | Current standing evidence | Missing evidence | Target artifact | Execute where (Local / Chimera) | Gate | Priority |
|---|---|---|---|---|---|---|---|
| S0 | Status hygiene / project state | `compiled-c3-r4`, `batch3c`, and `batch3d` are accepted; historical failures still clutter top-level status wording | current-state wording and archived-failure separation | `PROJECT_STATUS.md` | Local | Must be internally consistent before any new package is treated as active | P0 |
| S1 | Main clean path support for Section 7.4 (`\label{subsec:exp-main}`) | `compiled-c3-r4` clean baselines passed on `U00/U03/U12/U15 @ seed 17` | manuscript-facing clean table, aggregate statistics, explicit inclusion list | `results/tables/compiled_c3_r4_main.csv`; `results/tables/compiled_c3_r4_main.tex`; `results/processed/paper_stats/main_clean_summary.json`; `results/processed/paper_stats/run_inclusion_lists.json` | Local | No new experiment package should outrun the clean evidence table | P0 |
| S2 | Robustness support for Section 7.3 and Section 7.5 (`\label{subsec:exp-recovery}`, `\label{subsec:exp-limits}`) | `batch3c` and `batch3d` passed on accepted compiled-c3 baselines | manuscript-facing robustness tables, family summary, explicit inclusion list | `results/tables/batch3cd_appendix.csv`; `results/tables/batch3cd_appendix.tex`; `results/tables/batch3_family_summary.csv`; `results/tables/batch3_family_summary.tex`; `results/processed/paper_stats/robustness_summary.json`; `results/processed/paper_stats/run_inclusion_lists.json` | Local | Freeze any new attack-family or payload expansion until these artifacts exist | P0 |
| T1 | Theorem 1 / tokenizer-contextual alignment package (`Section 4`) | `theorem1_qwen7b/contextual_exact` and `theorem1_qwen7b_rerun_sequence_proxy_v2` are both accepted on the same compiled `U03` target; the repaired `sequence_proxy` arm now conditions on `Payload label` while keeping scaffolded non-contextual completion format | paper-facing T1 comparison table and explicit theorem inclusion list | `results/processed/paper_stats/t1_summary.json`; `results/tables/t1_contextual_alignment.csv`; `results/tables/t1_contextual_alignment.tex`; `docs/t1_chimera_package.md` | Local consolidation only | Freeze new `T1` reruns unless the paper claim changes or a stricter matched-verifier rerun is explicitly requested | P0 |
| T2 | Theorem 2 / objective comparison package (`Section 5`) | `T2-r1` now produced a discriminative Chimera result on a multi-member-bucket Qwen catalog: `fixed_representative` passes the exact-slot gate, while `bucket_mass` and `uniform_bucket` remain bucket-correct but fail exact-slot by selecting non-canonical members | manuscript-facing theorem table and supplementary bucket-level metrics table; explicit theorem run inclusion list | `results/processed/paper_stats/t2_r1_summary.json`; `results/tables/t2_r1_objective_comparison.csv`; `results/tables/t2_r1_objective_comparison.tex`; `results/tables/t2_r1_bucket_supplement.csv`; `results/tables/t2_r1_bucket_supplement.tex`; `docs/t2_chimera_package.md` | Local consolidation; no new Chimera sweep required | Freeze new theorem reruns unless the paper claim changes; keep same model family and current eval gate | P0 |
| C1 | Statistics / error bars / compute accounting (`Section 7` global reporting) | seeds, payloads, train/eval/attack summaries, resolved configs, and submission metadata already exist locally | CI-ready aggregation, compute accounting, exact inclusion manifests, manuscript-facing summary CSV | `results/processed/paper_stats/compute_accounting.json`; `results/processed/paper_stats/run_inclusion_lists.json`; `results/tables/stat_summary.csv`; `results/tables/compute_accounting.csv` | Local | Must reflect only accepted standing runs | P1 |
| B1 | Matched-budget benchmark scaffolding (`Section 7.4`) | no active benchmark package yet | benchmark protocol, config skeletons, and calibration plan only | `docs/baseline_protocol.md`; future `configs/experiment/prep/exp_benchmark_*` | Local prepare only | Blocked until `T1`, `T2`, and `C1` are stable | P2 |
| R1 | Llama-3.1-8B replication (`Section 7.4` replication claim) | no active clean replication package | clean replication prep package only | future `configs/experiment/prep/exp_*_llama3_1_8b_*` | Local prepare only | Blocked until Qwen theorem-package work is stable | P2 |

## Execution Order

1. `S0`: make `PROJECT_STATUS.md` internally consistent.
2. `S1` and `S2`: materialize the standing clean and robustness tables from accepted runs only.
3. `C1`: add statistics, compute accounting, and exact inclusion lists for those same standing runs.
4. `T1` and `T2`: consolidate the accepted theorem results into paper-facing artifacts; no new Chimera sweep unless the paper claim changes.
5. Keep `B1` and `R1` blocked.

## Non-Goals For This Stage

- no `Batch 3E` or larger robustness grid
- no new attack families beyond standing accepted results
- no new baselines
- no new model families
- no redesign of the repo
