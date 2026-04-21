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
| T1 | Theorem 1 / tokenizer-contextual alignment package (`Section 4`) | indirect support from compile-then-train success only | controlled package contrasting exact contextual compiled path vs sequence/non-contextual proxy, ready for dry-run manifest generation | `configs/experiment/prep/exp_train__qwen2_5_7b__t1_*.yaml`; `configs/experiment/prep/exp_eval__qwen2_5_7b__t1_*.yaml`; `results/processed/paper_stats/theorem_package_dry_runs.json` | Local prepare, Chimera later | Same `Qwen/Qwen2.5-7B-Instruct`; same compiled codebook; no sweep yet | P1 |
| T2 | Theorem 2 / objective comparison package (`Section 5`) | no paper-facing objective-contrast package exists yet | dry-run-ready package for `bucket_mass` vs `fixed_representative` vs `uniform_bucket` on the same compiled path | `configs/experiment/prep/exp_train__qwen2_5_7b__t2_*.yaml`; `configs/experiment/prep/exp_eval__qwen2_5_7b__t2_*.yaml`; `results/processed/paper_stats/theorem_package_dry_runs.json` | Local prepare, Chimera later | Same codebook; same model; no baseline opening; no full Chimera sweep yet | P1 |
| C1 | Statistics / error bars / compute accounting (`Section 7` global reporting) | seeds, payloads, train/eval/attack summaries, resolved configs, and submission metadata already exist locally | CI-ready aggregation, compute accounting, exact inclusion manifests, manuscript-facing summary CSV | `results/processed/paper_stats/compute_accounting.json`; `results/processed/paper_stats/run_inclusion_lists.json`; `results/tables/stat_summary.csv`; `results/tables/compute_accounting.csv` | Local | Must reflect only accepted standing runs | P1 |
| B1 | Matched-budget benchmark scaffolding (`Section 7.4`) | no active benchmark package yet | benchmark protocol, config skeletons, and calibration plan only | `docs/baseline_protocol.md`; future `configs/experiment/prep/exp_benchmark_*` | Local prepare only | Blocked until `T1`, `T2`, and `C1` are stable | P2 |
| R1 | Llama-3.1-8B replication (`Section 7.4` replication claim) | no active clean replication package | clean replication prep package only | future `configs/experiment/prep/exp_*_llama3_1_8b_*` | Local prepare only | Blocked until Qwen theorem-package work is stable | P2 |

## Execution Order

1. `S0`: make `PROJECT_STATUS.md` internally consistent.
2. `S1` and `S2`: materialize the standing clean and robustness tables from accepted runs only.
3. `C1`: add statistics, compute accounting, and exact inclusion lists for those same standing runs.
4. `T1` and `T2`: prepare dry-run packages only; no Chimera sweep yet.
5. Keep `B1` and `R1` blocked.

## Non-Goals For This Stage

- no `Batch 3E` or larger robustness grid
- no new attack families beyond standing accepted results
- no new baselines
- no new model families
- no redesign of the repo
