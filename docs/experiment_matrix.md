# Experiment Matrix

This matrix is the strict execution package for the paper-facing stage.

## Guardrails

- do not expand `Batch 3` beyond currently standing attack families until the scaling packages below are completed
- do not open broad baseline comparison yet
- do not open additional model families beyond the first planned replication until the Qwen7B scaling package is stable
- do not rerun already-standing successful grids unless aggregation or a missing paper artifact strictly requires it
- keep all new work centered on:
  1. manuscript-facing consolidation,
  2. theorem-package consolidation,
  3. scale/generalization evidence,
  4. statistical reporting,
  5. then matched-budget comparison

## What counts as paper-complete evidence

A result package is paper-complete only if it has:
- standing runs
- manuscript-facing table/figure artifacts
- exact run inclusion lists
- error bars / CI or explicit statistical aggregation
- compute accounting
- a clear claim that it supports

---

## Matrix

| ID | Paper claim / section | Current standing evidence | Missing evidence | Target artifact | Execute where | Gate | Priority |
|---|---|---|---|---|---|---|---|
| S0 | Status hygiene / project state | Qwen7B compiled path and Batch 3 standing results exist, but historical failures must remain clearly archived and not pollute current status | fully consistent status wording and archived-failure separation | `PROJECT_STATUS.md` | Local | Must be internally consistent before further paper packaging | P0 |
| S1 | Main clean path support for Section 7.4 | `compiled-c3-r4` standing on Qwen7B with representative payloads `U00/U03/U12/U15 @ seed 17`; representative multi-seed clean coverage also exists | manuscript-facing clean table, exact inclusion list, aggregate stats | `results/tables/compiled_main_clean.csv`, `results/tables/compiled_main_clean.tex`, `results/processed/paper_stats/main_clean_summary.json`, `results/processed/paper_stats/run_inclusion_lists.json` | Local | No new package should outrun the clean table | P0 |
| S2 | Robustness support for Section 7.3 / 7.5 | `batch3c` and `batch3d` standing on accepted clean baselines; payload / seed / family coverage exists | manuscript-facing robustness tables + condensed family summary + inclusion list | `results/tables/batch3_summary.csv`, `results/tables/batch3_summary.tex`, `results/tables/batch3_family_summary.csv`, `results/tables/batch3_family_summary.tex`, `results/processed/paper_stats/robustness_summary.json` | Local | Freeze new attack-family expansion until these artifacts exist | P0 |
| T1 | Theorem 1 / tokenizer-contextual alignment (Section 4) | `contextual_exact` and repaired `sequence_proxy` both standing on accepted runs | paper-facing theorem comparison that highlights exactness / repair burden / stability, not only final pass/fail | `results/tables/t1_alignment.csv`, `results/tables/t1_alignment.tex`, `results/processed/paper_stats/t1_summary.json` | Local first; Chimera only if a tiny missing rerun is strictly needed | Freeze reruns unless claim wording changes | P0 |
| T2 | Theorem 2 / bucket-level objective (Section 5) | `T2-r1` standing on multi-member-bucket Qwen package; discriminative result exists | paper-facing table with corrected metric hierarchy: bucket correctness + utility / lexical distortion primary, exact-slot secondary | `results/tables/t2_objective.csv`, `results/tables/t2_objective.tex`, `results/tables/t2_bucket_supplement.csv`, `results/processed/paper_stats/t2_summary.json` | Local first; Chimera only if a tiny missing rerun is strictly needed | Freeze reruns unless metric definitions change | P0 |
| G1 | Main-path scale package: payload × seed coverage | publication-scale Qwen7B clean package is now standing on `U00..U15 @ seeds 17,23,29`; final aggregate is `48/48` included, `0` pending, `0` excluded | none for the current Qwen7B payload x seed package beyond freeze-and-cite manuscript alignment | `configs/experiment/scale/exp_train__qwen2_5_7b__g1_payload_seed_scale_v1.yaml`; `configs/experiment/scale/exp_eval__qwen2_5_7b__g1_payload_seed_scale_v1.yaml`; `results/tables/g1_payload_seed_scale.csv`; `results/tables/g1_payload_seed_scale.tex`; `results/processed/paper_stats/g1_summary.json`; `results/processed/paper_stats/g1_run_inclusion_list.json` | Local review only | Freeze current package; remaining scale work should continue at `G3` after G2 | P0 |
| G2 | Prompt-family scale package | standing on `PF1/PF2/PF3 x U00/U03/U12/U15 x seeds 17,23,29`; final aggregate is `36/36` included, `0` pending, `0` excluded | none for the current Qwen7B prompt-family package beyond Chimera-side canonical artifact regeneration and manuscript alignment | `configs/experiment/scale/exp_train__qwen2_5_7b__g2_prompt_family_scale_v1.yaml`; `configs/experiment/scale/exp_eval__qwen2_5_7b__g2_prompt_family_scale_v1.yaml`; `results/tables/g2_prompt_family_scale.csv`; `results/tables/g2_prompt_family_scale.tex`; `results/processed/paper_stats/g2_summary.json`; `results/processed/paper_stats/g2_run_inclusion_list.json` | Local review only | Freeze current package; next scale work should start at `G3` without changing G2 prompt-family definitions | P0 |
| G3 | Codebook / slot / block scale package | package is prepared locally for fixed `4 x 4` compiled codebook with `block_count=1/2/4`; `B2` reuses standing evidence | Chimera execution for new `B1` and `B4` cells: `24 train + 24 eval` | `configs/experiment/scale/exp_train__qwen2_5_7b__g3_codebook_block_scale_v1.yaml`; `configs/experiment/scale/exp_eval__qwen2_5_7b__g3_codebook_block_scale_v1.yaml`; `results/tables/g3_codebook_block_scale.csv`; `results/processed/paper_stats/g3_summary.json` | Local prepare, Chimera run | Keep Qwen7B, the compiled codebook, fields, prompt family, and train payload labels fixed; vary only `block_count` in this package | P1 |
| G4 | Training-signal scale package | current standing path proves existence, but not yet a paper-facing curve over dataset / contract sample count | train-set scale curve (e.g., 16 / 32 / 64 / 128 contract samples) under fixed codebook and fixed model | `configs/experiment/scale/exp_qwen7b_train_scale_*`; `results/tables/g4_train_scale.csv`; `results/processed/paper_stats/g4_summary.json` | Local prepare, Chimera run | Keep all other factors fixed | P1 |
| C1 | Statistical reporting / compute accounting / reproducibility | seeds, payloads, summaries, configs, and submission metadata exist | paper-facing aggregation, CI/error bars, exact run inclusion, compute accounting | `results/processed/paper_stats/compute_accounting.json`; `results/processed/paper_stats/run_inclusion_lists.json`; `results/tables/stat_summary.csv`; `results/tables/compute_accounting.csv` | Local | Must reflect only accepted standing runs or clearly labeled scale packages | P1 |
| B0 | Benchmark protocol / matched-budget calibration | not yet built | benchmark protocol, calibration plan, null FAR protocol, utility protocol | `docs/baseline_protocol.md`; `docs/calibration_protocol.md`; `configs/experiment/prep/exp_benchmark_*` | Local only | Blocked until S1/S2/T1/T2/G1-G4/C1 are stable | P2 |
| B1 | Active fingerprint baseline comparison | none standing yet | one active ownership/fingerprinting baseline under matched utility + matched FAR | future benchmark table | Local prepare, Chimera later | Blocked until B0 and Qwen scale package are complete | P2 |
| B2 | Provenance / watermark control comparison | none standing yet | one provenance-style control baseline under the same comparison protocol, clearly labeled as task-mismatched control rather than primary ownership baseline | future benchmark table | Local prepare, Chimera later | Blocked until B0 and Qwen scale package are complete | P2 |
| R1 | Cross-family replication | no standing replication package yet | one minimal clean compiled path on a second model family (preferably Llama-3.1-8B-Instruct or a literature-aligned 7B/8B model) | future `results/tables/r1_replication.csv` | Local prepare, Chimera later | Blocked until Qwen theorem + scale package is stable | P2 |

---

## Publication-scale design for the new scale packages

### G1: payload × seed scale
Status:
- landed on the current Qwen7B compiled path
- final aggregate is `48/48` included, `0` pending, `0` excluded

Default target:
- payload labels: full `U00..U15`
- seeds: at least `17, 23, 29`
- same Qwen7B compiled main path
- same codebook
- same attack-disabled clean evaluation

Desired evidence:
- one training run per seed if the train set already covers all payloads
- evaluation across all payload labels
- aggregated acceptance / verifier_success / decoded payload correctness

### G2: prompt-family scale
Goal:
answer the “carefully hand-designed prompt” criticism.

Use 3 prompt families with preserved semantics:
- PF1: current deterministic exact-slot prefix
- PF2: semantically equivalent but delimiter-varied prefix
- PF3: semantically equivalent but instruction wording varied prefix

Constraints:
- same fields
- same codebook
- same model
- same block_count
- same train-set size

### G3: codebook / slot / block scale
Goal:
answer the “toy bucket / toy slot count” criticism.

Minimum scaling ladder:
- bucket cardinality: `1 -> 2 -> 4` members per bucket
- block_count: `1 -> 2 -> 4`
- keep fields fixed first
- do not vary too many axes at once

### G4: training-signal scale
Goal:
answer the “works only with tiny hand-crafted sample set” criticism.

Minimum scale ladder:
- contract sample counts: `16, 32, 64, 128`
- same model
- same codebook
- same prompt contract
- same eval path

---

## Comparison protocol principles

### What to compare for baselines
All baselines must be compared on:
1. clean ownership-verification success
2. false-accept rate under null probes
3. utility degradation on organic tasks
4. robustness under the same accepted-before attack protocol
5. compute cost / embedding cost / run cost

### What NOT to do
- do not compare raw thresholded scores at arbitrary operating points
- do not compare a provenance method as if it were a primary ownership-verification baseline
- do not compare on different tokenizers / backbones unless clearly labeled as literature-aligned but not apples-to-apples

---

## Execution order

1. `S0`: fix status hygiene.
2. `S1` + `S2`: materialize standing clean and robustness tables.
3. `T1` + `T2`: turn standing theorem results into manuscript-facing artifacts with correct metric hierarchy.
4. `C1`: add CI/error bars, compute accounting, and exact inclusion lists.
5. `G3` -> `G4`: run the remaining publication-scale packages on Qwen7B while keeping `G1` and `G2` frozen.
6. `B0`: write and freeze matched-budget benchmark protocol.
7. `B1` + `B2`: run minimal baseline comparison package.
8. `R1`: run one minimal replication package on a second model family.

---

## Non-goals for this stage

- no Batch 3E or larger robustness-grid expansion
- no new attack families beyond currently standing accepted results
- no H200 / multi-GPU / NCCL path
- no broad new model-family sweep
- no broad baseline zoo
- no repo redesign
