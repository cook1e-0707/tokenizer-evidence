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
| G1 | Main-path scale package: payload × seed coverage | publication-scale Qwen7B clean package is now standing on `U00..U15 @ seeds 17,23,29`; final aggregate is `48/48` included, `0` pending, `0` excluded | none for the current Qwen7B payload x seed package beyond freeze-and-cite manuscript alignment | `configs/experiment/scale/exp_train__qwen2_5_7b__g1_payload_seed_scale_v1.yaml`; `configs/experiment/scale/exp_eval__qwen2_5_7b__g1_payload_seed_scale_v1.yaml`; `results/tables/g1_payload_seed_scale.csv`; `results/tables/g1_payload_seed_scale.tex`; `results/processed/paper_stats/g1_summary.json`; `results/processed/paper_stats/g1_run_inclusion_list.json` | Local review only | Freeze current package; later scale packages must not change G1 accounting | P0 |
| G2 | Prompt-family scale package | standing on `PF1/PF2/PF3 x U00/U03/U12/U15 x seeds 17,23,29`; final aggregate is `36/36` included, `0` pending, `0` excluded | none for the current Qwen7B prompt-family package beyond Chimera-side canonical artifact regeneration and manuscript alignment | `configs/experiment/scale/exp_train__qwen2_5_7b__g2_prompt_family_scale_v1.yaml`; `configs/experiment/scale/exp_eval__qwen2_5_7b__g2_prompt_family_scale_v1.yaml`; `results/tables/g2_prompt_family_scale.csv`; `results/tables/g2_prompt_family_scale.tex`; `results/processed/paper_stats/g2_summary.json`; `results/processed/paper_stats/g2_run_inclusion_list.json` | Local review only | Freeze current package; later scale packages must not change G2 prompt-family definitions | P0 |
| G3a | Block-count scale package | `G3a-v3` is standing on held-out `B1/B2/B4 x U00..U15 x seeds 17,23,29`; final aggregate is `142/144` valid successes, `2` valid method failures, `0` invalid exclusions, `0` pending | none before manuscript alignment; do not rerun v3 unless claim changes | `configs/experiment/scale/g3a_v3/exp_train__qwen2_5_7b__g3a_block_scale_v3.yaml`; `configs/experiment/scale/g3a_v3/exp_eval__qwen2_5_7b__g3a_block_scale_v3.yaml`; `results/tables/g3a_v3_block_scale.csv`; `results/tables/g3a_v3_block_scale.tex`; `results/tables/g3a_v3_failure_cases.csv`; `results/tables/g3a_v3_slot_margin.csv`; `results/processed/paper_stats/g3a_v3_summary.json`; `results/processed/paper_stats/g3a_v3_run_inclusion_list.json`; `results/processed/paper_stats/g3a_v3_compute_accounting.json` | Local review only | Treat the `2/144` failures as valid method failures in the denominator; artifact-paper-ready is true, claim wording must report failures honestly | P0 |
| G4 | Training-signal scale package | standing on fixed Qwen7B, B2, codebook, prompt family, G3a-v3 hp04 objective, and `S16/S32/S64/S128 x U00/U03/U12/U15 x seeds 17,23,29`; final aggregate is `48/48` valid successes, `0` method failures, `0` invalid exclusions, `0` pending | none before manuscript alignment; do not rerun unless claim changes | `configs/experiment/scale/g4/exp_train__qwen2_5_7b__g4_train_signal_scale_v1.yaml`; `configs/experiment/scale/g4/exp_eval__qwen2_5_7b__g4_train_signal_scale_v1.yaml`; `configs/reporting/g4_train_signal_scale_v1.yaml`; `results/tables/g4_train_scale.csv`; `results/tables/g4_train_scale.tex`; `results/processed/paper_stats/g4_summary.json`; `results/processed/paper_stats/g4_run_inclusion_list.json` | Local review only | Only effective training-signal sample count varies; `S128` is explicitly `64` unique compiled samples repeated twice under `compiled_sample_repeats=2` | P0 |
| C1 | Statistical reporting / compute accounting / reproducibility | G3a-v3 and G4 standing compute/stat/inclusion rows are now represented in canonical C1 artifacts | maintain C1 consistency as B0/B1/B2 packages are added later | `results/processed/paper_stats/compute_accounting.json`; `results/processed/paper_stats/run_inclusion_lists.json`; `results/tables/stat_summary.csv`; `results/tables/compute_accounting.csv`; package-local `*_compute_accounting.json` | Local | Must separate validation/preliminary compute from final reported package compute and keep valid method failures in denominators | P1 |
| B0 | Benchmark protocol / matched-budget calibration | frozen matched-budget protocol exists for query budget, threshold selection, FAR, utility, and inclusion/exclusion semantics | none before B1/B2 package implementation | `docs/baseline_protocol.md`; `docs/calibration_protocol.md` | Local only | Baseline execution remains blocked until B1/B2 configs, calibration artifacts, and dry-run manifests are reviewed | P1 |
| B1 | Active fingerprint baseline comparison | protocol-ready only; no standing baseline result yet | one active ownership/fingerprinting baseline under matched utility + matched FAR | future `results/tables/matched_budget_baselines.csv`; future `results/processed/paper_stats/baseline_summary.json` | Local prepare, Chimera later | Can be prepared under frozen B0; cannot execute until dry-run and calibration artifacts exist | P2 |
| B2 | Provenance / watermark control comparison | protocol-ready only; no standing control result yet | one provenance-style control baseline under the same comparison protocol, clearly labeled as task-mismatched control rather than primary ownership baseline | future `results/tables/matched_budget_baselines.csv`; future `results/processed/paper_stats/baseline_summary.json` | Local prepare, Chimera later | Can be prepared under frozen B0; cannot execute until dry-run and calibration artifacts exist | P2 |
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

### G3a / G3b: block and bucket-member scale
Goal:
answer the “toy bucket / toy slot count” criticism.

Current split:
- `G3a-v3` varied only `block_count: 1 -> 2 -> 4` under the current single-member `4 x 4` compiled codebook and is now standing at `142/144` valid successes.
- `G3b` is reserved for `members_per_bucket: 1 -> 2 -> 4` at fixed `block_count = 2`; do not run it before G4 and the baseline protocol are complete unless the paper claim specifically needs bucket-member scale.

### G4: training-signal scale
Goal:
answer the “works only with tiny hand-crafted sample set” criticism.

Minimum scale ladder:
- effective contract sample counts: `16, 32, 64, 128`
- same model
- same codebook
- same prompt contract
- same eval path
- `128` is pre-registered as `64` unique compiled samples repeated twice because fixed `B2 x 16 payload labels x 2 fields` has only `64` unique compiled slot samples.

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
5. `B0`: write and freeze matched-budget benchmark protocol. Completed.
6. `B1` + `B2`: prepare the minimal baseline comparison package under frozen B0, then run only after configs, calibration artifacts, and dry-run manifests are reviewed.
7. `R1`: run one minimal replication package on a second model family.

---

## Non-goals for this stage

- no Batch 3E or larger robustness-grid expansion
- no new attack families beyond currently standing accepted results
- no H200 / multi-GPU / NCCL path
- no broad new model-family sweep
- no broad baseline zoo
- no repo redesign
