# Baseline Gap Audit

Status: audit generated from local artifacts on 2026-04-28.

## Inputs Read

- `results/processed/paper_stats/baseline_summary.json`
- `results/processed/paper_stats/baseline_calibration_summary.json`
- `results/tables/matched_budget_baselines.csv`
- `results/tables/baseline_calibration.csv`
- `results/tables/baseline_far_summary.csv`
- `results/tables/baseline_utility_summary.csv`
- `docs/baseline_protocol.md`
- `docs/calibration_protocol.md`
- `manuscripts/69db2644566dcc36c9da320e/section_02_related_work.tex`
- `manuscripts/69db2644566dcc36c9da320e/section_07_experiments.tex`
- `manuscripts/69db2644566dcc36c9da320e/appendix/extended_related_work.tex`
- `manuscripts/69db2644566dcc36c9da320e/references.bib`

The detailed method matrix is in `results/tables/baseline_gap_matrix.csv`.

## Current Baseline Artifact State

The baseline package is internally complete for the currently defined 36-row ownership denominator:

| Quantity | Value |
|---|---:|
| target_count | 36 |
| reporting_row_count | 48 |
| completed_count | 36 |
| valid_completed_count | 36 |
| success_count | 24 |
| method_failure_count | 12 |
| invalid_excluded_count | 0 |
| pending_count | 0 |
| control_unavailable_count | 12 |
| contract_hash_status_counts | `{"match": 36}` |
| baseline artifact paper_ready | `true` |

The 48 reporting rows break down as follows:

| Method | Rows | Artifact result |
|---|---:|---|
| fixed_representative | 12 | valid successes |
| uniform_bucket | 12 | valid successes |
| english_random_active_fingerprint | 12 | valid method failures |
| kgw_provenance_control | 12 | task-mismatched unavailable control rows |

Calibration artifacts are complete for the current minimal package:

| Quantity | Value |
|---|---:|
| calibration completed_count | 48 |
| calibration pending_count | 0 |
| thresholds_frozen | `true` |
| missing_negative_sets | `[]` |

This means the package is artifact-consistent for what it currently implements. It does not mean the baseline set is venue-sufficient.

## Main Gap

No strong external active ownership baseline is completed under matched false-accept and utility budgets.

The current implemented baselines are:

- `fixed_representative`: an internal objective and decoding ablation.
- `uniform_bucket`: an internal objective ablation.
- `english_random_active_fingerprint`: an executable weak natural-language proxy that fails under the payload-recovery verifier. It is not a faithful implementation of Instructional Fingerprinting.
- `kgw_provenance_control`: a task-mismatched watermark/provenance control that is explicitly unavailable and outside the ownership denominator.
- `baseline_chain_hash_qwen_v1`: a prepared Chain&Hash-style external ownership package with manifests and pending summary rows, but no completed final rows, FAR calibration, or utility suite yet.

The baseline protocol correctly states that provenance controls must not be treated as primary ownership baselines. It also states that CTCC and ESF adapters are safe placeholders unless real implementations are wired and audited. The source tree confirms this: `src/baselines/ctcc_adapter.py`, `src/baselines/esf_adapter.py`, and `src/baselines/kgw_adapter.py` are placeholder adapters.

## Method Classification

The audit classifies methods into six roles:

| Method | Classification | Main reason |
|---|---|---|
| fixed representative | internal ablation | Tests canonical representative forcing within this paper's own design. |
| uniform bucket | internal ablation | Tests uniform bucket supervision within this paper's own design. |
| English-random / Instructional Fingerprinting | weak legacy baseline | Current row is a weak proxy. It is not a full Instructional Fingerprinting implementation. |
| Chain & Hash | strong external ownership baseline | Direct active LLM ownership baseline. Package prepared; final rows and calibration pending. |
| Scalable Fingerprinting / Perinucleus | strong external ownership baseline | Closest scalable active fingerprint family. Not implemented. |
| MergePrint | strong external ownership baseline | Direct robust black-box ownership verification baseline. Not implemented. |
| CTCC | strong external ownership baseline | Direct robust fingerprinting framework. Only placeholder adapter exists. |
| EverTracer | strong external ownership baseline | Direct probabilistic stolen-model fingerprinting. Not implemented. |
| MEraser | attack/scrubbing evaluation | Erasure method, not an ownership baseline. |
| KGW | provenance/task-mismatched control | Text provenance watermark, not model ownership payload recovery. |
| PostMark | provenance/task-mismatched control | Robust black-box text watermark, not ownership payload recovery. |
| ESF/RESF | related work only | Black-box tamper detection, not currently a payload ownership baseline. |

## Manuscript Consistency

The related-work sections are mostly aligned with this audit. They separate text provenance, active fingerprinting, erasure, and passive provenance. They also cite the external active fingerprinting families that should be considered for stronger baselines.

The experiments section is stale relative to the current artifacts. It still says no complete paper-facing matched-budget baseline table is linked and keeps a placeholder baseline table. That statement is no longer accurate for the minimal internal baseline package, because `matched_budget_baselines.csv` now exists and is internally complete. However, the manuscript's conservative conclusion is still correct: it should not claim baseline superiority until at least one strong external active ownership baseline is implemented and calibrated.

## Minimum Additional Experiments

The next baseline package should not expand into a broad zoo before adding one real external ownership baseline. The minimum defensible package is:

1. Implement one strong external active ownership baseline under the frozen B0/B1 protocol. Recommended first choices are Chain & Hash or a faithful Instructional Fingerprinting implementation because they are closest to black-box text-only active verification.
2. Calibrate the baseline with the frozen split in `docs/calibration_protocol.md`, including `target_far = 0.01`, `M = 4`, and all negative sets.
3. Report final rows on the same Qwen backbone, final payloads `U00/U03/U12/U15`, seeds `17/23/29`, and the same utility budget.
4. Keep valid failures in the denominator.
5. Add provenance controls only as controls, not as main ownership baselines.
6. Add MEraser or related erasure work only as an attack or scrubbing evaluation after a real external fingerprint baseline exists.

For a stronger NeurIPS package, implement at least two external active ownership baselines from distinct families, for example Chain & Hash plus MergePrint or CTCC, and evaluate at matched FAR and utility budgets.

## Sufficiency Conclusion

Workshop: not sufficient for a workshop-level ownership-verification baseline package if the paper makes a baseline-superiority or state-of-the-art ownership-verification claim. The current artifacts can only be used as preliminary internal ablations plus a documented weak proxy, with explicit acknowledgement that no strong external active ownership baseline has completed matched-budget evaluation.

NeurIPS main conference: not sufficient. The artifact package is internally consistent, but it lacks a completed real external active ownership baseline under matched FAR and utility budgets.

NeurIPS Spotlight: not sufficient. Spotlight-level evidence requires at least one, and preferably multiple, strong external active ownership baselines completed under the frozen matched-budget protocol, plus utility and false-accept calibration and clear attack or scrubbing evaluation.

Required blocking statement: no current baseline package should be described as sufficient for NeurIPS main or Spotlight until at least one strong external active ownership baseline is completed under matched FAR and utility budgets.
