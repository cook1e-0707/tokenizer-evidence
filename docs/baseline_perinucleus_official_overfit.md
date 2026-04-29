# Perinucleus Official Single-Fingerprint Overfit Gate

This is a diagnostic gate only. It does not run an anchor or final baseline matrix.

## Decision

`OVERFIT_GATE_PASS: all configured ladder stages passed; anchor may be considered only after review.`

## Stages

| stage | fingerprints | pass | epochs | exact accuracy | rank1 accuracy | mean target rank | mean target probability |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| stage1_one_fingerprint | 1 | True | 1 | 1.0 | 1.0 | 1.0 | 0.9910222291946411 |
| stage2_four_fingerprints | 4 | True | 1 | 0.5 | 0.5 | 4.0 | 0.3394230892881751 |
| stage3_sixteen_fingerprints | 16 | True | 1 | 0.6875 | 0.6875 | 1.75 | 0.5554085455369204 |

## Fidelity Notes

- Fingerprint generation uses the official Scalable Fingerprinting repository at the recorded commit.
- Training is an adapted diagnostic LoRA overfit loop using the same chat-template key/response label contract.
- Target modules are all-linear Qwen modules, not q/v-only LoRA.
- Outputs are not paper baseline results and must not enter the main comparison table.

## Output Files

- Table: `/home/guanjie.lin001/tokenizer-evidence/results/tables/baseline_perinucleus_official_overfit.csv`
- Summary: `/home/guanjie.lin001/tokenizer-evidence/results/processed/paper_stats/baseline_perinucleus_official_overfit_summary.json`
- Compute: `/home/guanjie.lin001/tokenizer-evidence/results/processed/paper_stats/baseline_perinucleus_official_overfit_compute.json`
