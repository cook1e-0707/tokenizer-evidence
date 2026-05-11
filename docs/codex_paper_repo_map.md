# Codex Paper Repo Map

Generated after the required read-only scan from repo root:

```bash
pwd
find manuscript -maxdepth 3 -type f | sort
find . -maxdepth 3 -type f \( -name '*.tex' -o -name '*.bib' -o -name '*STATUS*' -o -name '*matrix*' -o -name '*.md' \) | sort
```

## Scan Result

- Repo root: `/Users/guanjie/Documents/tokenizer_alignment`
- Expected manuscript directory: `manuscript/`
- Status of expected manuscript directory: `UNKNOWN` / not present (`find manuscript ...` returned `No such file or directory`)
- Actual manuscript candidate found by scan: `manuscripts/69db2644566dcc36c9da320e/`
- Note: `manuscripts/69db2644566dcc36c9da320e/` is itself a nested git worktree/repository.

## LaTeX Entry

- Main LaTeX entry: `manuscripts/69db2644566dcc36c9da320e/main.tex`
- NeurIPS style/template files:
  - `manuscripts/69db2644566dcc36c9da320e/neurips_2026.sty`
  - `manuscripts/69db2644566dcc36c9da320e/neurips_2026.tex`
- Current compiled PDF artifact:
  - `manuscripts/69db2644566dcc36c9da320e/main.pdf`

## Bibliography

- BibTeX file: `manuscripts/69db2644566dcc36c9da320e/references.bib`
- Main file citation hook:
  - `\bibliographystyle{plainnat}`
  - `\bibliography{references}`

## Main Section Files

Included from `main.tex`:

- `manuscripts/69db2644566dcc36c9da320e/section_01_introduction.tex`
- `manuscripts/69db2644566dcc36c9da320e/section_02_related_work.tex`
- `manuscripts/69db2644566dcc36c9da320e/section_03_problem_setup.tex`
- `manuscripts/69db2644566dcc36c9da320e/section_04_tokenizer_alignment.tex`
- `manuscripts/69db2644566dcc36c9da320e/section_05_bucket_level_injection.tex`
- `manuscripts/69db2644566dcc36c9da320e/section_06_deterministic_verification.tex`
- `manuscripts/69db2644566dcc36c9da320e/section_07_experiments.tex`
- `manuscripts/69db2644566dcc36c9da320e/section_08_discussion_limitations.tex`
- `manuscripts/69db2644566dcc36c9da320e/section_09_conclusion.tex`

## Appendix and Checklist Files

Included from `main.tex`:

- `manuscripts/69db2644566dcc36c9da320e/appendix/results_main.tex`
- `manuscripts/69db2644566dcc36c9da320e/appendix/results_robustness.tex`
- `manuscripts/69db2644566dcc36c9da320e/appendix/proofs.tex`
- `manuscripts/69db2644566dcc36c9da320e/checklist.tex`

Present but not currently included from `main.tex`:

- `manuscripts/69db2644566dcc36c9da320e/appendix/algorithms.tex`

## Figures and Tables

Manuscript-local figure directory:

- `UNKNOWN` / no `figures/` directory found under `manuscripts/69db2644566dcc36c9da320e/`

Repo-level figure files:

- `results/figures/summary.svg`

Repo-level paper table files:

- `results/tables/batch3_family_summary.tex`
- `results/tables/batch3cd_appendix.tex`
- `results/tables/compiled_c3_r4_main.tex`
- `results/tables/g1_payload_seed_scale.tex`
- `results/tables/g2_prompt_family_scale.tex`
- `results/tables/g3a_block_scale.tex`
- `results/tables/g3a_v2_block_scale.tex`
- `results/tables/t1_contextual_alignment.tex`
- `results/tables/t2_r1_bucket_supplement.tex`
- `results/tables/t2_r1_objective_comparison.tex`

Repo-level table source/summary files:

- `results/tables/*.csv`
- `results/tables/summary.md`

## Current Experiment Artifact Paths

Primary processed paper statistics:

- `results/processed/paper_stats/`
- `results/processed/comparison_rows.jsonl`
- `results/processed/run_summaries.jsonl`
- `results/processed/audits/`

Primary raw/local run directories visible in the scan:

- `results/raw/`
- `runs/`
- `Untitled/`
- `batch2/`
- `batch25/`
- `batch26/`
- `batch27/`
- `batch28a_qwen3b/`
- `batch28b_qwen7b/`
- `batch28b_qwen7b_c3r2/`
- `batch28b_qwen7b_c3r3/`
- `batch28b_qwen7b_c3r4/`
- `batch28b_qwen7b_compiled/`
- `batch28b_qwen7b_compiled_c1/`
- `batch28b_qwen7b_compiled_c2/`
- `batch28b_qwen7b_compiled_c3/`
- `batch28b_qwen7b_main/`
- `batch28b_qwen7b_main_rerun/`
- `batch28b_qwen7b_repair/`
- `batch3_preflight_failed/`
- `batch3_preflight_reopen/`
- `batch3_preflight_reopen_rerun/`
- `batch3a_qwen7b/`
- `batch3b_qwen7b/`
- `batch3b_qwen7b_rerun_missing/`
- `batch3c_qwen7b/`
- `batch3d_qwen7b/`
- `foundation_qwen7b_f1/`
- `theorem1_qwen7b/`
- `theorem1_qwen7b_rerun_sequence_proxy/`
- `theorem1_qwen7b_rerun_sequence_proxy_v2/`
- `theorem2_qwen7b/`
- `theorem2_qwen7b_r1/`
- `theorem2_qwen7b_r1_rerun_missing/`

Manifest directories:

- `manifests/`
- `review/g1_chimera_review_2026-04-23/`
- `review/g2_chimera_review_2026-04-23/`

## Unknown or Needs Human Confirmation

- Whether `manuscripts/69db2644566dcc36c9da320e/` is the intended active paper directory, because the requested `manuscript/` path does not exist.
- Whether repo-level `results/tables/*.tex` should be copied, input directly, or manually transcribed into manuscript tables.
- Whether `results/figures/summary.svg` is intended for the paper, because no manuscript-local figure include was found.
- Which experiment batches are authoritative for the final paper narrative if multiple reruns exist for the same claim.
