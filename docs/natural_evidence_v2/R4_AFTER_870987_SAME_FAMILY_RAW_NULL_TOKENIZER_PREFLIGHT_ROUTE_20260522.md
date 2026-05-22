# R4 After-870987 Same-Family Raw-Null Tokenizer Preflight Route

Date: 2026-05-22

## Decision

The passed Qwen first-token event locked-scale and pre-FAR null package allows
artifact-only planning for same-family raw-null controls, but it does not allow
same-family generation yet. Before any Qwen 3B/7B/14B raw-null generation, the
same organic-null row bank must pass actual tokenizer boundary preflight under
each planned tokenizer.

## Scope

This route is tokenizer-only.

- Tokenizers: `Qwen/Qwen2.5-3B-Instruct`,
  `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`
- Row bank:
  `results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_row_bank_v2_plan_20260521/row_allocation_rows.jsonl`
- Rows per tokenizer: 262,144
- Total checked rows: 786,432
- Slurm wrapper:
  `scripts/natural_evidence_v2/slurm/r4_after_870987_same_family_raw_null_tokenizer_boundary_preflight_h200.sbatch`
- Array: `0-2%3`
- Partition/QOS/account/GRES: `pomplun` / `pomplun` / `cs_yinxin.wan` / `gpu:h200:1`
- Time limit: `30-00:00:00`

## Gates

Each tokenizer must satisfy:

- checked rows = 262,144
- failed rows = 0
- empty target token rows = 0
- empty other token rows = 0
- target/other overlap rows = 0

The route starts no model forward pass, scoring, generation, training, Llama,
sanitizer, FAR aggregation, payload-diversity claim, or paper-facing claim.

## Downstream Rule

Same-family raw-null generation remains blocked until this tokenizer preflight
array passes, is reviewed, and a separate exactly-one allowlist submission
preflight is recorded.
