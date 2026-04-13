# Result Schema

All stage-4 run outputs are schema-aware JSON objects with fixed provenance fields:

- `run_id`
- `experiment_name`
- `method_name`
- `model_name`
- `seed`
- `git_commit`
- `timestamp`
- `hostname`
- `slurm_job_id`
- `status`

## Primary schemas

- `train_run_summary`
- `eval_run_summary`
- `calibration_summary`
- `attack_run_summary`
- `aggregated_comparison_row`

## Eval outputs

Canonical eval runs write these fixed files inside `results/raw/<experiment>/<run_id>/`:

- `eval_summary.json`
  - schema: `eval_run_summary`
  - required stage-4 fields include:
    - `verification_mode`
    - `render_format`
    - `verifier_success`
    - `decoded_payload`
    - `decoded_unit_count`
    - `decoded_block_count`
    - `unresolved_field_count`
    - `malformed_count`
    - `utility_acceptance_rate`
    - `diagnostics`
- `verifier_result.json`
  - detailed parser/verifier output for inspection
- `rendered_evidence.txt`
  - canonical text passed into the verifier
- `rendered_evidence.json`
  - structured render metadata

## Calibration outputs

Calibration runs write:

- `calibration_summary.json`
  - schema: `calibration_summary`
  - required stage-4 fields include:
    - `target_far`
    - `threshold`
    - `observed_far`
    - `calibration_target`
    - `operating_point_name`
    - `threshold_candidates`
    - `selected_metric_name`
    - `selected_metric_value`

## Aggregation

`scripts/summarize.py` loads result files by `schema_name`, not by filename patterns. It writes:

- `results/processed/run_summaries.jsonl`
- `results/processed/comparison_rows.jsonl`

This keeps summarization compatible with both current stage-4 outputs and older legacy files that still carry known schema aliases such as `calibration_output`.
