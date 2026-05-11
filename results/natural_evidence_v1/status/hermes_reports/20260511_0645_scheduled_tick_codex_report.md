Implemented one allowed project-advancing action: the R3.2 same-contract `a55e` full wrapper aggregation path is now in place, with no generation, no allowlist enablement, and no Slurm submission.

Changed:
- Added [aggregate_r3_2_locked_scale_shards.py](/Users/guanjie/Documents/tokenizer_alignment/scripts/natural_evidence_v2/aggregate_r3_2_locked_scale_shards.py)
- Upgraded [r3_2_qwen_locked_scale_eval.sbatch](/Users/guanjie/Documents/tokenizer_alignment/scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch)
- Added focused aggregation coverage in [test_natural_evidence_v2_wp6_coordinate_majority.py](/Users/guanjie/Documents/tokenizer_alignment/tests/test_natural_evidence_v2_wp6_coordinate_majority.py)
- Recorded state in [R3_2_FULL_WRAPPER_AGGREGATION_PATH_20260511_0645.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v2/R3_2_FULL_WRAPPER_AGGREGATION_PATH_20260511_0645.md), [CURRENT_STATE.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v2/CURRENT_STATE.md), and v2 gate status.

Validation passed:
- `py_compile`
- `bash -n`
- `.venv/bin/python -m pytest tests/test_natural_evidence_v2_wp6_coordinate_majority.py` → `10 passed`
- local `VALIDATE_PLAN_ONLY=1` wrapper run into `results/natural_evidence_v2/status/r3_2_full_wrapper_plan_validation_20260511_0645`

Current gate remains conservative: `wrapper_review_passed=false`, allowlist disabled, no Slurm submitted. Next allowed action is review of this implemented aggregation path plus the existing exact `852426` replay artifacts.