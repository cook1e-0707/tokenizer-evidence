Recorded the R3.2 full same-contract wrapper review pass.

Changed/added:
- [R3_2_FULL_WRAPPER_REVIEW_20260511_0702.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v2/R3_2_FULL_WRAPPER_REVIEW_20260511_0702.md)
- [r3_2_full_wrapper_review_20260511_0702.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/status/r3_2_full_wrapper_review_20260511_0702.json)
- Updated [CURRENT_STATE.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v2/CURRENT_STATE.md)
- Updated v1/v2 gate status and added the Hermes review-pass report.

Validation run:
- `python3 -m py_compile ...` passed
- `bash -n scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch` passed
- `.venv/bin/python -m pytest tests/test_natural_evidence_v2_wp6_coordinate_majority.py` passed: `10 passed`

No allowlist was enabled, no Slurm job was submitted, and no generation/training/Llama/null/sanitizer/FAR/paper-claim work was started. The next recorded action is allowlist/notification/submission sequencing for exactly one reviewed R3.2 Slurm command.