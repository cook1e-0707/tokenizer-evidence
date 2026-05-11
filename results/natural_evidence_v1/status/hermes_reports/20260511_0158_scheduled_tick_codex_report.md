Recorded one R3.2 package/wrapper-readiness review only.

Added:
- [R3_2_QWEN_LOCKED_SCALE_PACKAGE_REVIEW_20260511.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v2/R3_2_QWEN_LOCKED_SCALE_PACKAGE_REVIEW_20260511.md)
- [r3_2_qwen_locked_scale_package_review_20260511_0200.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/status/r3_2_qwen_locked_scale_package_review_20260511_0200.json)

Outcome: R3.2 scope is fixed as P00/P01/P02/P03 x seeds 17/23/29, 8 blocks per cell, 96 protected blocks total, 5 arms, budget 64 primary with 16/32 diagnostics. The review explicitly marks the current R2 wrapper as not submission-ready for R3.2 because it is single-payload/single-8-block-cell.

Validation: `python3 -m json.tool` passed for the new status JSON. No Slurm, training, generation, Qwen E2E rerun, Llama, same-family null, sanitizer, FAR aggregation, or paper-facing claim was started.