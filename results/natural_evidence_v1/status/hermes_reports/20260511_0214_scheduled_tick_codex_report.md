Recorded one small R3.2 prep action: a disabled allowlist placeholder for `v2_r3_2_qwen_locked_scale_eval`, plus a review note making clear it is not submittable until the R3.2 wrapper, precommit contract, local plan-only validation, notification, and gate review are recorded.

Changed:
- [run_allowlist.yaml](/Users/guanjie/Documents/tokenizer_alignment/configs/natural_evidence_v2/run_allowlist.yaml:63)
- [R3_2_QWEN_LOCKED_SCALE_PACKAGE_REVIEW_20260511.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v2/R3_2_QWEN_LOCKED_SCALE_PACKAGE_REVIEW_20260511.md:112)
- [20260511_0214_r3_2_disabled_allowlist_placeholder_review.md](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_0214_r3_2_disabled_allowlist_placeholder_review.md:1)
- [20260511_0214_r3_2_disabled_allowlist_placeholder_review.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_0214_r3_2_disabled_allowlist_placeholder_review.json:1)

Validation passed:
- YAML parsed and confirmed the R3.2 allowlist entry exists exactly once and is `enabled: false`.
- Review JSON parsed and confirms no Slurm submission, no wrapper, and no precommit.
- Confirmed reserved wrapper path `scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch` does not yet exist.

No Slurm job, training, generation, Qwen E2E rerun, Llama, same-family null, sanitizer, FAR aggregation, or paper-facing positive claim was started.