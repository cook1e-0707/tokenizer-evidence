Blocked the submission instead of enabling/submitting.

The 13:48 notification JSON shows TG/email succeeded, but `configs/natural_evidence_v2/run_allowlist.yaml` still has forbidden `llama_v2_wp6_e2e_eval` enabled while `llama_allowed=false`, so the R3.2 “exactly one reviewed Slurm command” condition is not safe or unambiguous.

Wrote the blocker artifacts:

- [20260511_1349_r3_2_submission_blocker.md](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_1349_r3_2_submission_blocker.md)
- [20260511_1349_r3_2_submission_blocker.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_1349_r3_2_submission_blocker.json)

Validation: `jq` parsed the blocker JSON successfully. No allowlist entry was enabled, no Slurm job was submitted, and no CPU/GPU work was started.