Blocked and stopped as required.

I recorded the blocker report at [20260511_1404_r3_2_submission_blocker.md](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_1404_r3_2_submission_blocker.md) with machine-readable JSON at [20260511_1404_r3_2_submission_blocker.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_1404_r3_2_submission_blocker.json).

Reason: TG/email notification succeeded, but submission is still unsafe because `configs/natural_evidence_v2/run_allowlist.yaml` has forbidden `llama_v2_wp6_e2e_eval` enabled while Llama remains disallowed. I did not enable R3.2, did not submit Slurm, and did not run generation/training/Llama/sanitizer/FAR work.

Validation: blocker JSON parses with `python3 -m json.tool`; report files are nonempty.