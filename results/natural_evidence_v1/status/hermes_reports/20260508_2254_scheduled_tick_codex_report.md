Blocked the requested repair/rerun as stale instead of duplicating it.

Live status already records repaired scaffold `...2308` and Slurm job `850242` with configured-tokenizer stability `PASS`, `unstable_token_count=0`, density/mass still `NOT_EVALUATED`, and WP4 still locked. I added the blocker report here:

- [20260508_2254_wp3_repair_rerun_already_completed_blocker.md](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260508_2254_wp3_repair_rerun_already_completed_blocker.md:1)
- [20260508_2254_wp3_repair_rerun_already_completed_blocker.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260508_2254_wp3_repair_rerun_already_completed_blocker.json:1)

Validation: `python3 -m json.tool` passed for the blocker JSON. No Slurm job, training, generation, E2E, FAR, Llama, same-family null, sanitizer, or Chimera login-node CPU work was started.