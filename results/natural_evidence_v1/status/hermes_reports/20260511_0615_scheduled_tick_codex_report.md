Recorded a blocker instead of upgrading/submitting the wrapper.

The issue: the existing full generation/decode path is single-window with fixed WP6 artifact names, while R3.2 now needs a same-contract `a55e` 12-shard aggregate path with a 96-block gate and exact `852426` replay before any new Slurm submission.

Updated/recorded:
- [blocker report](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_0617_r3_2_full_wrapper_upgrade_blocker.md)
- [blocker JSON](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_0617_r3_2_full_wrapper_upgrade_blocker.json)
- v1/v2 gate status entries

Validation: `jq empty` passed for both gate status files and the new blocker JSON.

No Slurm submission, no allowlist enablement, no generation, no Qwen E2E rerun, no training, and no claims were started.