Recorded the artifact-only `852426` replay compatibility re-review under the repaired R3.2 prompt split contract. The replay remains compatible: `852426` used `wp3_r1_eval` rows `768..1279`, inside the repaired R3.2 eval-only rows `512..2559`, and the fresh replay exact-matched.

Updated:
- [R3_2_852426_REPLAY_COMPATIBILITY_REREVIEW_20260511_1817.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v2/R3_2_852426_REPLAY_COMPATIBILITY_REREVIEW_20260511_1817.md)
- [CURRENT_STATE.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v2/CURRENT_STATE.md)
- [r3_2_852426_replay_compatibility_rereview_20260511_1817.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/status/r3_2_852426_replay_compatibility_rereview_20260511_1817.json)
- both v1/v2 `gate_status.json`

Validation: reran the artifact-only replay utility into a fresh directory, verified JSON parses with `jq`, and confirmed no Slurm, generation, Qwen E2E rerun, training, Llama, FAR, sanitizer, or paper claim was started. Next allowed action is now allowlist safety recheck only; no R3.2 Slurm resubmission is unlocked.