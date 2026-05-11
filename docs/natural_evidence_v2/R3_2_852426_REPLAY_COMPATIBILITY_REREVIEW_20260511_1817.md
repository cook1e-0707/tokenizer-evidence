# R3.2 852426 Replay Compatibility Re-Review: 2026-05-11 18:17Z

## Decision

PASS: job `852426` replay compatibility remains valid under the repaired R3.2
prompt split contract.

This is artifact-only. It did not submit Slurm, enable an allowlist, start
generation, rerun Qwen E2E, train, start Llama, start same-family nulls, run a
sanitizer benchmark, aggregate FAR, or make paper-facing positive claims.

Machine-readable status:

```text
results/natural_evidence_v2/status/r3_2_852426_replay_compatibility_rereview_20260511_1817.json
```

Fresh replay artifact:

```text
results/natural_evidence_v2/status/r3_2_same_contract_852426_replay_rereview_20260511_1817/r3_2_852426_replay_summary.json
```

## Compatibility Check

The repaired R3.2 prompt split contract uses:

```text
selected_split = wp3_r1_eval
eval_prompt_file_rows = 512..2559
prompt_window_policy = deterministic_4_eval_window_circular_reuse_by_replicate_group_index
```

The reviewed `852426` precommit used:

```text
selected_split = wp3_r1_eval
selected_prompt_file_rows = 768..1279
```

Rows `768..1279` are a contiguous 512-row subset of the repaired R3.2 eval-only
allocation. The replay remains a same-contract `a55e` diagnostic compatibility
check over 8 reviewed WP6-R2 blocks, not the 96-block R3.2 locked-scale gate.

## Replay Result

The artifact-only replay utility was rerun into a fresh output directory:

```text
python3 scripts/natural_evidence_v2/replay_r3_2_same_contract_from_852426.py \
  --output-dir results/natural_evidence_v2/status/r3_2_same_contract_852426_replay_rereview_20260511_1817
```

Observed summary:

```text
replay_exact_match = true
contract_id = a55e
protected accepts at budget 64 = 7/8
raw/task-only/wrong-key/wrong-payload accepts at budget 64 = 0/8 each
min accepted-block support = 26
min accepted-block majority margin = 5
forbidden_public_surface_count = 0
status = PASS_R3_2_SAME_CONTRACT_852426_REPLAY_EXACT_MATCH_NO_SLURM
```

## Still Blocked

This re-review satisfies only the `852426` replay compatibility prerequisite.
Before any new R3.2 Slurm submission, allowlist safety must be rechecked and a
new single-job submission route must be recorded. The R3.2 allowlist entry
remains disabled.

## Status

```text
PASS_R3_2_852426_REPLAY_COMPATIBILITY_REREVIEW_UNDER_REPAIRED_PROMPT_SPLIT_NO_SLURM
```
