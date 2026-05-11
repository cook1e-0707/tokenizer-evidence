# WP6-R2 Option B Scale Eval 852426 Review

## Decision

Slurm job `852426` completed successfully, and its artifacts were synced to:

```text
results/natural_evidence_v2/status/wp6_r2_option_b_scale_eval_852426/
```

The precommitted WP6-R2 Option B robust-block scale gate passed.

This is not a FAR result, not a Llama result, not a same-family null, not a
sanitizer benchmark, and not a paper-facing positive claim.

## Slurm Result

```text
job_id = 852426
job_name = nat-ev-v2-wp6r2b
partition = DGXA100
node = chimera12
state = COMPLETED
exit_code = 0:0
elapsed = 01:27:40
start = 2026-05-10T08:34:12
end = 2026-05-10T10:01:52
```

Synced Slurm logs:

```text
results/natural_evidence_v2/status/wp6_r2_option_b_scale_eval_852426/slurm/nat-ev-v2-wp6r2b-852426.out
results/natural_evidence_v2/status/wp6_r2_option_b_scale_eval_852426/slurm/nat-ev-v2-wp6r2b-852426.err
```

## Artifact Completeness

All precommitted required outputs are present and nonempty:

```text
precommit/wp6_r2_option_b_contract.json
wp6_generation_summary.json
wp6_generated_outputs.jsonl
wp6_e2e_summary.json
wp6_slot_observations.jsonl
wp6_decode_decisions.jsonl
coordinate_majority_r2_option_b/wp6_r2_option_b_decode_rows.jsonl
coordinate_majority_r2_option_b/wp6_r2_option_b_summary.json
coordinate_majority_r2_option_b/wp6_r2_option_b_support_by_block_budget.csv
coordinate_majority_r2_option_b/wp6_r2_option_b_contract.json
```

Observed row counts:

```text
wp6_generated_outputs.jsonl = 1536
wp6_decode_decisions.jsonl = 2560
wp6_slot_observations.jsonl = 40785
coordinate_majority_r2_option_b/wp6_r2_option_b_decode_rows.jsonl = 160
```

## Precommit Check

The synced precommit and decode contracts match and record the intended locked
R2 Option B scope:

```text
precommit_hash_sha256 = 0d29b86e123c9cc0946ca62b50ecd74fb61efc58ba2448728f8847055c054eb6
protocol_id = natural_evidence_v2_wp6_r2_option_b_robust_block_scale
decoder_id = qwen_v2_wp6_r2_robust_block_coordinate_majority_decoder_v1
payload_plus_checksum_hex = a55e
selected_split = wp3_r1_eval
selected_prompt_file_rows = 768..1279
block_count = 8
block_size = 64
query_budgets_per_block = [8, 16, 32, 64]
transcript_precommitted_before_generation = true
transcript_provenance = precommitted_replacement_run
```

No synced artifact contains the stale marker
`post_hoc_not_precommitted_for_852086`.

## Scale Gate

Controlling budget: `64`.

| Gate | Requirement | Observed | Status |
|---|---:|---:|---|
| protected block accepts | >= 6 / 8 | 7 / 8 | pass |
| raw accepts | 0 / 8 | 0 / 8 | pass |
| task-only accepts | 0 / 8 | 0 / 8 | pass |
| wrong-key accepts | 0 / 8 | 0 / 8 | pass |
| wrong-payload accepts | 0 / 8 | 0 / 8 | pass |
| min support in accepted protected blocks | >= 16 | 26 | pass |
| min majority margin in accepted protected blocks | >= 3 | 5 | pass |
| forbidden public surface count | 0 | 0 | pass |
| output artifacts complete | required | complete | pass |

Protected block outcomes at budget `64`:

```text
block_0 a55e accepted support=28 margin=6
block_1 a55e accepted support=34 margin=14
block_2 a55e accepted support=31 margin=5
block_3 a55e accepted support=39 margin=21
block_4 a55e accepted support=41 margin=15
block_5 a55e accepted support=26 margin=16
block_6 a55e accepted support=34 margin=12
block_7 b014 rejected support=7 margin=1
```

Result:

```text
scale_gate_status = PASS_WP6_R2_OPTION_B_ROBUST_BLOCK_SCALE_GATE
scale_gate_pass = true
```

## Diagnostic Context

The legacy exact-frame WP6 E2E summary remains diagnostic context only for this
R2 review. It still fails its older proof-of-life gate with
`protected_accept_rate_at_64 = 0.0625`; that older exact-frame gate is not the
controlling precommitted R2 Option B robust-block gate.

## Validation

Local artifact checks:

```text
required synced files: present and nonempty
synced precommit/decode contracts: byte-identical
stale metadata marker search: pass
Slurm sacct terminal state: COMPLETED 0:0
```

## Next Allowed Action

Stop after this review until the next route is explicitly recorded. Do not
submit another WP6 job, train, rerun Qwen E2E, start Llama or same-family
nulls, run a sanitizer benchmark, aggregate FAR, or make a paper-facing
positive claim from this scale result.
