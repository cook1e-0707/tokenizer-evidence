# WP6-R1 Coordinate-Majority Scale Eval 852202 Review

## Decision

Slurm job `852202` completed successfully, and its artifacts were synced to:

```text
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_scale_eval_852202/
```

The precommitted WP6-R1 scale gate failed.

This is not a FAR result, not a Llama result, not a same-family null, not a
sanitizer benchmark, and not a paper-facing positive claim.

## Slurm Result

```text
job_id = 852202
job_name = nat-ev-v2-wp6r1scale
partition = DGXA100
node = chimera13
state = COMPLETED
exit_code = 0:0
elapsed = 00:44:39
```

Synced Slurm logs:

```text
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_scale_eval_852202/slurm/nat-ev-v2-wp6r1scale-852202.out
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_scale_eval_852202/slurm/nat-ev-v2-wp6r1scale-852202.err
```

## Artifact Completeness

All precommitted required outputs are present:

```text
precommit/wp6_r1_scale_contract.json
wp6_generation_summary.json
wp6_generated_outputs.jsonl
wp6_e2e_summary.json
wp6_slot_observations.jsonl
wp6_decode_decisions.jsonl
coordinate_majority_scale/wp6_r1_scale_decode_rows.jsonl
coordinate_majority_scale/wp6_r1_scale_summary.json
coordinate_majority_scale/wp6_r1_scale_support_by_block_budget.csv
coordinate_majority_scale/wp6_r1_scale_contract.json
```

Observed row counts:

```text
wp6_generated_outputs.jsonl = 768
wp6_decode_decisions.jsonl = 1280
wp6_slot_observations.jsonl = 20421
coordinate_majority_scale/wp6_r1_scale_decode_rows.jsonl = 80
```

## Precommit Check

The synced contract records the intended locked scale scope:

```text
precommit_hash_sha256 = e1997dcfaded1d24fd57a76df181454c7ad3320a46240d957549d3a775f2993b
prompt_source_sha256 = 20154c7b14851ce2116041176ab92acc727f1c49c343826eac9ecfc9430fc179
wp4_contract_sha256 = 69d1feb2b63f52db7cf1ca82bb9ccfcbeb056f2f4f5945b230fc8c44923ada07
selected_split = wp3_r1_eval
selected_prompt_file_rows = 512..767
block_count = 4
block_size = 64
query_budgets_per_block = [8, 16, 32, 64]
payload_plus_checksum_hex = a55e
```

The summary uses the cleaned metadata vocabulary:

```text
precommitted_transcript = true
post_hoc_artifact_replay = false
transcript_provenance = precommitted_replacement_run
```

No synced artifact contains the stale marker
`post_hoc_not_precommitted_for_852086`.

## Scale Gate

Controlling budget: `64`.

| Gate | Requirement | Observed | Status |
|---|---:|---:|---|
| protected block accepts | >= 3 / 4 | 4 / 4 | pass |
| raw accepts | 0 / 4 | 0 / 4 | pass |
| task-only accepts | 0 / 4 | 0 / 4 | pass |
| wrong-key accepts | 0 / 4 | 0 / 4 | pass |
| wrong-payload accepts | 0 / 4 | 0 / 4 | pass |
| min support in accepted protected blocks | >= 16 | 27 | pass |
| min majority margin in accepted protected blocks | >= 3 | 2 | fail |
| forbidden public surface count | 0 | 0 | pass |
| output artifacts complete | required | complete | pass |

Result:

```text
scale_gate_status = FAIL_WP6_R1_COORDINATE_MAJORITY_SCALE_GATE
scale_gate_pass = false
```

The gate failed only on the precommitted minimum majority-margin threshold.
Block `3` accepted the target codeword at budget `64`, but its minimum majority
margin was `2`; the scale contract required at least `3` in every accepted
protected block.

## Diagnostic Context

The legacy exact-frame decoder remains diagnostic only for WP6-R1 scale. In
the synced `wp6_e2e_summary.json`, it still fails its older proof-of-life gate
with protected accept rate `0.125` at budget `64`. That failure does not control
the WP6-R1 block-window majority scale gate, but it is preserved for context.

## Validation

Local artifact checks:

```text
required synced files: present and nonempty
jq scale summary assertion: pass
stale metadata marker search: pass
```

## Next Allowed Action

Artifact-only WP6-R1 scale failure diagnosis and repair planning only.

Do not submit another WP6 job, train, rerun Qwen E2E, start Llama or
same-family nulls, run a sanitizer benchmark, aggregate FAR, or make a
paper-facing positive claim from this failed scale review.
