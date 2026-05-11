# WP6-R2 Option B 852426 Canonical Review

## Decision

Slurm job `852426` is adopted into the canonical state as the Qwen-only
WP6-R2 Option B positive diagnostic for Route R3.

This adoption is deliberately narrow. It records that the controlled-natural
Qwen v2 micro-slot protocol passed the precommitted robust-block scale gate
under repeated-coordinate majority decoding. It does not authorize a
paper-facing positive claim, full FAR, Llama, same-family nulls, sanitizer
benchmarks, or cross-family generality.

Machine-readable summary:

```text
results/natural_evidence_v2/status/wp6_r2_option_b_852426_canonical_summary.json
```

## Source Job

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

Reviewed source artifacts:

```text
results/natural_evidence_v2/status/wp6_r2_option_b_scale_eval_852426/
```

Previous review:

```text
docs/natural_evidence_v2/WP6_R2_OPTION_B_SCALE_EVAL_852426_REVIEW.md
```

## Provenance

```text
git_head = 0c0d7607d649e457086d5b29c9425f9c7fc42614
precommit_contract_sha256 = b0d2dbdebb0f9e0d915fba114da515ca8b4a83cd005bedff04a01b6353323ec1
precommit_hash_sha256 = 0d29b86e123c9cc0946ca62b50ecd74fb61efc58ba2448728f8847055c054eb6
source_wp4_contract_sha256 = 69d1feb2b63f52db7cf1ca82bb9ccfcbeb056f2f4f5945b230fc8c44923ada07
source_wp4_precommit_hash_sha256 = d8ccb3697e79aca907e4badb098fea22225f5f9a4b27ff58063cd89f1938c5cb
prompt_source_sha256 = 20154c7b14851ce2116041176ab92acc727f1c49c343826eac9ecfc9430fc179
prompt_selection_sha256 = 522d4a879dcbeb9182493c2aed2bc7c85bcba5452bafe1f08380dbdcb2986dc1
selected_prompt_jsonl_sha256 = d3966ce5c43347df9c68dc6cd6118102fb0708484ddd53e9b08b7b42b1f12ddd
generated_outputs_sha256 = b2b6404dd1a8ac614d3bd72f6e956a691104b5a6f5b0d81bdf10221a66dcfbf4
slot_observations_sha256 = 2957c0ccbacfdbafa36612aabc3f4a345932423bbf67ed5d906f1a1c7e6f0ce4
decode_decisions_sha256 = c424bb4c641bb9503cd93f75cfa21e9a910a82c968a27e72739063a85b92e0d6
r2_decode_rows_sha256 = bf1c069c8300aa4f9920333882c8792b8ad7638436b8677205f694f8919dcf27
```

The precommit contract and decode contract are byte-identical. The transcript
was generated after the R2 Option B contract was written.

## Locked Contract

```text
protocol_id = natural_evidence_v2_wp6_r2_option_b_robust_block_scale
decoder_id = qwen_v2_wp6_r2_robust_block_coordinate_majority_decoder_v1
bucket_policy_id = qwen_v2_wp3_r2_primary_set_plan_vs_create_prepare_v1
slot_policy_id = strict_step_label_index_1_to_16
eval_split = wp3_r1_eval
payload_plus_checksum_hex = a55e
audit_key_id = KWP4_QWEN_PILOT_001
wrong_audit_key_id = KWP4_QWEN_PILOT_WRONG_001
wrong_payload_byte_hex = 5a
block_count = 8
block_size = 64
query_budgets_per_block = [8, 16, 32, 64]
selected_prompt_file_rows = 768..1279
```

Decoder policy:

```text
accept_rule = per_block majority codeword checksum_valid_and_payload_matches_expected
controlling_budget = 64
minimum_support_at_64 = 16
minimum_majority_margin_at_64 = 3
minimum_protected_block_accepts_at_64 = 6
```

Claim-control fields in the contract set training, Llama, same-family nulls,
sanitizer, FAR aggregation, and paper claims to false.

## Artifact Completeness

Required output files were present and nonempty:

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
generated outputs = 1536
slot observations = 40785
decode decisions = 2560
R2 decode rows = 160
```

## Gate Result

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

Canonical result:

```text
scale_gate_status = PASS_WP6_R2_OPTION_B_ROBUST_BLOCK_SCALE_GATE
scale_gate_pass = true
```

## Exact-Decoder Context

The older all-16-digit exact prompt-local decoder remains a failed diagnostic
on this artifact:

```text
legacy_exact_decoder_gate_status = FAIL_WP6_QWEN_V2_E2E_PROOF_OF_LIFE
legacy_exact_decoder_gate_pass = false
protected_accept_rate_at_64 = 0.0625
protected_slot_detection_rate_at_64 = 0.9765625
protected_target_bucket_hit_rate_at_64 = 0.7646484375
```

The canonical R3 adoption is therefore explicitly tied to the precommitted
repeated-coordinate majority decoder, not the older exact 16-digit decoder.

## Claim Control

Allowed after this review:

```text
qwen_v2_scale_allowed = true
```

Still forbidden:

```text
llama_allowed = false
same_family_null_allowed = false
far_aggregation_allowed = false
sanitizer_allowed = false
paper_claim_allowed = false
full_far_claim_allowed = false
cross_family_claim_allowed = false
robustness_claim_allowed = false
```

## Current Route State

This review opens Route R3 as an internal Qwen formalization route:

```text
current_phase = V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED
```

The immediate next work remains artifact-only unless a later wrapper and
allowlist review explicitly records a Slurm action.
