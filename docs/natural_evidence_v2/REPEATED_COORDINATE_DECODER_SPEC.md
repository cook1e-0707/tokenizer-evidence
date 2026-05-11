# Repeated-Coordinate Majority Decoder Spec

## Scope

This document defines the repeated-coordinate majority decoder used by the
Qwen v2 controlled-natural micro-slot Route R3. It replaces the brittle exact
all-16-digits prompt-local decoder for the R3 route.

The decoder is precommitted. It is not allowed to tune thresholds, keys,
payloads, prompt sets, or accepted coordinates after seeing transcripts.

## Inputs

The decoder consumes a locked contract and an observed transcript table.

The contract fixes:

```text
protocol_id
decoder_id
audit_key_id
payload_byte
checksum_byte
slot_policy_id
bucket_policy_id
prompt_set
query_budget
block_id
block_size
coordinate mapping
S_min support threshold
M_min majority-margin threshold
wrong-key control
wrong-payload control
```

For the reviewed WP6-R2 Option B job `852426`, the locked values were:

```text
protocol_id = natural_evidence_v2_wp6_r2_option_b_robust_block_scale
decoder_id = qwen_v2_wp6_r2_robust_block_coordinate_majority_decoder_v1
payload_plus_checksum_hex = a55e
slot_policy_id = strict_step_label_index_1_to_16
bucket_policy_id = qwen_v2_wp3_r2_primary_set_plan_vs_create_prepare_v1
query_budgets_per_block = [8, 16, 32, 64]
controlling_budget = 64
block_count = 8
block_size = 64
S_min = 16
M_min = 3
```

## Coordinate Model

Each prompt response contains 16 controlled-natural Step-label micro-slots.
Each micro-slot is mapped to one coordinate of a 16-bit codeword:

```text
8 payload bits || 8 checksum bits
```

Each coordinate is observed repeatedly across a precommitted block of prompts.
For a given block and budget, the decoder collects all observations assigned to
the same coordinate.

An observation contributes to a coordinate only when:

- the strict Step-label slot is detected;
- the observed first word is in the locked 2-way bucket set;
- the bucket maps to a bit under the committed key and bucket policy.

Out-of-bank first words, missing slots, duplicate slots, and malformed slots
are erasures. They do not create alternative buckets and do not lower the
threshold after the fact.

## Majority Rule

For each coordinate `c`, let:

```text
n0(c) = count of observed bucket-bit 0
n1(c) = count of observed bucket-bit 1
support(c) = n0(c) + n1(c)
margin(c) = abs(n1(c) - n0(c))
majority_bit(c) = argmax(n0(c), n1(c)) if n0(c) != n1(c)
```

A coordinate is complete iff:

```text
support(c) >= S_min
margin(c) >= M_min
majority_bit(c) is defined
```

The thresholds are fixed by the precommit. For the current R3 starting point:

```text
S_min = 16
M_min = 3
```

Do not lower `S_min` or `M_min` to rescue a transcript.

## Block Decode

A block decode is complete iff all 16 coordinates are complete.

The decoder then constructs:

```text
decoded_payload_byte = bits[0:8]
decoded_checksum_byte = bits[8:16]
```

The checksum is recomputed from the expected audit key and decoded payload using
the locked checksum domain:

```text
natural_evidence_v2_wp4_prompt_local_checksum_v1
```

A block accepts iff:

```text
all 16 coordinates are complete
and decoded_payload_byte == expected_payload_byte
and decoded_checksum_byte == checksum(audit_key_id, expected_payload_byte)
and the contract hash / prompt set / bucket policy / slot policy match
```

Wrong-key control evaluates the same observations with the wrong audit key and
must reject. Wrong-payload control evaluates the same observations against the
wrong expected payload and must reject. These controls may share observed bits
with the protected condition, but they do not accept unless their own
key/payload/checksum predicate passes.

## Scale Gate

For a multi-block scale run, the protected arm passes only if the precommitted
minimum number of blocks accept at the controlling budget.

For WP6-R2 Option B job `852426`:

```text
minimum_protected_block_accepts_at_64 = 6 / 8
null_accepts_per_condition_at_64 = 0
```

Route R3 locked-scale work may use a larger block count, but the thresholds,
query budget, checksum rule, and null rejection rules must be recorded before
generation.

## Claim Boundary

This decoder can support a Qwen controlled-natural micro-slot diagnostic or a
future locked-scale Qwen package if the scale gates pass. It does not by itself
establish:

- full FAR;
- Llama transfer;
- same-family null rejection;
- sanitizer robustness;
- arbitrary natural-output ownership evidence;
- superiority over Scalable or Perinucleus;
- a paper-facing positive claim.
