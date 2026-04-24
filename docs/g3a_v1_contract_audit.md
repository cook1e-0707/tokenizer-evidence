# G3a-v1 Contract Audit

This read-only audit checks whether every G3a-v1 train/eval pair uses the same semantic contract before any repair or rerun. It does not modify configs, rerun training, rerun evaluation, change thresholds, or overwrite the existing G3a summaries.

## Inputs And Limits

- Source run diagnostics: `results/tables/g3a_v1_run_diagnostics.csv`.
- Source slot diagnostics: `results/tables/g3a_v1_slot_diagnostics.csv`.
- Source G3a table: `results/tables/g3a_block_scale.csv`.
- Raw Chimera scratch mounted locally: `False`.
- Audit limitation: the local workstation cannot open `/hpcstor6/...`, so the audit uses the committed Chimera-derived run-file diagnostics rather than reopening raw train/eval files.

## Scope

- completed / target: `36` / `36`.
- included / excluded: `29` / `7`.
- paper_ready: `False`.
- failure clusters inspected first: B1 seed 23 U03/U12/U15 and B4 seed 29 U00/U03/U12/U15.

## Required Check Results

- `model_id`: `qwen2.5-7b-instruct`, status `MATCH` for all rows.
- `tokenizer_id`: `Qwen/Qwen2.5-7B-Instruct`, status `MATCH` for all rows.
- `block_count`: variant B1/B2/B4 matches block_count 1/2/4 for all rows.
- `fields_per_block` and `field_order`: `2` fields per block, `SECTION|TOPIC`, status `MATCH` for all rows.
- `codebook path and hash`: `configs/data/frozen/real_pilot_catalog__qwen2_5_7b_compiled__v1.yaml`, SHA256 `c09e3866128bcc2ceeb57dbc9fd34e78b321500fe05c869cded1f49ff737ec92`, status `MATCH` for all rows.
- `bucket partition hash`: `157b4ef7d9ff5364c4beb082152855ec1f853950947f31f9442902be925d5d90`, status `MATCH` for all rows.
- `payload label`, expected payload codeword, and payload-to-bucket mapping: status `MATCH`; each variant+payload has a single stable expected codeword across seeds.
- `prompt family`: one stable prompt-family hash within each block-count variant; B1/B2/B4 differ as expected because the rendered prompt contract includes block count.
- `generation max_new_tokens`: `1`, generation hash `5bbb502ea6b99b70e004602a5d5d8c16c0d219d647e47ee7daff32a231b497cf`, status `MATCH` for all rows.
- `decoding mode`: `greedy`, status `MATCH` for all rows.
- `verifier contract`: `compiled_gate`, one stable eval contract per variant+payload across seeds, status `MATCH`.
- `RS config`: uniformly `not_configured_or_missing` with `rs_config_hash=missing`; this is not a run-specific mismatch.
- `canonical render contract`: `canonical_v1` with stable field order/codebook/eval contract per variant+payload, status `MATCH`.
- `adapter checkpoint used by eval`: eval adapter path belongs to the same case root as the train summary for all rows, status `MATCH`.

## Failure Neighbor Checks

- `B1_U03_s23`: B1 seed `23` payload `U03`; parser=`True`, block_count=`True`, slot_errors=`b0:TOPIC:3->2`; neighbors same-block+seed=`B1_U00_s23` (MATCH), same-block+payload=`B1_U03_s17` (MATCH), same-seed+payload=`B2_U03_s23` (MATCH).
- `B1_U12_s23`: B1 seed `23` payload `U12`; parser=`True`, block_count=`True`, slot_errors=`b0:SECTION:3->0;b0:TOPIC:0->2`; neighbors same-block+seed=`B1_U00_s23` (MATCH), same-block+payload=`B1_U12_s17` (MATCH), same-seed+payload=`B2_U12_s23` (MATCH).
- `B1_U15_s23`: B1 seed `23` payload `U15`; parser=`True`, block_count=`True`, slot_errors=`b0:SECTION:3->0;b0:TOPIC:3->1`; neighbors same-block+seed=`B1_U00_s23` (MATCH), same-block+payload=`B1_U15_s17` (MATCH), same-seed+payload=`B2_U15_s23` (MATCH).
- `B4_U00_s29`: B4 seed `29` payload `U00`; parser=`True`, block_count=`True`, slot_errors=`b3:SECTION:0->3;b3:TOPIC:0->2`; neighbors same-block+seed=`none` (NO_PASSED_NEIGHBOR), same-block+payload=`B4_U00_s23` (MATCH), same-seed+payload=`B2_U00_s29` (MATCH).
- `B4_U03_s29`: B4 seed `29` payload `U03`; parser=`True`, block_count=`True`, slot_errors=`b3:SECTION:0->3;b3:TOPIC:3->2`; neighbors same-block+seed=`none` (NO_PASSED_NEIGHBOR), same-block+payload=`B4_U03_s23` (MATCH), same-seed+payload=`B2_U03_s29` (MATCH).
- `B4_U12_s29`: B4 seed `29` payload `U12`; parser=`True`, block_count=`True`, slot_errors=`b3:TOPIC:0->2`; neighbors same-block+seed=`none` (NO_PASSED_NEIGHBOR), same-block+payload=`B4_U12_s23` (MATCH), same-seed+payload=`B2_U12_s29` (MATCH).
- `B4_U15_s29`: B4 seed `29` payload `U15`; parser=`True`, block_count=`True`, slot_errors=`b3:TOPIC:3->2`; neighbors same-block+seed=`none` (NO_PASSED_NEIGHBOR), same-block+payload=`B4_U15_s23` (MATCH), same-seed+payload=`B2_U15_s29` (MATCH).

B4 seed 29 has no passed same-block+same-seed neighbor because all four B4 seed 29 payloads failed. It still has passed same-block+same-payload neighbors at seeds 17/23 and passed same-seed+same-payload neighbors in B1/B2.

## Contract Group Summary

```json
{
  "bucket_partition_hash_values": [
    "157b4ef7d9ff5364c4beb082152855ec1f853950947f31f9442902be925d5d90"
  ],
  "codebook_hash_values": [
    "c09e3866128bcc2ceeb57dbc9fd34e78b321500fe05c869cded1f49ff737ec92"
  ],
  "eval_contract_hash_count_by_variant_payload": {
    "B1:U00": 1,
    "B1:U03": 1,
    "B1:U12": 1,
    "B1:U15": 1,
    "B2:U00": 1,
    "B2:U03": 1,
    "B2:U12": 1,
    "B2:U15": 1,
    "B4:U00": 1,
    "B4:U03": 1,
    "B4:U12": 1,
    "B4:U15": 1
  },
  "expected_codeword_count_by_variant_payload": {
    "B1:U00": 1,
    "B1:U03": 1,
    "B1:U12": 1,
    "B1:U15": 1,
    "B2:U00": 1,
    "B2:U03": 1,
    "B2:U12": 1,
    "B2:U15": 1,
    "B4:U00": 1,
    "B4:U03": 1,
    "B4:U12": 1,
    "B4:U15": 1
  },
  "field_order_values": [
    "SECTION|TOPIC"
  ],
  "generation_config_hash_values": [
    "5bbb502ea6b99b70e004602a5d5d8c16c0d219d647e47ee7daff32a231b497cf"
  ],
  "model_id_values": [
    "qwen2.5-7b-instruct"
  ],
  "payload_map_hash_by_variant": {
    "B1": [
      "d06c6c5674c5f1b5ed11aad031f0c4a829c6548ca80aada118ef8aaf421e3216"
    ],
    "B2": [
      "f596eb5b1aa93c5178cd0d4e080566752399143f2665e871b2f832ac23327900"
    ],
    "B4": [
      "3dbe09240b2ec3501867cfc7272b95770f076f37fb1d617387f49f48946fb558"
    ]
  },
  "prompt_family_hash_by_variant": {
    "B1": [
      "34a728d2bf25cc54d7cd6a28721aa5470cc1b5bb908ed3207caa11e138fc285c"
    ],
    "B2": [
      "c09020fd75f3ed5ed7b9664abb4d97248ea7bb15ef8dbc321bdffc1f423ca456"
    ],
    "B4": [
      "52467f746c346054b65cde66b56158c34f6242bea163b0f0fc3690c3b0f95e80"
    ]
  },
  "rs_config_hash_values": [
    "missing"
  ],
  "tokenizer_id_values": [
    "Qwen/Qwen2.5-7B-Instruct"
  ],
  "train_contract_hash_count_by_variant_payload": {
    "B1:U00": 1,
    "B1:U03": 1,
    "B1:U12": 1,
    "B1:U15": 1,
    "B2:U00": 1,
    "B2:U03": 1,
    "B2:U12": 1,
    "B2:U15": 1,
    "B4:U00": 1,
    "B4:U03": 1,
    "B4:U12": 1,
    "B4:U15": 1
  }
}
```

## Mismatches

None.

## Interpretation

No contract/path/accounting mismatch is visible in the committed Chimera-derived diagnostics. The failures retain valid parser and block-count behavior, use the same model/tokenizer/codebook/bucket/generation/verifier contracts, and fail through wrong slot buckets or payload recovery. That points away from plumbing and toward method behavior under specific seed/block-count conditions.

## Conclusion

CONTRACT_MATCH_CONFIRMED

No semantic contract mismatch was found. G3a-v1 failures should be treated as method/training/verifier failures, not plumbing failures.
