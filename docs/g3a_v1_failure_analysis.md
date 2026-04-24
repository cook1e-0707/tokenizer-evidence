# G3a-v1 Failure Analysis

This report diagnoses the non-standing G3a-v1 block-count scale package without modifying the original G3a-v1 paper artifacts or rerunning training/evaluation.

## Package State

- `paper_ready`: `False`
- completed / target: `36` / `36`
- included / excluded / pending: `29` / `7` / `0`
- overall accepted/verifier: `29` / `36`

## Artifact And Path Check

- Missing outputs, missing files, or path/accounting bug: `no` from committed paper artifacts; `pending_case_count=0`, all 36 rows have train/eval summary paths, and the new-case roots point to scratch.
- Raw Chimera run files available in this execution environment: `yes`.
- Rows with missing hash inputs: `36` / `36`.

## Failure Pattern

- Failure count: `7`.
- Failure by variant: `{'B1': 3, 'B4': 4}`.
- Failure by seed: `{'23': 3, '29': 4}`.
- Failure by payload: `{'U03': 2, 'U12': 2, 'U15': 2, 'U00': 1}`.
- Failure by block_count: `{'1': 3, '4': 4}`.
- Failure reasons: `{'accepted,verifier_success,decoded_payload_correct': 7}`.

The excluded cases are:

- `B1_U03_s23`: block_count=`1`, seed=`23`, payload=`U03`, match_ratio=`0.5`, reasons=`accepted,verifier_success,decoded_payload_correct`
- `B1_U12_s23`: block_count=`1`, seed=`23`, payload=`U12`, match_ratio=`0.0`, reasons=`accepted,verifier_success,decoded_payload_correct`
- `B1_U15_s23`: block_count=`1`, seed=`23`, payload=`U15`, match_ratio=`0.0`, reasons=`accepted,verifier_success,decoded_payload_correct`
- `B4_U00_s29`: block_count=`4`, seed=`29`, payload=`U00`, match_ratio=`0.75`, reasons=`accepted,verifier_success,decoded_payload_correct`
- `B4_U03_s29`: block_count=`4`, seed=`29`, payload=`U03`, match_ratio=`0.75`, reasons=`accepted,verifier_success,decoded_payload_correct`
- `B4_U12_s29`: block_count=`4`, seed=`29`, payload=`U12`, match_ratio=`0.875`, reasons=`accepted,verifier_success,decoded_payload_correct`
- `B4_U15_s29`: block_count=`4`, seed=`29`, payload=`U15`, match_ratio=`0.875`, reasons=`accepted,verifier_success,decoded_payload_correct`

## Required Questions

- Is the failure due to missing outputs, missing files, or path/accounting bugs? No evidence for that in the paper artifacts; all runs are completed and accounted for. Raw file availability is environment-dependent and was `yes` for this diagnostic run.
- Is the failure due to parser failure? No evidence from the committed G3a table; all excluded cases have `decoded_block_count_correct=True`, and parser_success is `True` under available diagnostics/inference.
- Is the failure due to block-count mismatch? No; all excluded cases have `decoded_block_count_correct=True`.
- Is the failure due to slot-level bucket errors? Not confirmed from paper artifacts alone. Per-slot observed buckets are missing for `0` slot rows unless the script is rerun where Chimera run files are available.
- Is the failure due to payload/RS decoding despite high slot match ratio? Partially supported for `4` B4 failures with match ratio >= 0.75; not supported for `2` zero-match B1 failures. RS decoding is not instrumented/stored for this package.
- Are failures concentrated by seed, payload, block_count, field, or bucket? They concentrate by variant/seed: B1 seed 23 and B4 seed 29. Payload concentration is weaker because B4 seed 29 fails all four payloads, while B1 seed 23 fails U03/U12/U15 but passes U00.
- Are train/eval contract hashes consistent for failed runs? Not confirmed when raw run files are missing. Observed hash groups are summarized in `contract_hash_sets` in the JSON summary.
- Is B4 seed=29 a contract/config issue or an optimization/generalization issue? Current evidence favors seed-specific optimization/generalization instability over path/accounting, but contract/config root cause is not definitively excluded without raw contract hashes and per-slot diagnostics.
- Is B1 seed=23 payload-specific hardness? It is seed-specific with payload dependence: U00 passes while U03/U12/U15 fail. This is not enough to prove intrinsic payload hardness.
- What instrumentation is missing for a definitive answer? Persist generated slot values, compiled gate per-slot diagnostics in paper-facing diagnostics, top-k/logits or bucket mass per slot, adapter hash manifest, train/eval contract hashes in the G3a table, and explicit RS/no-RS decode trace.

## Contract Hash Sets

```json
{
  "B1": {
    "codebook_hash": [
      "c09e3866128bcc2ceeb57dbc9fd34e78b321500fe05c869cded1f49ff737ec92"
    ],
    "eval_contract_hash": [
      "bb1cd99a4da68164a185431698b895efb91f2be87e4bf91e53ab9add3e752645",
      "da613ecf886905b48fe20b73df5d15899ce38a0efc7081dbc67feda45901f87a",
      "f4cdcb7d290a605d43f1e782a21404b6cabd70f34a97dd0f334f7308b2265ae8",
      "fd8154906d0355638ea44a98d090fd5b7a0fe889ffaea762bccd73707fda2489"
    ],
    "generation_config_hash": [
      "5bbb502ea6b99b70e004602a5d5d8c16c0d219d647e47ee7daff32a231b497cf"
    ],
    "payload_map_hash": [
      "d06c6c5674c5f1b5ed11aad031f0c4a829c6548ca80aada118ef8aaf421e3216"
    ],
    "prompt_family_hash": [
      "34a728d2bf25cc54d7cd6a28721aa5470cc1b5bb908ed3207caa11e138fc285c"
    ],
    "train_contract_hash": [
      "780e82f04ceea4062f1c59da58676726c1b45ea4a0d1a7c916c97c7dcfd1cbef",
      "8277e738c9909a526c1a06fe71eaf85debf3e492c5fa41ff811b2fc5ac9b3234",
      "9d9050f917cf0cac63ad7102e34a17167170fbe6aaf18e4e474e38110b97960f",
      "b5c87a16fb9906072c6fd9c15f8b5649139c7f0db4b3d902954ba42c7853772b"
    ]
  },
  "B2": {
    "codebook_hash": [
      "c09e3866128bcc2ceeb57dbc9fd34e78b321500fe05c869cded1f49ff737ec92"
    ],
    "eval_contract_hash": [
      "136acbb7525f67cce34b237c8a7164c79a1782d433091e74e1bb7a05f0933ce2",
      "354a867a612b17eb8be9bfea3599299c081e670805202ab21cf3f2367e6b68cc",
      "56c198f01a5b173ded6a1d3dd298f6f33c2640dbdd0ab136a19c0c6e4117265e",
      "c554b4ec89297b7bc4fb4373927da587f6b67edb4d0436a198db19b27f849bba"
    ],
    "generation_config_hash": [
      "5bbb502ea6b99b70e004602a5d5d8c16c0d219d647e47ee7daff32a231b497cf"
    ],
    "payload_map_hash": [
      "f596eb5b1aa93c5178cd0d4e080566752399143f2665e871b2f832ac23327900"
    ],
    "prompt_family_hash": [
      "c09020fd75f3ed5ed7b9664abb4d97248ea7bb15ef8dbc321bdffc1f423ca456"
    ],
    "train_contract_hash": [
      "2036038442085b3b41d885f98e94627a560dc4d72eced8b15ddbeedd64b36315",
      "639b629ef2fa6827b2fb0fab37b5d8c0f59677e2bc5f3f50d0a8febe55751aee",
      "6cb08666d4420f53e07520dcfbc9c4ff4aaf0776f9b9914c4cd14e49840bae87",
      "d280407fc53da51263c3c81ae2d7ee30bb99020320cdcc4d7117203973fe54e6"
    ]
  },
  "B4": {
    "codebook_hash": [
      "c09e3866128bcc2ceeb57dbc9fd34e78b321500fe05c869cded1f49ff737ec92"
    ],
    "eval_contract_hash": [
      "101a0d60f6fa391cce21f90597051c4700bcb95d9244db52e7058b65bafec06c",
      "4895bdc6a2c128f57294f54e868b0b2a49a1f7d9da8945f76cbedbd379a0e9e2",
      "a4ddde7747b0560fea9237b8f8d3ec1fcb830730efa5c2249d56f15711d090fc",
      "b44b520ac3ac372e951c7b0cb8b5c5d8a2ec1c8e2ca2ddfc1eb3cd1bfd217864"
    ],
    "generation_config_hash": [
      "5bbb502ea6b99b70e004602a5d5d8c16c0d219d647e47ee7daff32a231b497cf"
    ],
    "payload_map_hash": [
      "3dbe09240b2ec3501867cfc7272b95770f076f37fb1d617387f49f48946fb558"
    ],
    "prompt_family_hash": [
      "52467f746c346054b65cde66b56158c34f6242bea163b0f0fc3690c3b0f95e80"
    ],
    "train_contract_hash": [
      "024ec5a9414e975890c11db16d71ea960f247621b56daf1fc031c20a71a448da",
      "2e49a12d809727032904986250ab81e400d1e0baa2c9d5aa9f7d9a08b94a4d1d",
      "e0f5fa3c25d0161cfcf13fe7b9fcc3e38630c40769dbd9c15d238e82f7fe6079",
      "eb5e82741b163219c87251b41891b2b7995d1b1fde035bb22f862e1cf64a013f"
    ]
  }
}
```

## Conclusion

ROOT_CAUSE_NOT_CONFIRMED: additional instrumentation required
