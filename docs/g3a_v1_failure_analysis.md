# G3a-v1 Failure Analysis

This report diagnoses the non-standing G3a-v1 block-count scale package without modifying the original G3a-v1 paper artifacts or rerunning training/evaluation.

## Package State

- `paper_ready`: `False`
- completed / target: `36` / `36`
- included / excluded / pending: `29` / `7` / `0`
- overall accepted/verifier: `29` / `36`

## Artifact And Path Check

- Missing outputs, missing files, or path/accounting bug: `no` from committed paper artifacts; `pending_case_count=0`, all 36 rows have train/eval summary paths, and the new-case roots point to scratch.
- Raw Chimera run files available in this execution environment: `no`.
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

- Is the failure due to missing outputs, missing files, or path/accounting bugs? No evidence for that in the paper artifacts; all runs are completed and accounted for. Raw file availability is environment-dependent and was `no` for this diagnostic run.
- Is the failure due to parser failure? No evidence from the committed G3a table; all excluded cases have `decoded_block_count_correct=True`, and parser_success is `True` under available diagnostics/inference.
- Is the failure due to block-count mismatch? No; all excluded cases have `decoded_block_count_correct=True`.
- Is the failure due to slot-level bucket errors? Not confirmed from paper artifacts alone. Per-slot observed buckets are missing for `168` slot rows unless the script is rerun where Chimera run files are available.
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
      "missing"
    ],
    "generation_config_hash": [
      "5bbb502ea6b99b70e004602a5d5d8c16c0d219d647e47ee7daff32a231b497cf"
    ],
    "payload_map_hash": [
      "f0966be88adb0773ea190986d65da9103fb323a9e9fbddc1e960f0f74742b591"
    ],
    "prompt_family_hash": [
      "a1905aa79faafc39495970de10e23b5614127708a883a19d29a99a1404ac5a72"
    ],
    "train_contract_hash": [
      "missing"
    ]
  },
  "B2": {
    "codebook_hash": [
      "c09e3866128bcc2ceeb57dbc9fd34e78b321500fe05c869cded1f49ff737ec92"
    ],
    "eval_contract_hash": [
      "missing"
    ],
    "generation_config_hash": [
      "5bbb502ea6b99b70e004602a5d5d8c16c0d219d647e47ee7daff32a231b497cf"
    ],
    "payload_map_hash": [
      "4d6e963108cc90ff35261a964982d23167146f17dc498e969e622a537783699a"
    ],
    "prompt_family_hash": [
      "a1905aa79faafc39495970de10e23b5614127708a883a19d29a99a1404ac5a72"
    ],
    "train_contract_hash": [
      "missing"
    ]
  },
  "B4": {
    "codebook_hash": [
      "c09e3866128bcc2ceeb57dbc9fd34e78b321500fe05c869cded1f49ff737ec92"
    ],
    "eval_contract_hash": [
      "missing"
    ],
    "generation_config_hash": [
      "5bbb502ea6b99b70e004602a5d5d8c16c0d219d647e47ee7daff32a231b497cf"
    ],
    "payload_map_hash": [
      "e88b9e8061f3f9b8b43c7e90042f182cfb1f83975d9964be3939723b7274eb1a"
    ],
    "prompt_family_hash": [
      "a1905aa79faafc39495970de10e23b5614127708a883a19d29a99a1404ac5a72"
    ],
    "train_contract_hash": [
      "missing"
    ]
  }
}
```

## Conclusion

ROOT_CAUSE_NOT_CONFIRMED: additional instrumentation required
