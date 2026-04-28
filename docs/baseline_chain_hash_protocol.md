# Chain&Hash-Style Active Fingerprint Baseline Protocol

Status: prepared as an adapted external ownership baseline protocol on
2026-04-28.

## Scope

This package implements a Chain&Hash-style active trigger-response ownership
baseline. The baseline uses natural prompt keys and assigns expected responses
from a public candidate set using a cryptographic hash over an owner secret and
the key. It is an external ownership baseline, not an internal ablation of the
bucket/RS method.

This package is not claimed to be an exact reproduction of the Chain & Hash
paper unless the external implementation is later matched and audited.

## Paths

Raw artifacts must live under:

```text
/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/chain_hash_qwen/
```

Paper-facing repo outputs:

- `docs/baseline_chain_hash_protocol.md`
- `results/tables/baseline_chain_hash.csv`
- `results/processed/paper_stats/baseline_chain_hash_summary.json`

## Protocol

| Field | Value |
|---|---|
| backbone | `Qwen/Qwen2.5-7B-Instruct` |
| tokenizer | Qwen/Qwen2.5 tokenizer |
| training | LoRA response-forcing on keyed trigger-response pairs |
| prompt keys | natural English ownership challenge prompts |
| response assignment | `hash(secret, payload, seed, key_index, key_text) mod candidate_count` |
| final seeds | `17`, `23`, `29` |
| final payload labels | `U00`, `U03`, `U12`, `U15` |
| query budgets | `1`, `3`, `5`, `10` |
| primary score | exact response match ratio |
| FAR target | `0.01` |

The B0 primary query budget is `M = 4`. Query budgets `1` and `3` are
under-budget diagnostics. Query budgets `5` and `10` are over-budget diagnostic
scaling rows and must not be used for matched-budget superiority claims.

## Enrollment And Training

For each payload and seed:

1. Build ten natural trigger keys.
2. Select one expected response per key from the public candidate set using the
   owner secret hash.
3. Write a scratch-local `chain_hash_contract.json`.
4. Write a scratch-local JSONL training set with repeated prompt/response
   examples.
5. Train a LoRA adapter under the same Qwen 7B training envelope used by the
   matched baseline package where possible.

The contract stores expected responses and a secret hash. It does not require
the verifier to know the raw secret after enrollment.

## Verification

For each query budget:

1. Load the trained adapter from the train/eval input artifact.
2. Query the model with the first `M` contract prompts.
3. Decode the generated first response word.
4. Count exact response matches.
5. Accept if `exact_response_match_ratio >= frozen_threshold`.

Valid failures remain in the denominator. A wrong response is a method failure,
not an invalid exclusion.

## Calibration And FAR

Thresholds must be frozen by the B0 calibration protocol before any final claim.

Calibration split:

| Field | Value |
|---|---|
| payloads | `U01`, `U05`, `U09`, `U13` |
| seed | `41` |
| negative sets | `foundation_null`, `wrong_payload_null`, `organic_prompt_null` |
| target FAR | `0.01` |

Until calibration is completed and linked, `baseline_chain_hash_summary.json`
must report `thresholds_frozen = false` and `paper_ready = false`.

## Required Reporting

Every final row must report:

- clean verification success,
- exact response match ratio,
- false accept / false claim status when calibration is available,
- prompt-family robustness or an explicit `not_evaluated` marker,
- utility degradation or an explicit `not_evaluated` marker,
- training and eval compute,
- failure examples.

## Guardrails

- Do not tune threshold on final test rows.
- Do not remove valid failures from the denominator.
- Do not treat this as an internal ablation.
- Do not claim exact equivalence to Chain & Hash without a matched external
  implementation audit.
