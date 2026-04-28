# Perinucleus-Style Baseline Protocol

Status: prepared as an adapted baseline protocol on 2026-04-28.

## Scope

This package implements a Perinucleus-style adapted scalable-fingerprinting
baseline for the Qwen/Qwen2.5-7B-Instruct setting. It is motivated by the
Scalable Fingerprinting / Perinucleus family cited in the manuscript, but it is
not claimed to be an exact reproduction of that paper.

The implemented object is a black-box active fingerprint baseline with natural
English instruction-style keys and first-token responses selected from the base
model next-token distribution near a nucleus boundary.

## Frozen Paths

Raw run artifacts must live under:

```text
/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_qwen/
```

Repo-local outputs are limited to:

- `configs/experiment/baselines/perinucleus/`
- `results/tables/baseline_perinucleus.csv`
- `results/processed/paper_stats/baseline_perinucleus_summary.json`
- `docs/baseline_perinucleus_protocol.md`

## Model And Access

| Field | Value |
|---|---|
| backbone | `Qwen/Qwen2.5-7B-Instruct` |
| access during verification | black-box first-token response scoring, implemented with local HF logits for reproducibility |
| training | no LoRA training; enrollment is base-model next-token distribution scoring |
| tokenizer | Qwen/Qwen2.5 tokenizer |
| prompt family | natural low-temperature English/instruction-style prompts |
| final seeds | `17`, `23`, `29` |
| final payload labels | `U00`, `U03`, `U12`, `U15` |
| query budgets | `1`, `3`, `5`, `10` |

The B0 primary query budget is `M = 4`. Therefore `q=1` and `q=3` are
under-budget comparisons; `q=5` and `q=10` are diagnostic scaling rows and must
not be used to claim matched-budget superiority over the primary method.

## Fingerprint Construction

For each payload, seed, query index, and prompt:

1. Compute the base model next-token distribution at the natural key prompt.
2. Sort candidate first tokens by probability.
3. Select a Perinucleus-style candidate window around the cumulative mass
   boundary: default inner mass `0.80`, outer mass `0.95`.
4. Filter empty, control-like, or non-English-looking token renderings.
5. Use a deterministic hash of payload, seed, query index, and prompt to select
   the expected response token and a keyed top-k response set.

This creates a stable owner-side response contract without using final-test
outcomes. The response contract records the expected token ids, top-k response
sets, prompt-bank hash, and selection parameters.

## Verification

Both gates are reported:

- `exact_response_match_ratio`: fraction of queries where the observed first
  token equals the keyed expected token.
- `top_k_response_match_ratio`: fraction of queries where the observed first
  token falls in the keyed top-k response set.

The primary score for frozen-threshold acceptance is the exact response match
ratio. The top-k score is diagnostic and must be reported alongside exact
matching.

## Calibration And FAR

Thresholds must follow `docs/calibration_protocol.md`.

Calibration split:

| Field | Value |
|---|---|
| payloads | `U01`, `U05`, `U09`, `U13` |
| seed | `41` |
| negative sets | `foundation_null`, `wrong_payload_null`, `organic_prompt_null` |
| target FAR | `0.01` |

No threshold may be selected from the final matrix. If calibration artifacts are
not yet complete, `baseline_perinucleus_summary.json` must mark
`thresholds_frozen = false` and `paper_ready = false`.

## Final Matrix

The final matrix is:

```text
payloads: U00, U03, U12, U15
seeds: 17, 23, 29
query_budgets: 1, 3, 5, 10
total rows: 48
```

Valid method failures remain in the denominator. Failed exact or top-k recovery
is not an exclusion reason.

## Metrics

Every row must report:

- verification success
- exact response match ratio
- top-k response match ratio
- frozen threshold
- FAR status at the fixed query budget
- utility acceptance rate
- utility degradation when the shared utility suite is available
- prompt-family identifier
- query budget
- training compute seconds
- embedding compute seconds
- model forward count
- expected response probability under the base model
- failure reason for valid failures

## Comparison Role

This is a strong external active ownership/fingerprinting baseline only after it
has completed calibration and final evaluation under the frozen protocol. Until
then it is an implemented adapted-baseline package with pending evidence.

It should be compared against:

- the bucket/RS method,
- the English-random active-fingerprint proxy,
- fixed representative if already available.

## Hard Guardrails

- Do not overwrite existing B1/B2 artifacts.
- Do not tune thresholds on final results.
- Do not remove failed runs from the denominator.
- Do not claim exact equivalence to Scalable Fingerprinting unless the external
  implementation is fully matched and audited.
