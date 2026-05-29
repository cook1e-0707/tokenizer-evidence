# R4 After-870987 Pre-FAR Null Package Review - 2026-05-22

Review status: `PASS_R4_AFTER_870987_PREFAR_NULL_PACKAGE_871250_PLUS_874308`

Reviewed at: `2026-05-22T13:29:27Z`

## Scope

This review combines the passed R4 first-token event locked-scale positive package with the pre-FAR null expansion artifacts:

- locked-scale positive aggregate: `870210 + 870987`
- standard-control pre-FAR aggregate: `871250` with contextual-window v2 artifact-only matcher repair
- organic-null raw-only aggregate: `874308`

This is still not a full FAR claim and not a paper-facing positive claim. It is a Qwen-only, same-contract, provider-side first-token event pre-FAR evidence package.

## Locked Positive Context

The locked-scale route remains the positive context for this null package. Its status is `PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE`. The protected first-token event route passed previously; full phrase/text-only decoding remains report-only and is not claimed.

## Standard Controls

Standard-control combined null evidence after expanding each arm to 256 block-equivalent rows:

| Arm | accepts | accepts ignoring quality |
| --- | ---: | ---: |
| `raw` | 0/256 | 0/256 |
| `task_only` | 0/256 | 0/256 |
| `wrong_key` | 0/256 | 0/256 |
| `wrong_payload` | 0/256 | 0/256 |

Status: `PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_GENERATION_GATE`

## Organic Null

Organic raw-null generation job `874308` completed all 256 shards and passed the aggregate gate.

```text
status: PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_GENERATION_GATE
raw accepts: 0/256
raw accepts ignoring quality: 0/256
generated rows: 262144
unique response hashes: 262144
global duplicate response hash extra rows: 0
duplicate generation id extra rows: 0
trace binding checked rows: 262144
trace binding invalid rows: 0
```

## Interpretation

The project now has a Qwen-only first-token event package with:

- protected locked-scale recovery already passed under strict quality gates;
- raw/task-only/wrong-key/wrong-payload standard controls at `0/256` each;
- organic raw null at `0/256`;
- duplicate and trace-binding gates clean for `874308`;
- no text-only phrase decoder success claim.

## Claim Control

Allowed internally:

```text
Qwen-only provider-side first-token event locked positive evidence.
Qwen-only preliminary pre-FAR null evidence for standard controls and organic raw nulls.
```

Still forbidden:

```text
full FAR
paper-facing positive claim without further claim review
text-only phrase decoder success
Llama or cross-family transfer
same-family raw null rejection
sanitizer robustness
payload diversity
superiority over external baselines
```

## Next Allowed Action

Prepare the next reviewed route decision package. Do not start Llama, same-family null, sanitizer, FAR aggregation, payload diversity, training, or paper-facing claims until their preconditions are recorded in a new route decision and preflight passes.
