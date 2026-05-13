# R3.2 H200 853524 repair decision package

Date: 2026-05-12

## Scope

This is an artifact-only decision package for completed H200 Slurm array job
`853524`. It does not authorize a rerun, training, Llama, same-family null,
sanitizer, FAR aggregation, payload-diversity claims, or paper-facing positive
claims.

Primary attribution artifact:

`results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/failure_attribution/r3_2_853524_failure_attribution.md`

Machine-readable summary:

`results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/failure_attribution/r3_2_853524_failure_attribution_summary.json`

## Result Status

`853524` is a clean Slurm completion and a negative locked-scale result:

- all 12 shard tasks completed with exit code `0:0`;
- all 12 prompt windows were unique;
- all 96 prompt blocks were unique;
- duplicate prompt-window reuse from `853430` is resolved;
- protected accepts at budget 64 were only `6/96`, below the `>=80/96` gate;
- raw/task-only/wrong-key/wrong-payload diagnostic null accepts were all `0/96`;
- min accepted-block support was `6`, below the required `>=16`;
- min accepted-block majority margin was `0`, below the required `>=3`;
- forbidden public surface count was `23`, above the required `0`.

This result should not be interpreted as a crash or control-plane failure. It
shows that the same-contract `a55e` Qwen v2 signal does not survive locked-scale
expanded prompt distribution under the current prompt variants, bank, and
decoder thresholds.

## Bottlenecks To Review

### 1. Protected Signal Survival

The largest erasure reason is:

`observed_first_word_not_in_primary_bucket_set = 30,021`

This means most protected observations do not resolve into the precommitted
primary bucket surfaces. The accepted blocks that do decode are too few and too
weakly supported for the locked-scale gate.

### 2. Prompt Variant Instability

The weakest prompt variant is:

`r1_strict_literal_16_step_lines`

Observed protected target-hit rate:

`0.307`

The other two variants are stronger but still insufficient:

- `r1_strict_numbered_step_label_lines`: `0.415`
- `r1_strict_no_heading_16_step_lines`: `0.434`

Some protected outputs duplicate labels, for example:

`Step 1: Create a Step 1: ...`

This contributes to:

- `missing_or_out_of_order_step_slots = 1,093`
- `duplicate_step_slots = 485`

### 3. Middle-Step Coordinate Weakness

Coordinate stability is uneven. Steps 1, 2, and 16 are strong, while many
middle steps have low majority margin. This makes a 16-coordinate full-block
majority decoder fragile even when total resolved observations are nontrivial.

Examples from the attribution table:

| Step | Mean resolved count | Mean majority margin |
|---:|---:|---:|
| 1 | 56.45 | 56.03 |
| 2 | 55.20 | 54.99 |
| 4 | 15.05 | 5.78 |
| 6 | 19.89 | 4.18 |
| 8 | 15.65 | 5.44 |
| 10 | 19.45 | 4.80 |
| 16 | 44.41 | 43.30 |

### 4. Forbidden-Surface Gate Failure

Literal forbidden-surface hits:

- `bucket = 21`
- `fingerprint = 1`
- `watermark = 1`

Most `bucket` hits appear to be ordinary task content, such as physical buckets
in gardening or cleaning checklists. The precommitted gate is literal, so the
run fails until matcher semantics are reviewed and, if revised, re-precommitted.

### 5. Nulls Remain Clean But Are Still Diagnostic

Diagnostic null arms stayed clean:

- raw: `0/96`
- task-only: `0/96`
- wrong-key: `0/96`
- wrong-payload: `0/96`

This is useful safety evidence for this diagnostic package, but it is not full
FAR, same-family null, sanitizer robustness, or payload diversity.

## Candidate Repair Axes For Expert Decision

These are decision axes, not approved execution routes:

1. Prompt-variant repair: remove or rewrite the weakest variant and prevent
   duplicated `Step N:` surfaces.
2. Bank/surface repair: audit whether the current primary bank is too narrow
   for the expanded prompt distribution.
3. Coordinate repair: reduce reliance on unstable middle-step coordinates or
   revise the repeated-coordinate schedule.
4. Forbidden matcher review: decide whether literal task-content uses of
   `bucket` should remain hard failures or be handled by a contextual matcher.
5. Training/signal repair: review whether protected target mass is too
   prompt-local and does not generalize across expanded prompt variants.

No axis is currently approved for Slurm submission.

## Locked Claim Policy

Allowed internal statement:

`853524 completed cleanly and failed the R3.2 same-contract locked-scale gate; duplicate prompt-window reuse is fixed, diagnostic nulls are clean, but protected signal survival and forbidden-surface gates fail.`

Still not allowed:

- R3.2 success;
- full FAR;
- payload diversity;
- Llama success or cross-family generality;
- same-family null rejection;
- sanitizer robustness;
- paper-facing positive claim;
- superiority over Scalable/Perinucleus.

## Next Allowed Action

Expert review / artifact-only repair decision. Do not submit a rerun or unlock
Llama, FAR, sanitizer, same-family null, payload-diversity claims, or
paper-facing positive claims until this package is reviewed and a new route is
recorded.
