# WP3-R2 Observed High-Mass Context-Mass Score 851233 Review

## Scope

Slurm job `851233` scored the WP3-R2 observed high-mass bank search plan:

```text
results/natural_evidence_v2/status/wp3_r2_observed_high_mass_bank_search_plan_20260509_054001/
```

Synced result directory:

```text
results/natural_evidence_v2/status/wp3_r2_observed_high_mass_context_mass_score_851233/
```

This was context-mass scoring only. It did not train, generate new transcripts,
run E2E, decode payloads, aggregate FAR, or make a paper-facing positive claim.

## Slurm Result

```text
job_id=851233
job_name=nat-ev-v2-wp3ctxm
partition=DGXA100
node=chimera12
state=COMPLETED
exit_code=0:0
runtime=00:00:47
```

The job completed cleanly, but the model-mass gate failed.

## Gate Result

```text
score_plan_rows=416
valid_context_score_rows=304
invalid_tokenization_rows=112
mass_rows=19
mass_gate_status=FAIL
```

Invalid tokenization was concentrated in banks containing surfaces that are not
single next tokens under the Qwen tokenizer in this context:

```text
Encourage -> [10751, 60040]
Organize -> [10762, 551]
```

Any repaired bank search should exclude those surfaces unless a different
single-token surface policy is explicitly approved.

## Best Scored Banks

The strongest scored candidates still fall far below the pilot absolute mass
threshold `min_bucket_mass >= 0.03`:

| Candidate bank | min bucket mass | ratio | Legacy pass |
|---|---:|---:|---|
| `set_prepare_vs_plan_create` | 0.005970 | 1.997 | true |
| `keep_choose_vs_check_develop` | 0.005208 | 1.279 | true |
| `keep_check_vs_choose_develop` | 0.005083 | 1.335 | true |
| `set_plan_vs_create_prepare` | 0.004110 | 3.353 | false |
| `set_create_vs_plan_prepare` | 0.003084 | 4.801 | false |

The previous primary-style bank family remains weak in absolute mass. Observed
surface frequency in generated outputs does not translate into high base-Qwen
next-token mass under the generic `Step N:` scoring prefix.

## Interpretation

This is a negative R2 result for the generic Step-label prefix policy, not a
training result. The likely issue is that the scored prefix is only:

```text
Step N:
```

It omits the owner prompt and the generated assistant prefix before the slot.
The `850885` outputs show that these action words occur in the full
prompt-conditioned answer context, but the current score plan did not score
that full context.

## Decision

Review status:

```text
R2_GENERIC_STEP_PREFIX_BANK_SEARCH_FAILED_PROMPT_CONDITIONED_REPAIR_PLAN_REQUIRED
```

WP4 remains blocked. Training, Qwen E2E, Llama, same-family null, sanitizer,
FAR aggregation, and paper-facing positive claims remain forbidden.

Next allowed action:

```text
Prepare a prompt-conditioned R2 context-mass repair plan that scores candidate
banks under owner prompt + assistant prefix contexts from the 850885 artifacts.
Any tokenizer/model scoring must be submitted through Chimera Slurm.
```
