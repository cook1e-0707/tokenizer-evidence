# WP3 Restricted Step-Label Expanded Mass Score 850483 Review

## Scope

Slurm job `850483` scored the restricted Step-label expanded action-verb bank
plan under base Qwen. This was context-mass scoring only: no text generation,
no training, no E2E, no payload recovery, no FAR aggregation, and no
paper-facing positive claim.

Artifacts:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_expanded_mass_score_850483/
```

## Slurm Result

```text
job_id=850483
job_name=nat-ev-v2-wp3ctxm
partition=DGXA100
node=chimera13
state=COMPLETED
exit_code=0:0
runtime=00:00:43
```

The wrapper used the Chimera virtual environment:

```text
/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3
```

## Summary

```text
status=WP3_CONTEXT_MASS_SCORED_NOT_TRAINING_NOT_GENERATION
mass_gate_status=FAIL
score_plan_rows=128
context_score_rows=112
invalid_tokenization_rows=16
mass_rows=7
wp4_allowed=false
```

Tokenizer validation skipped all `16` rows for:

```text
step_label_arrange_schedule_organize_plan_v1
```

Reason:

```text
Organize -> [10762, 551]
```

`Organize` is not one Qwen next token under the scoring tokenizer, so that bank
cannot be used in this form.

## Mass Gate Results

Gate thresholds:

```text
min_bucket_mass >= 0.005
max_bucket_mass_ratio <= 5.0
```

| Bank | Min mass | Ratio | Decision |
|---|---:|---:|---|
| `step_label_create_develop_establish_set_v1` | 0.0040764090 | 3.0936 | fail: min mass below gate |
| `step_label_ensure_confirm_check_review_v1` | 0.0009384600 | 6.6498 | fail: min mass and ratio |
| `step_label_identify_assess_research_review_v1` | 0.0035765931 | 1.5094 | fail: min mass below gate |
| `step_label_pack_bring_gather_collect_v1` | 0.0010932492 | 1.5718 | fail: min mass below gate |
| `step_label_prepare_plan_determine_define_v1` | 0.0030839754 | 3.4434 | fail: min mass below gate |
| `step_label_select_decide_choose_make_v1` | 0.0037812320 | 3.3194 | fail: min mass below gate |
| `step_label_use_take_send_document_v1` | 0.0003998556 | 99.1648 | fail: min mass and severe imbalance |

No expanded bank passes the configured mass gate.

## Interpretation

The expanded action-verb search improved surface coverage but did not produce a
mass-valid bank under the current `Step N: ` context. Several banks are close on
balance but below the `0.005` minimum bucket mass. The strongest near-miss is:

```text
step_label_create_develop_establish_set_v1
min_bucket_mass=0.0040764090
ratio=3.0936
```

The most balanced below-threshold bank is:

```text
step_label_identify_assess_research_review_v1
min_bucket_mass=0.0035765931
ratio=1.5094
```

This suggests the next repair should not start training or WP4. It should stay
artifact-only and focus on one of:

1. candidate repair: replace low-mass or multi-token surfaces such as
   `Organize` with single-token, high-probability step openers;
2. prefix repair: evaluate stricter or more natural prompt-local prefixes that
   raise full-vocab mass for the weaker bucket;
3. gate review: keep the threshold unchanged for now, but record near-misses
   separately rather than treating them as trainable banks.

## Decision

WP3 still fails. WP4 remains blocked.

Next allowed action:

```text
artifact-only mass-aware candidate repair plan from 850483 context scores
```

The repair plan should not submit Slurm automatically. Any future tokenizer or
model scoring must go through reviewed Slurm wrappers and the disabled
allowlist.
