# R4 After 864832 Two-Sided Qwen Tokenizer Preflight 865210 Review

Status:
`PASS_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT`

Job:

```text
job_id: 865210
job_name: nat-ev-v2-r4tsTok
state: COMPLETED
elapsed: 00:00:28
exit_code: 0:0
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
```

Tokenizer gate:

```text
score_row_count: 8192
checked_row_count: 8192
failed_row_count: 0
empty_target_id_row_count: 0
empty_other_id_row_count: 0
target_other_overlap_row_count: 0
first_failing_row: null
tokenizer_name: Qwen/Qwen2.5-7B-Instruct
```

Guarded actions:

```text
model_forward_pass_started: false
scoring_authorized: false
scoring_job_submitted: false
training_started: false
generation_started: false
llama_started: false
same_family_null_started: false
sanitizer_benchmark_started: false
far_aggregation_started: false
paper_claim_allowed: false
```

## Interpretation

The two-sided cover-bank rows are compatible with the actual Qwen tokenizer
boundary contract. This resolves the tokenizer-boundary prerequisite for a
future teacher-forced surface-mass scoring route.

This review does not itself unlock generation, training, Qwen E2E, Llama,
same-family null, sanitizer, FAR, payload diversity, or paper-facing claims.

## Next Allowed Action

Prepare a reviewed H200 teacher-forced surface-mass scoring route for the
two-sided rows. The route must remain scoring-only:

```text
arms: base, protected, task_only
rows: 8192
no generation
no training
no Llama
no null/FAR expansion
no sanitizer
no paper claim
```

The future scoring gate should remain:

```text
protected lift vs base >= +0.15
protected lift vs task-only >= +0.10
protected rank1 >= 0.75
protected median margin > 0
boundary failures = 0
target/other overlap = 0
```
