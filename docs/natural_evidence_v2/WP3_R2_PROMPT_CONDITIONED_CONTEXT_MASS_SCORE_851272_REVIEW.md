# WP3-R2 prompt-conditioned context-mass score review: job 851272

## Decision

`851272` completed successfully on Chimera DGXA100 and produced a valid
prompt-conditioned Qwen context-mass audit. The whole candidate-set audit is
still labeled `FAIL` because the scored set includes invalid and low-mass
candidates, but per-bank review finds one primary bank that passes the current
business gate for WP3-R2.

Selected primary bank:

```text
source_candidate_bank_id = step_label_r2_prompt_ctx_set_plan_vs_create_prepare_v1
bucket 0 = Set | Plan
bucket 1 = Create | Prepare
min_bucket_mass = 0.06311000335572636
combined_bank_mass = 0.14328484169406375
mass_ratio = 1.2703982582035882
context_count = 512
```

This satisfies the pilot threshold `min_bucket_mass >= 0.03`, the paper-ready
absolute-mass threshold `min_bucket_mass >= 0.05`, the preferred combined mass
threshold `combined_bank_mass >= 0.10`, and the balance threshold
`mass_ratio <= 3`.

## Source Artifacts

```text
results/natural_evidence_v2/status/wp3_r2_prompt_conditioned_context_mass_score_851272/qwen_v2_wp3_context_mass_score_summary.json
results/natural_evidence_v2/status/wp3_r2_prompt_conditioned_context_mass_score_851272/qwen_v2_wp3_context_mass_audit.json
results/natural_evidence_v2/status/wp3_r2_prompt_conditioned_context_mass_score_851272/qwen_v2_wp3_context_mass_artifact.json
results/natural_evidence_v2/status/wp3_r2_prompt_conditioned_context_mass_score_851272/qwen_v2_wp3_context_mass_context_scores.jsonl
results/natural_evidence_v2/status/wp3_r2_prompt_conditioned_context_mass_score_851272/qwen_v2_wp3_context_mass_invalid_tokenization_rows.jsonl
results/natural_evidence_v2/status/wp3_r2_prompt_conditioned_context_mass_score_851272/slurm/nat-ev-v2-wp3ctxm-851272.out
results/natural_evidence_v2/status/wp3_r2_prompt_conditioned_context_mass_score_851272/slurm/nat-ev-v2-wp3ctxm-851272.err
```

Derived review artifacts:

```text
results/natural_evidence_v2/status/wp3_r2_prompt_conditioned_context_mass_score_851272/wp3_r2_prompt_conditioned_context_mass_score_851272_review.json
results/natural_evidence_v2/status/wp3_r2_prompt_conditioned_context_mass_score_851272/wp3_r2_prompt_conditioned_bank_selection.csv
results/natural_evidence_v2/buckets/qwen_v2_primary_2way_bank.jsonl
results/natural_evidence_v2/buckets/qwen_v2_primary_2way_bank_audit.csv
results/natural_evidence_v2/buckets/qwen_v2_bank_rejections_851272.csv
```

## Slurm Status

```text
job_id = 851272
job_name = nat-ev-v2-wp3ctxm
partition = DGXA100
node = chimera13
state = COMPLETED
elapsed = 00:05:06
exit_code = 0:0
```

No training, transcript generation, E2E evaluation, FAR aggregation, sanitizer
benchmark, Llama run, or paper-facing positive claim was started.

## Score Summary

```text
score_plan_rows = 10240
context_score_rows = 9728
invalid_tokenization_rows = 512
mass_rows = 19
script_mass_gate_status = FAIL
model = Qwen/Qwen2.5-7B-Instruct
scoring_context_kind = chat_prompt_plus_assistant_prefix
```

The invalid tokenization rows all come from
`step_label_r2_prompt_ctx_provide_vs_summarize_v1`, because `Summarize` is not
one Qwen next token under the scoring prefix:

```text
Summarize -> [8116, 5612, 551]
```

The script-level gate is therefore not the final business decision. It is a
candidate-set audit result. The relevant WP3-R2 question is whether a
precommitted, prompt-conditioned, single-token 2-way bank exists with adequate
absolute mass and balance. Job `851272` answers that question yes for the
selected primary bank above.

## Candidate Review

Passing counts under the stricter business thresholds:

```text
pilot-passing banks = 5
paper-ready absolute-mass banks = 1
preferred-combined-mass banks = 2
```

Top reviewed banks:

| Role | Candidate | Bucket 0 | Bucket 1 | Min mass | Combined mass | Ratio |
|---|---|---|---|---:|---:|---:|
| primary | `step_label_r2_prompt_ctx_set_plan_vs_create_prepare_v1` | `Set, Plan` | `Create, Prepare` | 0.063110 | 0.143285 | 1.270 |
| secondary | `step_label_r2_prompt_ctx_set_create_vs_plan_prepare_v1` | `Set, Create` | `Plan, Prepare` | 0.049635 | 0.143285 | 1.887 |
| secondary | `step_label_r2_prompt_ctx_assign_schedule_vs_identify_establish_v1` | `Assign, Schedule` | `Identify, Establish` | 0.044221 | 0.090905 | 1.056 |
| secondary | `step_label_r2_prompt_ctx_assign_identify_vs_schedule_establish_v1` | `Assign, Identify` | `Schedule, Establish` | 0.043189 | 0.090905 | 1.105 |
| secondary | `step_label_r2_prompt_ctx_set_vs_plan_v1` | `Set` | `Plan` | 0.030188 | 0.080175 | 1.656 |
| near miss | `step_label_r2_prompt_ctx_keep_check_vs_choose_develop_v1` | `Keep, Check` | `Choose, Develop` | 0.028954 | 0.059359 | 1.050 |

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| WP3-R1 strict density | Pass with legacy top-level runner note | `850885` split-level dev/eval oracle completion passes; one eval response caused the legacy top-level fail due language drift |
| WP3-R2 high-mass bank | Pass by reviewed primary selection | `Set/Plan` vs `Create/Prepare` passes mass, balance, and combined-mass thresholds |
| WP3-R3 naturalness | Pass with language-drift note | `96` balanced examples: `PASS=88`, `BORDERLINE=8`, no forbidden/coding/semantic failures |
| WP4 | Artifact-only allowed next | Prompt-local contract and decoder oracle only |
| Training | Forbidden | Teacher-forced target-mass gate has not run for v2 |
| Qwen E2E | Forbidden | No WP5 teacher-forced lift evidence yet |

## Interpretation

The important change from job `851233` to job `851272` is that scoring the same
surfaces under the full prompt-conditioned assistant prefix materially raises
absolute mass. The generic bare `Step N:` prefix was too weak; the
prompt-conditioned context gives a usable high-mass 2-way action-verb bank.

This is still not payload recovery, not FAR, and not proof that training will
work. It only establishes that WP3 can move from structure/mass auditing into
artifact-only WP4 prompt-local contract compilation.

## Next Allowed Action

Prepare the artifact-only WP4 prompt-local payload contract:

```text
payload = 8-bit payload + 8-bit checksum
slots = 16 line-start Step-label action-verb micro-slots
bank = qwen_v2_wp3_r2_primary_set_plan_vs_create_prepare_v1
decoder oracle substitution accept target = 100%
wrong-key oracle reject target = 100%
wrong-payload oracle reject target = 100%
```

Training, Qwen E2E, Llama, same-family nulls, sanitizer benchmark, FAR
aggregation, and paper-facing positive claims remain forbidden.
