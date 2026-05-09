# Branch-Aware Compatibility And Local-Suffix Repair Preparation

This note records the artifact-only preparation step after the 846699 negative
diagnosis, R1 prefix-conditioned selector replay, and selector-contract
preflight.

## Decision Status

`COMPLETE_BRANCH_AWARE_SUFFIX_REPAIR_INPUTS_PREPARED_NOT_SCORED`

The step prepared inputs only. It did not load a model, score branch-aware
compatibility, regenerate suffixes, train, generate E2E transcripts, decode
payload recovery, estimate FAR, or create paper-facing positive claims.

## Inputs

- selector preflight:
  `results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_selector_contract_preflight/`
- R1 selector replay:
  `results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_prefix_conditioned_selector_replay/`
- variable-radix train metadata:
  `results/natural_evidence_v1/status/variable_radix_frame_policy_dry_run_20260506_1848/`

The selector contract remains a draft:

`DRAFT_NOT_ACTIVE_BLOCKED_BY_R1_NO_PROTECTED_LIFT`

## Outputs

Output directory:

`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_suffix_repair_preparation/`

Primary files:

- `branch_aware_compatibility_summary.json`
- `branch_aware_compatibility_by_token_class.csv`
- `branch_aware_compatibility_scoring_plan.jsonl`
- `regenerated_suffix_repair_manifest.json`
- `regenerated_suffix_repair_examples.jsonl`
- `branch_aware_suffix_repair_readiness.csv`
- `branch_aware_suffix_repair_preparation.md`

## Counts

| Item | Count |
|---|---:|
| Source R1 examples read | 240 |
| Planned branch-aware scoring rows | 68 |
| Regenerated/local-suffix repair input examples | 68 |
| Train metadata matched rows | 68 |
| `compatible_non_target` drift rows | 60 |
| `observed_token_not_candidate_set` drift rows | 8 |

The prepared rows are raw-only:

`model_condition_counts.raw=68`

This is a limitation, not a success signal. It is enough to define the scoring
schema and null-side branch-aware/local-suffix diagnostic inputs, but it is not
enough for protected-vs-task-only branch-aware comparison. That comparison
requires either a richer R1 replay example export or an expanded example
selection that includes protected and task-only rows.

## Planned Branch-Aware Questions

The prepared scoring plan asks a later Slurm-scored diagnostic to separate three
questions:

1. Does the original suffix remain compatible after replacing the observed
   token with a target-bucket candidate?
2. Does `prefix + candidate` admit a short natural continuation when the suffix
   is allowed to branch?
3. Would a regenerated or locally repaired suffix reduce CE conflict while
   preserving the prompt answer semantics?

These questions target the current bottleneck: the strict token-index anchor
and suffix-preserving training target do not create a protected signal over raw
or task-only nulls.

## Gate Effect

No gate is unlocked by this preparation. It only creates inputs for the next
diagnostic.

Allowed next actions:

- run one Slurm-scored branch-aware compatibility diagnostic from
  `branch_aware_compatibility_scoring_plan.jsonl`; or
- construct an artifact-only local-suffix repair dry-run from
  `regenerated_suffix_repair_examples.jsonl`.

Forbidden actions remain:

- no new Qwen training;
- no Qwen E2E rerun;
- no Llama, same-family null, or sanitizer benchmark;
- no payload-recovery, full-FAR, robustness, or cross-family claims.

## Local-Suffix Repair Dry-Run

The local-suffix repair dry-run is now complete:

`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_local_suffix_repair_dry_run/`

Status:

`COMPLETE_LOCAL_SUFFIX_REPAIR_DRY_RUN_NOT_GENERATED`

This dry-run performed deterministic text-level local substitutions only. It
did not regenerate suffixes, score branch-aware compatibility, load a model,
train, run E2E, decode payload recovery, or estimate FAR.

Counts:

| Item | Count |
|---|---:|
| Repair examples | 68 |
| Text-substitution-ready rows | 36 |
| Needs tokenizer-aligned or branch regeneration | 32 |

The same limitation remains: all rows are raw-only
(`model_condition_counts.raw=68`). The dry-run also shows that approximate text
replacement is not sufficient as a repair method. In `32/68` rows the observed
token text cannot even be located in the original response text, and the
`36/68` text-substitution-ready rows are not naturalness-scored. The next useful
step is therefore a Slurm-scored branch-aware compatibility diagnostic, or a
richer protected/task-only replay-example export before scoring.

## Balanced Protected/Task-Only Example Export

A richer example export completed as Slurm job `848405`:

```text
job_name=nat-ev-qwen-babr
state=COMPLETED
exit_code=0:0
elapsed=00:00:46
checked_at=2026-05-08T05:02:35Z
```

The export uses:

- `scripts/natural_evidence_v1/export_balanced_branch_aware_examples.py`
- `scripts/natural_evidence_v1/slurm/qwen_balanced_branch_aware_examples.sbatch`

It is still artifact-only. It reads existing generated transcripts,
variable-radix train metadata, and bucketized candidate artifacts, then selects
balanced examples across protected, task-only, and raw conditions. It does not
score compatibility, regenerate suffixes, train, generate new outputs, run E2E,
decode payload recovery, or estimate FAR.

Synced output:

`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_balanced_branch_aware_examples/`

Counts:

| Condition | Rows |
|---|---:|
| protected_trained | 288 |
| task_only_lora | 288 |
| raw | 192 |

All `768` selected examples include generated transcript response text.

Reviewed files:

- `balanced_branch_aware_example_export_summary.json`
- `prefix_conditioned_selector_replay_examples.jsonl`
- `balanced_branch_aware_examples.csv`
- `balanced_branch_aware_examples_by_slice.csv`

## Balanced Branch-Aware/Local-Suffix Preparation

The richer examples were used to regenerate branch-aware/local-suffix diagnostic
inputs:

`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_suffix_repair_preparation_balanced/`

Status:

`COMPLETE_BRANCH_AWARE_SUFFIX_REPAIR_INPUTS_PREPARED_NOT_SCORED`

Counts:

| Item | Count |
|---|---:|
| Planned branch-aware scoring rows | 209 |
| Regenerated/local-suffix repair input examples | 209 |
| protected_trained rows | 76 |
| task_only_lora rows | 59 |
| raw rows | 74 |

Drift reasons:

| Reason | Rows |
|---|---:|
| compatible_non_target | 68 |
| observed_bucket_not_compatible | 79 |
| observed_token_not_candidate_set | 62 |

The balanced local-suffix repair dry-run is also complete:

`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_local_suffix_repair_dry_run_balanced/`

All `209/209` rows are text-substitution-ready in the generated transcript
response text. This remains a dry-run only. It is not branch-aware compatibility
scoring, naturalness scoring, training-data approval, payload recovery, or FAR.

## Balanced Branch-Aware Compatibility Scoring

The Slurm-scored branch-aware compatibility proxy diagnostic is now complete:

```text
job_id=848414
job_name=nat-ev-qwen-brscore
state=COMPLETED
exit_code=0:0
elapsed=00:00:55
```

Output directory:

`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_compatibility_scored_balanced/`

Primary files:

- `branch_aware_compatibility_score_summary.json`
- `branch_aware_compatibility_score_rows.jsonl`
- `branch_aware_compatibility_score_rows.csv`
- `branch_aware_compatibility_score_by_group.csv`

Status:

`COMPLETE_BRANCH_AWARE_COMPATIBILITY_MODEL_SCORED_PROXY_NOT_GENERATED`

This job loaded Qwen/Qwen2.5-7B-Instruct as a reference-model scorer and used
NLL deltas as a branch-aware/local-suffix proxy. It did not generate branch
continuations, regenerate suffixes, train, run E2E, decode payload recovery, or
estimate FAR.

Aggregate metrics:

| Metric | Value |
|---|---:|
| Scored rows | 209 |
| Response naturalness proxy pass | 155 / 209 (`0.7416`) |
| Suffix-preserving proxy pass | 169 / 209 (`0.8086`) |
| Branch-aware proxy pass | 153 / 209 (`0.7321`) |
| Mean response delta NLL/token | `0.318643` |
| Mean suffix delta NLL/token | `0.554276` |

By condition:

| Condition | Rows | Branch-aware proxy pass |
|---|---:|---:|
| protected_trained | 76 | 57 (`0.7500`) |
| raw | 74 | 52 (`0.7027`) |
| task_only_lora | 59 | 44 (`0.7458`) |

By drift reason:

| Drift reason | Rows | Branch-aware proxy pass |
|---|---:|---:|
| compatible_non_target | 68 | 58 (`0.8529`) |
| observed_bucket_not_compatible | 79 | 54 (`0.6835`) |
| observed_token_not_candidate_set | 62 | 41 (`0.6613`) |

By token class:

| Token class | Rows | Branch-aware proxy pass |
|---|---:|---:|
| function_word | 22 | 21 (`0.9545`) |
| punctuation | 44 | 24 (`0.5455`) |
| word | 143 | 108 (`0.7552`) |

Interpretation:

- Many local target substitutions are plausible under the model-scored proxy.
- `compatible_non_target` rows are the cleanest repair candidates.
- Punctuation is the weakest class under the proxy.
- Protected is essentially tied with task-only (`0.7500` vs `0.7458`), so this
  does not establish protected-specific ownership signal.

The next allowed action is artifact-only branch-aware score interpretation and
repaired training-target preflight. Training and E2E remain forbidden.

## Score Interpretation And Repaired-Target Preflight

The artifact-only interpretation/preflight is complete:

`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_score_interpretation/`

Primary files:

- `branch_aware_score_interpretation_summary.json`
- `branch_aware_score_slice_summary.csv`
- `branch_aware_score_protected_vs_controls.csv`
- `repaired_target_mass_probe_candidates.jsonl`
- `repaired_target_mass_probe_candidates.csv`
- `repaired_target_mass_probe_secondary_candidates.jsonl`
- `branch_aware_score_interpretation.md`

Status:

`COMPLETE_BRANCH_AWARE_SCORE_INTERPRETATION_REPAIRED_TARGET_PREFLIGHT_NOT_TRAINING`

Candidate counts:

| Category | Rows |
|---|---:|
| Primary repaired target-mass probe candidates | 75 |
| Secondary / ablation candidates | 78 |
| Rejected rows | 56 |

Primary candidates by condition:

| Condition | Rows |
|---|---:|
| protected_trained | 39 |
| task_only_lora | 20 |
| raw | 16 |

Primary candidates by drift reason:

| Drift reason | Rows |
|---|---:|
| compatible_non_target | 58 |
| observed_token_not_candidate_set | 17 |

Primary candidates by token class:

| Token class | Rows |
|---|---:|
| word | 74 |
| function_word | 1 |

Decision:

`PRIMARY_CANDIDATES_EXIST_BUT_NO_TRAINING_GATE_PROTECTED_CONTROL_SEPARATION_WEAK`

This is a useful positive diagnostic for repairability, not a training gate.
The rows selected in `repaired_target_mass_probe_candidates.jsonl` can be used
for a future artifact-only teacher-forced repaired target-mass probe. They
cannot be used to justify new LoRA training or Qwen E2E rerun without first
showing a protected target-mass lift over base and task-only controls.
