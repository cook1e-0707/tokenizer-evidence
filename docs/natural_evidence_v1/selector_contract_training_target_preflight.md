# Selector Contract And Training-Target Preflight

This document records the artifact-only selector precommit contract draft and
branch-aware/regenerated-suffix training-target preflight after the R1
selector-contract analysis. It is not training, not generation, not E2E replay,
not payload recovery, and not FAR.

## Outputs

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_selector_contract_preflight/
```

Files:

```text
selector_contract_training_target_preflight_summary.json
selector_precommit_contract_draft.json
selector_contract_precommit_fields.csv
branch_aware_training_target_preflight_plan.csv
selector_contract_training_target_preflight.md
```

## Status

```text
COMPLETE_SELECTOR_PRECOMMIT_DRAFT_BRANCH_AWARE_PREFLIGHT_PLAN_READY
```

The draft selector contract is not active:

```text
DRAFT_NOT_ACTIVE_BLOCKED_BY_R1_NO_PROTECTED_LIFT
```

Reason: R1 had no protected lift over raw or task-only in any of the 64
comparison slices.

## Draft Selector Contract

| Field | Value |
|---|---|
| selector_id | `prefix_conditioned_observed_text_v0` |
| selector_mode | `prefix_conditioned_observed_text_event_scan` |
| match_policy | `UNSELECTED_BLOCKED_BY_R1` |
| reference_model | `Qwen/Qwen2.5-7B-Instruct` |
| reference_tokenizer | `Qwen/Qwen2.5-7B-Instruct` |
| bucket_policy_id | `expanded_actual_prefix_topk64_b4_minarity2_v1` |
| direct_replay_verifier_allowed | `false` |
| post_hoc_policy_selection_allowed | `false` |

Any active version must fix selector id, match policy, reference model/tokenizer,
bucket policy, audit key, payload, query budget, prompt split, thresholds,
decode rule, allowed trials, and multiple-testing rule before generation.

## Gate Plan

| Gate | Status | Required Artifact | Action |
|---|---|---|---|
| selector_contract_fields | PASS_DESIGNED_NOT_ACTIVE | `selector_precommit_contract_draft.json` | Keep as draft until repair gates pass |
| r1_protected_lift_over_raw | FAIL_BLOCKER | `r1_selector_contract_pairwise_lift.csv` | Do not use direct replay verifier |
| r1_protected_lift_over_task_only | FAIL_BLOCKER | `r1_selector_contract_pairwise_lift.csv` | Do not train from current target data |
| branch_aware_compatibility | NEEDS_RESULTS | `branch_aware_compatibility_summary.json` | Prepare Slurm-scored diagnostic only |
| regenerated_suffix_repair | NEEDS_RESULTS | `regenerated_suffix_repair_manifest.json` | Build artifact-only repair manifest/examples |
| teacher_forced_repaired_target_mass | NEEDS_RESULTS | `teacher_forced_repaired_target_mass_probe_summary.json` | Probe only after repaired data/objective preflight |
| sparse_coordinate_code | SYNTHETIC_PREFLIGHT_NEEDED | `sparse_coordinate_code_synthetic_preflight.json` | Keep decoder repair behind owner-specific coordinate survival |
| fresh_lockbox_or_locked_replay | NEEDS_RESULTS | `locked_selector_replay_or_lockbox_summary.json` | Use only after repair gates pass |

## Next Allowed Action

Prepare artifact-only branch-aware compatibility and regenerated/local-suffix
repair diagnostics under the draft selector contract. If this requires Chimera
CPU or GPU work, submit it through Slurm. Training remains forbidden.
