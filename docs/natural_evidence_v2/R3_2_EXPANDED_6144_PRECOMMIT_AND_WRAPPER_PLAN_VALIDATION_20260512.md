# R3.2 Expanded 6144 Precommit And Wrapper Plan Validation

Date: 2026-05-12

## Scope

This update validates the artifact-only control plane for the repaired Qwen v2
R3.2 same-contract locked-scale route. It does not submit Slurm, train, run
generation, run Llama, aggregate FAR, launch sanitizer work, or make any
paper-facing claim.

The route remains same-contract only:

- `contract_id = a55e`;
- `payload_diversity_tested = false`;
- route units are replicate groups, shards, and blocks, not payload labels.

## Expanded Config

Added and frozen:

`configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale_expanded_6144.yaml`

Key fields:

- prompt artifact:
  `results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_r3_2_6144_20260512/restricted_step_label_r1_eval_prompts.jsonl`;
- prompt source sha256:
  `8fcba10d2df1dae83eb03f8ce26fa45623c1918a9246c94b6b6868fc1204247a`;
- split: `wp3_r1_density_eval`;
- prompt-window policy: `distinct_eval_window_by_shard_index`;
- 12 replicate groups x 8 blocks x 64 prompts = 96 blocks per arm;
- selected prompt manifest sha256:
  `ce057a36ad75424919f4367eb3e2f0221725a9c6715d156ab4b2a377edb600ed`.

## Precommit Builder

Added:

`scripts/natural_evidence_v2/build_r3_2_expanded_locked_scale_precommit.py`

The builder is artifact-only. It validates:

- WP4 contract payload is `a55e`;
- R3.2 does not use payload IDs or distinct payload contracts;
- split is `wp3_r1_density_eval`;
- schedule is 12 x 8 x 64;
- prompt-window policy is `distinct_eval_window_by_shard_index`;
- decoder thresholds remain support `16` and majority margin `3`;
- claim-control gates remain false;
- prompt source hash matches the expanded config;
- selected prompt manifest hash matches the expanded config;
- all 12 512-row windows are unique;
- all 96 block hashes are unique;
- all selected rows have 16 expected structural slots.

Validated outputs:

`results/natural_evidence_v2/status/r3_2_expanded_6144_precommit_plan_20260512_verified_after_builder_fix/`

Summary:

| Metric | Value |
|---|---:|
| replicate groups | 12 |
| total blocks per arm | 96 |
| unique window hashes | 12 |
| unique block hashes | 96 |
| selected manifest hash matched config | yes |
| Slurm submitted | no |

The precommit hash is:

`6de7432ef3155100321affa30f677c2d88e17d5bc6323cde2670ab838d8a85ea`

## Wrapper Plan-Only Validation

Repaired the shard-array wrappers to use the expanded route:

- `scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_shard_array_h200.sbatch`;
- `scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_shard_array.sbatch`.

Both wrappers now default to:

- expanded 6144-row prompt artifact;
- expanded config;
- expanded precommit builder;
- selected prompt manifest hash
  `ce057a36ad75424919f4367eb3e2f0221725a9c6715d156ab4b2a377edb600ed`;
- split `wp3_r1_density_eval`;
- prompt-window policy `distinct_eval_window_by_shard_index`;
- shard row allocation `expected_start = SHARD_INDEX * 512`.

Both wrappers also support:

`VALIDATE_PLAN_ONLY=1`

This mode builds and validates the precommit artifacts, then exits before
adapter checks, model generation, or decode. It is intended for local and
remote control-plane review before any Slurm submission.

Plan-only smoke outputs:

- H200 wrapper:
  `results/natural_evidence_v2/status/r3_2_expanded_6144_h200_wrapper_plan_smoke_20260512/`;
- scavenger wrapper:
  `results/natural_evidence_v2/status/r3_2_expanded_6144_scavenger_wrapper_plan_smoke_20260512/`.

Both passed with:

- manifest hash matched;
- 12 unique windows;
- 96 unique blocks;
- no Slurm submission.

## Remaining Blockers Before Slurm

R3.2 Slurm submission is still blocked until a later reviewed submission route
records all of the following:

- allowlist safety passes with zero enabled entries before enablement;
- exactly one reviewed R3.2 shard-array allowlist entry is enabled for
  submission;
- local and Chimera remote hashes match for config, wrappers, precommit
  builder, prompt artifact, WP4 contract, gate status, and current state;
- Hermes TG/email notification succeeds before submission;
- the allowlist entry is disabled immediately after `sbatch` returns.

The legacy serial wrapper still contains old 4-window assumptions and is not
the canonical expanded route. Use the repaired shard-array wrappers for the
expanded 6144 route.

## Current Claim Policy

Allowed internal statement:

`The expanded 6144-row R3.2 same-contract precommit and shard-array plan-only wrapper validation passed locally.`

Still not allowed:

- full FAR;
- payload diversity;
- Llama success;
- cross-family generality;
- same-family null rejection;
- sanitizer robustness;
- paper-facing positive claim.
