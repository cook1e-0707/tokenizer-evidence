# R3.2 Expanded Prompt Plan 6144

Date: 2026-05-12

## Scope

Artifact-only prompt-bank expansion for the Qwen v2 R3.2 same-contract
locked-scale route. No model generation, training, Slurm submission, FAR
aggregation, Llama run, sanitizer benchmark, or paper-facing claim was started.

## Artifacts Created

Expanded WP2 prompt scaffold:

`results/natural_evidence_v2/prompts/wp2_controlled_natural_prompt_family_scaffold_r3_2_expanded_20260512/`

Expanded WP3 strict Step-label prompt plan:

`results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_r3_2_6144_20260512/`

R3.2 prompt allocation preflight:

`results/natural_evidence_v2/status/r3_2_repaired_prompt_allocation_preflight_expanded_6144_20260512/`

## Counts

The expanded WP2 scaffold wrote:

| Split | Rows |
|---|---:|
| train | 6,144 |
| dev | 6,144 |
| eval | 6,144 |
| organic_null | 2,048 |

The expanded WP3 strict Step-label prompt plan wrote:

| Split | Rows |
|---|---:|
| dev | 512 |
| eval | 6,144 |

The WP3 oracle prompt-local frame completion rate is `1.0` by construction.

## R3.2 Allocation Preflight

Using the expanded eval prompt artifact:

`results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_r3_2_6144_20260512/restricted_step_label_r1_eval_prompts.jsonl`

and split:

`wp3_r1_density_eval`

the repaired allocation preflight reports:

| Metric | Value |
|---|---:|
| desired shards | 12 |
| blocks per shard | 8 |
| block size | 64 |
| desired blocks | 96 |
| available eval rows | 6,144 |
| available unique shards | 12 |
| available unique blocks | 96 |
| duplicate prompt IDs | 0 |
| bad structural-slot rows | 0 |
| allocation status | PASS |

The 12 proposed windows are distinct 512-row windows. Variant composition is
balanced within each window, avoiding the previous artifact's late-window
variant concentration.

## Remaining Wrapper Repair Before Slurm

The expanded prompt plan is feasible, but no R3.2 Slurm submission is allowed
yet. The existing R3.2 wrappers and precommit builder still encode assumptions
from the old 2,560-row prompt artifact:

- old split: `wp3_r1_eval`;
- old selected prompt source shape: dev+eval in one 2,560-row file;
- old shard allocation: `SHARD_INDEX % 4`;
- old selected prompt manifest hash.

Before submission, wrappers must be reviewed and repaired to support:

- prompt artifact path:
  `results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_r3_2_6144_20260512/restricted_step_label_r1_eval_prompts.jsonl`;
- split: `wp3_r1_density_eval`;
- shard allocation: `expected_start = SHARD_INDEX * 512`;
- shard range: `0..11`, no modulo reuse;
- 12 unique selected prompt hashes;
- updated precommit manifest hash;
- aggregate duplicate-window guard enabled.

## Next Allowed Action

Artifact-only wrapper/precommit repair and local plan-only validation for the
expanded 6144-row prompt route. No Slurm submission until the repaired wrapper,
allowlist safety, local/remote hashes, and Hermes notification are reviewed.
