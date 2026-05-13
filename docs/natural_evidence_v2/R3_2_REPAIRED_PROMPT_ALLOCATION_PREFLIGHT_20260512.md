# R3.2 Repaired Prompt Allocation Preflight

Date: 2026-05-12

## Scope

This is an artifact-only preflight for repairing the R3.2 Qwen same-contract
locked-scale route after H200 job `853430`. It does not submit Slurm, generate
new transcripts, train, run Llama, aggregate FAR, or make paper-facing claims.

## Input Artifacts

- Prompt artifact:
  `results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/restricted_step_label_strict_density_audit_prompts.jsonl`
- Config:
  `configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale.yaml`
- Preflight script:
  `scripts/natural_evidence_v2/plan_r3_2_repaired_prompt_allocation.py`

## Result

The requested 96-block locked-scale schedule is not feasible with the current
prompt artifact if every block must be backed by a distinct prompt row.

| Quantity | Value |
|---|---:|
| total prompt rows | 2,560 |
| dev rows | 512 |
| eval rows | 2,048 |
| desired shards | 12 |
| blocks per shard | 8 |
| block size | 64 |
| required unique eval rows for 96 blocks | 6,144 |
| available unique eval rows | 2,048 |
| max feasible unique shards | 4 |
| max feasible unique blocks | 32 |
| additional eval rows needed for 96 blocks | 4,096 |

Machine-readable preflight:

`results/natural_evidence_v2/status/r3_2_repaired_prompt_allocation_preflight_20260512/r3_2_repaired_prompt_allocation_preflight.json`

Prompt window plan:

`results/natural_evidence_v2/status/r3_2_repaired_prompt_allocation_preflight_20260512/r3_2_repaired_prompt_window_plan.csv`

## Window Composition

| Proposed shard | Prompt rows | Variant composition |
|---|---:|---|
| `shard_00` | `512-1023` | `strict_literal_16_step_lines: 512` |
| `shard_01` | `1024-1535` | `strict_literal_16_step_lines: 171`, `strict_no_heading_16_step_lines: 341` |
| `shard_02` | `1536-2047` | `strict_no_heading_16_step_lines: 342`, `strict_numbered_step_label_lines: 170` |
| `shard_03` | `2048-2559` | `strict_numbered_step_label_lines: 512` |

This aligns with the 853430 attribution: later windows, especially the
`strict_numbered_step_label_lines`-heavy window, had much weaker target survival
and support/margin under the repeated-coordinate decoder.

## Consequence

The current prompt artifact can support a 32-block unique diagnostic package
without prompt row reuse. It cannot support a 96-block independent locked-scale
package. A future 96-block R3.2 rerun must either:

1. expand the prompt bank to at least 6,144 eval rows, then precommit 12
   distinct 512-row shard windows; or
2. redefine the route as a smaller 32-block unique diagnostic and avoid any
   96-block claim.

## Next Allowed Action

Artifact-only route decision or repair design. A Slurm resubmission remains
blocked until a reviewed route records either a 32-block unique diagnostic or a
larger expanded prompt bank with 12 genuinely distinct windows, plus local and
remote allowlist safety checks and Hermes notification.
