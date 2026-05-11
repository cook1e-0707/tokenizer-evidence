# R3.2 Full Wrapper Upgrade Blocker: 2026-05-11T06:17Z

## Status

`BLOCK_R3_2_FULL_WRAPPER_NEEDS_SHARD_AGGREGATION_PATH_NO_SLURM`

## Decision

No Slurm job was submitted. No generation, Qwen E2E rerun, training, Llama,
same-family null, sanitizer benchmark, FAR aggregation, or paper-facing positive
claim was started.

## Review Scope

Reviewed the current R3.2 wrapper and nearest existing full-generation/decode
paths:

```text
scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
scripts/natural_evidence_v2/slurm/wp6_r2_option_b_scale_eval.sbatch
scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py
scripts/natural_evidence_v2/decode_wp6_payload.py
scripts/natural_evidence_v2/decode_wp6_r1_scale_blocks.py
configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale.yaml
```

## Blocker

The existing full-generation/decode path is scoped to one contiguous prompt
window and fixed WP6 artifact names:

```text
wp6_generated_outputs.jsonl
wp6_slot_observations.jsonl
wp6_decode_decisions.jsonl
coordinate_majority_*/...summary.json
```

R3.2 requires a 12-shard same-contract package:

```text
fixed a55e contract
canonical units are shard_00..shard_11, not payload labels
12 replicate groups
8 blocks per group
96 protected blocks per arm
aggregate protected accepts at 64 >= 80/96
aggregate null accepts at 64 = 0/96 for each null arm
```

A wrapper-only loop over the existing scripts would either reuse fixed artifact
names in one output directory or produce per-shard subdirectories without a
reviewed R3.2 aggregate gate artifact. That is not sufficient for the approved
locked-scale contract.

## Required Next Action

Implement or review a minimal R3.2 same-contract shard aggregation path before enabling the
allowlist or submitting Slurm. The path must preserve the fixed `a55e` contract,
avoid fake payload/cell labels, refuse existing output artifacts, and produce an
explicit R3.2 aggregate 96-block gate review.

## Gate State

```text
allowlist_enabled = false
slurm_job_submitted = false
generation_started = false
qwen_e2e_rerun_started = false
paper_claim_allowed = false
```
