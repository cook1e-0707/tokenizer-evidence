# R3.2 Qwen Locked-Scale Package Review: 2026-05-11

## Decision

Route R3.2 package scope is recorded for wrapper review only.

No Slurm job was submitted. No training, generation, Qwen E2E rerun, Llama,
same-family null, sanitizer benchmark, FAR aggregation, or paper-facing positive
claim was started.

Machine-readable status:

```text
results/natural_evidence_v2/status/r3_2_qwen_locked_scale_package_review_20260511_0200.json
```

## Controlling Inputs

```text
docs/natural_evidence_v1/AUTOMATION_STATE.md
docs/natural_evidence_v1/next_step_codex_plan.md
results/natural_evidence_v1/status/gate_status.json
docs/natural_evidence_v2/PROTOCOL_CONTRACT.md
docs/natural_evidence_v2/CLAIM_GUARDRAILS.md
results/natural_evidence_v2/status/gate_status.json
docs/natural_evidence_v2/WP6_R2_OPTION_B_852426_CANONICAL_REVIEW.md
docs/natural_evidence_v2/REPEATED_COORDINATE_DECODER_SPEC.md
```

## Locked R3.2 Scope

```text
route = V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED
package_id = qwen_v2_r3_2_locked_scale_package_v1
model_family = Qwen only
payloads = P00, P01, P02, P03
seeds = 17, 23, 29
cells = 12
blocks_per_cell = 8
block_size = 64
protected_blocks_total = 96
arms = protected, raw, task_only, wrong_key, wrong_payload
query_budgets = 16, 32, 64
primary_budget = 64
diagnostic_budgets = 16, 32
support_threshold = 16
majority_margin_threshold = 3
forbidden_public_surface_count_required = 0
```

The R3.2 pass gate must be precommitted before transcript generation:

```text
protected accepts at budget 64 >= 80 / 96
raw accepts at budget 64 = 0 / 96
task_only accepts at budget 64 = 0 / 96
wrong_key accepts at budget 64 = 0 / 96
wrong_payload accepts at budget 64 = 0 / 96
accepted-block support >= 16
accepted-block majority margin >= 3
forbidden_public_surface_count = 0
```

## Wrapper Readiness Finding

The existing reviewed R2 wrapper is not sufficient as the R3.2 submission
wrapper without a separate implementation and validation pass.

Observed R2 wrapper constraints:

```text
script = scripts/natural_evidence_v2/slurm/wp6_r2_option_b_scale_eval.sbatch
protocol_id = natural_evidence_v2_wp6_r2_option_b_robust_block_scale
payload_plus_checksum_hex = a55e
block_count = 8
block_size = 64
max_prompts = 512
selected_prompt_file_rows = 768..1279
minimum_protected_block_accepts_at_64 = 6
```

R3.2 requires a package wrapper that fixes a 12-cell grid and 96 protected
blocks before generation. The wrapper review must explicitly validate:

```text
payload grid = P00/P01/P02/P03
seed grid = 17/23/29
cell count = 12
blocks per cell = 8
total blocks per arm = 96
arms = protected/raw/task_only/wrong_key/wrong_payload
primary budget = 64
diagnostic budgets = 16/32
support threshold = 16
majority margin threshold = 3
protected pass threshold = 80/96
null-arm pass threshold = 0/96 for each null arm
forbidden public surface count = 0
fresh output directory refusal on any pre-existing precommit, transcript,
decode summary, or gate-review artifact
```

## Required Before Any Slurm Submission

Before a later tick may submit R3.2, all of the following must be recorded:

```text
wrapper implementation review = missing
allowlist entry review = disabled placeholder recorded only
precommit contract artifact = missing
local plan-only validation = missing
gate review package = this document only, not sufficient for submission
Hermes Telegram/email notification = required at submission tick
```

The allowlist must remain disabled until a later notified submission tick
explicitly permits exactly one reviewed R3.2 Qwen Slurm job and disables the
entry immediately after submission.

Disabled allowlist placeholder recorded on 2026-05-11:

```text
name = v2_r3_2_qwen_locked_scale_eval
command = sbatch scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
enabled = false
enable_condition = disabled_until_r3_2_wrapper_precommit_and_gate_review_recorded
```

This is not a submission approval because the wrapper path does not yet exist,
and no R3.2 precommit contract or local plan-only validation has been recorded.

## Still Forbidden

```text
training = false
generation = false
qwen_e2e_rerun = false
llama = false
same_family_null = false
sanitizer = false
far_aggregation = false
paper_facing_positive_claim = false
```

## Status

```text
R3_2_QWEN_LOCKED_SCALE_PACKAGE_SCOPE_RECORDED_WRAPPER_NOT_READY_NO_SLURM
```
