# WP6-R2 Option B Wrapper Review: 2026-05-10

## Decision

The WP6-R2 Option B wrapper and contract-only/plan-only path are implemented
and locally validated.

No Slurm job was submitted. No generation, Qwen E2E rerun, new training,
Llama, same-family null, sanitizer, FAR aggregation, or paper-facing positive
claim was started.

## Implemented Files

```text
scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py
scripts/natural_evidence_v2/decode_wp6_r1_scale_blocks.py
scripts/natural_evidence_v2/slurm/wp6_r2_option_b_scale_eval.sbatch
tests/test_natural_evidence_v2_wp6_coordinate_majority.py
configs/natural_evidence_v2/run_allowlist.yaml
```

## Contract

```text
protocol_id = natural_evidence_v2_wp6_r2_option_b_robust_block_scale
decoder_id = qwen_v2_wp6_r2_robust_block_coordinate_majority_decoder_v1
payload_plus_checksum_hex = a55e
query_budgets_per_block = [8,16,32,64]
block_count = 8
block_size = 64
prompt_count = 512
controlling_budget = 64
protected_robust_block_accepts_required = >=6/8
null_robust_accepts_required = 0/8 per raw/task-only/wrong-key/wrong-payload
minimum_support_at_64 = 16
minimum_majority_margin_at_64 = 3
```

The wrapper fixes the fresh `wp3_r1_eval` prompt window to file rows
`768..1279`, disjoint from the failed `852202` rows `512..767`.

## Local Validation

Plan-only wrapper validation:

```text
results/natural_evidence_v2/status/wp6_r2_option_b_wrapper_validate_20260510_0312/
```

The plan-only run wrote:

```text
precommit/wp6_r2_option_b_contract.json
wp6_generation_plan_summary.json
```

Key validation fields:

```text
generation_started = false
max_prompts = 512
prompt_count = 512
selected_split = wp3_r1_eval
selected_prompt_file_rows = 768..1279
selected_prompt_jsonl_sha256 = d3966ce5c43347df9c68dc6cd6118102fb0708484ddd53e9b08b7b42b1f12ddd
block_file_rows = 768..831, 832..895, 896..959, 960..1023, 1024..1087, 1088..1151, 1152..1215, 1216..1279
transcript_precommitted_before_generation = true
```

Checks passed:

```text
.venv/bin/python -m py_compile scripts/natural_evidence_v2/decode_wp6_r1_scale_blocks.py scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py scripts/natural_evidence_v2/decode_wp6_payload.py
bash -n scripts/natural_evidence_v2/slurm/wp6_r2_option_b_scale_eval.sbatch
.venv/bin/python -m pytest tests/test_natural_evidence_v2_wp6_coordinate_majority.py tests/test_natural_evidence_v2_wp6_e2e_decode.py
VALIDATE_PLAN_ONLY=1 ... bash scripts/natural_evidence_v2/slurm/wp6_r2_option_b_scale_eval.sbatch
```

Result:

```text
9 passed
```

## Allowlist

Recorded disabled entry:

```text
name = v2_wp6_r2_option_b_scale_eval
enabled = false
command = sbatch scripts/natural_evidence_v2/slurm/wp6_r2_option_b_scale_eval.sbatch
enable_condition = wrapper_review_passed_pending_separate_r2_submission_tick
```

The entry must remain disabled until a later notified submission tick explicitly
permits generation/Qwen E2E, sends Telegram/email notification first, syncs the
reviewed wrapper to Chimera, submits exactly one Slurm job, and disables the
entry immediately after submission.

## Status

```text
PASS_LOCAL_WP6_R2_OPTION_B_WRAPPER_VALIDATION_ALLOWLIST_RECORDED_DISABLED_NO_SLURM
```

## Still Forbidden

- no Slurm submission from this tick;
- no generation while the current hard constraint remains active;
- no Qwen E2E rerun while the current hard constraint remains active;
- no new training;
- no Llama;
- no same-family null;
- no sanitizer;
- no FAR aggregation;
- no paper-facing positive claim.
