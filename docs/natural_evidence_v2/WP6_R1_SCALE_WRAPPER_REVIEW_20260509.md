# WP6-R1 Scale Wrapper Review: 2026-05-09

## Decision

The WP6-R1 scale wrapper and block-window majority decoder are implemented and
locally validated.

No Slurm job was submitted. No generation, new training, Llama, same-family
null, sanitizer, FAR aggregation, or paper-facing positive claim was started.

Because the active supervisor prompt still includes hard `no generation` and
`no Qwen E2E rerun` constraints, the scale allowlist entry is recorded but
remains disabled.

## Implemented Files

```text
scripts/natural_evidence_v2/decode_wp6_r1_scale_blocks.py
scripts/natural_evidence_v2/slurm/wp6_r1_coordinate_majority_scale_eval.sbatch
tests/test_natural_evidence_v2_wp6_coordinate_majority.py
configs/natural_evidence_v2/run_allowlist.yaml
```

## Decoder Contract

```text
decoder_id = qwen_v2_wp6_r1_block_window_coordinate_majority_decoder_v1
payload_cell = wp4_payload_a5_checksum_5e
payload_plus_checksum_hex = a55e
query_budgets_per_block = [8,16,32,64]
replicate_blocks = 4
replicate_block_size = 64
controlling_budget = 64
protected_block_accepts_required = >=3/4
null_accepts_required = 0/4 per null condition
minimum_support_at_64 = 16
minimum_majority_margin_at_64 = 3
```

The decoder evaluates each 64-frame block independently. For each block and
budget it majority-decodes strict Step `1..16` coordinates, then applies the
same checksum-and-payload accept rule to protected/raw/task-only/wrong-key and
wrong-payload decode conditions.

## Wrapper Checks

The scale wrapper:

- writes `precommit/wp6_r1_scale_contract.json` before generation;
- fixes `MAX_PROMPTS=256`, `BLOCK_COUNT=4`, and `BLOCK_SIZE=64`;
- fixes the selected prompt slice to the first 256 `wp3_r1_eval` prompts;
- records file rows `512..767` as four blocks: `512..575`, `576..639`,
  `640..703`, and `704..767`;
- preserves exact-frame decoder outputs as diagnostic artifacts;
- writes scale decode artifacts under `coordinate_majority_scale/`;
- refuses overwrite if generated outputs or a scale summary already exist;
- runs as a Chimera Slurm wrapper for any real CPU/GPU work.

## Local Validation

Plan-only wrapper validation:

```text
results/natural_evidence_v2/status/wp6_r1_scale_wrapper_validate_20260509_1904/
```

The plan-only run wrote:

```text
precommit/wp6_r1_scale_contract.json
wp6_generation_plan_summary.json
```

Key validation fields:

```text
generation_started = false
prompt_count = 256
max_prompts = 256
selected_split = wp3_r1_eval
selected_prompt_file_rows = 512..767
block_file_rows = 512..575, 576..639, 640..703, 704..767
transcript_precommitted_before_generation = true
```

Checks passed:

```text
.venv/bin/python -m py_compile scripts/natural_evidence_v2/decode_wp6_r1_scale_blocks.py scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py scripts/natural_evidence_v2/decode_wp6_payload.py
bash -n scripts/natural_evidence_v2/slurm/wp6_r1_coordinate_majority_scale_eval.sbatch
.venv/bin/python -m pytest tests/test_natural_evidence_v2_wp6_coordinate_majority.py tests/test_natural_evidence_v2_wp6_e2e_decode.py
```

Result:

```text
6 passed
```

## Allowlist

Recorded disabled entry:

```text
name = v2_wp6_r1_coordinate_majority_scale_eval
enabled = false
command = sbatch scripts/natural_evidence_v2/slurm/wp6_r1_coordinate_majority_scale_eval.sbatch
enable_condition = wrapper_review_passed_pending_separate_scale_submission_tick
```

The entry must only be enabled in a later submission tick that permits
generation/Qwen E2E, sends Telegram/email notification first, syncs the reviewed
wrapper to Chimera, submits exactly one Slurm job, and disables the entry
immediately after submission.

## Status

```text
PASS_LOCAL_WP6_R1_SCALE_WRAPPER_VALIDATION_ALLOWLIST_RECORDED_DISABLED_NO_SLURM
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
