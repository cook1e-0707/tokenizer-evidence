# WP6-R1 Scale/Reproducibility Decision Package: 2026-05-09

## Decision

Prepare a Qwen-only WP6-R1 reproducibility scale run, but do not submit it until
the scale wrapper and allowlist entry are reviewed.

This package is not a Slurm submission, not new training, not FAR aggregation,
not a Llama run, and not a paper-facing positive claim.

## Starting Point

Job `852094` is the controlling WP6-R1 proof-of-life result.

```text
job_id = 852094
result = internal pass for precommitted repeated-coordinate majority decoder
protected budget-64 decoded_hex = a55e
raw/task-only/wrong-key/wrong-payload = reject
exact decoder = still failed, not controlling for R1
```

Review:

```text
docs/natural_evidence_v2/WP6_R1_COORDINATE_MAJORITY_E2E_852094_REVIEW.md
docs/natural_evidence_v2/WP6_R1_METADATA_CLEANUP_20260509.md
```

## Scale Objective

The next run should answer a narrow reproducibility question:

```text
Does the precommitted WP6-R1 repeated-coordinate majority decoder recover the
same trained Qwen v2 payload across independent 64-query prompt blocks?
```

It should not answer:

```text
full FAR
multi-payload generality
Llama transfer
same-family nulls
sanitizer robustness
paper-facing success claim
```

## Prompt Count

Use `256` prompts from the locked WP3-R1 strict Step-label prompt source:

```text
prompt_source =
results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/restricted_step_label_strict_density_audit_prompts.jsonl

prompt_source_rows = 2560
prompt_source_sha256 = 20154c7b14851ce2116041176ab92acc727f1c49c343826eac9ecfc9430fc179
```

The scale wrapper must precommit the exact `wp3_r1_eval` prompt slice. This
matches `generate_wp6_e2e_outputs.py`, which filters by split before applying
`MAX_PROMPTS`.

```text
selected_split = wp3_r1_eval
selected_prompt_rows = first 256 rows after filtering prompt_source by selected_split
selected_prompt_file_rows = 512..767
replicate_block_count = 4
replicate_block_size = 64
blocks =
  block_0: selected eval indices 0..63, file rows 512..575
  block_1: selected eval indices 64..127, file rows 576..639
  block_2: selected eval indices 128..191, file rows 640..703
  block_3: selected eval indices 192..255, file rows 704..767
```

The `256` prompt count is a reproducibility sample size, not a new owner query
budget claim.

## Payload Cells

Use exactly one trained payload cell for this scale run:

```text
payload_cell_id = wp4_payload_a5_checksum_5e
payload_byte_hex = a5
checksum_byte_hex = 5e
payload_plus_checksum_hex = a55e
audit_key_id = KWP4_QWEN_PILOT_001
```

Source contract:

```text
results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/wp4_prompt_local_payload_contract.json
sha256 = 69d1feb2b63f52db7cf1ca82bb9ccfcbeb056f2f4f5945b230fc8c44923ada07
```

Rationale:

The current protected/task-only LoRA artifacts were trained for this one
payload cell. Additional payload cells would require a separate WP4/WP5 contract
and training gate, so they are out of scope for this reproducibility run.

## Query Budgets

Use query budgets within each 64-prompt replicate block:

```text
query_budgets_per_block = [8, 16, 32, 64]
```

Do not report `128` or `256` as owner query budgets from this run. A cumulative
256-prompt diagnostic may be written as non-controlling metadata only if the
wrapper clearly labels it as diagnostic and not part of the proof-of-life gate.

## Null Controls

Required decode conditions:

```text
protected
raw
task_only
wrong_key
wrong_payload
```

Null definitions:

```text
raw = base Qwen generation decoded under the committed contract
task_only = task-only LoRA generation decoded under the committed contract
wrong_key = protected generation decoded with KWP4_QWEN_PILOT_WRONG_001
wrong_payload = protected generation decoded with payload byte 5a
```

The wrapper must also preserve exact-frame decoder outputs as diagnostic
context, but exact-frame failure does not control the WP6-R1 scale gate.

## Scale Gate

Controlling gate at budget `64`, evaluated over four independent blocks:

| Gate | Requirement |
|---|---:|
| protected block accepts | `>= 3 / 4` |
| preferred protected block accepts | `4 / 4` |
| raw accepts | `0 / 4` |
| task-only accepts | `0 / 4` |
| wrong-key accepts | `0 / 4` |
| wrong-payload accepts | `0 / 4` |
| min support in accepted protected blocks | `>= 16` |
| min majority margin in accepted protected blocks | `>= 3` |
| forbidden public surface count | `0` |
| output artifacts complete | required |

Budget `32` is informative but not controlling. The result should report the
full curve for `[8, 16, 32, 64]` within each block.

## Output Requirements

The scale run must write:

```text
precommit/wp6_r1_scale_contract.json
wp6_generation_summary.json
wp6_generated_outputs.jsonl
wp6_e2e_summary.json
wp6_slot_observations.jsonl
wp6_decode_decisions.jsonl
coordinate_majority_scale/wp6_r1_scale_decode_rows.jsonl
coordinate_majority_scale/wp6_r1_scale_summary.json
coordinate_majority_scale/wp6_r1_scale_support_by_block_budget.csv
coordinate_majority_scale/wp6_r1_scale_contract.json
slurm logs
```

The scale summary must use the cleaned metadata vocabulary:

```text
precommitted_transcript = true
post_hoc_artifact_replay = false
transcript_provenance = precommitted_replacement_run
```

It must not emit:

```text
post_hoc_not_precommitted_for_852086
```

## Allowlist

Planned allowlist entry:

```text
name = v2_wp6_r1_coordinate_majority_scale_eval
enabled = false until wrapper review passes
command_pattern = sbatch scripts/natural_evidence_v2/slurm/wp6_r1_coordinate_majority_scale_eval.sbatch
claim_note = WP6-R1 Qwen V2 repeated-coordinate majority reproducibility scale eval only; no training, no FAR, no Llama, no paper positive claim.
```

After wrapper review passes, automation may enable this entry for exactly one
submission and disable it immediately after submission.

## Wrapper Review Requirements

Before any Slurm submission, Codex/Hermes must verify:

```text
scale wrapper exists
wrapper writes precommit contract before generation
wrapper fixes prompt slice to first 256 selected wp3_r1_eval prompts
wrapper uses MAX_PROMPTS=256
wrapper evaluates four non-overlapping 64-query blocks
wrapper preserves null controls
wrapper refuses overwrite
wrapper uses Chimera Slurm only
wrapper local plan-only validation passes
pytest for decoder/windowing passes
bash -n passes
allowlist entry exists and is enabled for one submission only
```

## Next Automation Action

Because the route is now explicitly approved, Hermes should not ask the user
for another approval before the next implementation step. The next Codex action
is:

```text
Implement and locally validate the WP6-R1 scale wrapper and block-window
majority decoder. Do not submit Slurm until wrapper review and allowlist update
are recorded.
```

## Still Forbidden

- no automatic scaled Slurm submission from this package alone;
- no new training;
- no Llama;
- no same-family null;
- no sanitizer;
- no FAR aggregation;
- no paper-facing positive claim;
- no multi-payload claim.
