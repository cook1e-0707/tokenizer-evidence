# WP6-R2 Option B Wrapper and Contract Plan: 2026-05-10

## Decision

The accepted WP6-R2 repair is Option B from the reviewed `852202` failure
diagnosis: keep budget `64`, keep majority margin `3`, increase independent
blocks to `8`, and count robust block accepts.

This is an artifact-only wrapper/contract planning record. It does not submit
Slurm, start generation, rerun Qwen E2E, train, start Llama or same-family
nulls, run a sanitizer, aggregate FAR, or make a paper-facing positive claim.
Job `852202` remains a failed precommitted WP6-R1 scale run.

## Inputs Reviewed

```text
docs/natural_evidence_v1/AUTOMATION_STATE.md
docs/natural_evidence_v1/next_step_codex_plan.md
results/natural_evidence_v1/status/gate_status.json
docs/natural_evidence_v2/PROTOCOL_CONTRACT.md
docs/natural_evidence_v2/CLAIM_GUARDRAILS.md
results/natural_evidence_v2/status/gate_status.json
docs/specs/stage4_real_integration_spec.md
docs/natural_evidence_v2/WP6_R1_SCALE_FAILURE_DIAGNOSIS_AND_REPAIR_PLAN_852202.md
docs/natural_evidence_v2/WP6_R1_SCALE_WRAPPER_REVIEW_20260509.md
```

## Planned R2 Contract

```text
protocol_id = natural_evidence_v2_wp6_r2_option_b_robust_block_scale
decoder_id = qwen_v2_wp6_r2_robust_block_coordinate_majority_decoder_v1
payload_plus_checksum_hex = a55e
payload_byte_hex = a5
checksum_byte_hex = 5e
route = same Qwen WP6-R1 route, no new training
block_count = 8
block_size = 64
prompt_count = 512
query_budgets_per_block = [8,16,32,64]
controlling_budget = 64
min_support_at_64 = 16
min_majority_margin_at_64 = 3
```

Robust block accept at budget `64` is fixed as:

```text
decoded_hex == a55e
payload_matches == true
checksum_valid == true
min_support >= 16
min_majority_margin >= 3
```

Planned R2 gate:

```text
protected robust block accepts >= 6 / 8
raw robust accepts = 0 / 8
task-only robust accepts = 0 / 8
wrong-key robust accepts = 0 / 8
wrong-payload robust accepts = 0 / 8
forbidden_public_surface_count = 0
required artifacts complete
```

## Fresh Prompt Window

The previous scale job `852202` used `wp3_r1_eval` file rows `512..767` from:

```text
results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/restricted_step_label_strict_density_audit_prompts.jsonl
```

The planned R2 prompt window is the next disjoint 512-row eval slice:

```text
selected_prompt_file_rows = 768..1279
selected_prompt_jsonl_sha256 = d3966ce5c43347df9c68dc6cd6118102fb0708484ddd53e9b08b7b42b1f12ddd
```

Planned blocks:

| block | file rows | row-jsonl sha256 |
|---|---:|---|
| block_0 | 768..831 | 67e869d94e33b659cc00ef0e10f50ddf7eb4e30e7a8e47bcaa156ec3227ff066 |
| block_1 | 832..895 | a911b65c16b5532e1b938d8f8a445b1030303f65d50659216d30adbba373270f |
| block_2 | 896..959 | 71d0fe09385b063a7386d474a4e483e99344341376e826abbdf249d5a2a3d4e5 |
| block_3 | 960..1023 | 7300594c6b469fd700c3a5830e01e6228bb9b8c4856b590907b7cdc5e53be045 |
| block_4 | 1024..1087 | d204650a48e8fdaf2c328114bd28ab74acfb1a3b36510117efdfe96a68be8111 |
| block_5 | 1088..1151 | 4659fc0ae69a3da744bb32ea7f0dd876868a18f38c883b4092c89e9d71eb8d8f |
| block_6 | 1152..1215 | f067ea7a1e8cdf709ab7141470ad96d9a9e7b667aef618d9d6bef2eada17b64e |
| block_7 | 1216..1279 | fb539266d645a9c25602f15d061886636cc58227fcdbf0be629985f104c9d2c4 |

## Wrapper Requirements

The current WP6-R1 scale wrapper cannot simply be reused with
`MAX_PROMPTS=512`: its decoder selects the first eval rows and would overlap
`852202`. The R2 wrapper review must therefore explicitly validate support for
the selected eval offset and row range above.

Required wrapper behavior:

- write the R2 precommit contract before any generation;
- fix `MAX_PROMPTS=512`, `BLOCK_COUNT=8`, `BLOCK_SIZE=64`, and selected file
  rows `768..1279`;
- refuse overwrite if a generation artifact, decoder summary, or precommit
  contract already exists in the output directory;
- run plan-only local validation with `generation_started=false`;
- preserve protected/raw/task-only/wrong-key/wrong-payload decode conditions;
- keep the allowlist disabled until a later notified submission tick explicitly
  permits one reviewed allowlisted Slurm submission.

## Status

```text
WP6_R2_OPTION_B_WRAPPER_CONTRACT_PLAN_RECORDED_NO_SLURM
```

Next allowed action: implement and locally validate the WP6-R2 Option B wrapper
and contract-only/plan-only path, then record a wrapper review. Do not submit
Slurm or start generation until a later notified tick permits one reviewed
allowlisted submission and disables the allowlist immediately afterward.

