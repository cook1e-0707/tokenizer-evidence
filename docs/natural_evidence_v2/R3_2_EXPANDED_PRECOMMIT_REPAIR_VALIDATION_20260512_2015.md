# R3.2 Expanded Precommit Repair Validation

Date: 2026-05-12 20:15Z

## Scope

Artifact-only precommit-builder repair and local plan-only validation for the
expanded 6,144-row R3.2 route. No Slurm submission, model generation, training,
Llama, same-family null, sanitizer benchmark, FAR aggregation, or paper-facing
claim was started.

## Repair

Updated:

`scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py`

The precommit builder now supports the reviewed expanded prompt allocation
contract:

- split: `wp3_r1_density_eval`;
- prompt policy: `distinct_eval_window_by_shard_index`;
- prompt source:
  `results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_r3_2_6144_20260512/restricted_step_label_r1_eval_prompts.jsonl`;
- 12 shard windows without modulo reuse;
- selected-index metadata in the expanded manifest;
- expected prompt source hash from config;
- expected selected prompt manifest hash from config when present.

The old four-window R3.2 policy remains accepted for compatibility with
existing review/replay utilities.

## Local Validation

Command:

```bash
python3 scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py \
  --prompts-jsonl results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_r3_2_6144_20260512/restricted_step_label_r1_eval_prompts.jsonl \
  --wp4-contract results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/wp4_prompt_local_payload_contract.json \
  --config-yaml configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale_expanded_6144.yaml \
  --output-dir results/natural_evidence_v2/status/r3_2_expanded_6144_precommit_local_validation_20260512_2015
```

Output:

`results/natural_evidence_v2/status/r3_2_expanded_6144_precommit_local_validation_20260512_2015/`

Result:

- selected prompt manifest SHA256:
  `ce057a36ad75424919f4367eb3e2f0221725a9c6715d156ab4b2a377edb600ed`;
- prompt rows: `6,144`;
- selected prompt rows: `6,144`;
- unique selected window hashes: `12`;
- unique block hashes: `96`;
- first shard prompt row start: `0`;
- last shard prompt row start: `5632`;
- status: `PASS_R3_2_EXPANDED_PRECOMMIT_LOCAL_PLAN_VALIDATION_NO_SLURM`.

## Remaining Blockers

Slurm remains blocked. The next allowed action is wrapper repair/review and
allowlist/local-remote hash safety review only. Do not submit R3.2, aggregate,
rerun Qwen E2E, start Llama/FAR/sanitizer work, train, or make paper-facing
positive claims from this state.
