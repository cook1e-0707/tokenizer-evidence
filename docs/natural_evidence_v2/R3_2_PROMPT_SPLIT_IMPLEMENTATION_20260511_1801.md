# R3.2 Prompt Split Implementation: 2026-05-11 18:01Z

## Scope

Implemented the recorded R3.2 prompt split contract repair artifact-only.

No Slurm job was submitted. No generation, Qwen E2E rerun, training, Llama,
same-family null, sanitizer benchmark, FAR aggregation, or paper-facing
positive claim was started.

## Code Paths Updated

- `scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py`
- `scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch`
- `configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale.yaml`

## Repaired Contract Now Enforced

```text
selected_split = wp3_r1_eval
prompt_window_policy = deterministic_4_eval_window_circular_reuse_by_replicate_group_index
eval_prompt_file_rows = 512..2559
eval_prompt_count = 2048
selected_prompt_manifest_sha256 = 3e50a08773c4c7dca3be976a762840a8d8a960ac63f4cfce382af3051a2b82d1
```

Shard assignment:

```text
shard_00, shard_04, shard_08 -> rows 512..1023
shard_01, shard_05, shard_09 -> rows 1024..1535
shard_02, shard_06, shard_10 -> rows 1536..2047
shard_03, shard_07, shard_11 -> rows 2048..2559
```

The wrapper now passes `--split wp3_r1_eval` to shard precommit decode,
generation, and post-generation shard decode calls.

## Validation

```text
bash -n scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch: PASS
python3 -m py_compile scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py: PASS
python3 scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py --output-dir results/natural_evidence_v2/status/r3_2_prompt_split_repair_precommit_20260511_1801: PASS
```

Plan-only precommit artifact:

```text
results/natural_evidence_v2/status/r3_2_prompt_split_repair_precommit_20260511_1801/precommit/r3_2_selected_prompt_manifest.json
```

## Still Blocked

R3.2 is not ready for another Slurm submission. Before any new R3.2 Slurm job,
`852426` replay compatibility must be re-reviewed or explicitly superseded,
allowlist safety must be rechecked, and a new single-job submission route must
be recorded.

## Status

```text
IMPLEMENTED_R3_2_PROMPT_SPLIT_REPAIR_PREFLIGHT_PASS_NO_SLURM_NO_GENERATION
```
