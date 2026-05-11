# Hermes natural_evidence_v1 Codex worker completed

phase:
V2_R3_2C_JOB_853070_FAILED_PROMPT_SPLIT_MISMATCH_NO_RESUBMIT

codex_report:
`results/natural_evidence_v1/status/hermes_reports/20260511_1801_scheduled_tick_codex_report.md`

summary:
```text
Implemented the R3.2 prompt split repair artifact-only.

Changed:
- `scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py`
- `scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch`
- `configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale.yaml`
- Recorded state in `docs/natural_evidence_v2/R3_2_PROMPT_SPLIT_IMPLEMENTATION_20260511_1801.md`, `CURRENT_STATE.md`, and both v1/v2 gate status files.

Result:
- R3.2 now uses `wp3_r1_eval` only.
- Shard windows are `512..1023`, `1024..1535`, `1536..2047`, `2048..2559`, reused across 12 shards.
- New manifest hash: `3e50a08773c4c7dca3be976a762840a8d8a960ac63f4cfce382af3051a2b82d1`.

Validated:
- `bash -n` wrapper passed.
- `py_compile` precommit builder passed.
- Local plan-only precommit passed at `results/natural_evidence_v2/status/r3_2_prompt_split_repair_precommit_20260511_1801`.

No Slurm submission, generation, training, Llama, nulls, sanitizer, FAR, or paper claim work was started. Next allowed action is artifact-only `852426` replay compatibility re-review or explicit supersession.
```

next_allowed_action:
Repair the R3.2 prompt allocation and wrapper split contract artifact-only. Do not submit another R3.2 Slurm job until the repaired allocation is recorded, plan-only preflight is rerun, 852426 replay compatibility is re-reviewed or superseded, allowlist safety is rechecked, and a new single-job submission route is recorded.

gate_controlled_actions_not_yet_unlocked:
Gate-controlled and not yet unlocked: training; Llama; same-family null; sanitizer benchmark; FAR aggregation; paper-facing positive claims. Hard-blocked unless explicitly allowed by the current next_allowed_action: generation and Qwen E2E reruns.
