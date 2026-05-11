# R3.2-A Allowlist Decontamination Pass

Timestamp UTC: `2026-05-11T14:45Z`

## Current Phase

`V2_R3_2B_QWEN_LOCKED_SCALE_SINGLE_JOB_SUBMISSION_READY`

## Action Executed

R3.2-A control-plane decontamination was completed. This was an artifact and
configuration safety action only.

## Files Updated

```text
configs/natural_evidence_v2/run_allowlist.yaml
scripts/natural_evidence_v2/check_allowlist_safety.py
docs/natural_evidence_v2/CURRENT_STATE.md
results/natural_evidence_v2/status/r3_2a_allowlist_decontamination_summary.json
results/natural_evidence_v2/status/r3_2a_allowlist_local_remote_diff.md
results/natural_evidence_v1/status/gate_status.json
results/natural_evidence_v2/status/gate_status.json
```

The local allowlist was synchronized to:

```text
/home/guanjie.lin001/tokenizer-evidence/configs/natural_evidence_v2/run_allowlist.yaml
```

## Safety Results

```text
enabled_entries = []
forbidden_enabled_entries = []
llama_entries_disabled = true
same_family_entries_disabled = true
sanitizer_entries_disabled = true
far_entries_disabled = true
paper_claim_entries_disabled = true
training_entries_disabled = true
r3_2_entry_disabled = true
local_remote_hashes_match = true
```

Remote grep for `enabled: true` returned no rows.

## No Slurm

No Slurm job was submitted. No generation, training, Llama, same-family null,
sanitizer, FAR aggregation, paper claim, or Chimera login-node CPU/GPU work was
started.

## Next Allowed Action

R3.2-B submission preflight may proceed in a later notified tick:

1. enable exactly `v2_r3_2_qwen_locked_scale_eval`;
2. verify it is the only enabled entry;
3. verify same-contract `a55e` preflight and `852426` replay gates still pass;
4. submit exactly one reviewed Chimera Slurm job;
5. immediately disable the entry after `sbatch` returns;
6. write `r3_2b_submission_record.json`.

