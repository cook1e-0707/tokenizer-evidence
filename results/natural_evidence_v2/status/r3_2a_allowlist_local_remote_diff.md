# R3.2-A Allowlist Local/Remote Diff

Timestamp UTC: `2026-05-11T14:45Z`

## Status

`PASS_LOCAL_REMOTE_ALLOWLIST_CONSISTENT`

No Slurm job was submitted. No training, generation, Llama, sanitizer, FAR, or
paper-claim work was started.

## Local Hashes

```text
20b589d09a78a6df234d90b67790af80bde8c463fb7fc84423800e3ed403208f  configs/natural_evidence_v2/run_allowlist.yaml
d9803e9a6ce9c4d65cdca19503383134aea281a0ba85850b6b8aafae63d69802  docs/natural_evidence_v2/CURRENT_STATE.md
eb98d26196398b468914a01dcf3273686487358ee47c16a7079a32f2f9a536e1  results/natural_evidence_v2/status/gate_status.json
8d6319c3096513fb5778cd55c345edc48f1d555bdfa4c0d30282e50af44886bb  scripts/natural_evidence_v2/check_allowlist_safety.py
```

## Remote Hashes

Remote root:

```text
/home/guanjie.lin001/tokenizer-evidence
```

```text
20b589d09a78a6df234d90b67790af80bde8c463fb7fc84423800e3ed403208f  configs/natural_evidence_v2/run_allowlist.yaml
d9803e9a6ce9c4d65cdca19503383134aea281a0ba85850b6b8aafae63d69802  docs/natural_evidence_v2/CURRENT_STATE.md
eb98d26196398b468914a01dcf3273686487358ee47c16a7079a32f2f9a536e1  results/natural_evidence_v2/status/gate_status.json
8d6319c3096513fb5778cd55c345edc48f1d555bdfa4c0d30282e50af44886bb  scripts/natural_evidence_v2/check_allowlist_safety.py
```

## Enabled Entry Check

Remote grep for `enabled: true` in `run_allowlist.yaml` returned no rows.
The local safety checker also reported:

```text
enabled_entries = []
forbidden_enabled_entries = []
r3_2_entry_disabled = true
```

## Next State

R3.2-A decontamination passed. The next route may be
`V2_R3_2B_QWEN_LOCKED_SCALE_SINGLE_JOB_SUBMISSION_READY`: a later notified tick
may enable exactly `v2_r3_2_qwen_locked_scale_eval`, submit exactly one reviewed
Chimera Slurm job, and immediately disable the entry after `sbatch` returns.

