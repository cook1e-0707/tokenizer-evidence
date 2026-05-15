# Hermes Sync: R4 Positive Selectivity Job Submitted

phase:

```text
V2_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_JOB_859491_SUBMITTED_MONITORING
```

job:

```text
859491
```

summary:

```text
Submitted exactly one H200/pomplun Slurm array job using allowlist entry
v2_r4_positive_selectivity_dev_diagnostic_h200. The allowlist entry was disabled
immediately after sbatch returned. Local and remote post-submit allowlist safety
both passed with zero enabled entries.

Command:
sbatch --export=ALL,ALLOW_STATIC_DEV_KEYS=1 scripts/natural_evidence_v2/slurm/r4_positive_selectivity_dev_diagnostic_h200.sbatch

Initial state:
- parent/array job 859491 seen as PENDING(Resources)
- array tasks 859492, 859493, 859494 seen RUNNING on chimera21
- expected output dir:
  /hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_positive_selectivity_dev_diagnostic_859491
```

next_allowed_action:

```text
Monitor job 859491. Do not submit another R4 selectivity dev diagnostic job
unless this one is reviewed and a new route decision is recorded.
```

