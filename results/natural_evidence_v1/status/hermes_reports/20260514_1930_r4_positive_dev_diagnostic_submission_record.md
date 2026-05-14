# R4 Positive Dev Diagnostic Submission Record

phase:
`V2_R4_POSITIVE_DEV_DIAGNOSTIC_SUBMITTED_MONITOR_JOB_859277`

summary:
```text
Submitted exactly one reviewed H200/pomplun Slurm array job for the R4
positive event-bank dev diagnostic.

Pre-submit controls:
- route doc recorded
- Hermes TG/email pre-submit notification sent
- local zero-enabled allowlist safety PASS
- remote zero-enabled allowlist safety PASS
- local/remote hash preflight PASS
- active-job preflight PASS
- exactly-one-enabled preflight PASS locally and remotely

Submission:
- allowlist entry enabled: v2_r4_positive_dev_diagnostic_h200
- command: sbatch --export=ALL,ALLOW_STATIC_DEV_KEYS=1 scripts/natural_evidence_v2/slurm/r4_positive_dev_diagnostic_h200.sbatch
- job id: 859277
- job name: nat-ev-v2-r4posdev
- array: 0-3%4
- partition/qos/account: pomplun / pomplun / cs_yinxin.wan
- node observed after submission: chimera21 for running tasks
- output dir: /hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_positive_event_bank_dev_diagnostic_859277

Post-submit controls:
- allowlist disabled immediately after sbatch returned
- local post-submit allowlist safety PASS, enabled entries []
- remote post-submit allowlist safety PASS, enabled entries []

No training/Llama/same-family null/sanitizer/FAR/payload-diversity/paper claim
action was started.
```

next_allowed_action:
Monitor job `859277` only. After all array tasks reach terminal state, sync and
review generated outputs plus keyed decode artifacts. Do not submit another R4
positive dev diagnostic job until `859277` is terminal and reviewed.

