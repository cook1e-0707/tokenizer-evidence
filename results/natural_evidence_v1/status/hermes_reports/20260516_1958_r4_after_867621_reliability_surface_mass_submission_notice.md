# Hermes Submission Notice: R4 After 867621 Reliability Surface-Mass Scoring

phase:
`V2_R4_AFTER_867621_RELIABILITY_SURFACE_MASS_ROUTE_VALIDATED_NO_SUBMIT`

remote preflight:
`PASS_R4_AFTER_867621_RELIABILITY_SURFACE_MASS_REMOTE_PREFLIGHT_NO_SUBMIT`

planned action:

```text
Enable exactly one allowlist entry:
v2_r4_after_867621_reliability_surface_mass_score_h200

Submit exactly one H200/pomplun teacher-forced surface-mass scoring Slurm job:
scripts/natural_evidence_v2/slurm/r4_after_867621_reliability_surface_mass_score_h200.sbatch

Immediately disable the allowlist entry after sbatch returns.
```

scope:

```text
teacher-forced surface-mass scoring only
conditions: base, protected, task_only
rows: 4096
generation: false
training: false
Llama/same-family/sanitizer/FAR/paper claim: false
```

preflight facts:

```text
local/remote hashes match: true
remote route validation: PASS
remote wrapper plan-only smoke: DRY_RUN_VALIDATED_INPUTS
remote zero-enabled allowlist safety: PASS
active Chimera jobs before submission: none
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
time: 30-00:00:00
```
