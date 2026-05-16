# R4 controller-only safety-bound H200 job submitted

phase:
V2_R4_CONTROLLER_ONLY_SAFETY_BOUND_H200_JOB_864117_RUNNING

summary:
```text
Submitted exactly one reviewed H200/pomplun teacher-forced scoring array.
job_id: 864117
array: 0-23%4
route_config: configs/natural_evidence_v2/r4_controller_only_safety_bound_pressure_route.yaml
condition_set: controller_only_controls
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gpu: h200
time_limit: 30-00:00:00

Post-submit allowlist safety passed locally and remotely with zero enabled entries.
No generation/training/Llama/null/sanitizer/FAR/paper claim was started.
```

next_allowed_action:
Monitor job 864117. After terminal completion, sync and review safety-bound controller scoring summary artifacts before any downstream action.
