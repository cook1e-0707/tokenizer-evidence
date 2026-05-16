# R4 After-868151 Quality-Repaired Generation Submission Route

Date: 2026-05-16

## Decision

The quality-repaired first-token event generation route is reviewed for one
H200/pomplun Slurm submission.

This route remains Qwen-only, same-contract `a55e`, and diagnostic only. It is
not a paper-facing positive claim and it does not unlock training, Llama,
same-family null, sanitizer, FAR aggregation, or payload diversity.

## Preconditions

```text
quality repair plan:
  PASS_R4_AFTER_868151_FIRST_TOKEN_EVENT_QUALITY_REPAIR_PLAN_ARTIFACT_ONLY
allocation manifest:
  PASS_DUPLICATE_SAFE_ROW_ALLOCATION_ARTIFACT_ONLY
local route validation:
  PASS_R4_AFTER_868016_CONTROLLER_GENERATION_ROUTE_VALIDATION_NO_SUBMIT
local wrapper plan-only:
  PASS_R4_AFTER_868016_CONTROLLER_GENERATION_WRAPPER_PLAN_ONLY
remote route validation:
  PASS_R4_AFTER_868016_CONTROLLER_GENERATION_ROUTE_VALIDATION_NO_SUBMIT
remote wrapper plan-only:
  PASS_R4_AFTER_868016_CONTROLLER_GENERATION_WRAPPER_PLAN_ONLY
local/remote allowlist safety:
  PASS zero-enabled
active Chimera jobs:
  none observed before route record
```

## Allowed Action

Exactly one submission is allowed:

```text
allowlist entry:
  v2_r4_after_868016_controller_generation_h200
command:
  sbatch scripts/natural_evidence_v2/slurm/r4_after_868016_controller_generation_h200.sbatch
partition/qos/account:
  pomplun / pomplun / cs_yinxin.wan
gres:
  gpu:h200:1
time:
  30-00:00:00
array:
  0-3%4
```

The allowlist entry must be disabled immediately after `sbatch` returns.

## Required Review After Completion

The resulting job must be reviewed before any further route:

```text
primary decoder:
  first-token event decoder with token-id traces
quality gates:
  contextual technical public literal count = 0
  duplicate response hash count = 0
controls:
  raw/task-only/wrong-key/wrong-payload accept counts = 0
status:
  diagnostic only until review passes
```
