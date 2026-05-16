# R4 After-864832 Coordinate-Unique Reliability Dev-Generation Route

Date: 2026-05-16

## Decision

The prior reliability dev-generation route is no longer blocked by surface
coordinate ambiguity. Codex built a coordinate-identifiable surface bank for the
selected reliability-codebook coordinates and revalidated the route without
starting Slurm, model scoring, generation, or training.

This route is still not a paper-facing result. It is a 32-block Qwen dev
generation diagnostic under the same `a55e` contract, using the
reliability-weighted codebook and `format_scrub=all` as the primary decode path.

## Repaired Artifacts

```text
surface bank:
  results/natural_evidence_v2/precommit/r4_after_864832_coordinate_unique_surface_bank_20260516/surface_bank.json
surface bank sha256:
  4a0f07af15ade41d51655352976e18d17e095b60a4850814ade231ec9f5fe1ac
surface bank status:
  PASS_COORDINATE_UNIQUE_SURFACE_BANK_BUILT_ARTIFACT_ONLY

surface uniqueness audit:
  results/natural_evidence_v2/status/r4_after_864832_coordinate_unique_surface_uniqueness_audit_20260516/
surface uniqueness status:
  PASS_R4_RELIABILITY_SURFACE_UNIQUENESS

route config:
  configs/natural_evidence_v2/r4_after_864832_reliability_dev_generation_route.yaml
route validation:
  results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_route_validation_unique_v2_20260516/
route validation status:
  PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_ROUTE_VALIDATION_NO_SUBMIT

wrapper:
  scripts/natural_evidence_v2/slurm/r4_after_864832_reliability_dev_generation_h200.sbatch
wrapper plan-only smoke:
  results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_wrapper_plan_smoke_unique_20260516/
wrapper plan-only status:
  PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_WRAPPER_PLAN_ONLY
```

## Scope

```text
model family: Qwen only
contract: a55e
payload diversity tested: false
blocks: 32
prompts per block: 64
array shards: 4
conditions generated: protected, raw, task_only
decode controls: protected, raw, task_only, wrong_key, wrong_payload
primary scrub mode: all
```

Compute policy:

```text
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
max time: 30-00:00:00
allowlist entry: v2_r4_after_864832_reliability_dev_generation_h200
```

## Required Before Submission

```text
remote file-hash preflight: pending
remote route validation: pending
remote zero-enabled allowlist safety: pending
remote wrapper plan-only smoke: pending
Hermes notification: pending
exactly-one allowlist enablement: pending
immediate allowlist disablement after sbatch: required
```

Do not submit Slurm until all remote preflight checks pass.

## Dev Gate After Completion

```text
protected accepts, format_scrub=all: >= 26/32
raw accepts: 0/32
task-only accepts: 0/32
wrong-key accepts: 0/32
wrong-payload accepts: 0/32
forbidden public surface count: 0
duplicate response hashes: 0
duplicate decode-row hashes: 0
```

## Not Unlocked

This route does not unlock:

```text
training
Llama
same-family null
sanitizer
FAR
payload diversity claim
paper-facing positive claim
```

These are conditionally authorized only after their own recorded prerequisite
gates pass.
