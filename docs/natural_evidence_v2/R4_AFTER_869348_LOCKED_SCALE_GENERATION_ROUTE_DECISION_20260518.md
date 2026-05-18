# R4 After-869348 Locked-Scale Generation Route Decision

Date: 2026-05-18

## Decision

The `869348` 32-block dev diagnostic passed, and `870078` validated actual Qwen
tokenizer boundaries for the held-out locked row bank. The next canonical route
is a 96-block Qwen-only, same-contract `a55e`, first-token-event locked-scale
generation diagnostic.

This is not a paper-facing positive claim by itself. It does not test payload
diversity, Llama transfer, sanitizer robustness, full FAR, or text-only phrase
decoding success.

## Scope

```text
row bank:
  results/natural_evidence_v2/status/r4_after_869348_global_unique_locked_scale_row_bank_plan_20260518/
split:
  locked
blocks/shards:
  96
row cylinders per block:
  1024
expected generated rows:
  294912
conditions generated:
  protected, raw, task_only
decode controls:
  protected, raw, task_only, wrong_key, wrong_payload
contract:
  a55e
payload diversity:
  not tested
```

## Gate

```text
protected strict accepts >= 85/96
protected accepts ignoring quality >= 90/96
raw accepts = 0/96
task-only accepts = 0/96
wrong-key accepts = 0/96
wrong-payload accepts = 0/96
within-block duplicate response hash count = 0
global duplicate response hash count = 0
technical forbidden public surface count = 0
ambiguous forbidden surface count = 0
trace binding validity = 100%
full phrase decoder = report-only, not a success claim
```

## Submission Rule

Only one H200 array job may be submitted after local and remote preflights pass:

```text
allowlist entry:
  v2_r4_after_869348_locked_scale_generation_h200
wrapper:
  scripts/natural_evidence_v2/slurm/r4_after_869348_locked_scale_generation_h200.sbatch
command:
  PLAN_ONLY=0 VALIDATE_PLAN_ONLY=0 sbatch scripts/natural_evidence_v2/slurm/r4_after_869348_locked_scale_generation_h200.sbatch
partition/qos/account/gres:
  pomplun / pomplun / cs_yinxin.wan / gpu:h200:1
array:
  0-95%4
time:
  30-00:00:00
```

The allowlist entry must be disabled immediately after `sbatch` returns.

## Not Unlocked

This route does not unlock training, Llama, same-family null, sanitizer, FAR,
payload-diversity claims, text-only phrase-decoder claims, or paper-facing
positive claims.
