# R4 After-869348 Locked-Scale Tokenizer Preflight Route Decision

Date: 2026-05-18

## Decision

The `869348` Qwen first-token-event 32-block dev diagnostic is adopted as a
reviewed dev diagnostic pass, not as a paper-facing positive claim. The next
canonical route is the held-out locked-scale tokenizer boundary preflight for
the same first-token-event protocol.

This route is tokenizer-only. It does not run model forward passes, model
scoring, generation, training, Llama, same-family null, sanitizer, FAR,
payload-diversity, or paper-facing claim jobs.

## Reviewed Inputs

```text
source dev diagnostic:
  869348, PASS_R4_AFTER_868348_GLOBAL_UNIQUE_DEV_DIAGNOSTIC_GATE
protected strict accepts:
  32/32
protected accepts ignoring quality:
  32/32
controls:
  raw=0/32, task_only=0/32, wrong_key=0/32, wrong_payload=0/32
global duplicate response hashes:
  0
protected forbidden public surface count:
  0
trace binding:
  98304 checked, 0 invalid
full phrase decoder:
  report-only, 0 protected accepts under format_scrub=all
```

## Locked Row Bank

```text
row bank:
  results/natural_evidence_v2/status/r4_after_869348_global_unique_locked_scale_row_bank_plan_20260518/
split:
  locked
contract:
  a55e
payload diversity:
  not tested
shards:
  96
prompts per shard:
  64
row cylinders:
  98304
selected coordinates:
  16
unique content prompt/prefix pairs:
  98304
duplicate content prompt/prefix extra rows:
  0
```

The locked row bank route validation passed and the local static boundary
preflight passed with `98304` checked rows and `0` failed rows.

## Next Allowed Action

Submit exactly one H200 tokenizer-only Slurm job after local/remote hash
preflight and exactly-one allowlist enablement:

```text
allowlist entry:
  v2_r4_after_869348_locked_scale_qwen_tokenizer_boundary_preflight_h200
wrapper:
  scripts/natural_evidence_v2/slurm/r4_after_869348_locked_scale_qwen_tokenizer_boundary_preflight_h200.sbatch
command:
  sbatch scripts/natural_evidence_v2/slurm/r4_after_869348_locked_scale_qwen_tokenizer_boundary_preflight_h200.sbatch
partition/qos/account/gres:
  pomplun / pomplun / cs_yinxin.wan / gpu:h200:1
time:
  30-00:00:00
```

The allowlist entry must be disabled immediately after `sbatch` returns.

## Gate

The tokenizer preflight passes only if all hold:

```text
checked rows = 98304
failed rows = 0
empty target id rows = 0
empty other id rows = 0
target/other first-token overlap rows = 0
model forward started = false
scoring started = false
generation started = false
training started = false
```

If this tokenizer preflight fails, no locked-scale generation route may be
submitted. The candidate row bank or tokenizer boundary contract must be
repaired first.

## Still Not Claimed

This route does not establish text-only phrase decoding, full FAR, sanitizer
robustness, payload diversity, Llama transfer, cross-family generality, or a
paper-facing positive result.
