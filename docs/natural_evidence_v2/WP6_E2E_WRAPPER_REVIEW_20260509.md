# WP6 E2E Wrapper Review: 2026-05-09

## Decision

WP6 Qwen V2 proof-of-life is authorized for one Chimera Slurm submission after
user approval on 2026-05-09.

This authorization is limited to:

- Qwen protected WP5-R2 adapter;
- Qwen raw base model;
- Qwen task-only WP5-R2 adapter;
- wrong-key decode over protected outputs;
- wrong-payload decode over protected outputs;
- query budgets `[8,16,32,64]`;
- one allowlisted `sbatch scripts/natural_evidence_v2/slurm/wp6_e2e_eval.sbatch` job.

It does not authorize Llama, same-family nulls, sanitizer benchmarks, FAR
aggregation, new training, or paper-facing positive claims.

## Contract Alignment

WP6 uses the WP4 prompt-local payload contract that WP5-R2 actually trained
against:

```text
results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/wp4_prompt_local_payload_contract.json
```

The controlling payload+checksum bits are:

```text
a55e = [1,0,1,0,0,1,0,1,0,1,0,1,1,1,1,0]
```

The older P00/P01 seeded oracle contracts under
`results/natural_evidence_v2/contracts/wp4_prompt_local_contract_20260509_0610/`
are not used for this WP6 run because they were not the target schedule learned
by the WP5-R2 adapters.

## New Wrapper Artifacts

```text
scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py
scripts/natural_evidence_v2/decode_wp6_payload.py
scripts/natural_evidence_v2/slurm/wp6_e2e_eval.sbatch
tests/test_natural_evidence_v2_wp6_e2e_decode.py
```

Generation conditions:

```text
protected
raw
task_only
```

Decode conditions:

```text
protected
raw
task_only
wrong_key
wrong_payload
```

Wrong-key and wrong-payload are decoded from the protected transcript, so the
job does not waste GPU time generating duplicate protected responses.

## Local Validation

Commands passed:

```text
.venv/bin/python -m pytest \
  tests/test_natural_evidence_v2_wp6_e2e_decode.py \
  tests/test_natural_evidence_v2_wp5_launch_plan.py \
  tests/test_natural_evidence_v2_restricted_density.py
```

Result:

```text
11 passed
```

Additional checks passed:

```text
.venv/bin/python -m py_compile \
  scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py \
  scripts/natural_evidence_v2/decode_wp6_payload.py

bash -n scripts/natural_evidence_v2/slurm/wp6_e2e_eval.sbatch
```

Plan-only validation output:

```text
results/natural_evidence_v2/status/wp6_e2e_local_plan_validation_20260509_1710/wp6_generation_plan_summary.json
```

The validation selected 64 `wp3_r1_eval` strict Step-label prompts and confirmed
the WP5-aligned `a55e` 16-bit contract.

## Allowlist

The `v2_wp6_e2e_eval` allowlist entry is enabled only for this reviewed single
submission:

```text
command = sbatch scripts/natural_evidence_v2/slurm/wp6_e2e_eval.sbatch
```

After submission, the entry should be disabled again with the submitted job id
recorded in state.

## Review Status

```text
PASS_READY_TO_SUBMIT_ONE_ALLOWLISTED_WP6_SLURM_JOB
```
