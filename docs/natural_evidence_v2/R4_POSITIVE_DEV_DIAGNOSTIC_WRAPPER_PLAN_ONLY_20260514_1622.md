# R4 Positive Dev Diagnostic Wrapper Plan-Only Review

Timestamp: `2026-05-14T16:22:00Z`

## Scope

Added a fail-closed H200/pomplun wrapper for the future R4 positive event-bank
dev diagnostic:

```text
scripts/natural_evidence_v2/slurm/r4_positive_dev_diagnostic_h200.sbatch
```

This wrapper currently supports only `VALIDATE_PLAN_ONLY=1`. Non-plan full mode
exits with:

```text
FULL_R4_POSITIVE_DEV_DIAGNOSTIC_IMPLEMENTATION_PENDING_NO_SUBMIT
```

Therefore this review does not unlock Slurm submission.

## Wrapper Policy

The wrapper records the future cluster policy:

```text
partition=pomplun
qos=pomplun
account=cs_yinxin.wan
gpu=h200
time=30-00:00:00
```

Plan-only mode validates:

```text
route scope config
extractor/decoder/route py_compile
format_scrub=all extractor smoke
full_mode_enabled=false
generation_started=false
training_started=false
slurm_submission_started=false
```

## Local Validation

Commands:

```text
bash -n scripts/natural_evidence_v2/slurm/r4_positive_dev_diagnostic_h200.sbatch
REPO_HOME=/Users/guanjie/Documents/tokenizer_alignment VALIDATE_PLAN_ONLY=1 OUTPUT_DIR=results/natural_evidence_v2/status/r4_positive_dev_diagnostic_wrapper_plan_smoke_20260514 bash scripts/natural_evidence_v2/slurm/r4_positive_dev_diagnostic_h200.sbatch
```

Results:

```text
bash -n: pass
wrapper plan-only status: PASS_R4_POSITIVE_DEV_DIAGNOSTIC_WRAPPER_PLAN_ONLY
extractor smoke event count: 3
```

Plan-only summary:

```text
results/natural_evidence_v2/status/r4_positive_dev_diagnostic_wrapper_plan_smoke_20260514/plan_validation/wrapper_plan_only_summary.json
```

## Next Allowed Action

Run remote plan-only wrapper validation, remote zero-enabled allowlist safety,
and local/remote hash preflight. Do not enable allowlist or submit Slurm until
those preflights pass and a submission record is created.
