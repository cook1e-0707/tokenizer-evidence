# Chimera Runbook

## Local dry-run

Validate the repository locally before touching the cluster:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/freeze_catalog.py \
  --source-catalog configs/data/real_pilot_catalog.yaml \
  --tokenizer-backend mock \
  --frozen-catalog-output artifacts/carrier_catalog_freeze_v1.yaml \
  --audit-report-output artifacts/tokenizer_audit_report_mock.json \
  --change-log-output artifacts/catalog_change_log.md \
  --data-config-output artifacts/real_pilot_frozen.yaml
```

The stage-4 pilot path does not require GPU.

## Generate the pilot manifest

Create the manifest locally from an experiment config that points to a frozen catalog:

```bash
.venv/bin/python scripts/make_manifest.py \
  --config /path/to/generated_frozen_experiment_config.yaml \
  --output manifests/exp_recovery/manifest.json
```

Dry-run inspection is usually enough before submission:

```bash
.venv/bin/python scripts/submit_slurm.py \
  --manifest manifests/exp_recovery/manifest.json
```

## Submit to Chimera

When the rendered sbatch command looks correct:

```bash
.venv/bin/python scripts/submit_slurm.py \
  --manifest manifests/exp_recovery/manifest.json \
  --submit
```

The pilot config requests modest CPU resources through `configs/runtime/pilot_cpu.yaml`.

## Inspect logs and metadata

Each run directory stores:

- `config.resolved.yaml`
- `environment.json`
- `submission.json`
- `run.log`
- `stdout.log`
- `stderr.log`
- `eval_summary.json`
- optional `rendered_evidence.txt`, `rendered_evidence.json`, `verifier_result.json`

Submission registry state is appended to:

- `manifests/job_registry.jsonl`

## Failure handling

Failed or timed-out entries can be requeued with:

```bash
.venv/bin/python scripts/resubmit_failed.py --manifest manifests/exp_recovery/manifest.json
.venv/bin/python scripts/resubmit_failed.py --manifest manifests/exp_recovery/manifest.json --submit
```

## Tokenizer audit note

Real Hugging Face tokenizer audit requires `transformers` to be installed in the environment. The cluster submission path does not depend on GPU for the stage-4 pilot, but pilot manifest/eval are blocked unless the catalog has frozen-catalog provenance from a strict-passed freeze workflow.
