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

## GPU resource preflight

Before submitting any Chimera GPU job that can be sharded, parallelized, or run
as multiple independent seeds/payloads, first inspect the target partitions for
available full H200/A100 devices. Prefer filling the remaining full-GPU capacity
on the relevant partition instead of launching a single long serial job, subject
to the experiment gate, account/QOS limits, and artifact-isolation rules.

Full-GPU here means non-MIG `gpu:h200` or `gpu:A100` resources. Do not count or
request MIG GRES for paper-facing runs unless a task explicitly documents that
MIG is acceptable; MIG slices have different memory and throughput behavior and
should not be mixed into comparable H200/A100 runs.

Start every GPU submission session with a remote resource check:

```bash
ssh chimera
sinfo -p pomplun,DGXA100 -N -o "%N %P %t %G %C"
squeue -p pomplun,DGXA100 -o "%.18i %.9P %.24j %.8T %.10M %.6D %.16b %R"
for p in pomplun DGXA100; do
  scontrol show partition "$p" | egrep 'PartitionName=|MaxTime=|DefaultTime=|AllowQos=|State=|TRES=|Gres='
done
scontrol show node -d | egrep 'NodeName=|State=|Gres=|CfgTRES=|AllocTRES='
```

Interpretation rules:

- Count only nodes whose GRES exposes full `gpu:h200:<n>` or `gpu:A100:<n>`
  devices and does not report MIG-only GPU resources.
- For idle nodes, the full GPU count is the GRES count. For mixed nodes, subtract
  the allocated full GPUs from the configured full GPUs using `AllocTRES` and
  `CfgTRES`; if this is ambiguous, treat the node as unavailable.
- If the same experiment can use both H200 and A100, submit separate shard ranges
  or arrays per partition so outputs remain disjoint and resource provenance
  stays explicit.
- For independent work such as reference scoring, candidate scoring, FAR shards,
  payload/seed cells, or null arms, set shard count or Slurm array concurrency to
  the number of available full GPUs, capped by the remaining task count and
  account/QOS limits.
- For actual training, request multiple GPUs in one job only when the training
  launcher is configured for multi-GPU execution. Otherwise fill available GPUs
  with independent single-GPU jobs over disjoint seeds, payloads, shards, or
  model arms.
- Never use resource availability as permission to bypass method gates. The job
  must still be allowlisted, must not overwrite existing artifacts, and must not
  expand old structured carrier-slot artifacts unless explicitly requested.

The practical goal is to avoid slow one-GPU serial execution when a partition has
idle full H200/A100 capacity. Prefer saturating the remaining eligible full GPUs
with auditable independent work over leaving the partition idle.

## Slurm mail notifications

Future Chimera sbatch scripts should request Slurm email notifications so long
jobs are visible without repeatedly polling the cluster. Chimera's script
examples list the mail options as optional commented directives; enabled scripts
must use single-`#SBATCH` lines:

```bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
```

Keep these directives in the sbatch header, before the first non-`#SBATCH`
command. For diagnostic CPU tests, submit a short job first to confirm the
cluster accepts the mail settings before using the script for long GPU work.

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
