# Chimera H200 Submission Defaults

## Default GPU Target

New GPU submissions should target the H200 partition:

```bash
--partition=pomplun
--gres=gpu:h200:1
```

This requests one full H200 GPU when Chimera exposes H200s as non-MIG whole-GPU
resources. H200 HBM is controlled by the GPU allocation, not by Slurm `--mem`.

Default CPU RAM for GPU jobs is now:

```bash
--mem=240G
```

`--mem` is host CPU memory. If a specific run truly needs all node CPU RAM, pass
`MEM=0` manually when submitting, but do not use that as the default because it
can reduce scheduling flexibility.

Current comparison wrappers also default to this target, including plan or
artifact-backed jobs, so Chimera submission behavior stays consistent across the
next experiment phase.

## Manual Wrapper Override

All hand-written GPU submit wrappers accept these overrides:

```bash
PARTITION=pomplun \
GRES=gpu:h200:1 \
MEM=240G \
bash scripts/submit_ours_tinybench_utility.sh
```

For multi-GPU H200 jobs where NVLink matters, request multiple H200s explicitly:

```bash
PARTITION=pomplun GRES=gpu:h200:2 MEM=480G bash <submit-wrapper>
```

Only use multi-GPU requests after the runner is known to support multi-GPU
execution.

## Manifest-Based Jobs

The Slurm renderer now respects `runtime.resources.gpu_type`.

For H200:

```yaml
runtime:
  resources:
    partition: pomplun
    gpu_type: h200
    num_gpus: 1
    mem_gb: 240
```

This renders:

```bash
#SBATCH --partition=pomplun
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=240G
```

The legacy runtime files `gpu_a100_batch1.yaml` and `gpu_a100_batch28.yaml`
are kept for compatibility with existing experiment includes, but their current
contents now point to `pomplun` / `h200`.
