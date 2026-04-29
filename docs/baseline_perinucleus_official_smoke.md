# Official Perinucleus Smoke Status

Date: 2026-04-28

## Status

`NOT_RUN`

The official repository was cloned and inspected, but the smoke test was not run
on this local machine. Local blockers:

- `python3 -c "import torch"` fails with `ModuleNotFoundError: No module named 'torch'`.
- The official Perinucleus generation and checking paths call CUDA directly.
- Running the smoke therefore requires a Chimera GPU environment.

## Chimera Runner

Use the single-entry smoke manifest:

```bash
cd "$REPO_HOME"
git pull --ff-only origin main

export EXP_SCRATCH=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence
mkdir -p "$EXP_SCRATCH/baselines/perinucleus_official_smoke"

python3 scripts/submit_slurm.py \
  --manifest manifests/baseline_perinucleus_official_smoke/smoke_manifest.json \
  --registry manifests/baseline_perinucleus_official_smoke/smoke_job_registry.jsonl \
  --all-pending \
  --submit
```

The job entry point is `scripts/run_perinucleus_official_smoke.py`; it writes
large generated fingerprints, model checkpoints, and logs under scratch only.
Small repo-local outputs are:

- `docs/baseline_perinucleus_official_smoke_result.md`
- `results/tables/baseline_perinucleus_official_smoke.csv`
- `results/processed/paper_stats/baseline_perinucleus_official_smoke_summary.json`
- `results/processed/paper_stats/baseline_perinucleus_official_smoke_compute.json`

The manifest environment setup verifies that official-code runtime imports are
available before launching the pipeline. It checks `wandb`, `deepspeed`,
`lm_eval`, and `peft`; if any are missing, it installs the official
requirement-compatible packages into the active Chimera venv. This is an
environment repair only. Missing dependencies must not be reported as evidence
against Scalable Fingerprinting.

The runner also applies smoke-only compatibility patches after checking out the
fixed official commit and before running official scripts. These patches are
recorded in the smoke summary under `compatibility_patches` and in the stage
list as `apply_*_patch`.

- `peft_lora_task_type`: `finetune_multigpu.py` uses the legacy PEFT LoRA
  setting `task_type="lm"`, while the active Chimera PEFT build rejects that
  value and requires `task_type="CAUSAL_LM"`.
- `deepspeed_disable_cpu_offload`: the official stage-2 DeepSpeed config
  enables CPU optimizer/parameter offload, which triggers DeepSpeed CPUAdam JIT
  compilation. Chimera currently exposes CUDA 10.1 to DeepSpeed while PyTorch is
  compiled with CUDA 12.1, so CPUAdam compilation fails before training starts.
  The smoke disables CPU offload for the tiny LoRA run to avoid that environment
  mismatch.
- `accelerate_skip_tokenizer_unwrap`: after training, the official script calls
  `trainer.accelerator.unwrap_model(tokenizer)`. Current Accelerate expects a
  torch module and fails on `Qwen2TokenizerFast`. The smoke saves the tokenizer
  directly, which is equivalent for tokenizer serialization.

These are API/environment repairs for smoke only. They do not change
fingerprint generation, training examples, verification, utility metrics, query
budgets, FAR logic, or thresholds.

## Official Repo Record

| Field | Value |
|---|---|
| repository | `https://github.com/SewoongLab/scalable-fingerprinting-of-llms` |
| local clone | `external_baselines/scalable_fingerprinting_official` |
| commit | `fdceaba14bd3e89340916a6a40e27c945d48460e` |
| license | MIT |
| README commands run | no |

## Smoke Gate Checklist

- [ ] Generate 8 or 16 fingerprints with `key_response_strategy=perinucleus`.
- [ ] Use `nucleus_t=0.8` and `nucleus_k=3`.
- [ ] Use `--use_chat_template` for Qwen/Qwen2.5-7B-Instruct or any instruct model unless a documented official setting says otherwise.
- [ ] Fine-tune a model with the generated fingerprints.
- [ ] Check fingerprints on the fingerprinted model with official `check_fingerprints.py`.
- [ ] Compare fingerprinted-model responses against base-model responses for the same keys.
- [ ] Run cheap utility evaluation if feasible.
- [ ] Write the fingerprint file, final model path, check output, and utility output under scratch.
- [ ] Record any compatibility patch applied to the official clone.

## Pass Criteria

Smoke passes only if:

- fingerprint checking is above random chance;
- at least one fingerprint response changes from the base model to the
  fingerprinted model;
- no raw prompt/chat-template mismatch is observed;
- utility command either completes or has a documented cheap-eval limitation.

## Current Decision

Full Perinucleus final matrices remain blocked. The current no-train v0 result
is renamed `perinucleus_no_train_diagnostic` and cannot be used as a Scalable
Fingerprinting baseline.
