# Perinucleus Llama Anchor Protocol

Status: pending execution.

This package is an official-code anchor reproduction for Scalable Fingerprinting
/ Perinucleus on a Llama-family model close to the paper's main setting. It is
not a Qwen matched-budget final matrix.

## Fidelity Scope

- Official repository: `https://github.com/SewoongLab/scalable-fingerprinting-of-llms`.
- Official commit: `fdceaba14bd3e89340916a6a40e27c945d48460e`.
- License: MIT.
- Official scripts used: `generate_finetuning_data.py`,
  `finetune_multigpu.py`, `check_fingerprints.py`, and `eval_utility.py`.
- Model: `meta-llama/Meta-Llama-3.1-8B`.
- Fingerprint strategy: `perinucleus`.
- Query/fingerprint ladder: 16, 64, then 128 fingerprints.
- Response length: 1 token.
- Perinucleus settings: `nucleus_t=0.8`, `nucleus_k=3`.

## Training Mode

The anchor uses official `finetune_multigpu.py` with `--use_lora` because full
8B fine-tuning is not assumed feasible on a single Chimera A100 under the
current environment constraints. This must be reported as a LoRA adaptation, not
as a full-fine-tune reproduction.

The run applies environment/API compatibility patches only:

- PEFT legacy `task_type="lm"` is mapped to `task_type="CAUSAL_LM"`.
- DeepSpeed CPU offload is disabled to avoid CPUAdam CUDA-extension compilation
  under the Chimera CUDA/PyTorch mismatch.
- Tokenizer `accelerator.unwrap_model` is skipped because tokenizers are not
  torch modules.
- Utility evaluation loads PEFT adapters as `pretrained=<base>,peft=<adapter>`.

These patches do not alter fingerprint generation, fingerprint targets,
verification logic, or utility task definitions.

## Gate Conditions

Each ladder stage passes only if:

- official base fingerprint checking completes;
- official fine-tuning completes and saves a model or adapter;
- official fingerprint checking on the trained model completes;
- trained exact fingerprint accuracy is above base accuracy and above zero;
- trained target-response probability is above the base probability.

Utility sanity should complete on tinyBenchmarks. If it fails only because of an
environment dependency, the run records `pending_environment_dependency`; that
does not by itself count as evidence against Scalable Fingerprinting.

## Outputs

- `docs/baseline_perinucleus_llama_anchor_protocol.md`
- `docs/baseline_perinucleus_llama_anchor_result.md`
- `results/tables/baseline_perinucleus_llama_anchor.csv`
- `results/processed/paper_stats/baseline_perinucleus_llama_anchor_summary.json`
- `results/processed/paper_stats/baseline_perinucleus_llama_anchor_compute.json`

## Execution

```bash
python3 scripts/submit_slurm.py \
  --manifest manifests/baseline_perinucleus_llama_anchor/anchor_manifest.json \
  --registry manifests/baseline_perinucleus_llama_anchor/anchor_job_registry.jsonl \
  --manifest-id perinucleus-llama-anchor-s17 \
  --force \
  --submit
```
