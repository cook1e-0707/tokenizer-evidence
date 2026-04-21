# T2 Chimera Package

This is the direct-submit Chimera package for `T2` only.

Scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- codebook: unchanged compiled Qwen 7B codebook
- contrasts:
  - `bucket_mass`
  - `fixed_representative`
  - `uniform_bucket`
- payload target: `U03`
- no `Batch 3` expansion
- no new baselines
- no new model family

This package tests the objective layer only. It keeps the compiled path, codebook, model family, and payload target fixed.

## 1. Prepare Environment

```bash
cd /home/guanjie.lin001/tokenizer-evidence
git pull origin main
source /hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/activate

export CHIMERA_ENV_SETUP=$'source ~/.bashrc\nsource /hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/activate\nexport HF_HOME=/hpcstor6/scratch01/g/guanjie.lin001/huggingface\nexport HF_TOKEN="$(tr -d \'\\r\\n\' </hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token)"'

export T2_ROOT=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/theorem2_qwen7b
mkdir -p "$T2_ROOT"
```

## 2. Define Submit Helpers

```bash
t2_case_root() {
  local variant="$1"
  echo "$T2_ROOT/$variant"
}

t2_submit_train() {
  local variant="$1"
  local config="$2"
  local root
  root="$(t2_case_root "$variant")"
  mkdir -p "$root/manifests" "$root/runs" "$root/processed"

  python scripts/make_manifest.py \
    --config "$config" \
    --override runtime.output_root="$root/runs" \
    --override runtime.environment_setup="$CHIMERA_ENV_SETUP" \
    --output "$root/manifests/$(basename "$config" .yaml).json"

  python scripts/submit_slurm.py \
    --manifest "$root/manifests/$(basename "$config" .yaml).json" \
    --registry "$root/manifests/job_registry.jsonl" \
    --submit \
    --force
}

t2_submit_eval() {
  local variant="$1"
  local config="$2"
  local root
  root="$(t2_case_root "$variant")"

  test -f "$root/runs/exp_train/latest_eval_input.json" || {
    echo "missing eval input for $variant" >&2
    return 1
  }

  python scripts/make_manifest.py \
    --config "$config" \
    --override data.eval_path="$root/runs/exp_train/latest_eval_input.json" \
    --override runtime.output_root="$root/runs" \
    --override runtime.environment_setup="$CHIMERA_ENV_SETUP" \
    --output "$root/manifests/$(basename "$config" .yaml).json"

  python scripts/submit_slurm.py \
    --manifest "$root/manifests/$(basename "$config" .yaml).json" \
    --registry "$root/manifests/job_registry.jsonl" \
    --submit \
    --force
}

t2_show_eval() {
  local variant="$1"
  local root
  root="$(t2_case_root "$variant")"
  find "$root/runs/exp_eval" -name eval_summary.json | tail -n 1 | xargs sed -n '1,240p'
}
```

## 3. Submit The Three T2 Train Runs

```bash
t2_submit_train bucket_mass configs/experiment/prep/exp_train__qwen2_5_7b__t2_bucket_mass_v1.yaml
t2_submit_train fixed_representative configs/experiment/prep/exp_train__qwen2_5_7b__t2_fixed_representative_v1.yaml
t2_submit_train uniform_bucket configs/experiment/prep/exp_train__qwen2_5_7b__t2_uniform_bucket_v1.yaml
```

## 4. Monitor Train Jobs

```bash
squeue -u "$USER" -o "%.18i %.9P %.24j %.8T %.10M %.6D %R"
```

```bash
tail -n 5 "$(t2_case_root bucket_mass)/manifests/job_registry.jsonl"
tail -n 5 "$(t2_case_root fixed_representative)/manifests/job_registry.jsonl"
tail -n 5 "$(t2_case_root uniform_bucket)/manifests/job_registry.jsonl"
```

Train-completion checks:

```bash
find "$(t2_case_root bucket_mass)/runs/exp_train" -name train_summary.json -o -name latest_eval_input.json -o -name training_health.json | sort
find "$(t2_case_root fixed_representative)/runs/exp_train" -name train_summary.json -o -name latest_eval_input.json -o -name training_health.json | sort
find "$(t2_case_root uniform_bucket)/runs/exp_train" -name train_summary.json -o -name latest_eval_input.json -o -name training_health.json | sort
```

## 5. Submit The Three T2 Eval Runs

```bash
t2_submit_eval bucket_mass configs/experiment/prep/exp_eval__qwen2_5_7b__t2_bucket_mass_v1.yaml
t2_submit_eval fixed_representative configs/experiment/prep/exp_eval__qwen2_5_7b__t2_fixed_representative_v1.yaml
t2_submit_eval uniform_bucket configs/experiment/prep/exp_eval__qwen2_5_7b__t2_uniform_bucket_v1.yaml
```

## 6. Inspect Eval Results

```bash
t2_show_eval bucket_mass
t2_show_eval fixed_representative
t2_show_eval uniform_bucket
```

Additional diagnostics:

```bash
find "$(t2_case_root bucket_mass)/runs/exp_eval" -name compiled_gate_result.json | tail -n 1 | xargs sed -n '1,260p'
find "$(t2_case_root fixed_representative)/runs/exp_eval" -name compiled_gate_result.json | tail -n 1 | xargs sed -n '1,260p'
find "$(t2_case_root uniform_bucket)/runs/exp_eval" -name compiled_gate_result.json | tail -n 1 | xargs sed -n '1,260p'
```

## 7. T2 Gate

The package is usable only if:
- all three train runs complete with `State=COMPLETED` and `ExitCode=0:0`
- all three eval runs complete with `State=COMPLETED` and `ExitCode=0:0`
- all three variants remain on the same payload target and compiled codebook
- the resulting acceptance and verifier outcomes are directly comparable across objectives

This package is for controlled objective comparison only. It is not a baseline opening, a new-model replication, or a larger robustness sweep.
