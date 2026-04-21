# T1 Chimera Package

This is the direct-submit Chimera package for `T1` only.

Scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- codebook: unchanged compiled Qwen 7B codebook
- contrasts:
  - `contextual_exact`
  - `sequence_proxy`
- payload target: `U03`
- no `Batch 3` expansion
- no new baselines
- no new model family

`T2` remains prep-only at this stage. The objective-contrast configs exist locally, but the current direct-submit package is `T1`.

## 1. Prepare Environment

```bash
cd /home/guanjie.lin001/tokenizer-evidence
git pull origin main
source /hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/activate

export CHIMERA_ENV_SETUP=$'source ~/.bashrc\nsource /hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/activate\nexport HF_HOME=/hpcstor6/scratch01/g/guanjie.lin001/huggingface\nexport HF_TOKEN="$(tr -d \'\\r\\n\' </hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token)"'

export T1_ROOT=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/theorem1_qwen7b
mkdir -p "$T1_ROOT"
```

## 2. Define Submit Helpers

```bash
t1_case_root() {
  local variant="$1"
  echo "$T1_ROOT/$variant"
}

t1_submit_train() {
  local variant="$1"
  local config="$2"
  local root
  root="$(t1_case_root "$variant")"
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

t1_submit_eval() {
  local variant="$1"
  local config="$2"
  local root
  root="$(t1_case_root "$variant")"

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

t1_show_eval() {
  local variant="$1"
  local root
  root="$(t1_case_root "$variant")"
  find "$root/runs/exp_eval" -name eval_summary.json | tail -n 1 | xargs sed -n '1,240p'
}
```

## 3. Submit The Two T1 Train Runs

```bash
t1_submit_train contextual_exact configs/experiment/prep/exp_train__qwen2_5_7b__t1_contextual_exact_v1.yaml
t1_submit_train sequence_proxy configs/experiment/prep/exp_train__qwen2_5_7b__t1_sequence_proxy_v1.yaml
```

## 4. Monitor Train Jobs

```bash
squeue -u "$USER" -o "%.18i %.9P %.24j %.8T %.10M %.6D %R"
```

```bash
tail -n 5 "$(t1_case_root contextual_exact)/manifests/job_registry.jsonl"
tail -n 5 "$(t1_case_root sequence_proxy)/manifests/job_registry.jsonl"
```

Train-completion checks:

```bash
find "$(t1_case_root contextual_exact)/runs/exp_train" -name train_summary.json -o -name latest_eval_input.json -o -name training_health.json | sort
find "$(t1_case_root sequence_proxy)/runs/exp_train" -name train_summary.json -o -name latest_eval_input.json -o -name training_health.json | sort
```

## 5. Submit The Two T1 Eval Runs

```bash
t1_submit_eval contextual_exact configs/experiment/prep/exp_eval__qwen2_5_7b__t1_contextual_exact_v1.yaml
t1_submit_eval sequence_proxy configs/experiment/prep/exp_eval__qwen2_5_7b__t1_sequence_proxy_v1.yaml
```

## 6. Inspect Eval Results

```bash
t1_show_eval contextual_exact
t1_show_eval sequence_proxy
```

Additional diagnostics:

```bash
find "$(t1_case_root contextual_exact)/runs/exp_eval" -name compiled_gate_result.json | tail -n 1 | xargs sed -n '1,260p'
find "$(t1_case_root sequence_proxy)/runs/exp_eval" -name compiled_gate_result.json | tail -n 1 | xargs sed -n '1,260p'
```

## 7. T1 Gate

The package is usable only if:
- both train runs complete with `State=COMPLETED` and `ExitCode=0:0`
- both eval runs complete with `State=COMPLETED` and `ExitCode=0:0`
- the `contextual_exact` path is accepted and verifier-successful
- the `sequence_proxy` path yields a directly comparable train/eval artifact under the same payload and codebook

This package is for controlled theorem comparison, not for reopening baselines or enlarging the robustness grid.
