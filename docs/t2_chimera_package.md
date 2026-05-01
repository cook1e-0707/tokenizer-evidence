# T2 Chimera Package

This is the repaired direct-submit Chimera package for `T2-r1`.

Why `r1`:
- the first `T2` package validated the execution path
- but it used a single-token-per-bucket compiled catalog, so `bucket_mass`, `fixed_representative`, and `uniform_bucket` collapsed to the same effective supervision problem
- `T2-r1` keeps the same `Qwen/Qwen2.5-7B-Instruct` family and the same compiled training path, but switches to a strict-passed frozen Qwen catalog with multi-member buckets and targets `U15` under `block_count=1`

Scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- objective contrasts:
  - `bucket_mass`
  - `fixed_representative`
  - `uniform_bucket`
- catalog: `configs/data/frozen/real_pilot_catalog__qwen2_5_7b__v1.yaml`
- payload target: `U15`
- block_count: `1`
- no `Batch 3` expansion
- no baselines
- no new model family

## 1. Prepare Environment

```bash
cd /home/guanjie.lin001/tokenizer-evidence
git pull origin main
source /hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/activate

export CHIMERA_ENV_SETUP=$'if [ -f /etc/profile ]; then . /etc/profile; fi\nsource /hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/activate\nexport HF_HOME=/hpcstor6/scratch01/g/guanjie.lin001/huggingface\nexport HF_TOKEN="$(tr -d \'\\r\\n\' </hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token)"'

export T2R1_ROOT=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/theorem2_qwen7b_r1
mkdir -p "$T2R1_ROOT"
```

## 2. Define Submit Helpers

```bash
t2r1_case_root() {
  local variant="$1"
  echo "$T2R1_ROOT/$variant"
}

t2r1_submit_train() {
  local variant="$1"
  local config="$2"
  local root
  root="$(t2r1_case_root "$variant")"
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

t2r1_submit_eval() {
  local variant="$1"
  local config="$2"
  local root
  root="$(t2r1_case_root "$variant")"

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

t2r1_show_eval() {
  local variant="$1"
  local root
  root="$(t2r1_case_root "$variant")"
  find "$root/runs/exp_eval" -name eval_summary.json | tail -n 1 | xargs sed -n '1,260p'
}
```

## 3. Submit The Three T2-r1 Train Runs

```bash
t2r1_submit_train bucket_mass configs/experiment/prep/exp_train__qwen2_5_7b__t2r1_bucket_mass_v1.yaml
t2r1_submit_train fixed_representative configs/experiment/prep/exp_train__qwen2_5_7b__t2r1_fixed_representative_v1.yaml
t2r1_submit_train uniform_bucket configs/experiment/prep/exp_train__qwen2_5_7b__t2r1_uniform_bucket_v1.yaml
```

## 4. Monitor Train Jobs

```bash
squeue -u "$USER" -o "%.18i %.9P %.24j %.8T %.10M %.6D %R"
```

```bash
tail -n 5 "$(t2r1_case_root bucket_mass)/manifests/job_registry.jsonl"
tail -n 5 "$(t2r1_case_root fixed_representative)/manifests/job_registry.jsonl"
tail -n 5 "$(t2r1_case_root uniform_bucket)/manifests/job_registry.jsonl"
```

Train-completion checks:

```bash
find "$(t2r1_case_root bucket_mass)/runs/exp_train" -name train_summary.json -o -name training_health.json -o -name latest_eval_input.json | sort
find "$(t2r1_case_root fixed_representative)/runs/exp_train" -name train_summary.json -o -name training_health.json -o -name latest_eval_input.json | sort
find "$(t2r1_case_root uniform_bucket)/runs/exp_train" -name train_summary.json -o -name training_health.json -o -name latest_eval_input.json | sort
```

## 5. Submit The Three T2-r1 Eval Runs

```bash
t2r1_submit_eval bucket_mass configs/experiment/prep/exp_eval__qwen2_5_7b__t2r1_bucket_mass_v1.yaml
t2r1_submit_eval fixed_representative configs/experiment/prep/exp_eval__qwen2_5_7b__t2r1_fixed_representative_v1.yaml
t2r1_submit_eval uniform_bucket configs/experiment/prep/exp_eval__qwen2_5_7b__t2r1_uniform_bucket_v1.yaml
```

## 6. Inspect Eval Results

```bash
t2r1_show_eval bucket_mass
t2r1_show_eval fixed_representative
t2r1_show_eval uniform_bucket
```

Additional diagnostics:

```bash
find "$(t2r1_case_root bucket_mass)/runs/exp_eval" -name compiled_gate_result.json | tail -n 1 | xargs sed -n '1,260p'
find "$(t2r1_case_root fixed_representative)/runs/exp_eval" -name compiled_gate_result.json | tail -n 1 | xargs sed -n '1,260p'
find "$(t2r1_case_root uniform_bucket)/runs/exp_eval" -name compiled_gate_result.json | tail -n 1 | xargs sed -n '1,260p'
```

## 7. T2-r1 Gate

The package is usable only if:
- all three train runs complete with `State=COMPLETED` and `ExitCode=0:0`
- all three eval runs complete with `State=COMPLETED` and `ExitCode=0:0`
- all three variants remain on the same model family, catalog, payload target, and runtime envelope
- `training_health.json` reports the correct compiled objective mode for each variant
- at least one comparison axis becomes informative:
  - different `slot_exact_rate`
  - different chosen token pattern within the same accepted bucket
  - or different accept/reject behavior under the same target

This is still a controlled theorem package, not a baseline opening or a robustness-grid expansion.
