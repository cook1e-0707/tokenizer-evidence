#!/bin/bash
#SBATCH --job-name=sanitizer-eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/sanitizer_benchmark/slurm/sanitizer-eval-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/sanitizer_benchmark/slurm/sanitizer-eval-%j.err

# Sanitizer benchmark: Apply text sanitizers and re-decode payload
# PREREQUISITE: Both Qwen AND Llama must pass WP6 positive recovery
# This wrapper applies sanitizers to generated outputs and measures
# payload recovery rate under each transformation.

set -euo pipefail

export REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
export PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
export VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
export HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"
export RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/sanitizer_benchmark}"

# Source model/adapter from the best-passing WP6 run
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export PRIMARY_BANK="${PRIMARY_BANK:-results/natural_evidence_v2/buckets/qwen_v2_primary_2way_bank.jsonl}"
export PROTECTED_ADAPTER="${PROTECTED_ADAPTER:-}"
export GENERATION_DIR="${GENERATION_DIR:-}"
export CONTRACT="${CONTRACT:-results/natural_evidence_v2/status/wp6_r2_option_b_wrapper_contract_plan_20260510_0251/wp6_r2_option_b_wrapper_contract_plan.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/sanitizer_eval_${SLURM_JOB_ID:-manual}}"
export BLOCK_COUNT="${BLOCK_COUNT:-8}"
export BLOCK_SIZE="${BLOCK_SIZE:-64}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$OUTPUT_DIR"

set +u
if [ -f /etc/profile ]; then
  . /etc/profile
fi
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
fi
set -u

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\r\n' <"$HF_TOKEN_FILE")"
fi

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "model_name=$MODEL_NAME"
echo "output_dir=$OUTPUT_DIR"
echo "claim_note=Sanitizer robustness benchmark; requires both Qwen and Llama positive recovery"

# Validate inputs
if [ -z "$GENERATION_DIR" ] || [ ! -d "$GENERATION_DIR" ]; then
  echo "ERROR: GENERATION_DIR not set or not found" >&2
  exit 2
fi

for required in "$PRIMARY_BANK" "$CONTRACT"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

# Step 1: Apply sanitizers to generated outputs
echo "=== Applying text sanitizers ==="
SANITIZER_DIR="$OUTPUT_DIR/sanitized"
"$PYTHON" scripts/natural_evidence_v2/apply_sanitizers.py \
  --generation-dir "$GENERATION_DIR" \
  --output-dir "$SANITIZER_DIR"

# Step 2: Decode each sanitized variant
echo "=== Decoding sanitized outputs ==="
for sanitizer in case_normalize step_label_strip transition_strip hedge_strip sentence_reorder; do
  SAN_INPUT="$SANITIZER_DIR/$sanitizer/sanitized_outputs.jsonl"
  if [ ! -s "$SAN_INPUT" ]; then
    echo "Skipping $sanitizer: no sanitized outputs"
    continue
  fi

  DECODE_DIR="$OUTPUT_DIR/decode_$sanitizer"
  echo "--- Decoding: $sanitizer ---"

  # Create a temporary generation dir with sanitized text
  TEMP_GEN="$OUTPUT_DIR/temp_gen_$sanitizer"
  mkdir -p "$TEMP_GEN"
  cp "$SAN_INPUT" "$TEMP_GEN/wp6_generated_outputs.jsonl"

  "$PYTHON" scripts/natural_evidence_v2/decode_wp6_payload.py \
    --generation-dir "$TEMP_GEN" \
    --primary-bank "$PRIMARY_BANK" \
    --contract "$CONTRACT" \
    --output-dir "$DECODE_DIR" \
    --block-count "$BLOCK_COUNT" \
    --block-size "$BLOCK_SIZE" \
    || echo "WARNING: decode failed for $sanitizer"

  # Clean up temp
  rm -rf "$TEMP_GEN"
done

# Step 3: Aggregate results
echo "=== Aggregating sanitizer results ==="
"$PYTHON" -c "
import json
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
results = {}
for sanitizer in ['case_normalize', 'step_label_strip', 'transition_strip', 'hedge_strip', 'sentence_reorder']:
    decode_dir = output_dir / f'decode_{sanitizer}'
    summary_path = decode_dir / 'wp6_decode_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        results[sanitizer] = {
            'protected_accept_rate': summary.get('protected_accept_rate_at_64', 'N/A'),
            'null_accepts': summary.get('null_accept_counts_at_64', {}),
        }
    else:
        results[sanitizer] = {'status': 'NO_DECODE_ARTIFACTS'}

aggregation = {
    'schema_name': 'sanitizer_benchmark_aggregation',
    'results_by_sanitizer': results,
    'status': 'COMPLETE',
}
out_path = output_dir / 'sanitizer_benchmark_aggregation.json'
out_path.write_text(json.dumps(aggregation, indent=2, sort_keys=True) + '\n')
print(f'Wrote: {out_path}')
"

echo "Sanitizer benchmark complete: $OUTPUT_DIR"
