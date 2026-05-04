#!/usr/bin/env bash
#SBATCH --job-name=nat-ev-phase-a
#SBATCH --partition=pomplun
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=48:00:00

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"
PROMPT_COUNT="${PROMPT_COUNT:-4096}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
MAX_PREFIXES_PER_RESPONSE="${MAX_PREFIXES_PER_RESPONSE:-16}"
PREFIX_STRIDE="${PREFIX_STRIDE:-8}"
MIN_RESPONSE_PREFIX_TOKENS="${MIN_RESPONSE_PREFIX_TOKENS:-1}"
TARGET_ENTRIES="${TARGET_ENTRIES:-24576}"

mkdir -p "$SCRATCH_ROOT/slurm" \
  "$SCRATCH_ROOT/reference_outputs" \
  "$SCRATCH_ROOT/reference_candidates" \
  "$SCRATCH_ROOT/bucket_banks"

cd "$REPO_HOME"

set +u
if [ -f /etc/profile ]; then
  # shellcheck disable=SC1091
  . /etc/profile
fi
if [ -f "$VENV_PATH/bin/activate" ]; then
  # shellcheck disable=SC1091
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

REFERENCE_OUTPUTS="$SCRATCH_ROOT/reference_outputs/${TOKENIZER_KEY}_reference_outputs.jsonl"
REFERENCE_CANDIDATES="$SCRATCH_ROOT/reference_candidates/${TOKENIZER_KEY}_topk_candidates.jsonl"
BUCKET_BANK_DIR="$SCRATCH_ROOT/bucket_banks"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "config=$CONFIG"
echo "tokenizer_key=$TOKENIZER_KEY"
echo "scratch=$SCRATCH_ROOT"
echo "python=$(command -v python3)"

python3 - <<'PY'
import os
import torch

print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("device count =", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA_NOT_AVAILABLE_ON_ALLOCATED_NODE")
print("device name =", torch.cuda.get_device_name(0))
PY

python3 scripts/natural_evidence_v1/validate_static.py \
  --config "$CONFIG" \
  --summary "$SCRATCH_ROOT/static_validation_summary_${TOKENIZER_KEY}.json"

python3 scripts/natural_evidence_v1/generate_reference_outputs.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --output-jsonl "$REFERENCE_OUTPUTS" \
  --prompt-count "$PROMPT_COUNT" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --require-cuda

python3 scripts/natural_evidence_v1/score_reference_candidates.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --input-jsonl "$REFERENCE_OUTPUTS" \
  --output-jsonl "$REFERENCE_CANDIDATES" \
  --max-prefixes-per-response "$MAX_PREFIXES_PER_RESPONSE" \
  --prefix-stride "$PREFIX_STRIDE" \
  --min-response-prefix-tokens "$MIN_RESPONSE_PREFIX_TOKENS" \
  --require-cuda

python3 scripts/natural_evidence_v1/build_bucket_bank.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --candidate-jsonl "$REFERENCE_CANDIDATES" \
  --output-dir "$BUCKET_BANK_DIR" \
  --target-entries "$TARGET_ENTRIES"

echo "reference_outputs=$REFERENCE_OUTPUTS"
echo "reference_candidates=$REFERENCE_CANDIDATES"
echo "bucket_bank_dir=$BUCKET_BANK_DIR"
