#!/usr/bin/env bash
#SBATCH --job-name=nat-ev-qwen-cx-shard
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=pomplun
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --array=0-2

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

SOURCE_ROOT="${SOURCE_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192}"
EXPANSION_ROOT="${EXPANSION_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3}"

CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-64}"
MAX_PREFIXES_PER_RESPONSE="${MAX_PREFIXES_PER_RESPONSE:-64}"
PREFIX_STRIDE="${PREFIX_STRIDE:-2}"
MIN_RESPONSE_PREFIX_TOKENS="${MIN_RESPONSE_PREFIX_TOKENS:-1}"
SHARD_COUNT="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-3}}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
PROGRESS_EVERY="${PROGRESS_EVERY:-256}"

REFERENCE_OUTPUTS="$SOURCE_ROOT/reference_outputs/qwen_reference_outputs.jsonl"
REFERENCE_CANDIDATES="$EXPANSION_ROOT/reference_candidates/shards/qwen_topk_candidates_shard_${SHARD_INDEX}_of_${SHARD_COUNT}.jsonl"
PROGRESS_JSON="$EXPANSION_ROOT/status/shards/qwen_candidate_shard_${SHARD_INDEX}_of_${SHARD_COUNT}_progress.json"
STATUS_DIR="$EXPANSION_ROOT/status"

mkdir -p "$EXPANSION_ROOT/slurm" \
  "$EXPANSION_ROOT/reference_candidates/shards" \
  "$STATUS_DIR/shards"

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

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "config=$CONFIG"
echo "source_root=$SOURCE_ROOT"
echo "expansion_root=$EXPANSION_ROOT"
echo "reference_outputs=$REFERENCE_OUTPUTS"
echo "reference_candidates=$REFERENCE_CANDIDATES"
echo "candidate_top_k=$CANDIDATE_TOP_K"
echo "max_prefixes_per_response=$MAX_PREFIXES_PER_RESPONSE"
echo "prefix_stride=$PREFIX_STRIDE"
echo "shard_index=$SHARD_INDEX"
echo "shard_count=$SHARD_COUNT"
echo "progress_json=$PROGRESS_JSON"
echo "python=$(command -v python3)"

if [ ! -s "$REFERENCE_OUTPUTS" ]; then
  echo "Missing reference outputs: $REFERENCE_OUTPUTS" >&2
  exit 2
fi

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
  --summary "$STATUS_DIR/static_validation_summary_qwen_candidate_expansion_sharded.json"

python3 scripts/natural_evidence_v1/score_reference_candidates.py \
  --config "$CONFIG" \
  --tokenizer-key qwen \
  --input-jsonl "$REFERENCE_OUTPUTS" \
  --output-jsonl "$REFERENCE_CANDIDATES" \
  --candidate-top-k "$CANDIDATE_TOP_K" \
  --max-prefixes-per-response "$MAX_PREFIXES_PER_RESPONSE" \
  --prefix-stride "$PREFIX_STRIDE" \
  --min-response-prefix-tokens "$MIN_RESPONSE_PREFIX_TOKENS" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --stream-output \
  --progress-json "$PROGRESS_JSON" \
  --progress-every "$PROGRESS_EVERY" \
  --require-cuda
