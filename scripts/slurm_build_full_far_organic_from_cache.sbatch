#!/usr/bin/env bash
#SBATCH --job-name=full-far-build-org
#SBATCH --partition=Intel6240
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
FULL_FAR_CONFIG="${FULL_FAR_CONFIG:-configs/experiment/comparison/full_far_payload_claim.yaml}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/comparison/full_far_payload_claim}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH_ROOT/shards/organic-prompt-cache-10way}"
ROW_SHARD_DIR="${ROW_SHARD_DIR:-$SCRATCH_ROOT/shards/organic-prompts-10way-from-cache}"
EXPECTED_SHARD_COUNT="${EXPECTED_SHARD_COUNT:-10}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
SHARD_COUNT="${SHARD_COUNT:-$EXPECTED_SHARD_COUNT}"
FORCE="${FORCE:-1}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

mkdir -p "$SCRATCH_ROOT/slurm" "$ROW_SHARD_DIR"
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
export PYTHONUNBUFFERED=1
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\r\n' <"$HF_TOKEN_FILE")"
fi

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "config=$FULL_FAR_CONFIG"
echo "scratch=$SCRATCH_ROOT"
echo "cache_dir=$CACHE_DIR"
echo "row_shard_dir=$ROW_SHARD_DIR"
echo "expected_shard_count=$EXPECTED_SHARD_COUNT"
echo "shard_index=${SHARD_INDEX:-all}"
echo "shard_count=$SHARD_COUNT"
echo "force=$FORCE"
echo "hf_home=$HF_HOME"
echo "transformers_cache=$TRANSFORMERS_CACHE"
echo "cpus=${SLURM_CPUS_PER_TASK:-unknown}"
echo "python=$(command -v python3)"
git rev-parse --short HEAD || true
nl -ba scripts/build_full_far_organic_from_cache.py | sed -n '55,105p'

EXTRA_ARGS=()
if [ "$FORCE" = "1" ]; then
  EXTRA_ARGS+=(--force)
fi
if [ -n "$SHARD_INDEX" ]; then
  EXTRA_ARGS+=(--shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT")
fi

python3 scripts/build_full_far_organic_from_cache.py \
  --config "$FULL_FAR_CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --row-shard-output-dir "$ROW_SHARD_DIR" \
  --expected-shard-count "$EXPECTED_SHARD_COUNT" \
  "${EXTRA_ARGS[@]}"
