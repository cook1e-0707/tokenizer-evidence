#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="${REPO_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FULL_FAR_CONFIG="${FULL_FAR_CONFIG:-configs/experiment/comparison/full_far_payload_claim.yaml}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/comparison/full_far_payload_claim}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH_ROOT/shards/organic-prompt-cache-10way}"
ROW_SHARD_DIR="${ROW_SHARD_DIR:-$SCRATCH_ROOT/shards/organic-prompts-10way-from-cache}"
EXPECTED_SHARD_COUNT="${EXPECTED_SHARD_COUNT:-10}"
GLOBAL_SHARD_COUNT="${GLOBAL_SHARD_COUNT:-${SHARD_COUNT:-$EXPECTED_SHARD_COUNT}}"
LOCAL_SHARD_COUNT="${LOCAL_SHARD_COUNT:-$GLOBAL_SHARD_COUNT}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
ARRAY="${ARRAY:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-$LOCAL_SHARD_COUNT}"
FORCE="${FORCE:-1}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
PARTITION="${PARTITION:-Intel6240}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-120G}"
TIME_LIMIT="${TIME_LIMIT:-4-00:00:00}"

mkdir -p "$SCRATCH_ROOT/slurm" "$ROW_SHARD_DIR"
unset SBATCH_QOS
unset SLURM_QOS
unset SBATCH_ACCOUNT

SBATCH_ARGS=(
  --partition="$PARTITION"
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="$MEM"
  --time="$TIME_LIMIT"
  --export=HOME,USER,LOGNAME,REPO_HOME="$REPO_HOME",FULL_FAR_CONFIG="$FULL_FAR_CONFIG",SCRATCH_ROOT="$SCRATCH_ROOT",CACHE_DIR="$CACHE_DIR",ROW_SHARD_DIR="$ROW_SHARD_DIR",EXPECTED_SHARD_COUNT="$EXPECTED_SHARD_COUNT",GLOBAL_SHARD_COUNT="$GLOBAL_SHARD_COUNT",LOCAL_SHARD_COUNT="$LOCAL_SHARD_COUNT",SHARD_OFFSET="$SHARD_OFFSET",FORCE="$FORCE",VENV_PATH="$VENV_PATH",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$HF_HOME"
)

if [ "$ARRAY" = "1" ]; then
  if ! [[ "$EXPECTED_SHARD_COUNT" =~ ^[0-9]+$ ]] || [ "$EXPECTED_SHARD_COUNT" -le 0 ]; then
    echo "EXPECTED_SHARD_COUNT must be a positive integer; got $EXPECTED_SHARD_COUNT" >&2
    exit 2
  fi
  if ! [[ "$GLOBAL_SHARD_COUNT" =~ ^[0-9]+$ ]] || [ "$GLOBAL_SHARD_COUNT" -le 0 ]; then
    echo "GLOBAL_SHARD_COUNT must be a positive integer; got $GLOBAL_SHARD_COUNT" >&2
    exit 2
  fi
  if ! [[ "$LOCAL_SHARD_COUNT" =~ ^[0-9]+$ ]] || [ "$LOCAL_SHARD_COUNT" -le 0 ]; then
    echo "LOCAL_SHARD_COUNT must be a positive integer; got $LOCAL_SHARD_COUNT" >&2
    exit 2
  fi
  if ! [[ "$SHARD_OFFSET" =~ ^[0-9]+$ ]]; then
    echo "SHARD_OFFSET must be a non-negative integer; got $SHARD_OFFSET" >&2
    exit 2
  fi
  if [ $((SHARD_OFFSET + LOCAL_SHARD_COUNT)) -gt "$GLOBAL_SHARD_COUNT" ]; then
    echo "SHARD_OFFSET + LOCAL_SHARD_COUNT must be <= GLOBAL_SHARD_COUNT" >&2
    exit 2
  fi
  if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [ "$MAX_PARALLEL" -le 0 ]; then
    echo "MAX_PARALLEL must be a positive integer; got $MAX_PARALLEL" >&2
    exit 2
  fi
  SBATCH_ARGS+=(
    --array="0-$((LOCAL_SHARD_COUNT - 1))%$MAX_PARALLEL"
    --output="$SCRATCH_ROOT/slurm/%x-%A_%a.out"
    --error="$SCRATCH_ROOT/slurm/%x-%A_%a.err"
  )
else
  if ! [[ "$SHARD_OFFSET" =~ ^[0-9]+$ ]]; then
    echo "SHARD_OFFSET must be a non-negative integer; got $SHARD_OFFSET" >&2
    exit 2
  fi
  SBATCH_ARGS[${#SBATCH_ARGS[@]}-1]+=",SHARD_INDEX=$SHARD_OFFSET,SHARD_COUNT=$GLOBAL_SHARD_COUNT"
  SBATCH_ARGS+=(
    --output="$SCRATCH_ROOT/slurm/%x-%j.out"
    --error="$SCRATCH_ROOT/slurm/%x-%j.err"
  )
fi

if [ -n "$ACCOUNT" ]; then
  SBATCH_ARGS+=(--account="$ACCOUNT")
fi

if [ -n "$QOS" ]; then
  SBATCH_ARGS+=(--qos="$QOS")
fi

sbatch "${SBATCH_ARGS[@]}" "$REPO_HOME/scripts/slurm_build_full_far_organic_from_cache.sbatch"
