#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="${REPO_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FULL_FAR_CONFIG="${FULL_FAR_CONFIG:-configs/experiment/comparison/full_far_payload_claim.yaml}"
RUN_MODE="${RUN_MODE:-generate-organic-cache-array}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/comparison/full_far_payload_claim}"
LOCAL_SHARD_COUNT="${LOCAL_SHARD_COUNT:-${SHARD_COUNT:-5}}"
GLOBAL_SHARD_COUNT="${GLOBAL_SHARD_COUNT:-$LOCAL_SHARD_COUNT}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-$LOCAL_SHARD_COUNT}"
if [ "$RUN_MODE" = "generate-non-owner-cache-array" ]; then
  DEFAULT_SHARD_OUTPUT_DIR="$SCRATCH_ROOT/shards/non-owner-probes"
  DEFAULT_CACHE_OUTPUT_DIR="$SCRATCH_ROOT/shards/non-owner-probe-cache"
else
  DEFAULT_SHARD_OUTPUT_DIR="$SCRATCH_ROOT/shards/organic-prompts"
  DEFAULT_CACHE_OUTPUT_DIR="$SCRATCH_ROOT/shards/organic-prompt-cache"
fi
SHARD_OUTPUT_DIR="${SHARD_OUTPUT_DIR:-$DEFAULT_SHARD_OUTPUT_DIR}"
CACHE_OUTPUT_DIR="${CACHE_OUTPUT_DIR:-$DEFAULT_CACHE_OUTPUT_DIR}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
ALLOW_SLOW_ORGANIC_ROW_EXECUTION="${ALLOW_SLOW_ORGANIC_ROW_EXECUTION:-0}"
PARTITION="${PARTITION:-pomplun}"
ACCOUNT="${ACCOUNT:-cs_yinxin.wan}"
QOS="${QOS:-pomplun}"
GRES="${GRES:-gpu:h200:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-240G}"
TIME_LIMIT="${TIME_LIMIT-30-00:00:00}"

if ! [[ "$LOCAL_SHARD_COUNT" =~ ^[0-9]+$ ]] || [ "$LOCAL_SHARD_COUNT" -le 0 ]; then
  echo "LOCAL_SHARD_COUNT/SHARD_COUNT must be a positive integer; got $LOCAL_SHARD_COUNT" >&2
  exit 2
fi

if ! [[ "$GLOBAL_SHARD_COUNT" =~ ^[0-9]+$ ]] || [ "$GLOBAL_SHARD_COUNT" -le 0 ]; then
  echo "GLOBAL_SHARD_COUNT must be a positive integer; got $GLOBAL_SHARD_COUNT" >&2
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

mkdir -p "$SCRATCH_ROOT/slurm" "$SHARD_OUTPUT_DIR" "$CACHE_OUTPUT_DIR"
unset SBATCH_QOS
unset SLURM_QOS
unset SBATCH_ACCOUNT

ARRAY_SPEC="0-$((LOCAL_SHARD_COUNT - 1))%$MAX_PARALLEL"
SBATCH_ARGS=(
  --partition="$PARTITION"
  --gres="$GRES"
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="$MEM"
  --array="$ARRAY_SPEC"
  --output="$SCRATCH_ROOT/slurm/%x-%A_%a.out"
  --error="$SCRATCH_ROOT/slurm/%x-%A_%a.err"
  --export=HOME,REPO_HOME="$REPO_HOME",FULL_FAR_CONFIG="$FULL_FAR_CONFIG",RUN_MODE="$RUN_MODE",SCRATCH_ROOT="$SCRATCH_ROOT",LOCAL_SHARD_COUNT="$LOCAL_SHARD_COUNT",GLOBAL_SHARD_COUNT="$GLOBAL_SHARD_COUNT",SHARD_OFFSET="$SHARD_OFFSET",SHARD_OUTPUT_DIR="$SHARD_OUTPUT_DIR",CACHE_OUTPUT_DIR="$CACHE_OUTPUT_DIR",CHECKPOINT_INTERVAL="$CHECKPOINT_INTERVAL",VENV_PATH="$VENV_PATH",ALLOW_SLOW_ORGANIC_ROW_EXECUTION="$ALLOW_SLOW_ORGANIC_ROW_EXECUTION"
)

if [ -n "$TIME_LIMIT" ]; then
  SBATCH_ARGS+=(--time="$TIME_LIMIT")
fi

if [ -n "$ACCOUNT" ]; then
  SBATCH_ARGS+=(--account="$ACCOUNT")
fi

if [ -n "$QOS" ]; then
  SBATCH_ARGS+=(--qos="$QOS")
fi

sbatch "${SBATCH_ARGS[@]}" "$REPO_HOME/scripts/slurm_full_far_payload_claim_benchmark_array.sbatch"
