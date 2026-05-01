#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="${REPO_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FULL_FAR_CONFIG="${FULL_FAR_CONFIG:-configs/experiment/comparison/full_far_payload_claim.yaml}"
RUN_MODE="${RUN_MODE:-execute-organic-null-array}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/comparison/full_far_payload_claim}"
SHARD_COUNT="${SHARD_COUNT:-5}"
MAX_PARALLEL="${MAX_PARALLEL:-$SHARD_COUNT}"
SHARD_OUTPUT_DIR="${SHARD_OUTPUT_DIR:-$SCRATCH_ROOT/shards/organic-prompts}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
PARTITION="${PARTITION:-pomplun}"
ACCOUNT="${ACCOUNT:-cs_yinxin.wan}"
QOS="${QOS:-pomplun}"
GRES="${GRES:-gpu:h200:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-240G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

if ! [[ "$SHARD_COUNT" =~ ^[0-9]+$ ]] || [ "$SHARD_COUNT" -le 0 ]; then
  echo "SHARD_COUNT must be a positive integer; got $SHARD_COUNT" >&2
  exit 2
fi

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [ "$MAX_PARALLEL" -le 0 ]; then
  echo "MAX_PARALLEL must be a positive integer; got $MAX_PARALLEL" >&2
  exit 2
fi

mkdir -p "$SCRATCH_ROOT/slurm" "$SHARD_OUTPUT_DIR"
unset SBATCH_QOS
unset SLURM_QOS
unset SBATCH_ACCOUNT

ARRAY_SPEC="0-$((SHARD_COUNT - 1))%$MAX_PARALLEL"
SBATCH_ARGS=(
  --partition="$PARTITION"
  --gres="$GRES"
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="$MEM"
  --time="$TIME_LIMIT"
  --array="$ARRAY_SPEC"
  --output="$SCRATCH_ROOT/slurm/%x-%A_%a.out"
  --error="$SCRATCH_ROOT/slurm/%x-%A_%a.err"
  --export=HOME,REPO_HOME="$REPO_HOME",FULL_FAR_CONFIG="$FULL_FAR_CONFIG",RUN_MODE="$RUN_MODE",SCRATCH_ROOT="$SCRATCH_ROOT",SHARD_COUNT="$SHARD_COUNT",SHARD_OUTPUT_DIR="$SHARD_OUTPUT_DIR",VENV_PATH="$VENV_PATH"
)

if [ -n "$ACCOUNT" ]; then
  SBATCH_ARGS+=(--account="$ACCOUNT")
fi

if [ -n "$QOS" ]; then
  SBATCH_ARGS+=(--qos="$QOS")
fi

sbatch "${SBATCH_ARGS[@]}" "$REPO_HOME/scripts/slurm_full_far_payload_claim_benchmark_array.sbatch"
