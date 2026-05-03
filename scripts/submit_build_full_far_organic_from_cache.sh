#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="${REPO_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FULL_FAR_CONFIG="${FULL_FAR_CONFIG:-configs/experiment/comparison/full_far_payload_claim.yaml}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/comparison/full_far_payload_claim}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH_ROOT/shards/organic-prompt-cache-10way}"
ROW_SHARD_DIR="${ROW_SHARD_DIR:-$SCRATCH_ROOT/shards/organic-prompts-10way-from-cache}"
EXPECTED_SHARD_COUNT="${EXPECTED_SHARD_COUNT:-10}"
FORCE="${FORCE:-1}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
PARTITION="${PARTITION:-Intel6240}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-120G}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"

mkdir -p "$SCRATCH_ROOT/slurm" "$ROW_SHARD_DIR"
unset SBATCH_QOS
unset SLURM_QOS
unset SBATCH_ACCOUNT

SBATCH_ARGS=(
  --partition="$PARTITION"
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="$MEM"
  --time="$TIME_LIMIT"
  --output="$SCRATCH_ROOT/slurm/%x-%j.out"
  --error="$SCRATCH_ROOT/slurm/%x-%j.err"
  --export=HOME,USER,LOGNAME,REPO_HOME="$REPO_HOME",FULL_FAR_CONFIG="$FULL_FAR_CONFIG",SCRATCH_ROOT="$SCRATCH_ROOT",CACHE_DIR="$CACHE_DIR",ROW_SHARD_DIR="$ROW_SHARD_DIR",EXPECTED_SHARD_COUNT="$EXPECTED_SHARD_COUNT",FORCE="$FORCE",VENV_PATH="$VENV_PATH",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$HF_HOME"
)

if [ -n "$ACCOUNT" ]; then
  SBATCH_ARGS+=(--account="$ACCOUNT")
fi

if [ -n "$QOS" ]; then
  SBATCH_ARGS+=(--qos="$QOS")
fi

sbatch "${SBATCH_ARGS[@]}" "$REPO_HOME/scripts/slurm_build_full_far_organic_from_cache.sbatch"
