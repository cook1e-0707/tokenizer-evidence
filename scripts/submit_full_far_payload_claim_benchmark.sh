#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="${REPO_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FULL_FAR_CONFIG="${FULL_FAR_CONFIG:-configs/experiment/comparison/full_far_payload_claim.yaml}"
RUN_MODE="${RUN_MODE:-write-plan}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/comparison/full_far_payload_claim}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
ALLOW_SLOW_ORGANIC_ROW_EXECUTION="${ALLOW_SLOW_ORGANIC_ROW_EXECUTION:-0}"
PARTITION="${PARTITION:-pomplun}"
ACCOUNT="${ACCOUNT:-cs_yinxin.wan}"
QOS="${QOS:-pomplun}"
GRES="${GRES:-gpu:h200:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-240G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

mkdir -p "$SCRATCH_ROOT/slurm"
unset SBATCH_QOS
unset SLURM_QOS
unset SBATCH_ACCOUNT

SBATCH_ARGS=(
  --partition="$PARTITION"
  --gres="$GRES"
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="$MEM"
  --time="$TIME_LIMIT"
  --output="$SCRATCH_ROOT/slurm/%x-%j.out"
  --error="$SCRATCH_ROOT/slurm/%x-%j.err"
  --export=HOME,REPO_HOME="$REPO_HOME",FULL_FAR_CONFIG="$FULL_FAR_CONFIG",RUN_MODE="$RUN_MODE",SCRATCH_ROOT="$SCRATCH_ROOT",VENV_PATH="$VENV_PATH",ALLOW_SLOW_ORGANIC_ROW_EXECUTION="$ALLOW_SLOW_ORGANIC_ROW_EXECUTION"
)

if [ -n "$ACCOUNT" ]; then
  SBATCH_ARGS+=(--account="$ACCOUNT")
fi

if [ -n "$QOS" ]; then
  SBATCH_ARGS+=(--qos="$QOS")
fi

sbatch "${SBATCH_ARGS[@]}" "$REPO_HOME/scripts/slurm_full_far_payload_claim_benchmark.sbatch"
