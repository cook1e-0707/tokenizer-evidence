#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="${REPO_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
UTILITY_CONFIG="${UTILITY_CONFIG:-configs/experiment/baselines/perinucleus_official/qwen_candidate_utility__baseline_perinucleus_official.yaml}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_qwen_candidate_utility}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
PARTITION="${PARTITION:-pomplun}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
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
  --export=HOME,REPO_HOME="$REPO_HOME",UTILITY_CONFIG="$UTILITY_CONFIG",SCRATCH_ROOT="$SCRATCH_ROOT",VENV_PATH="$VENV_PATH"
)

if [ -n "$ACCOUNT" ]; then
  SBATCH_ARGS+=(--account="$ACCOUNT")
fi

if [ -n "$QOS" ]; then
  SBATCH_ARGS+=(--qos="$QOS")
fi

sbatch "${SBATCH_ARGS[@]}" "$REPO_HOME/scripts/slurm_perinucleus_qwen_candidate_utility.sbatch"
