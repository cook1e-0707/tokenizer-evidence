#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="${REPO_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SWEEP_CONFIG="${SWEEP_CONFIG:-configs/experiment/baselines/perinucleus_official/qwen_capacity_sweep__baseline_perinucleus_official.yaml}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_qwen_capacity_sweep}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
PARTITION="${PARTITION:-pomplun}"
QOS="${QOS:-pomplun}"
GRES="${GRES:-gpu:h200:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-240G}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"

mkdir -p "$SCRATCH_ROOT/slurm"
unset SBATCH_QOS
unset SLURM_QOS

sbatch \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --gres="$GRES" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --time="$TIME_LIMIT" \
  --output="$SCRATCH_ROOT/slurm/%x-%j.out" \
  --error="$SCRATCH_ROOT/slurm/%x-%j.err" \
  --export=ALL,REPO_HOME="$REPO_HOME",SWEEP_CONFIG="$SWEEP_CONFIG",SCRATCH_ROOT="$SCRATCH_ROOT",VENV_PATH="$VENV_PATH" \
  "$REPO_HOME/scripts/slurm_perinucleus_qwen_capacity_sweep.sbatch"
