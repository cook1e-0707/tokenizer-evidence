#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="${REPO_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMPARISON_CONFIG="${COMPARISON_CONFIG:-configs/experiment/comparison/far_utility_compute_ours.yaml}"
RUN_MODE="${RUN_MODE:-write-plan}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/comparison/far_utility_compute}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
PARTITION="${PARTITION:-Intel}"
GRES="${GRES:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-16G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"

mkdir -p "$SCRATCH_ROOT/slurm"

SBATCH_ARGS=(
  --partition="$PARTITION" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --time="$TIME_LIMIT" \
  --output="$SCRATCH_ROOT/slurm/%x-%j.out" \
  --error="$SCRATCH_ROOT/slurm/%x-%j.err" \
  --export=ALL,REPO_HOME="$REPO_HOME",COMPARISON_CONFIG="$COMPARISON_CONFIG",RUN_MODE="$RUN_MODE",SCRATCH_ROOT="$SCRATCH_ROOT",VENV_PATH="$VENV_PATH"
)

if [ -n "$GRES" ]; then
  SBATCH_ARGS+=(--gres="$GRES")
fi

sbatch "${SBATCH_ARGS[@]}" "$REPO_HOME/scripts/slurm_matched_far_utility_compute.sbatch"
