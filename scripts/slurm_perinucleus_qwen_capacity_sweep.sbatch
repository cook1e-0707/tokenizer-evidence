#!/usr/bin/env bash
#SBATCH --job-name=perinuc-qwen-sweep
#SBATCH --partition=DGXA100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=48:00:00

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
SWEEP_CONFIG="${SWEEP_CONFIG:-configs/experiment/baselines/perinucleus_official/qwen_capacity_sweep__baseline_perinucleus_official.yaml}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_qwen_capacity_sweep}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

mkdir -p "$SCRATCH_ROOT/slurm"
cd "$REPO_HOME"

if [ -f "$HOME/.bashrc" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc"
fi

if [ -f "$VENV_PATH/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\r\n' <"$HF_TOKEN_FILE")"
fi

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "config=$SWEEP_CONFIG"
echo "scratch=$SCRATCH_ROOT"
echo "python=$(command -v python3)"

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

python3 scripts/run_perinucleus_qwen_capacity_sweep.py \
  --config "$SWEEP_CONFIG" \
  --force

python3 - <<'PY'
import json
from pathlib import Path

p = Path("results/processed/paper_stats/baseline_perinucleus_qwen_capacity_sweep_summary.json")
d = json.loads(p.read_text())
print("decision =", d["decision"])
print("candidate_found =", d["candidate_found"])
print("strong_candidate_found =", d["strong_candidate_found"])
print("run_root =", d["run_root"])
recommended = d.get("recommended_candidate")
print("recommended =", recommended.get("arm_id") if isinstance(recommended, dict) else None)
for arm in d["arms"]:
    final = arm["final"]
    print(
        arm["arm_id"],
        "exact=", final.get("exact_accuracy"),
        "rank=", final.get("target_rank_mean"),
        "prob=", final.get("target_probability_mean"),
    )
PY
