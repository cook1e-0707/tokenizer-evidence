#!/usr/bin/env bash
#SBATCH --job-name=full-far-claim
#SBATCH --partition=pomplun
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
FULL_FAR_CONFIG="${FULL_FAR_CONFIG:-configs/experiment/comparison/full_far_payload_claim.yaml}"
RUN_MODE="${RUN_MODE:-write-plan}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/comparison/full_far_payload_claim}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

mkdir -p "$SCRATCH_ROOT/slurm"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
echo "run_mode=$RUN_MODE"
echo "python=$(command -v python3)"

python3 - <<'PY'
import os

try:
    import torch
except Exception as exc:
    print("torch import failed:", type(exc).__name__, exc)
else:
    print("torch =", torch.__version__)
    print("torch cuda =", torch.version.cuda)
    print("cuda available =", torch.cuda.is_available())
    print("device count =", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if torch.cuda.is_available():
        print("device name =", torch.cuda.get_device_name(0))
PY

case "$RUN_MODE" in
  dry-run)
    python3 scripts/run_full_far_payload_claim_benchmark.py \
      --config "$FULL_FAR_CONFIG" \
      --dry-run
    ;;
  write-plan)
    python3 scripts/run_full_far_payload_claim_benchmark.py \
      --config "$FULL_FAR_CONFIG" \
      --write-plan \
      --force
    ;;
  execute)
    python3 scripts/run_full_far_payload_claim_benchmark.py \
      --config "$FULL_FAR_CONFIG" \
      --execute \
      --force
    ;;
  execute-registered-null)
    python3 scripts/run_full_far_payload_claim_benchmark.py \
      --config "$FULL_FAR_CONFIG" \
      --execute \
      --fresh-null-mode registered-probes \
      --force
    ;;
  execute-organic-null)
    python3 scripts/run_full_far_payload_claim_benchmark.py \
      --config "$FULL_FAR_CONFIG" \
      --execute \
      --fresh-null-mode organic-prompts \
      --force
    ;;
  execute-registered-and-organic-null)
    python3 scripts/run_full_far_payload_claim_benchmark.py \
      --config "$FULL_FAR_CONFIG" \
      --execute \
      --fresh-null-mode registered-and-organic \
      --force
    ;;
  *)
    echo "Unsupported RUN_MODE=$RUN_MODE; expected dry-run, write-plan, execute, execute-registered-null, execute-organic-null, or execute-registered-and-organic-null" >&2
    exit 2
    ;;
esac
