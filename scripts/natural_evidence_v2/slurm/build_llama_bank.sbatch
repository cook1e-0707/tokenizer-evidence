#!/bin/bash
#SBATCH --job-name=llama-bank
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/llama_migration/slurm/llama-bank-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/llama_migration/slurm/llama-bank-%j.err

set -euo pipefail

export REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
export PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
export VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
export HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

cd "$REPO_HOME"

set +u
if [ -f /etc/profile ]; then . /etc/profile; fi
if [ -f "$VENV_PATH/bin/activate" ]; then source "$VENV_PATH/bin/activate"; fi
set -u

export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\r\n' <"$HF_TOKEN_FILE")"
fi

echo "Building Llama-3.1-8B tokenizer bucket bank..."
"$PYTHON" scripts/natural_evidence_v2/build_llama_v2_bucket_bank.py
echo "Done: $SLURM_JOB_ID"
