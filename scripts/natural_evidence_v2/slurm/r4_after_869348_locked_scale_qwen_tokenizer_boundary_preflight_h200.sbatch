#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4lTok
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=pomplun
#SBATCH --account=cs_yinxin.wan
#SBATCH --qos=pomplun
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4lTok-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4lTok-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

SCORE_ROWS="${SCORE_ROWS:-results/natural_evidence_v2/status/r4_after_869348_global_unique_locked_scale_row_bank_plan_20260518/row_allocation_rows.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/r4_after_869348_locked_scale_qwen_tokenizer_boundary_preflight_${SLURM_JOB_ID:-manual}}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MAX_ROWS="${MAX_ROWS:-98304}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm"

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
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\r\n' <"$HF_TOKEN_FILE")"
fi

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=natural_evidence_v2 R4 after-869348 held-out locked-scale actual Qwen tokenizer boundary preflight only; no model forward, no scoring, no generation, no training, no Llama, no same-family null, no sanitizer, no FAR, no paper claim"
echo "partition=pomplun"
echo "account=cs_yinxin.wan"
echo "qos=pomplun"
echo "gres=gpu:h200:1"
echo "score_rows=$SCORE_ROWS"
echo "output_dir=$OUTPUT_DIR"
echo "tokenizer_name=$TOKENIZER_NAME"
echo "max_rows=$MAX_ROWS"
echo "model_forward_started=false"
echo "scoring_started=false"
echo "generation_started=false"
echo "training_started=false"

for required in "$SCORE_ROWS" \
  "scripts/natural_evidence_v2/preflight_r4_surface_tokenizer_boundary.py" \
  "scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

if [ -e "$OUTPUT_DIR" ]; then
  echo "R4_AFTER_869348_LOCKED_SCALE_QWEN_TOKENIZER_PREFLIGHT_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" scripts/natural_evidence_v2/preflight_r4_surface_tokenizer_boundary.py \
  --score-rows "$SCORE_ROWS" \
  --output-dir "$OUTPUT_DIR" \
  --max-rows "$MAX_ROWS" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --run-qwen-tokenizer

echo "R4 after-869348 locked-scale actual Qwen tokenizer boundary preflight complete: $OUTPUT_DIR"
