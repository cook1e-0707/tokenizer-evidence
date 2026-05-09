#!/bin/bash
#SBATCH --job-name=nat-ev-v2-wp3ctxm
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp3ctxm-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp3ctxm-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"

SCORE_PLAN="${SCORE_PLAN:-results/natural_evidence_v2/status/wp3_context_mass_plan_prefix_boundary_repair_20260509_0024/qwen_v2_wp3_context_mass_score_plan.jsonl}"
CONFIG="${CONFIG:-configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/wp3_context_mass_score_${SLURM_JOB_ID:-manual}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$OUTPUT_DIR"

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

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=natural_evidence_v2 WP3 context-specific base-Qwen mass scoring only; no training, no generation, no E2E, no FAR, no positive claim"
echo "score_plan=$SCORE_PLAN"
echo "config=$CONFIG"
echo "output_dir=$OUTPUT_DIR"
echo "model_name=$MODEL_NAME"
echo "tokenizer_name=$TOKENIZER_NAME"

for required in "$SCORE_PLAN" "$CONFIG"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

summary_json="$OUTPUT_DIR/qwen_v2_wp3_context_mass_score_summary.json"
mass_json="$OUTPUT_DIR/qwen_v2_wp3_context_mass_artifact.json"
audit_json="$OUTPUT_DIR/qwen_v2_wp3_context_mass_audit.json"
if [ -e "$summary_json" ] || [ -e "$mass_json" ] || [ -e "$audit_json" ]; then
  echo "WP3_CONTEXT_MASS_ARTIFACT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" -m py_compile scripts/natural_evidence_v2/score_wp3_context_mass.py

"$PYTHON" scripts/natural_evidence_v2/score_wp3_context_mass.py \
  --score-plan "$SCORE_PLAN" \
  --config "$CONFIG" \
  --validate-plan-only

"$PYTHON" scripts/natural_evidence_v2/score_wp3_context_mass.py \
  --score-plan "$SCORE_PLAN" \
  --config "$CONFIG" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --validate-tokenizer-only \
  --skip-invalid-tokenization

"$PYTHON" scripts/natural_evidence_v2/score_wp3_context_mass.py \
  --score-plan "$SCORE_PLAN" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --skip-invalid-tokenization \
  --require-cuda
