#!/bin/bash
#SBATCH --job-name=nat-ev-v2-wp5tf
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
#SBATCH --time=08:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp5tf-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp5tf-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"

PLAN_DIR="${PLAN_DIR:-results/natural_evidence_v2/status/wp5_teacher_forced_launch_plan_20260509_0226}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/wp5_teacher_forced_train_and_score_${SLURM_JOB_ID:-manual}}"
PRIMARY_BANK="${PRIMARY_BANK:-results/natural_evidence_v2/buckets/qwen_v2_primary_2way_bank.jsonl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MAX_ROWS="${MAX_ROWS:-512}"
SCORE_MAX_ROWS="${SCORE_MAX_ROWS:-8192}"
MAX_STEPS="${MAX_STEPS:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
PROTECTED_LR="${PROTECTED_LR:-5e-5}"
TASK_ONLY_LR="${TASK_ONLY_LR:-5e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
MARGIN_LAMBDA="${MARGIN_LAMBDA:-5.0}"
MARGIN_TAU="${MARGIN_TAU:-0.15}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"

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

PROTECTED_ROWS="$PLAN_DIR/wp5_protected_training_rows.jsonl"
TASK_ONLY_ROWS="$PLAN_DIR/wp5_task_only_training_rows.jsonl"
SCORE_ROWS="$PLAN_DIR/wp5_teacher_forced_score_rows.jsonl"
PLAN_SUMMARY="$PLAN_DIR/wp5_teacher_forced_launch_plan_summary.json"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=natural_evidence_v2 WP5 teacher-forced train-and-score only; no E2E, no payload recovery, no FAR, no paper positive claim"
echo "plan_dir=$PLAN_DIR"
echo "output_dir=$OUTPUT_DIR"
echo "model_name=$MODEL_NAME"
echo "tokenizer_name=$TOKENIZER_NAME"
echo "max_rows=$MAX_ROWS"
echo "score_max_rows=$SCORE_MAX_ROWS"
echo "max_steps=$MAX_STEPS"
echo "dry_run_only=$DRY_RUN_ONLY"

for required in "$PRIMARY_BANK" "$PROTECTED_ROWS" "$TASK_ONLY_ROWS" "$SCORE_ROWS" "$PLAN_SUMMARY"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

if [ -e "$OUTPUT_DIR/wp5_teacher_forced_bucket_mass_summary.json" ]; then
  echo "WP5_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

PROTECTED_OUT="$OUTPUT_DIR/protected_train"
TASK_ONLY_OUT="$OUTPUT_DIR/task_only_train"
SCORE_OUT="$OUTPUT_DIR/teacher_forced_score"

DRY_RUN_FLAG=""
CUDA_FLAG="--require-cuda"
if [ "$DRY_RUN_ONLY" = "1" ]; then
  DRY_RUN_FLAG="--dry-run"
  CUDA_FLAG=""
fi

"$PYTHON" scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py \
  --train-rows "$PROTECTED_ROWS" \
  --primary-bank "$PRIMARY_BANK" \
  --output-dir "$PROTECTED_OUT" \
  --arm protected \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --max-rows "$MAX_ROWS" \
  --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$PROTECTED_LR" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --margin-lambda "$MARGIN_LAMBDA" \
  --margin-tau "$MARGIN_TAU" \
  $CUDA_FLAG \
  $DRY_RUN_FLAG

"$PYTHON" scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py \
  --train-rows "$TASK_ONLY_ROWS" \
  --primary-bank "$PRIMARY_BANK" \
  --output-dir "$TASK_ONLY_OUT" \
  --arm task_only \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --max-rows "$MAX_ROWS" \
  --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$TASK_ONLY_LR" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  $CUDA_FLAG \
  $DRY_RUN_FLAG

SCORE_ADAPTER_ARGS=""
if [ "$DRY_RUN_ONLY" != "1" ]; then
  SCORE_ADAPTER_ARGS="--protected-adapter $PROTECTED_OUT/adapter --task-only-adapter $TASK_ONLY_OUT/adapter"
fi

"$PYTHON" scripts/natural_evidence_v2/score_wp5_teacher_forced_bucket_mass.py \
  --score-rows "$SCORE_ROWS" \
  --primary-bank "$PRIMARY_BANK" \
  --output-dir "$SCORE_OUT" \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --max-rows "$SCORE_MAX_ROWS" \
  --batch-size 8 \
  $CUDA_FLAG \
  $DRY_RUN_FLAG \
  $SCORE_ADAPTER_ARGS

echo "WP5 teacher-forced train-and-score wrapper complete: $OUTPUT_DIR"
