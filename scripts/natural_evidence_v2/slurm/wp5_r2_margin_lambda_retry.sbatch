#!/bin/bash
#SBATCH --job-name=nat-ev-v2-wp5r2
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp5r2-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp5r2-%j.err

# WP5-R2: Margin lambda sweep retry
# Root cause: WP5 margin loss was only 7% of total loss, adapter learned
# rank ordering but not mass shift. Increase margin_lambda 5→30, steps 64→256.

set -euo pipefail

export REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
export PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
export VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
export HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"
export RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"

# WP5-R2 overrides
export PLAN_DIR="${PLAN_DIR:-results/natural_evidence_v2/status/wp5_teacher_forced_launch_plan_20260509_0226}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/wp5_r2_teacher_forced_train_and_score_${SLURM_JOB_ID:-manual}}"
export PRIMARY_BANK="${PRIMARY_BANK:-results/natural_evidence_v2/buckets/qwen_v2_primary_2way_bank.jsonl}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export MAX_ROWS="${MAX_ROWS:-512}"
export SCORE_MAX_ROWS="${SCORE_MAX_ROWS:-8192}"
export MAX_STEPS="${MAX_STEPS:-256}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export PROTECTED_LR="${PROTECTED_LR:-1e-4}"
export TASK_ONLY_LR="${TASK_ONLY_LR:-1e-4}"
export LORA_R="${LORA_R:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export MARGIN_LAMBDA="${MARGIN_LAMBDA:-30.0}"
export MARGIN_TAU="${MARGIN_TAU:-0.15}"
export DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$OUTPUT_DIR"

set +u
if [ -f /etc/profile ]; then
  . /etc/profile
fi
if [ -f "$VENV_PATH/bin/activate" ]; then
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
echo "wp5_r2_changes=margin_lambda=30.0, max_steps=256, lr=1e-4"
echo "claim_note=natural_evidence_v2 WP5-R2 margin lambda retry; no E2E, no payload recovery, no FAR, no paper positive claim"
echo "plan_dir=$PLAN_DIR"
echo "output_dir=$OUTPUT_DIR"
echo "model_name=$MODEL_NAME"
echo "max_rows=$MAX_ROWS"
echo "max_steps=$MAX_STEPS"
echo "margin_lambda=$MARGIN_LAMBDA"
echo "learning_rate=$PROTECTED_LR"

for required in "$PRIMARY_BANK" "$PROTECTED_ROWS" "$TASK_ONLY_ROWS" "$SCORE_ROWS" "$PLAN_SUMMARY"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

if [ -e "$OUTPUT_DIR/wp5_teacher_forced_bucket_mass_summary.json" ]; then
  echo "WP5_R2_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

PROTECTED_OUT="$OUTPUT_DIR/protected_train"
TASK_ONLY_OUT="$OUTPUT_DIR/task_only_train"
SCORE_OUT="$OUTPUT_DIR/teacher_forced_score"

CUDA_FLAG="--require-cuda"

echo "=== WP5-R2: Training protected arm (margin_lambda=30.0) ==="
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
  $CUDA_FLAG

echo "=== WP5-R2: Training task-only arm ==="
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
  $CUDA_FLAG

echo "=== WP5-R2: Scoring teacher-forced bucket mass ==="
"$PYTHON" scripts/natural_evidence_v2/score_wp5_teacher_forced_bucket_mass.py \
  --score-rows "$SCORE_ROWS" \
  --primary-bank "$PRIMARY_BANK" \
  --output-dir "$SCORE_OUT" \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --max-rows "$SCORE_MAX_ROWS" \
  --batch-size 8 \
  --protected-adapter "$PROTECTED_OUT/adapter" \
  --task-only-adapter "$TASK_ONLY_OUT/adapter" \
  $CUDA_FLAG

echo "WP5-R2 teacher-forced train-and-score wrapper complete: $OUTPUT_DIR"
