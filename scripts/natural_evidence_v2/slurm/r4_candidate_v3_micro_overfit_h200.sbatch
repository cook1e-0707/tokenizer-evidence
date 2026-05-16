#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4mof
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=pomplun
#SBATCH --account=cs_yinxin.wan
#SBATCH --qos=pomplun
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=30-00:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4mof-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4mof-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

SPLIT_DIR="${SPLIT_DIR:-results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_split_20260514_codex}"
TRAIN_ROWS="${TRAIN_ROWS:-$SPLIT_DIR/train_rows.jsonl}"
HELDOUT_ROWS="${HELDOUT_ROWS:-$SPLIT_DIR/heldout_rows.jsonl}"
SCORE_ROWS="${SCORE_ROWS:-$SPLIT_DIR/score_rows.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/r4_candidate_v3_micro_overfit_${SLURM_JOB_ID:-manual}}"
PROTECTED_TRAIN_DIR="${PROTECTED_TRAIN_DIR:-$OUTPUT_DIR/protected_micro_overfit_train}"
SCORE_OUTPUT_DIR="${SCORE_OUTPUT_DIR:-$OUTPUT_DIR/teacher_forced_score}"
TASK_ONLY_ADAPTER="${TASK_ONLY_ADAPTER:-$RUN_ROOT/status/wp5_r2_teacher_forced_train_and_score_851481/task_only_train/adapter}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-512}"
MAX_SCORE_ROWS="${MAX_SCORE_ROWS:-8192}"
MAX_STEPS="${MAX_STEPS:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MARGIN_LAMBDA="${MARGIN_LAMBDA:-5.0}"
MARGIN_TAU="${MARGIN_TAU:-0.15}"
SURFACE_MARGIN_LOSS_MODE="${SURFACE_MARGIN_LOSS_MODE:-mass_relu}"
TARGET_MASS_FLOOR="${TARGET_MASS_FLOOR:-0.20}"
TARGET_MASS_FLOOR_LAMBDA="${TARGET_MASS_FLOOR_LAMBDA:-5.0}"
TARGET_MASS_CEILING="${TARGET_MASS_CEILING:-0.45}"
TARGET_MASS_CEILING_LAMBDA="${TARGET_MASS_CEILING_LAMBDA:-5.0}"
STRATUM_WEIGHTING_MODE="${STRATUM_WEIGHTING_MODE:-r4_candidate_v3_failed_surface}"
STRATUM_WEIGHT_MAX="${STRATUM_WEIGHT_MAX:-3.0}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
VALIDATE_PLAN_ONLY="${VALIDATE_PLAN_ONLY:-0}"

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
echo "claim_note=natural_evidence_v2 R4 candidate v3 micro-overfit protected-only teacher-forced training/scoring; no generation, no Qwen E2E rerun, no Llama, no same-family null, no sanitizer, no FAR, no payload diversity, no paper positive claim"
echo "partition=pomplun"
echo "account=cs_yinxin.wan"
echo "qos=pomplun"
echo "gres=gpu:h200:1"
echo "time_limit=30-00:00:00"
echo "split_dir=$SPLIT_DIR"
echo "train_rows=$TRAIN_ROWS"
echo "heldout_rows=$HELDOUT_ROWS"
echo "score_rows=$SCORE_ROWS"
echo "output_dir=$OUTPUT_DIR"
echo "surface_margin_loss_mode=$SURFACE_MARGIN_LOSS_MODE"
echo "target_mass_floor=$TARGET_MASS_FLOOR"
echo "target_mass_floor_lambda=$TARGET_MASS_FLOOR_LAMBDA"
echo "target_mass_ceiling=$TARGET_MASS_CEILING"
echo "target_mass_ceiling_lambda=$TARGET_MASS_CEILING_LAMBDA"
echo "stratum_weighting_mode=$STRATUM_WEIGHTING_MODE"
echo "stratum_weight_max=$STRATUM_WEIGHT_MAX"
echo "validate_plan_only=$VALIDATE_PLAN_ONLY"

for required in "$SPLIT_DIR/split_summary.json" "$TRAIN_ROWS" "$HELDOUT_ROWS" "$SCORE_ROWS" \
  "scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py" \
  "scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

"$PYTHON" - "$SPLIT_DIR/split_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("schema_name") != "natural_evidence_v2_r4_candidate_v3_micro_overfit_split_summary_v1":
    raise SystemExit("split summary schema mismatch")
if summary.get("train_heldout_overlap_count") != 0:
    raise SystemExit("split summary records train/heldout overlap")
for key in (
    "training_allowed",
    "generation_allowed",
    "qwen_e2e_rerun_allowed",
    "llama_allowed",
    "same_family_null_allowed",
    "sanitizer_benchmark_allowed",
    "far_aggregation_allowed",
    "paper_claim_allowed",
    "allowlist_enablement_allowed",
    "slurm_submission_allowed",
    "remote_cpu_gpu_work_allowed",
):
    if summary.get(key) is not False:
        raise SystemExit(f"split summary locked-action flag not false: {key}")
PY

if [ "$VALIDATE_PLAN_ONLY" = "1" ]; then
  echo "VALIDATE_PLAN_ONLY=1: exiting before model/tokenizer loading, CUDA initialization, adapter loading, training, scoring, remote sync, or Slurm submission."
  exit 0
fi

for planned in "$PROTECTED_TRAIN_DIR" "$SCORE_OUTPUT_DIR" "$OUTPUT_DIR/COMPLETED"; do
  if [ -e "$planned" ]; then
    echo "R4_CANDIDATE_V3_MICRO_OVERFIT_OUTPUT_EXISTS_REFUSING_OVERWRITE: $planned" >&2
    exit 74
  fi
done
if [ ! -s "$TASK_ONLY_ADAPTER/adapter_config.json" ]; then
  echo "TASK_ONLY_ADAPTER_MISSING_OR_EMPTY: $TASK_ONLY_ADAPTER/adapter_config.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

"$PYTHON" scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py \
  --train-rows "$TRAIN_ROWS" \
  --output-dir "$PROTECTED_TRAIN_DIR" \
  --arm protected \
  --row-mode r4_prefix_native_surface \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --max-rows "$MAX_TRAIN_ROWS" \
  --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --margin-lambda "$MARGIN_LAMBDA" \
  --margin-tau "$MARGIN_TAU" \
  --surface-margin-loss-mode "$SURFACE_MARGIN_LOSS_MODE" \
  --target-mass-floor "$TARGET_MASS_FLOOR" \
  --target-mass-floor-lambda "$TARGET_MASS_FLOOR_LAMBDA" \
  --target-mass-ceiling "$TARGET_MASS_CEILING" \
  --target-mass-ceiling-lambda "$TARGET_MASS_CEILING_LAMBDA" \
  --stratum-weighting-mode "$STRATUM_WEIGHTING_MODE" \
  --stratum-weight-max "$STRATUM_WEIGHT_MAX" \
  --max-length "$MAX_LENGTH" \
  --require-cuda

"$PYTHON" scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py \
  --score-rows "$SCORE_ROWS" \
  --output-dir "$SCORE_OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --protected-adapter "$PROTECTED_TRAIN_DIR/adapter" \
  --task-only-adapter "$TASK_ONLY_ADAPTER" \
  --protected-adapter-gains "1.0" \
  --max-rows "$MAX_SCORE_ROWS" \
  --batch-size "$SCORE_BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --require-cuda

date -u +"%Y-%m-%dT%H:%M:%SZ" >"$OUTPUT_DIR/COMPLETED"
echo "R4 candidate v3 micro-overfit training/scoring complete: $OUTPUT_DIR"
