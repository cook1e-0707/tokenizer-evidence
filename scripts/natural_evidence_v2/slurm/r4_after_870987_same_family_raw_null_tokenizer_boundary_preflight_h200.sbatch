#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4sfTok
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=pomplun
#SBATCH --account=cs_yinxin.wan
#SBATCH --qos=pomplun
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=30-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4sfTok-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4sfTok-%A_%a.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
if [ ! -d "$REPO_HOME" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  CANDIDATE_REPO_HOME="$(cd "$SCRIPT_DIR/../../.." && pwd)"
  if [ -d "$CANDIDATE_REPO_HOME/.git" ]; then
    REPO_HOME="$CANDIDATE_REPO_HOME"
  fi
fi
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"
ROUTE_CONFIG="${ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_870987_same_family_raw_null_tokenizer_preflight_route.yaml}"
ROUTE_VALIDATOR="${ROUTE_VALIDATOR:-scripts/natural_evidence_v2/validate_r4_after_870987_same_family_raw_null_tokenizer_route.py}"
PLAN_ONLY="${PLAN_ONLY:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm"

case "$TASK_ID" in
  0)
    TOKENIZER_NAME="Qwen/Qwen2.5-3B-Instruct"
    MODEL_SLUG="qwen2_5_3b_instruct_raw"
    ;;
  1)
    TOKENIZER_NAME="Qwen/Qwen2.5-7B-Instruct"
    MODEL_SLUG="qwen2_5_7b_instruct_raw"
    ;;
  2)
    TOKENIZER_NAME="Qwen/Qwen2.5-14B-Instruct"
    MODEL_SLUG="qwen2_5_14b_instruct_raw"
    ;;
  *)
    echo "INVALID_SAME_FAMILY_TOKENIZER_TASK_ID: $TASK_ID" >&2
    exit 2
    ;;
esac

SCORE_ROWS="${SCORE_ROWS:-results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_row_bank_v2_plan_20260521/row_allocation_rows.jsonl}"
MAX_ROWS="${MAX_ROWS:-262144}"
DEFAULT_OUTPUT_LABEL="r4_after_870987_same_family_raw_null_tokenizer_preflight"
if [[ "$ROUTE_CONFIG" == *"r4_after_877840_same_family_raw_null_v8"* ]]; then
  DEFAULT_OUTPUT_LABEL="r4_after_877840_same_family_raw_null_v8_tokenizer_preflight"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/status/${DEFAULT_OUTPUT_LABEL}_${SLURM_ARRAY_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$MODEL_SLUG}"
ROUTE_VALIDATION_OUTPUT_DIR="${ROUTE_VALIDATION_OUTPUT_DIR:-$OUTPUT_ROOT/route_validation_${MODEL_SLUG}}"

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
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-manual}"
echo "task_id=$TASK_ID"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=natural_evidence_v2 R4 after-870987 same-family raw-null tokenizer boundary preflight only; no model forward, no scoring, no generation, no training, no Llama, no sanitizer, no FAR, no paper claim"
echo "partition=pomplun"
echo "account=cs_yinxin.wan"
echo "qos=pomplun"
echo "gres=gpu:h200:1"
echo "tokenizer_name=$TOKENIZER_NAME"
echo "model_slug=$MODEL_SLUG"
echo "score_rows=$SCORE_ROWS"
echo "output_dir=$OUTPUT_DIR"
echo "route_validation_output_dir=$ROUTE_VALIDATION_OUTPUT_DIR"
echo "max_rows=$MAX_ROWS"
echo "plan_only=$PLAN_ONLY"
echo "model_forward_started=false"
echo "scoring_started=false"
echo "generation_started=false"
echo "training_started=false"

for required in "$SCORE_ROWS" \
  "$ROUTE_CONFIG" \
  "$ROUTE_VALIDATOR" \
  "scripts/natural_evidence_v2/preflight_r4_surface_tokenizer_boundary.py" \
  "scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

if [[ "$PLAN_ONLY" == "1" ]]; then
  ROUTE_VALIDATION_ARGS="${ROUTE_VALIDATION_EXTRA_ARGS:-}"
else
  ROUTE_VALIDATION_ARGS="${ROUTE_VALIDATION_EXTRA_ARGS:---skip-allowlist-state-check}"
fi

"$PYTHON" "$ROUTE_VALIDATOR" \
  --config "$ROUTE_CONFIG" \
  --output-dir "$ROUTE_VALIDATION_OUTPUT_DIR" \
  $ROUTE_VALIDATION_ARGS

if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "R4 same-family raw-null tokenizer wrapper plan-only validation passed"
  exit 0
fi

if [ -e "$OUTPUT_DIR" ]; then
  echo "R4_SAME_FAMILY_RAW_NULL_TOKENIZER_PREFLIGHT_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" scripts/natural_evidence_v2/preflight_r4_surface_tokenizer_boundary.py \
  --score-rows "$SCORE_ROWS" \
  --output-dir "$OUTPUT_DIR" \
  --max-rows "$MAX_ROWS" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --run-qwen-tokenizer

echo "R4 after-870987 same-family raw-null tokenizer boundary preflight complete: $OUTPUT_DIR"
