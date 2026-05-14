#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4gain
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4gain-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4gain-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

SCORE_ROWS="${SCORE_ROWS:-results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_v3_20260513/r4_prefix_native_surface_probe_rows_v3.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/r4_candidate_v3_adapter_gain_sweep_${SLURM_JOB_ID:-manual}}"
PROTECTED_ADAPTER="${PROTECTED_ADAPTER:-$RUN_ROOT/status/wp5_r2_teacher_forced_train_and_score_851481/protected_train/adapter}"
TASK_ONLY_ADAPTER="${TASK_ONLY_ADAPTER:-$RUN_ROOT/status/wp5_r2_teacher_forced_train_and_score_851481/task_only_train/adapter}"
PROTECTED_ADAPTER_GAINS="${PROTECTED_ADAPTER_GAINS:-0.0,0.5,1.0,1.5,2.0,3.0,4.0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_ROWS="${MAX_ROWS:-8192}"
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
echo "claim_note=natural_evidence_v2 R4 candidate v3 protected-adapter gain sweep teacher-forced scoring only; no generation, no training, no Llama, no same-family null, no sanitizer, no FAR, no paper positive claim"
echo "partition=pomplun"
echo "account=cs_yinxin.wan"
echo "qos=pomplun"
echo "gres=gpu:h200:1"
echo "score_rows=$SCORE_ROWS"
echo "output_dir=$OUTPUT_DIR"
echo "protected_adapter=$PROTECTED_ADAPTER"
echo "task_only_adapter=$TASK_ONLY_ADAPTER"
echo "protected_adapter_gains=$PROTECTED_ADAPTER_GAINS"
echo "validate_plan_only=$VALIDATE_PLAN_ONLY"

for required in "$SCORE_ROWS" \
  "scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

if [ "$VALIDATE_PLAN_ONLY" != "1" ]; then
  for required in "$PROTECTED_ADAPTER/adapter_config.json" "$TASK_ONLY_ADAPTER/adapter_config.json"; do
    if [ ! -s "$required" ]; then
      echo "REQUIRED_ADAPTER_MISSING_OR_EMPTY: $required" >&2
      exit 2
    fi
  done
fi

if [ -e "$OUTPUT_DIR/r4_teacher_forced_surface_mass_summary.json" ] || \
   [ -e "$OUTPUT_DIR/r4_teacher_forced_surface_mass_rows.jsonl" ]; then
  echo "R4_CANDIDATE_V3_ADAPTER_GAIN_SWEEP_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

PLAN_FLAG=""
CUDA_FLAG="--require-cuda"
if [ "$VALIDATE_PLAN_ONLY" = "1" ]; then
  PLAN_FLAG="--dry-run"
  CUDA_FLAG=""
fi

"$PYTHON" scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py \
  --score-rows "$SCORE_ROWS" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --protected-adapter "$PROTECTED_ADAPTER" \
  --task-only-adapter "$TASK_ONLY_ADAPTER" \
  --protected-adapter-gains "$PROTECTED_ADAPTER_GAINS" \
  --max-rows "$MAX_ROWS" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  $CUDA_FLAG \
  $PLAN_FLAG

echo "R4 candidate v3 adapter-gain sweep complete: $OUTPUT_DIR"
