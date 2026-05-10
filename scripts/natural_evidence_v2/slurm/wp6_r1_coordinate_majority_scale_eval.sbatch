#!/bin/bash
#SBATCH --job-name=nat-ev-v2-wp6r1scale
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp6r1scale-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp6r1scale-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"

PROMPTS_JSONL="${PROMPTS_JSONL:-results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/restricted_step_label_strict_density_audit_prompts.jsonl}"
WP4_CONTRACT="${WP4_CONTRACT:-results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/wp4_prompt_local_payload_contract.json}"
PROTECTED_ADAPTER="${PROTECTED_ADAPTER:-$RUN_ROOT/status/wp5_r2_teacher_forced_train_and_score_851481/protected_train/adapter}"
TASK_ONLY_ADAPTER="${TASK_ONLY_ADAPTER:-$RUN_ROOT/status/wp5_r2_teacher_forced_train_and_score_851481/task_only_train/adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/wp6_r1_coordinate_majority_scale_eval_${SLURM_JOB_ID:-manual}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MAX_PROMPTS="${MAX_PROMPTS:-256}"
BLOCK_COUNT="${BLOCK_COUNT:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-896}"
VALIDATE_PLAN_ONLY="${VALIDATE_PLAN_ONLY:-0}"

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
echo "claim_note=natural_evidence_v2 WP6-R1 Qwen repeated-coordinate majority reproducibility scale eval only; no training, no FAR, no Llama, no paper positive claim"
echo "prompts_jsonl=$PROMPTS_JSONL"
echo "wp4_contract=$WP4_CONTRACT"
echo "protected_adapter=$PROTECTED_ADAPTER"
echo "task_only_adapter=$TASK_ONLY_ADAPTER"
echo "output_dir=$OUTPUT_DIR"
echo "model_name=$MODEL_NAME"
echo "max_prompts=$MAX_PROMPTS"
echo "block_count=$BLOCK_COUNT"
echo "block_size=$BLOCK_SIZE"
echo "validate_plan_only=$VALIDATE_PLAN_ONLY"

for required in "$PROMPTS_JSONL" "$WP4_CONTRACT" \
  "scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py" \
  "scripts/natural_evidence_v2/decode_wp6_payload.py" \
  "scripts/natural_evidence_v2/decode_wp6_r1_scale_blocks.py"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

if [ "$MAX_PROMPTS" != "256" ] || [ "$BLOCK_COUNT" != "4" ] || [ "$BLOCK_SIZE" != "64" ]; then
  echo "WP6_R1_SCALE_SCOPE_MISMATCH: MAX_PROMPTS=256 BLOCK_COUNT=4 BLOCK_SIZE=64 required" >&2
  exit 2
fi

if [ "$VALIDATE_PLAN_ONLY" != "1" ]; then
  for required in "$PROTECTED_ADAPTER/adapter_config.json" "$TASK_ONLY_ADAPTER/adapter_config.json"; do
    if [ ! -s "$required" ]; then
      echo "REQUIRED_ADAPTER_MISSING_OR_EMPTY: $required" >&2
      exit 2
    fi
  done
fi

if [ -e "$OUTPUT_DIR/coordinate_majority_scale/wp6_r1_scale_summary.json" ] || \
   [ -e "$OUTPUT_DIR/wp6_generated_outputs.jsonl" ]; then
  echo "WP6_R1_SCALE_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

PRECOMMIT_DIR="$OUTPUT_DIR/precommit"
SCALE_DIR="$OUTPUT_DIR/coordinate_majority_scale"
mkdir -p "$PRECOMMIT_DIR" "$SCALE_DIR"

"$PYTHON" scripts/natural_evidence_v2/decode_wp6_r1_scale_blocks.py \
  --input-dir "$OUTPUT_DIR" \
  --prompts-jsonl "$PROMPTS_JSONL" \
  --wp4-contract "$WP4_CONTRACT" \
  --output-dir "$PRECOMMIT_DIR" \
  --precommitted-transcript \
  --contract-only

PLAN_FLAG=""
CUDA_FLAG="--require-cuda"
if [ "$VALIDATE_PLAN_ONLY" = "1" ]; then
  PLAN_FLAG="--validate-plan-only"
  CUDA_FLAG=""
fi

"$PYTHON" scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py \
  --prompts-jsonl "$PROMPTS_JSONL" \
  --wp4-contract "$WP4_CONTRACT" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --protected-adapter "$PROTECTED_ADAPTER" \
  --task-only-adapter "$TASK_ONLY_ADAPTER" \
  --max-prompts "$MAX_PROMPTS" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  $CUDA_FLAG \
  $PLAN_FLAG

if [ "$VALIDATE_PLAN_ONLY" = "1" ]; then
  echo "WP6-R1 scale plan validation complete: $OUTPUT_DIR"
  exit 0
fi

"$PYTHON" scripts/natural_evidence_v2/decode_wp6_payload.py \
  --generated-outputs "$OUTPUT_DIR/wp6_generated_outputs.jsonl" \
  --wp4-contract "$WP4_CONTRACT" \
  --output-dir "$OUTPUT_DIR"

"$PYTHON" scripts/natural_evidence_v2/decode_wp6_r1_scale_blocks.py \
  --input-dir "$OUTPUT_DIR" \
  --prompts-jsonl "$PROMPTS_JSONL" \
  --wp4-contract "$WP4_CONTRACT" \
  --output-dir "$SCALE_DIR" \
  --precommitted-transcript

echo "WP6-R1 coordinate-majority scale wrapper complete: $OUTPUT_DIR"
