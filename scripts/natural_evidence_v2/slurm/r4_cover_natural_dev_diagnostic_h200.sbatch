#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4dev
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
#SBATCH --array=0-3%4
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4dev-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4dev-%A_%a.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

CONFIG_YAML="${CONFIG_YAML:-configs/natural_evidence_v2/r4_cover_natural_ecc.yaml}"
PROMPTS_JSONL="${PROMPTS_JSONL:-results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/dev_prompts.jsonl}"
SURFACE_BANK="${SURFACE_BANK:-results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/surface_bank.json}"
CODEBOOK="${CODEBOOK:-results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/codebook.json}"
DECODER_SPEC="${DECODER_SPEC:-results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/decoder_spec.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/r4_cover_natural_dev_diagnostic_h200_array_${SLURM_ARRAY_JOB_ID:-manual}}"
PROTECTED_ADAPTER="${PROTECTED_ADAPTER:-$RUN_ROOT/status/wp5_r2_teacher_forced_train_and_score_851481/protected_train/adapter}"
TASK_ONLY_ADAPTER="${TASK_ONLY_ADAPTER:-$RUN_ROOT/status/wp5_r2_teacher_forced_train_and_score_851481/task_only_train/adapter}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
VALIDATE_PLAN_ONLY="${VALIDATE_PLAN_ONLY:-0}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-}}"

if [ -z "$SHARD_INDEX" ]; then
  echo "SHARD_INDEX_REQUIRED: set SLURM_ARRAY_TASK_ID or SHARD_INDEX" >&2
  exit 2
fi
if [ "$SHARD_INDEX" -lt 0 ] || [ "$SHARD_INDEX" -gt 3 ]; then
  echo "INVALID_R4_DEV_SHARD_INDEX: $SHARD_INDEX" >&2
  exit 2
fi

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

expected_start=$((SHARD_INDEX * 512))
expected_end=$((expected_start + 511))
shard_id="$(printf 'shard_%02d' "$SHARD_INDEX")"
shard_dir="$OUTPUT_DIR/shards/$shard_id"

echo "array_job_id=${SLURM_ARRAY_JOB_ID:-manual}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-manual}"
echo "shard_index=$SHARD_INDEX"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=natural_evidence_v2 R4 cover-natural dev diagnostic; no training, no Llama, no same-family null, no sanitizer, no FAR, no paper positive claim"
echo "partition=pomplun"
echo "account=cs_yinxin.wan"
echo "qos=pomplun"
echo "gres=gpu:h200:1"
echo "config_yaml=$CONFIG_YAML"
echo "prompts_jsonl=$PROMPTS_JSONL"
echo "output_dir=$OUTPUT_DIR"
echo "shard_dir=$shard_dir"
echo "replicate_group_id=$shard_id"
echo "prompt_rows=$expected_start..$expected_end"
echo "validate_plan_only=$VALIDATE_PLAN_ONLY"

for required in "$CONFIG_YAML" "$PROMPTS_JSONL" "$SURFACE_BANK" "$CODEBOOK" "$DECODER_SPEC" \
  "scripts/natural_evidence_v2/generate_r4_cover_natural_outputs.py" \
  "scripts/natural_evidence_v2/decode_r4_cover_natural_ecc.py"; do
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

if [ -e "$shard_dir/r4_generated_outputs.jsonl" ] || \
   [ -e "$shard_dir/decode_all/decode_summary.json" ] || \
   [ -e "$shard_dir/decode_none/decode_summary.json" ]; then
  echo "R4_DEV_SHARD_OUTPUT_EXISTS_REFUSING_OVERWRITE: $shard_dir" >&2
  exit 74
fi

mkdir -p "$shard_dir"
PLAN_FLAG=""
CUDA_FLAG="--require-cuda"
if [ "$VALIDATE_PLAN_ONLY" = "1" ]; then
  PLAN_FLAG="--validate-plan-only"
  CUDA_FLAG=""
fi

"$PYTHON" scripts/natural_evidence_v2/generate_r4_cover_natural_outputs.py \
  --prompts-jsonl "$PROMPTS_JSONL" \
  --output-dir "$shard_dir" \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --protected-adapter "$PROTECTED_ADAPTER" \
  --task-only-adapter "$TASK_ONLY_ADAPTER" \
  --split dev \
  --max-prompts 512 \
  --prompt-file-row-start "$expected_start" \
  --prompt-file-row-end "$expected_end" \
  --replicate-group-id "$shard_id" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  $CUDA_FLAG \
  $PLAN_FLAG

if [ "$VALIDATE_PLAN_ONLY" = "1" ]; then
  echo "R4 dev diagnostic shard plan validation complete: $shard_id $shard_dir"
  exit 0
fi

"$PYTHON" scripts/natural_evidence_v2/decode_r4_cover_natural_ecc.py \
  --generated-outputs "$shard_dir/r4_generated_outputs.jsonl" \
  --surface-bank "$SURFACE_BANK" \
  --codebook "$CODEBOOK" \
  --decoder-spec "$DECODER_SPEC" \
  --output-dir "$shard_dir/decode_all" \
  --format-scrub all \
  --prompts-per-block 64 \
  --split-label dev \
  --include-protected-controls

"$PYTHON" scripts/natural_evidence_v2/decode_r4_cover_natural_ecc.py \
  --generated-outputs "$shard_dir/r4_generated_outputs.jsonl" \
  --surface-bank "$SURFACE_BANK" \
  --codebook "$CODEBOOK" \
  --decoder-spec "$DECODER_SPEC" \
  --output-dir "$shard_dir/decode_none" \
  --format-scrub none \
  --prompts-per-block 64 \
  --split-label dev \
  --include-protected-controls

echo "R4 dev diagnostic shard complete: $shard_id $shard_dir"
