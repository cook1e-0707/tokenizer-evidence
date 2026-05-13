#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r32h200
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
#SBATCH --array=0-11%8
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r32h200-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r32h200-%A_%a.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

PROMPTS_JSONL="${PROMPTS_JSONL:-results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_r3_2_6144_20260512/restricted_step_label_r1_eval_prompts.jsonl}"
WP4_CONTRACT="${WP4_CONTRACT:-results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/wp4_prompt_local_payload_contract.json}"
CONFIG_YAML="${CONFIG_YAML:-configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale_expanded_6144.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/r3_2_qwen_locked_scale_h200_array_${SLURM_ARRAY_JOB_ID:-manual}}"
PRECOMMIT_BUILDER="${PRECOMMIT_BUILDER:-scripts/natural_evidence_v2/build_r3_2_expanded_locked_scale_precommit.py}"
EXPECTED_SELECTED_PROMPT_MANIFEST_SHA256="${EXPECTED_SELECTED_PROMPT_MANIFEST_SHA256:-ce057a36ad75424919f4367eb3e2f0221725a9c6715d156ab4b2a377edb600ed}"
PROMPT_SPLIT="${PROMPT_SPLIT:-wp3_r1_density_eval}"
PROTECTED_ADAPTER="${PROTECTED_ADAPTER:-$RUN_ROOT/status/wp5_r2_teacher_forced_train_and_score_851481/protected_train/adapter}"
TASK_ONLY_ADAPTER="${TASK_ONLY_ADAPTER:-$RUN_ROOT/status/wp5_r2_teacher_forced_train_and_score_851481/task_only_train/adapter}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-896}"
VALIDATE_PLAN_ONLY="${VALIDATE_PLAN_ONLY:-0}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-}}"

if [ -z "$SHARD_INDEX" ]; then
  echo "SHARD_INDEX_REQUIRED: set SLURM_ARRAY_TASK_ID or SHARD_INDEX" >&2
  exit 2
fi
if [ "$SHARD_INDEX" -lt 0 ] || [ "$SHARD_INDEX" -gt 11 ]; then
  echo "INVALID_SHARD_INDEX: $SHARD_INDEX" >&2
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

echo "array_job_id=${SLURM_ARRAY_JOB_ID:-manual}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-manual}"
echo "shard_index=$SHARD_INDEX"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=natural_evidence_v2 R3.2 Qwen locked-scale H200 shard array; no training, no Llama, no same-family null, no sanitizer, no FAR, no paper positive claim"
echo "partition=pomplun"
echo "account=cs_yinxin.wan"
echo "qos=pomplun"
echo "gres=gpu:h200:1"
echo "prompts_jsonl=$PROMPTS_JSONL"
echo "wp4_contract=$WP4_CONTRACT"
echo "config_yaml=$CONFIG_YAML"
echo "output_dir=$OUTPUT_DIR"
echo "selected_prompt_manifest_sha256=$EXPECTED_SELECTED_PROMPT_MANIFEST_SHA256"
echo "prompt_split=$PROMPT_SPLIT"
echo "prompt_window_policy=distinct_eval_window_by_shard_index"
echo "validate_plan_only=$VALIDATE_PLAN_ONLY"
echo "protected_adapter=$PROTECTED_ADAPTER"
echo "task_only_adapter=$TASK_ONLY_ADAPTER"
echo "model_name=$MODEL_NAME"

for required in "$PROMPTS_JSONL" "$WP4_CONTRACT" "$CONFIG_YAML" "$PRECOMMIT_BUILDER" \
  "scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py" \
  "scripts/natural_evidence_v2/decode_wp6_payload.py" \
  "scripts/natural_evidence_v2/decode_wp6_r1_scale_blocks.py"; do
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

precommit_manifest="$OUTPUT_DIR/precommit/r3_2_selected_prompt_manifest.json"
precommit_contract="$OUTPUT_DIR/precommit/r3_2_qwen_locked_scale_contract.json"
precommit_lock="$OUTPUT_DIR/.precommit.lock"
mkdir -p "$OUTPUT_DIR"
have_precommit_lock=0

cleanup_precommit_lock() {
  if [ "${have_precommit_lock:-0}" = "1" ] && [ -d "$precommit_lock" ]; then
    rmdir "$precommit_lock" 2>/dev/null || true
  fi
}
trap cleanup_precommit_lock EXIT

if [ ! -s "$precommit_manifest" ] || [ ! -s "$precommit_contract" ]; then
  if [ -s "$precommit_manifest" ] && [ -s "$precommit_contract" ]; then
    :
  else
    while true; do
      if mkdir "$precommit_lock" 2>/dev/null; then
        have_precommit_lock=1
        break
      fi
      if [ -s "$precommit_manifest" ] && [ -s "$precommit_contract" ]; then
        break
      fi
      echo "waiting_for_precommit_lock=$precommit_lock"
      sleep 10
    done
  fi
  if [ "$have_precommit_lock" = "1" ]; then
    if [ ! -s "$precommit_manifest" ] || [ ! -s "$precommit_contract" ]; then
      "$PYTHON" "$PRECOMMIT_BUILDER" \
        --config-yaml "$CONFIG_YAML" \
        --output-dir "$OUTPUT_DIR"
    fi
    rmdir "$precommit_lock"
    have_precommit_lock=0
  fi
fi

if [ ! -s "$precommit_manifest" ] || [ ! -s "$precommit_contract" ]; then
  echo "R3_2_PRECOMMIT_NOT_READY: $OUTPUT_DIR/precommit" >&2
  exit 2
fi

if [ "$VALIDATE_PLAN_ONLY" = "1" ]; then
  echo "R3.2 H200 expanded shard-array plan validation complete: $OUTPUT_DIR"
  exit 0
fi

shard_id="$(printf 'shard_%02d' "$SHARD_INDEX")"
expected_start=$((SHARD_INDEX * 512))
expected_end=$((expected_start + 511))
shard_dir="$OUTPUT_DIR/shards/$shard_id"
shard_decode_dir="$shard_dir/coordinate_majority_r3_2_shard"
mkdir -p "$shard_dir/precommit" "$shard_decode_dir"

if [ -s "$shard_decode_dir/r3_2_shard_summary.json" ]; then
  echo "R3_2_SHARD_ALREADY_COMPLETE_SKIP: $shard_id"
  exit 0
fi
if [ -e "$shard_dir/wp6_generated_outputs.jsonl" ]; then
  echo "R3_2_SHARD_PARTIAL_OUTPUT_EXISTS_REFUSING_MIX: $shard_dir" >&2
  exit 74
fi

"$PYTHON" scripts/natural_evidence_v2/decode_wp6_r1_scale_blocks.py \
  --input-dir "$shard_dir" \
  --prompts-jsonl "$PROMPTS_JSONL" \
  --wp4-contract "$WP4_CONTRACT" \
  --output-dir "$shard_dir/precommit" \
  --split "$PROMPT_SPLIT" \
  --max-prompts 512 \
  --block-count 8 \
  --block-size 64 \
  --expected-file-row-start "$expected_start" \
  --expected-file-row-end "$expected_end" \
  --query-budgets 16,32,64 \
  --min-protected-block-accepts-at-64 0 \
  --protocol-id natural_evidence_v2_r3_2_qwen_same_contract_locked_scale \
  --decoder-id qwen_v2_r3_2_locked_scale_coordinate_majority_decoder_v1 \
  --output-prefix r3_2_shard \
  --artifact-role r3_2_same_contract_locked_scale_shard \
  --decode-artifact-dir coordinate_majority_r3_2_shard \
  --pass-gate-status PASS_R3_2_SHARD_LOCAL_DECODE \
  --fail-gate-status FAIL_R3_2_SHARD_LOCAL_DECODE \
  --precommitted-transcript \
  --contract-only

"$PYTHON" scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py \
  --prompts-jsonl "$PROMPTS_JSONL" \
  --wp4-contract "$WP4_CONTRACT" \
  --output-dir "$shard_dir" \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --protected-adapter "$PROTECTED_ADAPTER" \
  --task-only-adapter "$TASK_ONLY_ADAPTER" \
  --split "$PROMPT_SPLIT" \
  --max-prompts 512 \
  --prompt-file-row-start "$expected_start" \
  --prompt-file-row-end "$expected_end" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --require-cuda

"$PYTHON" scripts/natural_evidence_v2/decode_wp6_payload.py \
  --generated-outputs "$shard_dir/wp6_generated_outputs.jsonl" \
  --wp4-contract "$WP4_CONTRACT" \
  --output-dir "$shard_dir"

"$PYTHON" scripts/natural_evidence_v2/decode_wp6_r1_scale_blocks.py \
  --input-dir "$shard_dir" \
  --prompts-jsonl "$PROMPTS_JSONL" \
  --wp4-contract "$WP4_CONTRACT" \
  --output-dir "$shard_decode_dir" \
  --split "$PROMPT_SPLIT" \
  --max-prompts 512 \
  --block-count 8 \
  --block-size 64 \
  --expected-file-row-start "$expected_start" \
  --expected-file-row-end "$expected_end" \
  --query-budgets 16,32,64 \
  --min-protected-block-accepts-at-64 0 \
  --protocol-id natural_evidence_v2_r3_2_qwen_same_contract_locked_scale \
  --decoder-id qwen_v2_r3_2_locked_scale_coordinate_majority_decoder_v1 \
  --output-prefix r3_2_shard \
  --artifact-role r3_2_same_contract_locked_scale_shard \
  --decode-artifact-dir coordinate_majority_r3_2_shard \
  --pass-gate-status PASS_R3_2_SHARD_LOCAL_DECODE \
  --fail-gate-status FAIL_R3_2_SHARD_LOCAL_DECODE \
  --precommitted-transcript

echo "R3.2 H200 shard complete: $shard_id $shard_dir"
