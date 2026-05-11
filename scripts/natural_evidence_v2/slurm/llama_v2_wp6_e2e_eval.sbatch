#!/bin/bash
#SBATCH --job-name=llama-v2-wp6
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
#SBATCH --time=04:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/llama_migration/slurm/llama-v2-wp6-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/llama_migration/slurm/llama-v2-wp6-%j.err

# Llama-3.1-8B WP6: E2E eval with robust-block coordinate-majority decoder
# Adapted from Qwen WP6-R2 Option B but with Llama model + adapter

set -euo pipefail

export REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
export PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
export VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
export HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"
export RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/llama_migration}"

# Llama-specific
export MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
export TOKENIZER_NAME="${TOKENIZER_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/llama_v2_wp6_e2e_eval_${SLURM_JOB_ID:-manual}}"

# WP5 adapter paths (must be set by caller)
export PROTECTED_ADAPTER="${PROTECTED_ADAPTER:-}"
export TASK_ONLY_ADAPTER="${TASK_ONLY_ADAPTER:-}"

# Contract and prompts (reuse Qwen WP6 contract, same payload 0xa55e)
export CONTRACT="${CONTRACT:-results/natural_evidence_v2/status/wp6_r2_option_b_wrapper_contract_plan_20260510_0251/wp6_r2_option_b_wrapper_contract_plan.json}"
export PROMPTS_JSONL="${PROMPTS_JSONL:-results/natural_evidence_v2/status/wp5_teacher_forced_launch_plan_20260509_0226/wp5_teacher_forced_score_rows.jsonl}"
export SPLIT="${SPLIT:-wp3_r1_eval}"
export MAX_PROMPTS="${MAX_PROMPTS:-64}"
export PROMPT_START="${PROMPT_START:-768}"
export PROMPT_END="${PROMPT_END:-1279}"

# Replay params
export QUERY_BUDGETS="${QUERY_BUDGETS:-8,16,32,64}"
export MIN_SUPPORT="${MIN_SUPPORT:-16}"
export MIN_MARGIN="${MIN_MARGIN:-3}"

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

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "model_name=$MODEL_NAME"
echo "output_dir=$OUTPUT_DIR"
echo "contract=$CONTRACT"
echo "claim_note=Llama-3.1-8B V2 WP6 robust-block E2E eval; no sanitizer, no paper positive claim"

# Validate adapters
if [ -z "$PROTECTED_ADAPTER" ] || [ ! -d "$PROTECTED_ADAPTER" ]; then
  echo "ERROR: PROTECTED_ADAPTER not set or not found: $PROTECTED_ADAPTER" >&2
  exit 2
fi
if [ -z "$TASK_ONLY_ADAPTER" ] || [ ! -d "$TASK_ONLY_ADAPTER" ]; then
  echo "ERROR: TASK_ONLY_ADAPTER not set or not found: $TASK_ONLY_ADAPTER" >&2
  exit 2
fi

for required in "$CONTRACT" "$PROMPTS_JSONL"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

if [ -e "$OUTPUT_DIR/wp6_e2e_summary.json" ]; then
  echo "LLAMA_WP6_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

# Step 1: Generate outputs
echo "=== Llama WP6: Generating E2E outputs ==="
GENERATION_DIR="$OUTPUT_DIR/generation"
mkdir -p "$GENERATION_DIR"

GEN_ARGS=(
  --output-dir "$GENERATION_DIR"
  --model-name "$MODEL_NAME"
  --tokenizer-name "$TOKENIZER_NAME"
  --protected-adapter "$PROTECTED_ADAPTER"
  --task-only-adapter "$TASK_ONLY_ADAPTER"
  --wp4-contract "$CONTRACT"
  --split "$SPLIT"
  --max-prompts "$MAX_PROMPTS"
  --require-cuda
)

if [ -n "$PROMPT_START" ]; then
  GEN_ARGS+=(--prompt-file-row-start "$PROMPT_START")
fi
if [ -n "$PROMPT_END" ]; then
  GEN_ARGS+=(--prompt-file-row-end "$PROMPT_END")
fi

"$PYTHON" scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py "${GEN_ARGS[@]}"

# Step 2: Decode payloads
echo "=== Llama WP6: Decoding payloads ==="
DECODE_DIR="$OUTPUT_DIR/decode"
mkdir -p "$DECODE_DIR"

"$PYTHON" scripts/natural_evidence_v2/decode_wp6_payload.py \
  --generated-outputs "$GENERATION_DIR/wp6_generated_outputs.jsonl" \
  --wp4-contract "$CONTRACT" \
  --output-dir "$DECODE_DIR"

# Step 3: Coordinate majority replay
echo "=== Llama WP6: Coordinate majority replay ==="
REPLAY_DIR="$OUTPUT_DIR/coordinate_majority"
mkdir -p "$REPLAY_DIR"

"$PYTHON" scripts/natural_evidence_v2/replay_wp6_coordinate_majority_decoder.py \
  --input-dir "$DECODE_DIR" \
  --wp4-contract "$CONTRACT" \
  --output-dir "$REPLAY_DIR" \
  --query-budgets "$QUERY_BUDGETS" \
  --min-support-at-64 "$MIN_SUPPORT" \
  --min-majority-margin-at-64 "$MIN_MARGIN"

echo "Llama V2 WP6 E2E eval wrapper complete: $OUTPUT_DIR"
