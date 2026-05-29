#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4sfRaw
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
#SBATCH --array=0-191
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4sfRaw-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4sfRaw-%A_%a.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
if [ ! -d "$REPO_HOME" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  CANDIDATE_REPO_HOME="$(cd "$SCRIPT_DIR/../../.." && pwd)"
  if [ -d "$CANDIDATE_REPO_HOME/.git" ]; then
    REPO_HOME="$CANDIDATE_REPO_HOME"
  fi
fi
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
PYTHON_BIN="${PYTHON_BIN:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
ROUTE_CONFIG="${ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_870987_same_family_raw_null_generation_route.yaml}"
ROUTE_VALIDATOR="${ROUTE_VALIDATOR:-scripts/natural_evidence_v2/validate_r4_after_870987_same_family_raw_null_generation_route.py}"
PLAN_ONLY="${PLAN_ONLY:-1}"
VALIDATE_PLAN_ONLY="${VALIDATE_PLAN_ONLY:-1}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
SHARDS_PER_MODEL="${SHARDS_PER_MODEL:-64}"

cd "$REPO_HOME"

MODEL_GROUP_INDEX=$((TASK_ID / SHARDS_PER_MODEL))
LOCAL_SHARD_INDEX=$((TASK_ID % SHARDS_PER_MODEL))
case "$MODEL_GROUP_INDEX" in
  0)
    MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
    TOKENIZER_NAME="Qwen/Qwen2.5-3B-Instruct"
    MODEL_SLUG="qwen2_5_3b_instruct_raw"
    ;;
  1)
    MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    TOKENIZER_NAME="Qwen/Qwen2.5-7B-Instruct"
    MODEL_SLUG="qwen2_5_7b_instruct_raw"
    ;;
  2)
    MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
    TOKENIZER_NAME="Qwen/Qwen2.5-14B-Instruct"
    MODEL_SLUG="qwen2_5_14b_instruct_raw"
    ;;
  *)
    echo "INVALID_MODEL_GROUP_INDEX: $MODEL_GROUP_INDEX" >&2
    exit 2
    ;;
esac

if [[ "$ROUTE_CONFIG" == *"r4_after_877840_same_family_raw_null_generation_v8"* ]]; then
  DEFAULT_ROUTE_OUTPUT_ROOT="$RUN_ROOT/status/r4_after_877840_same_family_raw_null_v8_generation_${SLURM_ARRAY_JOB_ID:-manual}"
else
  DEFAULT_ROUTE_OUTPUT_ROOT="$RUN_ROOT/status/r4_after_870987_same_family_raw_null_generation_${SLURM_ARRAY_JOB_ID:-manual}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_ROUTE_OUTPUT_ROOT/$MODEL_SLUG}"
export MODEL_GROUP_INDEX LOCAL_SHARD_INDEX MODEL_NAME TOKENIZER_NAME MODEL_SLUG OUTPUT_DIR

echo "claim_note=R4 after-870987 same-family raw-null generation; no full FAR, no Llama transfer, no paper claim"
echo "route_config=$ROUTE_CONFIG"
echo "model_group_index=$MODEL_GROUP_INDEX"
echo "local_shard_index=$LOCAL_SHARD_INDEX"
echo "model_name=$MODEL_NAME"
echo "tokenizer_name=$TOKENIZER_NAME"
echo "model_slug=$MODEL_SLUG"
echo "output_dir=$OUTPUT_DIR"
echo "plan_only=$PLAN_ONLY"

if [[ "$PLAN_ONLY" == "1" ]]; then
  ROUTE_VALIDATION_ARGS="${ROUTE_VALIDATION_EXTRA_ARGS:-}"
else
  ROUTE_VALIDATION_ARGS="${ROUTE_VALIDATION_EXTRA_ARGS:---skip-allowlist-state-check}"
fi

"$PYTHON_BIN" "$ROUTE_VALIDATOR" \
  --config "$ROUTE_CONFIG" \
  --output-dir "${OUTPUT_DIR}/route_validation_shard_${LOCAL_SHARD_INDEX}" \
  $ROUTE_VALIDATION_ARGS

if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "R4 same-family raw-null wrapper plan-only validation passed"
  exit 0
fi

export ROUTE_CONFIG
export ROUTE_VALIDATOR
export PLAN_ONLY=0
export VALIDATE_PLAN_ONLY="${VALIDATE_PLAN_ONLY:-0}"
export SCORE_ROWS="${SCORE_ROWS:-results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_row_bank_v2_plan_20260521/row_allocation_rows.jsonl}"
export ALLOCATION_ROWS="${ALLOCATION_ROWS:-results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_row_bank_v2_plan_20260521/row_allocation_rows.jsonl}"
export CONTEXTUAL_LITERAL_POLICY="${CONTEXTUAL_LITERAL_POLICY:-results/natural_evidence_v2/precommit/r4_after_868260_quality_gate_repair_package_20260517/contextual_forbidden_surface_policy_v2.json}"
export DUPLICATE_SAFE_POLICY="${DUPLICATE_SAFE_POLICY:-configs/natural_evidence_v2/r4_first_token_event_duplicate_safe_generation_v2.yaml}"
export PUBLIC_RUN_SALT="${PUBLIC_RUN_SALT:-r4_after_870987_same_family_raw_null_generation_v1_${MODEL_SLUG}}"
export TOKENIZER_REVIEW="${TOKENIZER_REVIEW:-results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_tokenizer_preflight_874778_review/review_summary.json}"
export PRECOMMIT_DIR="${PRECOMMIT_DIR:-results/natural_evidence_v2/precommit/r4_after_868212_repaired_first_token_event_precommit_20260516}"
export CODEBOOK="${CODEBOOK:-$PRECOMMIT_DIR/codebook.json}"
export DECODER_SPEC="${DECODER_SPEC:-$PRECOMMIT_DIR/decoder_spec.json}"
export FULL_PHRASE_DECODER_SPEC="${FULL_PHRASE_DECODER_SPEC:-results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516/decoder_spec.json}"
export DECODER_ROUTE_CONFIG="${DECODER_ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_868212_repaired_first_token_event_decoder_route.yaml}"
export ROWS_PER_SHARD=1024
export EXPECTED_SELECTED_COORDINATE_COUNT=16
export MAX_SHARD_INDEX=63
export GENERATION_CONDITIONS=raw
export SHARD_INDEX="$LOCAL_SHARD_INDEX"
export ROUTE_VALIDATION_EXTRA_ARGS="${ROUTE_VALIDATION_EXTRA_ARGS:---skip-allowlist-state-check}"
export MAX_SHARD_INDEX=$((SHARDS_PER_MODEL - 1))

# The delegated wrapper normally reads SLURM_ARRAY_TASK_ID directly. Clear it so
# SHARD_INDEX is the local shard within each model group.
unset SLURM_ARRAY_TASK_ID
exec bash scripts/natural_evidence_v2/slurm/r4_after_868016_controller_generation_h200.sbatch
