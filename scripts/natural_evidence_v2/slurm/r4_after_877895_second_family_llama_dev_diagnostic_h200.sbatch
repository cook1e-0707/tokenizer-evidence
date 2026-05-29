#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4ll32
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
#SBATCH --array=0-31
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4ll32-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4ll32-%A_%a.err

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
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/r4_after_877895_second_family_llama_dev_diagnostic_${SLURM_ARRAY_JOB_ID:-manual}/llama3_1_8b_instruct}"

export ROUTE_CONFIG="${ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_877895_second_family_llama_dev_diagnostic_generation_route.yaml}"
export ROUTE_VALIDATOR="${ROUTE_VALIDATOR:-scripts/natural_evidence_v2/validate_r4_after_877895_second_family_llama_dev_diagnostic_route.py}"
export OUTPUT_DIR
export SCORE_ROWS="${SCORE_ROWS:-results/natural_evidence_v2/status/r4_after_877895_second_family_llama_row_bank_plan_20260526/row_allocation_rows.jsonl}"
export ALLOCATION_ROWS="${ALLOCATION_ROWS:-results/natural_evidence_v2/status/r4_after_877895_second_family_llama_row_bank_plan_20260526/row_allocation_rows.jsonl}"
export TOKENIZER_REVIEW="${TOKENIZER_REVIEW:-results/natural_evidence_v2/status/r4_after_877895_second_family_llama_tokenizer_preflight_879100_review/review_summary.json}"
export MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
export TOKENIZER_NAME="${TOKENIZER_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
export MODEL_SLUG="${MODEL_SLUG:-llama3_1_8b_instruct}"
export PRECOMMIT_DIR="${PRECOMMIT_DIR:-results/natural_evidence_v2/precommit/r4_after_868212_repaired_first_token_event_precommit_20260516}"
export CODEBOOK="${CODEBOOK:-$PRECOMMIT_DIR/codebook.json}"
export DECODER_SPEC="${DECODER_SPEC:-$PRECOMMIT_DIR/decoder_spec.json}"
export FULL_PHRASE_DECODER_SPEC="${FULL_PHRASE_DECODER_SPEC:-results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516/decoder_spec.json}"
export DECODER_ROUTE_CONFIG="${DECODER_ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_868212_repaired_first_token_event_decoder_route.yaml}"
export CONTEXTUAL_LITERAL_POLICY="${CONTEXTUAL_LITERAL_POLICY:-results/natural_evidence_v2/precommit/r4_after_868260_quality_gate_repair_package_20260517/contextual_forbidden_surface_policy_v2.json}"
export DUPLICATE_SAFE_POLICY="${DUPLICATE_SAFE_POLICY:-configs/natural_evidence_v2/r4_first_token_event_duplicate_safe_generation_v2.yaml}"
export PUBLIC_RUN_SALT="${PUBLIC_RUN_SALT:-r4_after_877895_second_family_llama_dev_diagnostic}"
export GENERATION_CONDITIONS="${GENERATION_CONDITIONS:-protected,raw}"
export TASK_ONLY_ADAPTER="${TASK_ONLY_ADAPTER:-}"
export ROWS_PER_SHARD="${ROWS_PER_SHARD:-1024}"
export EXPECTED_SELECTED_COORDINATE_COUNT="${EXPECTED_SELECTED_COORDINATE_COUNT:-16}"
export MAX_SHARD_INDEX="${MAX_SHARD_INDEX:-31}"
export SHARD_INDEX="$TASK_ID"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
export CONTROLLER_BONUS_NATS="${CONTROLLER_BONUS_NATS:-4.0}"
export CONTROLLER_PENALTY_NATS="${CONTROLLER_PENALTY_NATS:-0.5}"
export CONTROLLER_MAX_TARGET_MASS="${CONTROLLER_MAX_TARGET_MASS:-0.5}"
export CONTROLLER_MAX_KL_BUDGET="${CONTROLLER_MAX_KL_BUDGET:-0.5}"
export PLAN_ONLY="${PLAN_ONLY:-1}"
export VALIDATE_PLAN_ONLY="${VALIDATE_PLAN_ONLY:-1}"
if [ "$VALIDATE_PLAN_ONLY" != "1" ]; then
  export ROUTE_VALIDATION_EXTRA_ARGS="${ROUTE_VALIDATION_EXTRA_ARGS:---skip-allowlist-state-check}"
fi

echo "claim_note=R4 after-877895 second-family Llama 32-block dev diagnostic; no paper claim, no FAR, no sanitizer, no payload diversity"
echo "task_id=$TASK_ID"
echo "model_name=$MODEL_NAME"
echo "tokenizer_name=$TOKENIZER_NAME"
echo "generation_conditions=$GENERATION_CONDITIONS"
echo "task_only_adapter_intentionally_empty=true"
echo "slurm_policy=H200 pomplun cs_yinxin.wan time 30-00:00:00"

exec bash scripts/natural_evidence_v2/slurm/r4_after_868016_controller_generation_h200.sbatch
