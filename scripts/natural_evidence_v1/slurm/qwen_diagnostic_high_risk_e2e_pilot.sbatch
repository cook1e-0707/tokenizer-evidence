#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-diag-e2e
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=pomplun
#SBATCH --account=cs_yinxin.wan
#SBATCH --qos=pomplun
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-diag-e2e-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-diag-e2e-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e}"
EXPANSION_ROOT="${EXPANSION_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192}"

STATUS="DIAGNOSTIC_HIGH_RISK"
CLAIM_STATUS="NO_PAPER_CLAIM"
MODEL="Qwen/Qwen2.5-7B-Instruct"
TOKENIZER="Qwen/Qwen2.5-7B-Instruct"
ARMS="qwen_protected,qwen_raw,qwen_task_only_lora,wrong_key,wrong_payload"
PAYLOADS="P0421,P1729"
SEEDS="17,23"
QUERY_BUDGETS="64,128,256,512"
EVAL_OWNER_PROBES="2048"
ORGANIC_NULL_PROMPTS="2048"
BUCKET_BANK="qwen_4way_min1_compatible"
FALLBACK_BANK="qwen_2way_if_available"

BANK_ENTRIES="${BANK_ENTRIES:-$EXPANSION_ROOT/bucket_banks_4way_strict/qwen_bucket_bank_entries.jsonl}"
COMPATIBILITY_BY_ENTRY_CSV="${COMPATIBILITY_BY_ENTRY_CSV:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/probability_preserving_compatibility_repair/compatibility_filtered_bank_dry_run_by_entry.csv}"
RAW_PRE_NULL_SUMMARY="${RAW_PRE_NULL_SUMMARY:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/raw_wrong_key_pre_null/raw_wrong_key_pre_null_summary.json}"
INVALID_SUFFIX_SUMMARY="${INVALID_SUFFIX_SUMMARY:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/invalid_suffix_review/invalid_suffix_review_summary.json}"
HELDOUT_DENSITY_SUMMARY="${HELDOUT_DENSITY_SUMMARY:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/frozen_heldout_min1_compatible_density/compatible_density_summary.json}"
REFERENCE_OUTPUTS="${REFERENCE_OUTPUTS:-$REFERENCE_ROOT/reference_outputs/qwen_reference_outputs.jsonl}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$RUN_ROOT/data}"
PROMPT_SPLIT_ID="${PROMPT_SPLIT_ID:-qwen_density_split_v1}"
BUDGET_CAP="${BUDGET_CAP:-qwen_diagnostic_high_risk_e2e_v0_max_steps64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-64}"
LEARNING_RATE="${LEARNING_RATE:-5.0e-5}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"
START_DIAGNOSTIC_E2E="${START_DIAGNOSTIC_E2E:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$RUN_ROOT/status"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "status=$STATUS"
echo "claim_status=$CLAIM_STATUS"
echo "model=$MODEL"
echo "tokenizer=$TOKENIZER"
echo "arms=$ARMS"
echo "payloads=$PAYLOADS"
echo "seeds=$SEEDS"
echo "query_budgets=$QUERY_BUDGETS"
echo "eval_owner_probes=$EVAL_OWNER_PROBES"
echo "organic_null_prompts=$ORGANIC_NULL_PROMPTS"
echo "bucket_bank=$BUCKET_BANK"
echo "fallback_bank=$FALLBACK_BANK"
echo "train_data_dir=$TRAIN_DATA_DIR"
echo "prompt_split_id=$PROMPT_SPLIT_ID"
echo "budget_cap=$BUDGET_CAP"
echo "batch_size=$BATCH_SIZE"
echo "max_steps=$MAX_STEPS"
echo "learning_rate=$LEARNING_RATE"
echo "dry_run_only=$DRY_RUN_ONLY"
echo "start_diagnostic_e2e=$START_DIAGNOSTIC_E2E"

for required in \
  "$BANK_ENTRIES" \
  "$COMPATIBILITY_BY_ENTRY_CSV" \
  "$RAW_PRE_NULL_SUMMARY" \
  "$INVALID_SUFFIX_SUMMARY" \
  "$HELDOUT_DENSITY_SUMMARY" \
  "$REFERENCE_OUTPUTS"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

cat >"$RUN_ROOT/status/qwen_diagnostic_e2e_wrapper_preflight_${SLURM_JOB_ID:-manual}.json" <<EOF
{
  "schema_name": "natural_evidence_qwen_diagnostic_e2e_wrapper_preflight_v1",
  "status": "$STATUS",
  "claim_status": "$CLAIM_STATUS",
  "model": "$MODEL",
  "tokenizer": "$TOKENIZER",
  "arms": "$ARMS",
  "payloads": "$PAYLOADS",
  "seeds": "$SEEDS",
  "query_budgets": "$QUERY_BUDGETS",
  "eval_owner_probes": $EVAL_OWNER_PROBES,
  "organic_null_prompts": $ORGANIC_NULL_PROMPTS,
  "bucket_bank": "$BUCKET_BANK",
  "fallback_bank": "$FALLBACK_BANK",
  "train_data_dir": "$TRAIN_DATA_DIR",
  "prompt_split_id": "$PROMPT_SPLIT_ID",
  "budget_cap": "$BUDGET_CAP",
  "batch_size": $BATCH_SIZE,
  "max_steps_per_training_run": $MAX_STEPS,
  "learning_rate": "$LEARNING_RATE",
  "dry_run_only": "$DRY_RUN_ONLY",
  "start_diagnostic_e2e": "$START_DIAGNOSTIC_E2E",
  "paper_claim_allowed": false,
  "training_started": false,
  "result_claim": "diagnostic_wrapper_preflight_not_payload_recovery"
}
EOF

if [ "$DRY_RUN_ONLY" != "0" ] || [ "$START_DIAGNOSTIC_E2E" != "1" ]; then
  echo "DRY_RUN_ONLY_OR_START_FLAG_NOT_SET: wrapper preflight completed; no training started"
  exit 0
fi

IFS=',' read -r -a PAYLOAD_ARRAY <<< "$PAYLOADS"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"

for payload_id in "${PAYLOAD_ARRAY[@]}"; do
  train_jsonl="$TRAIN_DATA_DIR/qwen_diagnostic_train_${payload_id}.jsonl"
  train_contract_json="$TRAIN_DATA_DIR/qwen_diagnostic_train_contract_${payload_id}.json"
  for required in "$train_jsonl" "$train_contract_json"; do
    if [ ! -s "$required" ]; then
      echo "REQUIRED_TRAINING_DATASET_MISSING_OR_EMPTY: $required" >&2
      exit 2
    fi
  done
  for seed in "${SEED_ARRAY[@]}"; do
    for arm in qwen_protected qwen_task_only_lora; do
      lambda_bucket="1.0"
      if [ "$arm" = "qwen_task_only_lora" ]; then
        lambda_bucket="0.0"
      fi
      "$PYTHON" scripts/natural_evidence_v1/train_natural_bucket_lora.py \
        --config "$CONFIG" \
        --train-jsonl "$train_jsonl" \
        --contract-json "$train_contract_json" \
        --output-dir "$RUN_ROOT/${arm}_${payload_id}_seed${seed}" \
        --model-name "$MODEL" \
        --tokenizer-name "$TOKENIZER" \
        --arm "$arm" \
        --payload-id "$payload_id" \
        --seed "$seed" \
        --prompt-split-id "$PROMPT_SPLIT_ID" \
        --budget-cap "$BUDGET_CAP" \
        --condition diagnostic_high_risk \
        --paper-claim-status "$CLAIM_STATUS" \
        --query-budgets "$QUERY_BUDGETS" \
        --eval-owner-probes "$EVAL_OWNER_PROBES" \
        --organic-null-prompts "$ORGANIC_NULL_PROMPTS" \
        --batch-size "$BATCH_SIZE" \
        --max-steps "$MAX_STEPS" \
        --learning-rate "$LEARNING_RATE" \
        --lambda-bucket "$lambda_bucket" \
        --start-training \
        --require-cuda
    done
  done
done
