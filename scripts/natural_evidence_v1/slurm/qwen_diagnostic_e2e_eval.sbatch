#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-diag-eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=pomplun
#SBATCH --account=cs_yinxin.wan
#SBATCH --qos=pomplun
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-diag-eval-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-diag-eval-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e}"
EXPANSION_ROOT="${EXPANSION_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3}"

STATUS="DIAGNOSTIC_HIGH_RISK"
CLAIM_STATUS="NO_PAPER_CLAIM"
MODEL="Qwen/Qwen2.5-7B-Instruct"
TOKENIZER="Qwen/Qwen2.5-7B-Instruct"
PAYLOADS="P0421,P1729"
SEEDS="17,23"
QUERY_BUDGETS="64,128,256,512"
EVAL_OWNER_PROBES="2048"
MAX_PROMPTS="${MAX_PROMPTS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"
START_DIAGNOSTIC_EVAL="${START_DIAGNOSTIC_EVAL:-0}"

HELDOUT_REFERENCE_OUTPUTS="${HELDOUT_REFERENCE_OUTPUTS:-$EXPANSION_ROOT/frozen_density_splits/qwen_density_split_v1/heldout_reference_outputs.jsonl}"
BUCKET_BANK_ENTRIES="${BUCKET_BANK_ENTRIES:-$EXPANSION_ROOT/bucket_banks_4way_strict/qwen_bucket_bank_entries.jsonl}"
COMPATIBILITY_JSONL="${COMPATIBILITY_JSONL:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/probability_preserving_compatibility_repair/qwen_counterfactual_compatibility_with_probabilities.jsonl}"
COMPATIBILITY_BY_ENTRY_CSV="${COMPATIBILITY_BY_ENTRY_CSV:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/probability_preserving_compatibility_repair/compatibility_filtered_bank_dry_run_by_entry.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/eval/qwen_diagnostic_e2e_eval_${SLURM_JOB_ID:-manual}}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$RUN_ROOT/status" "$RUN_ROOT/eval"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "status=$STATUS"
echo "claim_status=$CLAIM_STATUS"
echo "model=$MODEL"
echo "tokenizer=$TOKENIZER"
echo "payloads=$PAYLOADS"
echo "seeds=$SEEDS"
echo "query_budgets=$QUERY_BUDGETS"
echo "eval_owner_probes=$EVAL_OWNER_PROBES"
echo "max_prompts=$MAX_PROMPTS"
echo "batch_size=$BATCH_SIZE"
echo "max_new_tokens=$MAX_NEW_TOKENS"
echo "temperature=$TEMPERATURE"
echo "dry_run_only=$DRY_RUN_ONLY"
echo "start_diagnostic_eval=$START_DIAGNOSTIC_EVAL"
echo "output_dir=$OUTPUT_DIR"

for required in \
  "$HELDOUT_REFERENCE_OUTPUTS" \
  "$BUCKET_BANK_ENTRIES" \
  "$COMPATIBILITY_JSONL" \
  "$COMPATIBILITY_BY_ENTRY_CSV"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

for payload_id in P0421 P1729; do
  for seed in 17 23; do
    for arm in qwen_protected qwen_task_only_lora; do
      checkpoint="$RUN_ROOT/${arm}_${payload_id}_seed${seed}/checkpoints/natural_bucket_lora_last"
      if [ ! -s "$checkpoint/adapter_model.safetensors" ]; then
        echo "REQUIRED_CHECKPOINT_MISSING_OR_EMPTY: $checkpoint" >&2
        exit 2
      fi
    done
  done
done

cat >"$RUN_ROOT/status/qwen_diagnostic_e2e_eval_preflight_${SLURM_JOB_ID:-manual}.json" <<EOF
{
  "schema_name": "natural_evidence_qwen_diagnostic_e2e_eval_wrapper_preflight_v1",
  "status": "$STATUS",
  "claim_status": "$CLAIM_STATUS",
  "model": "$MODEL",
  "tokenizer": "$TOKENIZER",
  "payloads": "$PAYLOADS",
  "seeds": "$SEEDS",
  "query_budgets": "$QUERY_BUDGETS",
  "eval_owner_probes": $EVAL_OWNER_PROBES,
  "max_prompts": $MAX_PROMPTS,
  "batch_size": $BATCH_SIZE,
  "max_new_tokens": $MAX_NEW_TOKENS,
  "temperature": "$TEMPERATURE",
  "dry_run_only": "$DRY_RUN_ONLY",
  "start_diagnostic_eval": "$START_DIAGNOSTIC_EVAL",
  "paper_claim_allowed": false,
  "training_started": false,
  "eval_started": false,
  "result_claim": "diagnostic_eval_wrapper_preflight_not_payload_recovery"
}
EOF

if [ "$DRY_RUN_ONLY" != "0" ] || [ "$START_DIAGNOSTIC_EVAL" != "1" ]; then
  "$PYTHON" scripts/natural_evidence_v1/evaluate_diagnostic_e2e.py \
    --config "$CONFIG" \
    --heldout-reference-outputs "$HELDOUT_REFERENCE_OUTPUTS" \
    --bucket-bank-entries "$BUCKET_BANK_ENTRIES" \
    --compatibility-jsonl "$COMPATIBILITY_JSONL" \
    --compatibility-by-entry-csv "$COMPATIBILITY_BY_ENTRY_CSV" \
    --run-root "$RUN_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL" \
    --tokenizer-name "$TOKENIZER" \
    --payload-ids "$PAYLOADS" \
    --seeds "$SEEDS" \
    --query-budgets "$QUERY_BUDGETS" \
    --eval-owner-probes "$EVAL_OWNER_PROBES" \
    --max-prompts "$MAX_PROMPTS" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --condition diagnostic_high_risk \
    --paper-claim-status "$CLAIM_STATUS"
  echo "DRY_RUN_ONLY_OR_START_FLAG_NOT_SET: diagnostic eval preflight completed; no generation started"
  exit 0
fi

"$PYTHON" scripts/natural_evidence_v1/evaluate_diagnostic_e2e.py \
  --config "$CONFIG" \
  --heldout-reference-outputs "$HELDOUT_REFERENCE_OUTPUTS" \
  --bucket-bank-entries "$BUCKET_BANK_ENTRIES" \
  --compatibility-jsonl "$COMPATIBILITY_JSONL" \
  --compatibility-by-entry-csv "$COMPATIBILITY_BY_ENTRY_CSV" \
  --run-root "$RUN_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL" \
  --tokenizer-name "$TOKENIZER" \
  --payload-ids "$PAYLOADS" \
  --seeds "$SEEDS" \
  --query-budgets "$QUERY_BUDGETS" \
  --eval-owner-probes "$EVAL_OWNER_PROBES" \
  --max-prompts "$MAX_PROMPTS" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --condition diagnostic_high_risk \
  --paper-claim-status "$CLAIM_STATUS" \
  --start-eval \
  --require-cuda
