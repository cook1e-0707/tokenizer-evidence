#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-nat-e2e
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-nat-e2e-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-nat-e2e-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$RUN_ROOT/data/variable_radix_frame_policy_dry_run_20260506_1848}"
PROOF_GATE_REVIEW_JSON="${PROOF_GATE_REVIEW_JSON:-$RUN_ROOT/status/qwen_proof_of_life_gate_review.json}"

STATUS="QWEN_NATURAL_VARIABLE_RADIX_PROOF_OF_LIFE_PREFLIGHT"
CLAIM_STATUS="NO_PAPER_CLAIM"
MODEL="Qwen/Qwen2.5-7B-Instruct"
TOKENIZER="Qwen/Qwen2.5-7B-Instruct"
ARMS="qwen_protected,qwen_raw,qwen_task_only_lora,wrong_key,wrong_payload"
TRAINING_ARMS="qwen_protected,qwen_task_only_lora"
PAYLOADS="${PAYLOADS:-P0421,P1729}"
SEEDS="${SEEDS:-17,23}"
QUERY_BUDGETS="${QUERY_BUDGETS:-64,128,256,512}"
EVAL_OWNER_PROBES="${EVAL_OWNER_PROBES:-2048}"
ORGANIC_NULL_PROMPTS="${ORGANIC_NULL_PROMPTS:-2048}"
PROMPT_SPLIT_ID="${PROMPT_SPLIT_ID:-qwen_density_split_v1}"
BUDGET_CAP="${BUDGET_CAP:-qwen_variable_radix_proof_of_life_v0_max_steps64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-64}"
LEARNING_RATE="${LEARNING_RATE:-5.0e-5}"
CONDITION="${CONDITION:-diagnostic_high_risk}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"
START_QWEN_NATURAL_E2E="${START_QWEN_NATURAL_E2E:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$RUN_ROOT/status" "$RUN_ROOT/preflight"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "status=$STATUS"
echo "claim_status=$CLAIM_STATUS"
echo "model=$MODEL"
echo "tokenizer=$TOKENIZER"
echo "arms=$ARMS"
echo "training_arms=$TRAINING_ARMS"
echo "payloads=$PAYLOADS"
echo "seeds=$SEEDS"
echo "query_budgets=$QUERY_BUDGETS"
echo "eval_owner_probes=$EVAL_OWNER_PROBES"
echo "organic_null_prompts=$ORGANIC_NULL_PROMPTS"
echo "prompt_split_id=$PROMPT_SPLIT_ID"
echo "budget_cap=$BUDGET_CAP"
echo "train_data_dir=$TRAIN_DATA_DIR"
echo "proof_gate_review_json=$PROOF_GATE_REVIEW_JSON"
echo "dry_run_only=$DRY_RUN_ONLY"
echo "start_qwen_natural_e2e=$START_QWEN_NATURAL_E2E"

IFS=',' read -r -a PAYLOAD_ARRAY <<< "$PAYLOADS"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=',' read -r -a TRAINING_ARM_ARRAY <<< "$TRAINING_ARMS"

for payload_id in "${PAYLOAD_ARRAY[@]}"; do
  train_jsonl="$TRAIN_DATA_DIR/$payload_id/variable_radix_train.jsonl"
  train_contract_json="$TRAIN_DATA_DIR/$payload_id/variable_radix_train_contract.json"
  for required in "$train_jsonl" "$train_contract_json"; do
    if [ ! -s "$required" ]; then
      echo "REQUIRED_VARIABLE_RADIX_TRAIN_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
      exit 2
    fi
  done
  "$PYTHON" - "$train_contract_json" <<'PY'
import json
import sys
from pathlib import Path

contract_path = Path(sys.argv[1])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
errors = []
if contract.get("schema_name") != "natural_evidence_variable_radix_train_contract_v1":
    errors.append("schema_name")
if contract.get("encoding_mode") != "variable_radix":
    errors.append("encoding_mode")
if contract.get("variable_radix_frame_policy") != "repeat_payload":
    errors.append("variable_radix_frame_policy")
if contract.get("variable_radix_min_positions_satisfied") is not True:
    errors.append("variable_radix_min_positions_satisfied")
claim_control = dict(contract.get("claim_control", {}))
if claim_control.get("contains_field_value_outputs") is not False:
    errors.append("contains_field_value_outputs")
if claim_control.get("contains_structured_evidence_blocks") is not False:
    errors.append("contains_structured_evidence_blocks")
if errors:
    raise SystemExit(f"VARIABLE_RADIX_CONTRACT_REVIEW_FAILED {contract_path}: {errors}")
PY
done

cat >"$RUN_ROOT/status/qwen_natural_e2e_wrapper_preflight_${SLURM_JOB_ID:-manual}.json" <<EOF
{
  "schema_name": "natural_evidence_qwen_natural_e2e_wrapper_preflight_v1",
  "status": "$STATUS",
  "claim_status": "$CLAIM_STATUS",
  "model": "$MODEL",
  "tokenizer": "$TOKENIZER",
  "arms": "$ARMS",
  "training_arms": "$TRAINING_ARMS",
  "payloads": "$PAYLOADS",
  "seeds": "$SEEDS",
  "query_budgets": "$QUERY_BUDGETS",
  "eval_owner_probes": $EVAL_OWNER_PROBES,
  "organic_null_prompts": $ORGANIC_NULL_PROMPTS,
  "prompt_split_id": "$PROMPT_SPLIT_ID",
  "budget_cap": "$BUDGET_CAP",
  "train_data_dir": "$TRAIN_DATA_DIR",
  "proof_gate_review_json": "$PROOF_GATE_REVIEW_JSON",
  "dry_run_only": "$DRY_RUN_ONLY",
  "start_qwen_natural_e2e": "$START_QWEN_NATURAL_E2E",
  "paper_claim_allowed": false,
  "training_started": false,
  "e2e_eval_started": false,
  "result_claim": "qwen_natural_e2e_wrapper_preflight_not_payload_recovery"
}
EOF

for payload_id in "${PAYLOAD_ARRAY[@]}"; do
  train_jsonl="$TRAIN_DATA_DIR/$payload_id/variable_radix_train.jsonl"
  train_contract_json="$TRAIN_DATA_DIR/$payload_id/variable_radix_train_contract.json"
  for seed in "${SEED_ARRAY[@]}"; do
    for arm in "${TRAINING_ARM_ARRAY[@]}"; do
      lambda_bucket="1.0"
      if [ "$arm" = "qwen_task_only_lora" ]; then
        lambda_bucket="0.0"
      fi
      "$PYTHON" scripts/natural_evidence_v1/train_natural_bucket_lora.py \
        --config "$CONFIG" \
        --train-jsonl "$train_jsonl" \
        --contract-json "$train_contract_json" \
        --output-dir "$RUN_ROOT/preflight/${arm}_${payload_id}_seed${seed}_${SLURM_JOB_ID:-manual}" \
        --model-name "$MODEL" \
        --tokenizer-name "$TOKENIZER" \
        --arm "$arm" \
        --payload-id "$payload_id" \
        --seed "$seed" \
        --prompt-split-id "$PROMPT_SPLIT_ID" \
        --budget-cap "$BUDGET_CAP" \
        --condition "$CONDITION" \
        --paper-claim-status "$CLAIM_STATUS" \
        --query-budgets "$QUERY_BUDGETS" \
        --eval-owner-probes "$EVAL_OWNER_PROBES" \
        --organic-null-prompts "$ORGANIC_NULL_PROMPTS" \
        --batch-size "$BATCH_SIZE" \
        --max-steps "$MAX_STEPS" \
        --learning-rate "$LEARNING_RATE" \
        --lambda-bucket "$lambda_bucket"
    done
  done
done

if [ "$DRY_RUN_ONLY" != "0" ] || [ "$START_QWEN_NATURAL_E2E" != "1" ]; then
  echo "DRY_RUN_ONLY_OR_START_FLAG_NOT_SET: wrapper preflight completed; no training started"
  exit 0
fi

if [ ! -s "$PROOF_GATE_REVIEW_JSON" ]; then
  echo "PROOF_GATE_REVIEW_JSON_MISSING_OR_EMPTY: $PROOF_GATE_REVIEW_JSON" >&2
  exit 2
fi

"$PYTHON" - "$PROOF_GATE_REVIEW_JSON" <<'PY'
import json
import sys
from pathlib import Path

review_path = Path(sys.argv[1])
review = json.loads(review_path.read_text(encoding="utf-8"))
if review.get("status") != "READY_FOR_EXPLICIT_LAUNCH_REVIEW":
    raise SystemExit(
        "PROOF_GATE_REVIEW_NOT_READY_FOR_LAUNCH: "
        f"status={review.get('status')} blockers={review.get('blocker_gates')}"
    )
if review.get("paper_claim_allowed") is not False:
    raise SystemExit("PROOF_GATE_REVIEW_MUST_NOT_ALLOW_PAPER_CLAIM")
PY

for payload_id in "${PAYLOAD_ARRAY[@]}"; do
  train_jsonl="$TRAIN_DATA_DIR/$payload_id/variable_radix_train.jsonl"
  train_contract_json="$TRAIN_DATA_DIR/$payload_id/variable_radix_train_contract.json"
  for seed in "${SEED_ARRAY[@]}"; do
    for arm in "${TRAINING_ARM_ARRAY[@]}"; do
      lambda_bucket="1.0"
      if [ "$arm" = "qwen_task_only_lora" ]; then
        lambda_bucket="0.0"
      fi
      output_dir="$RUN_ROOT/training/${arm}_${payload_id}_seed${seed}_${SLURM_JOB_ID:-manual}"
      if [ -e "$output_dir" ]; then
        echo "QWEN_NATURAL_E2E_OUTPUT_EXISTS_REFUSING_OVERWRITE: $output_dir" >&2
        exit 2
      fi
      "$PYTHON" scripts/natural_evidence_v1/train_natural_bucket_lora.py \
        --config "$CONFIG" \
        --train-jsonl "$train_jsonl" \
        --contract-json "$train_contract_json" \
        --output-dir "$output_dir" \
        --model-name "$MODEL" \
        --tokenizer-name "$TOKENIZER" \
        --arm "$arm" \
        --payload-id "$payload_id" \
        --seed "$seed" \
        --prompt-split-id "$PROMPT_SPLIT_ID" \
        --budget-cap "$BUDGET_CAP" \
        --condition "$CONDITION" \
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
