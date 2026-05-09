#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-nat-eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-nat-eval-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-nat-eval-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$RUN_ROOT/data/variable_radix_frame_policy_dry_run_20260506_1848}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUN_ROOT/training}"
PROOF_GATE_REVIEW_JSON="${PROOF_GATE_REVIEW_JSON:-$RUN_ROOT/status/qwen_proof_of_life_gate_review.json}"

STATUS="QWEN_NATURAL_VARIABLE_RADIX_FIVE_ARM_EVAL_PREFLIGHT"
CLAIM_STATUS="NO_PAPER_CLAIM"
MODEL="Qwen/Qwen2.5-7B-Instruct"
TOKENIZER="Qwen/Qwen2.5-7B-Instruct"
ARMS="qwen_protected,qwen_raw,qwen_task_only_lora,wrong_key,wrong_payload"
TRAINED_ARMS="qwen_protected,qwen_task_only_lora"
PAYLOADS="${PAYLOADS:-P0421,P1729}"
SEEDS="${SEEDS:-17,23}"
QUERY_BUDGETS="${QUERY_BUDGETS:-64,128,256,512}"
EVAL_OWNER_PROBES="${EVAL_OWNER_PROBES:-2048}"
ORGANIC_NULL_PROMPTS="${ORGANIC_NULL_PROMPTS:-2048}"
PROMPT_SPLIT_ID="${PROMPT_SPLIT_ID:-qwen_density_split_v1}"
BUDGET_CAP="${BUDGET_CAP:-qwen_variable_radix_proof_of_life_v0_max_steps64}"
CONDITION="${CONDITION:-diagnostic_high_risk}"
MAX_PROMPTS="${MAX_PROMPTS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TRAINING_JOB_ID="${TRAINING_JOB_ID:-}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"
START_QWEN_NATURAL_E2E_EVAL="${START_QWEN_NATURAL_E2E_EVAL:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/eval/qwen_natural_e2e_eval_${SLURM_JOB_ID:-manual}}"

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
echo "arms=$ARMS"
echo "trained_arms=$TRAINED_ARMS"
echo "payloads=$PAYLOADS"
echo "seeds=$SEEDS"
echo "query_budgets=$QUERY_BUDGETS"
echo "eval_owner_probes=$EVAL_OWNER_PROBES"
echo "organic_null_prompts=$ORGANIC_NULL_PROMPTS"
echo "prompt_split_id=$PROMPT_SPLIT_ID"
echo "budget_cap=$BUDGET_CAP"
echo "train_data_dir=$TRAIN_DATA_DIR"
echo "checkpoint_root=$CHECKPOINT_ROOT"
echo "proof_gate_review_json=$PROOF_GATE_REVIEW_JSON"
echo "output_dir=$OUTPUT_DIR"
echo "dry_run_only=$DRY_RUN_ONLY"
echo "start_qwen_natural_e2e_eval=$START_QWEN_NATURAL_E2E_EVAL"

IFS=',' read -r -a PAYLOAD_ARRAY <<< "$PAYLOADS"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"

for payload_id in "${PAYLOAD_ARRAY[@]}"; do
  train_jsonl="$TRAIN_DATA_DIR/$payload_id/variable_radix_train.jsonl"
  train_contract_json="$TRAIN_DATA_DIR/$payload_id/variable_radix_train_contract.json"
  for required in "$train_jsonl" "$train_contract_json"; do
    if [ ! -s "$required" ]; then
      echo "REQUIRED_VARIABLE_RADIX_EVAL_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
      exit 2
    fi
  done
done

cat >"$RUN_ROOT/status/qwen_natural_e2e_eval_wrapper_preflight_${SLURM_JOB_ID:-manual}.json" <<EOF
{
  "schema_name": "natural_evidence_qwen_natural_e2e_eval_wrapper_preflight_v1",
  "status": "$STATUS",
  "claim_status": "$CLAIM_STATUS",
  "model": "$MODEL",
  "tokenizer": "$TOKENIZER",
  "arms": "$ARMS",
  "trained_arms": "$TRAINED_ARMS",
  "payloads": "$PAYLOADS",
  "seeds": "$SEEDS",
  "query_budgets": "$QUERY_BUDGETS",
  "eval_owner_probes": $EVAL_OWNER_PROBES,
  "organic_null_prompts": $ORGANIC_NULL_PROMPTS,
  "prompt_split_id": "$PROMPT_SPLIT_ID",
  "budget_cap": "$BUDGET_CAP",
  "train_data_dir": "$TRAIN_DATA_DIR",
  "checkpoint_root": "$CHECKPOINT_ROOT",
  "proof_gate_review_json": "$PROOF_GATE_REVIEW_JSON",
  "output_dir": "$OUTPUT_DIR",
  "dry_run_only": "$DRY_RUN_ONLY",
  "start_qwen_natural_e2e_eval": "$START_QWEN_NATURAL_E2E_EVAL",
  "paper_claim_allowed": false,
  "training_started": false,
  "eval_started": false,
  "result_claim": "qwen_natural_e2e_eval_wrapper_preflight_not_payload_recovery"
}
EOF

if [ "$DRY_RUN_ONLY" != "0" ] || [ "$START_QWEN_NATURAL_E2E_EVAL" != "1" ]; then
  "$PYTHON" scripts/natural_evidence_v1/evaluate_qwen_natural_e2e.py \
    --config "$CONFIG" \
    --train-data-dir "$TRAIN_DATA_DIR" \
    --run-root "$RUN_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL" \
    --tokenizer-name "$TOKENIZER" \
    --payload-ids "$PAYLOADS" \
    --seeds "$SEEDS" \
    --query-budgets "$QUERY_BUDGETS" \
    --eval-owner-probes "$EVAL_OWNER_PROBES" \
    --organic-null-prompts "$ORGANIC_NULL_PROMPTS" \
    --prompt-split-id "$PROMPT_SPLIT_ID" \
    --budget-cap "$BUDGET_CAP" \
    --condition "$CONDITION" \
    --paper-claim-status "$CLAIM_STATUS" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --training-job-id "$TRAINING_JOB_ID" \
    --max-prompts "$MAX_PROMPTS" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE"
  echo "DRY_RUN_ONLY_OR_START_FLAG_NOT_SET: five-arm natural E2E eval preflight completed; no generation started"
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

review = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if review.get("status") != "READY_FOR_EXPLICIT_LAUNCH_REVIEW":
    raise SystemExit(
        "PROOF_GATE_REVIEW_NOT_READY_FOR_EVAL: "
        f"status={review.get('status')} blockers={review.get('blocker_gates')}"
    )
if review.get("paper_claim_allowed") is not False:
    raise SystemExit("PROOF_GATE_REVIEW_MUST_NOT_ALLOW_PAPER_CLAIM")
PY

if [ -e "$OUTPUT_DIR/qwen_natural_e2e_eval_summary.json" ] || \
   [ -e "$OUTPUT_DIR/qwen_natural_e2e_decode_trace.csv" ] || \
   [ -e "$OUTPUT_DIR/qwen_natural_e2e_generated_outputs.jsonl" ]; then
  echo "QWEN_NATURAL_E2E_EVAL_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 2
fi

"$PYTHON" scripts/natural_evidence_v1/evaluate_qwen_natural_e2e.py \
  --config "$CONFIG" \
  --train-data-dir "$TRAIN_DATA_DIR" \
  --run-root "$RUN_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL" \
  --tokenizer-name "$TOKENIZER" \
  --payload-ids "$PAYLOADS" \
  --seeds "$SEEDS" \
  --query-budgets "$QUERY_BUDGETS" \
  --eval-owner-probes "$EVAL_OWNER_PROBES" \
  --organic-null-prompts "$ORGANIC_NULL_PROMPTS" \
  --prompt-split-id "$PROMPT_SPLIT_ID" \
  --budget-cap "$BUDGET_CAP" \
  --condition "$CONDITION" \
  --paper-claim-status "$CLAIM_STATUS" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --training-job-id "$TRAINING_JOB_ID" \
  --max-prompts "$MAX_PROMPTS" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --start-eval \
  --require-cuda
