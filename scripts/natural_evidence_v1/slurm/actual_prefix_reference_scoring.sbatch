#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-apscore
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
#SBATCH --time=24:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-apscore-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-apscore-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_diagnostic_e2e_eval_844121}"
PLAN_DIR="${PLAN_DIR:-$EVAL_DIR/actual_prefix_scoring_plan_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_DIR/actual_prefix_reference_scoring_topk64_v1}"

INPUT_JSONL="${INPUT_JSONL:-$PLAN_DIR/actual_prefix_scoring_input.jsonl}"
PLAN_MANIFEST="${PLAN_MANIFEST:-$PLAN_DIR/actual_prefix_scoring_manifest.json}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$OUTPUT_DIR/qwen_actual_prefix_topk_candidates.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-$OUTPUT_DIR/qwen_actual_prefix_reference_scoring_summary.json}"
PROGRESS_JSON="${PROGRESS_JSON:-$OUTPUT_DIR/qwen_actual_prefix_reference_scoring_progress.json}"
STATIC_VALIDATION_SUMMARY="${STATIC_VALIDATION_SUMMARY:-$OUTPUT_DIR/static_validation_summary_actual_prefix_reference_scoring.json}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:-$RUN_ROOT/status/qwen_actual_prefix_reference_scoring_preflight_${SLURM_JOB_ID:-manual}.json}"

TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-64}"
BUCKET_COUNT="${BUCKET_COUNT:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_RECORDS="${MAX_RECORDS:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-256}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"
START_ACTUAL_PREFIX_SCORING="${START_ACTUAL_PREFIX_SCORING:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$RUN_ROOT/status" "$OUTPUT_DIR"

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
echo "config=$CONFIG"
echo "run_root=$RUN_ROOT"
echo "eval_dir=$EVAL_DIR"
echo "plan_dir=$PLAN_DIR"
echo "input_jsonl=$INPUT_JSONL"
echo "plan_manifest=$PLAN_MANIFEST"
echo "output_dir=$OUTPUT_DIR"
echo "output_jsonl=$OUTPUT_JSONL"
echo "summary_json=$SUMMARY_JSON"
echo "progress_json=$PROGRESS_JSON"
echo "tokenizer_key=$TOKENIZER_KEY"
echo "candidate_top_k=$CANDIDATE_TOP_K"
echo "bucket_count=$BUCKET_COUNT"
echo "batch_size=$BATCH_SIZE"
echo "max_records=$MAX_RECORDS"
echo "shard_index=$SHARD_INDEX"
echo "shard_count=$SHARD_COUNT"
echo "dry_run_only=$DRY_RUN_ONLY"
echo "start_actual_prefix_scoring=$START_ACTUAL_PREFIX_SCORING"
echo "claim_note=actual-prefix reference-model scoring only; no training, no E2E eval, no payload recovery"

for required in "$INPUT_JSONL" "$PLAN_MANIFEST"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ACTUAL_PREFIX_SCORING_INPUT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

cat >"$PREFLIGHT_JSON" <<EOF
{
  "schema_name": "natural_evidence_qwen_actual_prefix_reference_scoring_preflight_v1",
  "status": "WRAPPER_PREFLIGHT",
  "tokenizer_key": "$TOKENIZER_KEY",
  "candidate_top_k": $CANDIDATE_TOP_K,
  "bucket_count": $BUCKET_COUNT,
  "batch_size": $BATCH_SIZE,
  "max_records": $MAX_RECORDS,
  "shard_index": $SHARD_INDEX,
  "shard_count": $SHARD_COUNT,
  "input_jsonl": "$INPUT_JSONL",
  "plan_manifest": "$PLAN_MANIFEST",
  "output_jsonl": "$OUTPUT_JSONL",
  "summary_json": "$SUMMARY_JSON",
  "progress_json": "$PROGRESS_JSON",
  "dry_run_only": "$DRY_RUN_ONLY",
  "start_actual_prefix_scoring": "$START_ACTUAL_PREFIX_SCORING",
  "paper_claim_allowed": false,
  "training_started": false,
  "e2e_eval_started": false,
  "result_claim": "actual_prefix_reference_scoring_preflight_not_payload_recovery"
}
EOF

"$PYTHON" scripts/natural_evidence_v1/validate_static.py \
  --config "$CONFIG" \
  --summary "$STATIC_VALIDATION_SUMMARY"

"$PYTHON" -m py_compile scripts/natural_evidence_v1/score_actual_prefix_reference_candidates.py

if [ "$DRY_RUN_ONLY" != "0" ] || [ "$START_ACTUAL_PREFIX_SCORING" != "1" ]; then
  echo "DRY_RUN_ONLY_OR_START_FLAG_NOT_SET: wrapper preflight completed; no reference scoring started"
  exit 0
fi

if [ -e "$OUTPUT_JSONL" ] || [ -e "$SUMMARY_JSON" ]; then
  echo "ACTUAL_PREFIX_REFERENCE_SCORING_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" - <<'PY'
import os
import torch

print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("device count =", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA_NOT_AVAILABLE_ON_ALLOCATED_NODE")
print("device name =", torch.cuda.get_device_name(0))
PY

"$PYTHON" scripts/natural_evidence_v1/score_actual_prefix_reference_candidates.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --input-jsonl "$INPUT_JSONL" \
  --manifest-json "$PLAN_MANIFEST" \
  --output-jsonl "$OUTPUT_JSONL" \
  --summary-json "$SUMMARY_JSON" \
  --candidate-top-k "$CANDIDATE_TOP_K" \
  --batch-size "$BATCH_SIZE" \
  --max-records "$MAX_RECORDS" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --stream-output \
  --progress-json "$PROGRESS_JSON" \
  --progress-every "$PROGRESS_EVERY" \
  --require-cuda
