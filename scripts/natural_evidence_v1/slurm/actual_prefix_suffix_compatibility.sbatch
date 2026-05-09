#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-apsuf
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-apsuf-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-apsuf-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_diagnostic_e2e_eval_844121}"
BUCKETIZATION_DIR="${BUCKETIZATION_DIR:-$EVAL_DIR/actual_prefix_bucketization_topk64_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_DIR/actual_prefix_suffix_compatibility_topk64_v1}"

GENERATED_OUTPUTS="${GENERATED_OUTPUTS:-$EVAL_DIR/qwen_diagnostic_generated_outputs.jsonl}"
BUCKETIZED_CANDIDATES_JSONL="${BUCKETIZED_CANDIDATES_JSONL:-$BUCKETIZATION_DIR/qwen_actual_prefix_bucketized_candidates.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$OUTPUT_DIR/qwen_actual_prefix_suffix_compatibility.jsonl}"
BY_ENTRY_CSV="${BY_ENTRY_CSV:-$OUTPUT_DIR/qwen_actual_prefix_suffix_compatibility_by_entry.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-$OUTPUT_DIR/qwen_actual_prefix_suffix_compatibility_summary.json}"
PROGRESS_JSON="${PROGRESS_JSON:-$OUTPUT_DIR/qwen_actual_prefix_suffix_compatibility_progress.json}"
STATIC_VALIDATION_SUMMARY="${STATIC_VALIDATION_SUMMARY:-$OUTPUT_DIR/static_validation_summary_actual_prefix_suffix_compatibility.json}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:-$RUN_ROOT/status/qwen_actual_prefix_suffix_compatibility_preflight_${SLURM_JOB_ID:-manual}.json}"

TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
SUFFIX_WINDOW_TOKENS="${SUFFIX_WINDOW_TOKENS:-16}"
MAX_CANDIDATES_PER_BUCKET="${MAX_CANDIDATES_PER_BUCKET:-4}"
MAX_RECORDS="${MAX_RECORDS:-0}"
DELTA_NLL_THRESHOLD="${DELTA_NLL_THRESHOLD:-0.5}"
PROGRESS_EVERY="${PROGRESS_EVERY:-64}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"
START_ACTUAL_PREFIX_SUFFIX_COMPATIBILITY="${START_ACTUAL_PREFIX_SUFFIX_COMPATIBILITY:-0}"

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
echo "bucketization_dir=$BUCKETIZATION_DIR"
echo "output_dir=$OUTPUT_DIR"
echo "generated_outputs=$GENERATED_OUTPUTS"
echo "bucketized_candidates_jsonl=$BUCKETIZED_CANDIDATES_JSONL"
echo "output_jsonl=$OUTPUT_JSONL"
echo "by_entry_csv=$BY_ENTRY_CSV"
echo "summary_json=$SUMMARY_JSON"
echo "progress_json=$PROGRESS_JSON"
echo "tokenizer_key=$TOKENIZER_KEY"
echo "suffix_window_tokens=$SUFFIX_WINDOW_TOKENS"
echo "max_candidates_per_bucket=$MAX_CANDIDATES_PER_BUCKET"
echo "max_records=$MAX_RECORDS"
echo "delta_nll_threshold=$DELTA_NLL_THRESHOLD"
echo "dry_run_only=$DRY_RUN_ONLY"
echo "start_actual_prefix_suffix_compatibility=$START_ACTUAL_PREFIX_SUFFIX_COMPATIBILITY"
echo "claim_note=actual-prefix suffix compatibility scoring only; no training, no E2E eval, no payload recovery"

for required in "$GENERATED_OUTPUTS" "$BUCKETIZED_CANDIDATES_JSONL"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ACTUAL_PREFIX_SUFFIX_INPUT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

cat >"$PREFLIGHT_JSON" <<EOF
{
  "schema_name": "natural_evidence_qwen_actual_prefix_suffix_compatibility_preflight_v1",
  "status": "WRAPPER_PREFLIGHT",
  "tokenizer_key": "$TOKENIZER_KEY",
  "suffix_window_tokens": $SUFFIX_WINDOW_TOKENS,
  "max_candidates_per_bucket": $MAX_CANDIDATES_PER_BUCKET,
  "max_records": $MAX_RECORDS,
  "delta_nll_threshold": $DELTA_NLL_THRESHOLD,
  "generated_outputs": "$GENERATED_OUTPUTS",
  "bucketized_candidates_jsonl": "$BUCKETIZED_CANDIDATES_JSONL",
  "output_jsonl": "$OUTPUT_JSONL",
  "by_entry_csv": "$BY_ENTRY_CSV",
  "summary_json": "$SUMMARY_JSON",
  "progress_json": "$PROGRESS_JSON",
  "dry_run_only": "$DRY_RUN_ONLY",
  "start_actual_prefix_suffix_compatibility": "$START_ACTUAL_PREFIX_SUFFIX_COMPATIBILITY",
  "paper_claim_allowed": false,
  "training_started": false,
  "e2e_eval_started": false,
  "result_claim": "actual_prefix_suffix_compatibility_preflight_not_payload_recovery"
}
EOF

"$PYTHON" scripts/natural_evidence_v1/validate_static.py \
  --config "$CONFIG" \
  --summary "$STATIC_VALIDATION_SUMMARY"

"$PYTHON" -m py_compile scripts/natural_evidence_v1/score_actual_prefix_suffix_compatibility.py

if [ "$DRY_RUN_ONLY" != "0" ] || [ "$START_ACTUAL_PREFIX_SUFFIX_COMPATIBILITY" != "1" ]; then
  echo "DRY_RUN_ONLY_OR_START_FLAG_NOT_SET: wrapper preflight completed; no suffix compatibility scoring started"
  exit 0
fi

if [ -e "$OUTPUT_JSONL" ] || [ -e "$BY_ENTRY_CSV" ] || [ -e "$SUMMARY_JSON" ]; then
  echo "ACTUAL_PREFIX_SUFFIX_COMPATIBILITY_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
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

"$PYTHON" scripts/natural_evidence_v1/score_actual_prefix_suffix_compatibility.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --generated-outputs "$GENERATED_OUTPUTS" \
  --bucketized-candidates-jsonl "$BUCKETIZED_CANDIDATES_JSONL" \
  --output-jsonl "$OUTPUT_JSONL" \
  --by-entry-csv "$BY_ENTRY_CSV" \
  --summary-json "$SUMMARY_JSON" \
  --progress-json "$PROGRESS_JSON" \
  --suffix-window-tokens "$SUFFIX_WINDOW_TOKENS" \
  --max-candidates-per-bucket "$MAX_CANDIDATES_PER_BUCKET" \
  --max-records "$MAX_RECORDS" \
  --delta-nll-threshold "$DELTA_NLL_THRESHOLD" \
  --progress-every "$PROGRESS_EVERY" \
  --require-cuda
