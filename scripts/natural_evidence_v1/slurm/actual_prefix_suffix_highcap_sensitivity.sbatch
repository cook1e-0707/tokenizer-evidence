#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-apcap
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-apcap-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-apcap-%j.err

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
BASELINE_DIR="${BASELINE_DIR:-$EVAL_DIR/actual_prefix_suffix_compatibility_topk64_v1}"
SENSITIVITY_DIR="${SENSITIVITY_DIR:-$EVAL_DIR/actual_prefix_suffix_sensitivity_topk64_v1}"

GENERATED_OUTPUTS="${GENERATED_OUTPUTS:-$EVAL_DIR/qwen_diagnostic_generated_outputs.jsonl}"
BUCKETIZED_CANDIDATES_JSONL="${BUCKETIZED_CANDIDATES_JSONL:-$BUCKETIZATION_DIR/qwen_actual_prefix_bucketized_candidates.jsonl}"
BASELINE_SUMMARY_JSON="${BASELINE_SUMMARY_JSON:-$BASELINE_DIR/qwen_actual_prefix_suffix_compatibility_summary.json}"
BASELINE_BY_ENTRY_CSV="${BASELINE_BY_ENTRY_CSV:-$BASELINE_DIR/qwen_actual_prefix_suffix_compatibility_by_entry.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-$SENSITIVITY_DIR/qwen_actual_prefix_suffix_sensitivity_summary.json}"
BY_CAP_CSV="${BY_CAP_CSV:-$SENSITIVITY_DIR/qwen_actual_prefix_suffix_sensitivity_by_cap.csv}"
BY_ENTRY_CSV="${BY_ENTRY_CSV:-$SENSITIVITY_DIR/qwen_actual_prefix_suffix_sensitivity_by_entry.csv}"
STATIC_VALIDATION_SUMMARY="${STATIC_VALIDATION_SUMMARY:-$SENSITIVITY_DIR/static_validation_summary_actual_prefix_suffix_sensitivity.json}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:-$RUN_ROOT/status/qwen_actual_prefix_suffix_sensitivity_preflight_${SLURM_JOB_ID:-manual}.json}"

TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
SUFFIX_WINDOW_TOKENS="${SUFFIX_WINDOW_TOKENS:-16}"
CAP_VALUES="${CAP_VALUES:-8,16}"
MAX_RECORDS="${MAX_RECORDS:-0}"
DELTA_NLL_THRESHOLD="${DELTA_NLL_THRESHOLD:-0.5}"
PROGRESS_EVERY="${PROGRESS_EVERY:-64}"
GENERATED_OUTPUT_COUNT="${GENERATED_OUTPUT_COUNT:-14336}"
BUCKET_COUNT="${BUCKET_COUNT:-4}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"
START_ACTUAL_PREFIX_SUFFIX_SENSITIVITY="${START_ACTUAL_PREFIX_SUFFIX_SENSITIVITY:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$RUN_ROOT/status" "$SENSITIVITY_DIR"

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
echo "sensitivity_dir=$SENSITIVITY_DIR"
echo "cap_values=$CAP_VALUES"
echo "tokenizer_key=$TOKENIZER_KEY"
echo "suffix_window_tokens=$SUFFIX_WINDOW_TOKENS"
echo "delta_nll_threshold=$DELTA_NLL_THRESHOLD"
echo "generated_output_count=$GENERATED_OUTPUT_COUNT"
echo "bucket_count=$BUCKET_COUNT"
echo "dry_run_only=$DRY_RUN_ONLY"
echo "start_actual_prefix_suffix_sensitivity=$START_ACTUAL_PREFIX_SUFFIX_SENSITIVITY"
echo "claim_note=higher-cap suffix sensitivity only; no training, no E2E eval, no payload recovery"

for required in "$GENERATED_OUTPUTS" "$BUCKETIZED_CANDIDATES_JSONL" "$BASELINE_SUMMARY_JSON" "$BASELINE_BY_ENTRY_CSV"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_SUFFIX_SENSITIVITY_INPUT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

cat >"$PREFLIGHT_JSON" <<EOF
{
  "schema_name": "natural_evidence_qwen_actual_prefix_suffix_sensitivity_preflight_v1",
  "status": "WRAPPER_PREFLIGHT",
  "tokenizer_key": "$TOKENIZER_KEY",
  "cap_values": "$CAP_VALUES",
  "suffix_window_tokens": $SUFFIX_WINDOW_TOKENS,
  "max_records": $MAX_RECORDS,
  "delta_nll_threshold": $DELTA_NLL_THRESHOLD,
  "generated_outputs": "$GENERATED_OUTPUTS",
  "bucketized_candidates_jsonl": "$BUCKETIZED_CANDIDATES_JSONL",
  "baseline_summary_json": "$BASELINE_SUMMARY_JSON",
  "baseline_by_entry_csv": "$BASELINE_BY_ENTRY_CSV",
  "summary_json": "$SUMMARY_JSON",
  "by_cap_csv": "$BY_CAP_CSV",
  "by_entry_csv": "$BY_ENTRY_CSV",
  "dry_run_only": "$DRY_RUN_ONLY",
  "start_actual_prefix_suffix_sensitivity": "$START_ACTUAL_PREFIX_SUFFIX_SENSITIVITY",
  "paper_claim_allowed": false,
  "training_started": false,
  "e2e_eval_started": false,
  "result_claim": "actual_prefix_suffix_sensitivity_preflight_not_payload_recovery"
}
EOF

"$PYTHON" scripts/natural_evidence_v1/validate_static.py \
  --config "$CONFIG" \
  --summary "$STATIC_VALIDATION_SUMMARY"

"$PYTHON" -m py_compile \
  scripts/natural_evidence_v1/score_actual_prefix_suffix_compatibility.py \
  scripts/natural_evidence_v1/summarize_actual_prefix_suffix_sensitivity.py

if [ "$DRY_RUN_ONLY" != "0" ] || [ "$START_ACTUAL_PREFIX_SUFFIX_SENSITIVITY" != "1" ]; then
  echo "DRY_RUN_ONLY_OR_START_FLAG_NOT_SET: wrapper preflight completed; no suffix sensitivity scoring started"
  exit 0
fi

if [ -e "$SUMMARY_JSON" ] || [ -e "$BY_CAP_CSV" ] || [ -e "$BY_ENTRY_CSV" ]; then
  echo "ACTUAL_PREFIX_SUFFIX_SENSITIVITY_OUTPUT_EXISTS_REFUSING_OVERWRITE: $SENSITIVITY_DIR" >&2
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

COMPARISON_ARGS=()
IFS=',' read -r -a CAP_ARRAY <<< "$CAP_VALUES"
for cap in "${CAP_ARRAY[@]}"; do
  cap_dir="$SENSITIVITY_DIR/cap${cap}"
  mkdir -p "$cap_dir"
  output_jsonl="$cap_dir/qwen_actual_prefix_suffix_compatibility_cap${cap}.jsonl"
  by_entry_csv="$cap_dir/qwen_actual_prefix_suffix_compatibility_cap${cap}_by_entry.csv"
  summary_json="$cap_dir/qwen_actual_prefix_suffix_compatibility_cap${cap}_summary.json"
  progress_json="$cap_dir/qwen_actual_prefix_suffix_compatibility_cap${cap}_progress.json"
  echo "running_cap=$cap output_dir=$cap_dir"
  "$PYTHON" scripts/natural_evidence_v1/score_actual_prefix_suffix_compatibility.py \
    --config "$CONFIG" \
    --tokenizer-key "$TOKENIZER_KEY" \
    --generated-outputs "$GENERATED_OUTPUTS" \
    --bucketized-candidates-jsonl "$BUCKETIZED_CANDIDATES_JSONL" \
    --output-jsonl "$output_jsonl" \
    --by-entry-csv "$by_entry_csv" \
    --summary-json "$summary_json" \
    --progress-json "$progress_json" \
    --suffix-window-tokens "$SUFFIX_WINDOW_TOKENS" \
    --max-candidates-per-bucket "$cap" \
    --max-records "$MAX_RECORDS" \
    --delta-nll-threshold "$DELTA_NLL_THRESHOLD" \
    --progress-every "$PROGRESS_EVERY" \
    --require-cuda
  COMPARISON_ARGS+=(--comparison "${cap}:${summary_json}:${by_entry_csv}")
done

"$PYTHON" scripts/natural_evidence_v1/summarize_actual_prefix_suffix_sensitivity.py \
  --baseline-summary-json "$BASELINE_SUMMARY_JSON" \
  --baseline-by-entry-csv "$BASELINE_BY_ENTRY_CSV" \
  "${COMPARISON_ARGS[@]}" \
  --bucketized-candidates-jsonl "$BUCKETIZED_CANDIDATES_JSONL" \
  --generated-output-count "$GENERATED_OUTPUT_COUNT" \
  --bucket-count "$BUCKET_COUNT" \
  --output-summary-json "$SUMMARY_JSON" \
  --output-by-cap-csv "$BY_CAP_CSV" \
  --output-by-entry-csv "$BY_ENTRY_CSV"
