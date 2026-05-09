#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-expsuf
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-expsuf-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-expsuf-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_diagnostic_e2e_eval_844121}"
TOPK_DIR="${TOPK_DIR:-$EVAL_DIR/actual_prefix_reference_scoring_topk64_v1}"
BUCKETIZED_DIR="${BUCKETIZED_DIR:-$EVAL_DIR/expanded_actual_prefix_bucketized_topk64_b4_minarity2_v1}"
SCORING_DIR="${SCORING_DIR:-$EVAL_DIR/expanded_actual_prefix_suffix_compatibility_topk64_b4_cap2_v1}"
VARIABLE_ARITY_DIR="${VARIABLE_ARITY_DIR:-$EVAL_DIR/expanded_actual_prefix_variable_arity_after_suffix_topk64_b4_cap2_v1}"

GENERATED_OUTPUTS="${GENERATED_OUTPUTS:-$EVAL_DIR/qwen_diagnostic_generated_outputs.jsonl}"
CANDIDATE_JSONL="${CANDIDATE_JSONL:-$TOPK_DIR/qwen_actual_prefix_topk_candidates.jsonl}"
BUCKETIZED_CANDIDATES_JSONL="${BUCKETIZED_CANDIDATES_JSONL:-$BUCKETIZED_DIR/expanded_actual_prefix_bucketized_candidates.jsonl}"
STATIC_VALIDATION_SUMMARY="${STATIC_VALIDATION_SUMMARY:-$SCORING_DIR/static_validation_summary_expanded_suffix.json}"

TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
GENERATED_OUTPUT_COUNT="${GENERATED_OUTPUT_COUNT:-14336}"
BUCKET_COUNT="${BUCKET_COUNT:-4}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-64}"
MIN_ARITY="${MIN_ARITY:-2}"
SUFFIX_WINDOW_TOKENS="${SUFFIX_WINDOW_TOKENS:-16}"
MAX_CANDIDATES_PER_BUCKET="${MAX_CANDIDATES_PER_BUCKET:-2}"
MAX_RECORDS="${MAX_RECORDS:-0}"
DELTA_NLL_THRESHOLD="${DELTA_NLL_THRESHOLD:-0.5}"
PROGRESS_EVERY="${PROGRESS_EVERY:-128}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"
START_EXPANDED_SUFFIX_COMPATIBILITY="${START_EXPANDED_SUFFIX_COMPATIBILITY:-0}"

OUTPUT_JSONL="${SCORING_DIR}/expanded_actual_prefix_suffix_compatibility.jsonl"
BY_ENTRY_CSV="${SCORING_DIR}/expanded_actual_prefix_suffix_compatibility_by_entry.csv"
SUMMARY_JSON="${SCORING_DIR}/expanded_actual_prefix_suffix_compatibility_summary.json"
PROGRESS_JSON="${SCORING_DIR}/expanded_actual_prefix_suffix_compatibility_progress.json"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$BUCKETIZED_DIR" "$SCORING_DIR" "$VARIABLE_ARITY_DIR"

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
echo "candidate_jsonl=$CANDIDATE_JSONL"
echo "bucketized_dir=$BUCKETIZED_DIR"
echo "scoring_dir=$SCORING_DIR"
echo "variable_arity_dir=$VARIABLE_ARITY_DIR"
echo "tokenizer_key=$TOKENIZER_KEY"
echo "bucket_count=$BUCKET_COUNT"
echo "candidate_top_k=$CANDIDATE_TOP_K"
echo "min_arity=$MIN_ARITY"
echo "suffix_window_tokens=$SUFFIX_WINDOW_TOKENS"
echo "max_candidates_per_bucket=$MAX_CANDIDATES_PER_BUCKET"
echo "delta_nll_threshold=$DELTA_NLL_THRESHOLD"
echo "generated_output_count=$GENERATED_OUTPUT_COUNT"
echo "dry_run_only=$DRY_RUN_ONLY"
echo "start_expanded_suffix_compatibility=$START_EXPANDED_SUFFIX_COMPATIBILITY"
echo "claim_note=expanded suffix compatibility scoring only; no training, no E2E, no payload recovery"

for required in "$GENERATED_OUTPUTS" "$CANDIDATE_JSONL"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_EXPANDED_SUFFIX_INPUT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

"$PYTHON" scripts/natural_evidence_v1/validate_static.py \
  --config "$CONFIG" \
  --summary "$STATIC_VALIDATION_SUMMARY"

"$PYTHON" -m py_compile \
  scripts/natural_evidence_v1/build_expanded_actual_prefix_bucketized_candidates.py \
  scripts/natural_evidence_v1/score_actual_prefix_suffix_compatibility.py \
  scripts/natural_evidence_v1/build_variable_arity_diagnostic.py

if [ "$DRY_RUN_ONLY" != "0" ] || [ "$START_EXPANDED_SUFFIX_COMPATIBILITY" != "1" ]; then
  echo "DRY_RUN_ONLY_OR_START_FLAG_NOT_SET: wrapper preflight completed; no expanded suffix scoring started"
  exit 0
fi

if [ -e "$BUCKETIZED_DIR/expanded_actual_prefix_bucketized_manifest.json" ]; then
  echo "EXPANDED_BUCKETIZED_OUTPUT_EXISTS_REFUSING_OVERWRITE: $BUCKETIZED_DIR" >&2
  exit 74
fi
if [ -e "$SUMMARY_JSON" ] || [ -e "$BY_ENTRY_CSV" ] || [ -e "$OUTPUT_JSONL" ]; then
  echo "EXPANDED_SUFFIX_OUTPUT_EXISTS_REFUSING_OVERWRITE: $SCORING_DIR" >&2
  exit 74
fi
if [ -e "$VARIABLE_ARITY_DIR/variable_arity_manifest.json" ]; then
  echo "EXPANDED_VARIABLE_ARITY_OUTPUT_EXISTS_REFUSING_OVERWRITE: $VARIABLE_ARITY_DIR" >&2
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

"$PYTHON" scripts/natural_evidence_v1/build_expanded_actual_prefix_bucketized_candidates.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --candidate-jsonl "$CANDIDATE_JSONL" \
  --output-dir "$BUCKETIZED_DIR" \
  --bucket-count "$BUCKET_COUNT" \
  --candidate-top-k "$CANDIDATE_TOP_K" \
  --min-arity "$MIN_ARITY" \
  --max-records "$MAX_RECORDS"

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

"$PYTHON" scripts/natural_evidence_v1/build_variable_arity_diagnostic.py \
  --compatibility-by-entry-csv "$BY_ENTRY_CSV" \
  --bucketized-candidates-jsonl "$BUCKETIZED_CANDIDATES_JSONL" \
  --output-dir "$VARIABLE_ARITY_DIR" \
  --generated-output-count "$GENERATED_OUTPUT_COUNT" \
  --bucket-count "$BUCKET_COUNT" \
  --min-arity "$MIN_ARITY" \
  --configured-min-members 2
