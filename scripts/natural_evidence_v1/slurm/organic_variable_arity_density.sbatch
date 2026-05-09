#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-orgdens
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
#SBATCH --time=12:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-orgdens-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-orgdens-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e}"
PHASE_ROOT="${PHASE_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_diagnostic_e2e_eval_844121}"
ORGANIC_ROOT="${ORGANIC_ROOT:-$RUN_ROOT/eval/qwen_organic_variable_arity_density_topk64_b4_cap2_v1}"

ORGANIC_PROMPTS="${ORGANIC_PROMPTS:-$PHASE_ROOT/frozen_density_splits/qwen_density_split_v1/organic_prompts.jsonl}"
BASE_GENERATED_OUTPUTS="${BASE_GENERATED_OUTPUTS:-$EVAL_DIR/qwen_diagnostic_generated_outputs.jsonl}"
BASE_COMPATIBILITY_BY_ENTRY_CSV="${BASE_COMPATIBILITY_BY_ENTRY_CSV:-$EVAL_DIR/expanded_actual_prefix_suffix_compatibility_topk64_b4_cap2_v1/expanded_actual_prefix_suffix_compatibility_by_entry.csv}"

ORGANIC_OUTPUT_DIR="${ORGANIC_OUTPUT_DIR:-$ORGANIC_ROOT/organic_outputs}"
PLAN_DIR="${PLAN_DIR:-$ORGANIC_ROOT/actual_prefix_scoring_plan}"
TOPK_DIR="${TOPK_DIR:-$ORGANIC_ROOT/actual_prefix_reference_scoring_topk64}"
BUCKETIZED_DIR="${BUCKETIZED_DIR:-$ORGANIC_ROOT/expanded_actual_prefix_bucketized_topk64_b4_minarity2}"
SCORING_DIR="${SCORING_DIR:-$ORGANIC_ROOT/expanded_actual_prefix_suffix_compatibility_topk64_b4_cap2}"
COMBINED_DIR="${COMBINED_DIR:-$ORGANIC_ROOT/combined_density_inputs}"
DENSITY_DIR="${DENSITY_DIR:-$ORGANIC_ROOT/full_density_with_organic}"

ORGANIC_GENERATED_OUTPUTS="${ORGANIC_GENERATED_OUTPUTS:-$ORGANIC_OUTPUT_DIR/qwen_organic_generated_outputs.jsonl}"
PLAN_INPUT_JSONL="${PLAN_INPUT_JSONL:-$PLAN_DIR/actual_prefix_scoring_input.jsonl}"
PLAN_MANIFEST_JSON="${PLAN_MANIFEST_JSON:-$PLAN_DIR/actual_prefix_scoring_manifest.json}"
TOPK_JSONL="${TOPK_JSONL:-$TOPK_DIR/qwen_actual_prefix_topk_candidates.jsonl}"
TOPK_SUMMARY_JSON="${TOPK_SUMMARY_JSON:-$TOPK_DIR/qwen_actual_prefix_reference_scoring_summary.json}"
TOPK_PROGRESS_JSON="${TOPK_PROGRESS_JSON:-$TOPK_DIR/qwen_actual_prefix_reference_scoring_progress.json}"
BUCKETIZED_JSONL="${BUCKETIZED_JSONL:-$BUCKETIZED_DIR/expanded_actual_prefix_bucketized_candidates.jsonl}"
SCORING_JSONL="${SCORING_JSONL:-$SCORING_DIR/expanded_actual_prefix_suffix_compatibility.jsonl}"
SCORING_BY_ENTRY_CSV="${SCORING_BY_ENTRY_CSV:-$SCORING_DIR/expanded_actual_prefix_suffix_compatibility_by_entry.csv}"
SCORING_SUMMARY_JSON="${SCORING_SUMMARY_JSON:-$SCORING_DIR/expanded_actual_prefix_suffix_compatibility_summary.json}"
SCORING_PROGRESS_JSON="${SCORING_PROGRESS_JSON:-$SCORING_DIR/expanded_actual_prefix_suffix_compatibility_progress.json}"
COMBINED_GENERATED_OUTPUTS="${COMBINED_GENERATED_OUTPUTS:-$COMBINED_DIR/generated_outputs_with_organic.jsonl}"
COMBINED_COMPATIBILITY_BY_ENTRY_CSV="${COMBINED_COMPATIBILITY_BY_ENTRY_CSV:-$COMBINED_DIR/compatibility_by_entry_with_organic.csv}"
COMBINED_SUMMARY_JSON="${COMBINED_SUMMARY_JSON:-$COMBINED_DIR/density_input_combiner_summary.json}"
STATIC_VALIDATION_SUMMARY="${STATIC_VALIDATION_SUMMARY:-$ORGANIC_ROOT/static_validation_summary_organic_density.json}"

TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
ORGANIC_COUNT="${ORGANIC_COUNT:-2048}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-64}"
SCORING_BATCH_SIZE="${SCORING_BATCH_SIZE:-4}"
BUCKET_COUNT="${BUCKET_COUNT:-4}"
MIN_ARITY="${MIN_ARITY:-2}"
SUFFIX_WINDOW_TOKENS="${SUFFIX_WINDOW_TOKENS:-16}"
MAX_CANDIDATES_PER_BUCKET="${MAX_CANDIDATES_PER_BUCKET:-2}"
DELTA_NLL_THRESHOLD="${DELTA_NLL_THRESHOLD:-0.5}"
PROGRESS_EVERY="${PROGRESS_EVERY:-128}"
MAX_RECORDS="${MAX_RECORDS:-0}"
TOKEN_COUNT_MODE="${TOKEN_COUNT_MODE:-tokenizer}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"
START_ORGANIC_DENSITY="${START_ORGANIC_DENSITY:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$ORGANIC_OUTPUT_DIR" "$PLAN_DIR" "$TOPK_DIR" "$BUCKETIZED_DIR" "$SCORING_DIR" "$COMBINED_DIR" "$DENSITY_DIR"

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
echo "organic_prompts=$ORGANIC_PROMPTS"
echo "organic_root=$ORGANIC_ROOT"
echo "base_generated_outputs=$BASE_GENERATED_OUTPUTS"
echo "base_compatibility_by_entry_csv=$BASE_COMPATIBILITY_BY_ENTRY_CSV"
echo "organic_generated_outputs=$ORGANIC_GENERATED_OUTPUTS"
echo "topk_jsonl=$TOPK_JSONL"
echo "bucketized_jsonl=$BUCKETIZED_JSONL"
echo "scoring_by_entry_csv=$SCORING_BY_ENTRY_CSV"
echo "combined_generated_outputs=$COMBINED_GENERATED_OUTPUTS"
echo "combined_compatibility_by_entry_csv=$COMBINED_COMPATIBILITY_BY_ENTRY_CSV"
echo "density_dir=$DENSITY_DIR"
echo "organic_count=$ORGANIC_COUNT"
echo "candidate_top_k=$CANDIDATE_TOP_K"
echo "bucket_count=$BUCKET_COUNT"
echo "min_arity=$MIN_ARITY"
echo "max_candidates_per_bucket=$MAX_CANDIDATES_PER_BUCKET"
echo "dry_run_only=$DRY_RUN_ONLY"
echo "start_organic_density=$START_ORGANIC_DENSITY"
echo "claim_note=organic generated-output density audit only; no training, no E2E, no payload recovery, no FAR"

for required in "$ORGANIC_PROMPTS" "$BASE_GENERATED_OUTPUTS" "$BASE_COMPATIBILITY_BY_ENTRY_CSV"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ORGANIC_DENSITY_INPUT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

"$PYTHON" scripts/natural_evidence_v1/validate_static.py \
  --config "$CONFIG" \
  --summary "$STATIC_VALIDATION_SUMMARY"

"$PYTHON" -m py_compile \
  scripts/natural_evidence_v1/generate_reference_outputs.py \
  scripts/natural_evidence_v1/prepare_actual_prefix_scoring_plan.py \
  scripts/natural_evidence_v1/score_actual_prefix_reference_candidates.py \
  scripts/natural_evidence_v1/build_expanded_actual_prefix_bucketized_candidates.py \
  scripts/natural_evidence_v1/score_actual_prefix_suffix_compatibility.py \
  scripts/natural_evidence_v1/combine_variable_arity_density_inputs.py \
  scripts/natural_evidence_v1/audit_variable_arity_full_density.py

if [ "$DRY_RUN_ONLY" != "0" ] || [ "$START_ORGANIC_DENSITY" != "1" ]; then
  echo "DRY_RUN_ONLY_OR_START_FLAG_NOT_SET: wrapper preflight completed; no organic density job started"
  exit 0
fi

for output in \
  "$ORGANIC_GENERATED_OUTPUTS" \
  "$PLAN_INPUT_JSONL" \
  "$PLAN_MANIFEST_JSON" \
  "$TOPK_JSONL" \
  "$TOPK_SUMMARY_JSON" \
  "$BUCKETIZED_DIR/expanded_actual_prefix_bucketized_manifest.json" \
  "$SCORING_JSONL" \
  "$SCORING_BY_ENTRY_CSV" \
  "$SCORING_SUMMARY_JSON" \
  "$COMBINED_GENERATED_OUTPUTS" \
  "$COMBINED_COMPATIBILITY_BY_ENTRY_CSV" \
  "$DENSITY_DIR/variable_arity_full_density_summary.json"; do
  if [ -e "$output" ]; then
    echo "ORGANIC_DENSITY_OUTPUT_EXISTS_REFUSING_OVERWRITE: $output" >&2
    exit 74
  fi
done

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

"$PYTHON" scripts/natural_evidence_v1/generate_reference_outputs.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --prompt-bank-jsonl "$ORGANIC_PROMPTS" \
  --output-jsonl "$ORGANIC_GENERATED_OUTPUTS" \
  --batch-size "$GEN_BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature 0.0 \
  --require-cuda \
  --max-records "$ORGANIC_COUNT"

"$PYTHON" scripts/natural_evidence_v1/prepare_actual_prefix_scoring_plan.py \
  --config "$CONFIG" \
  --generated-outputs "$ORGANIC_GENERATED_OUTPUTS" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --output-dir "$PLAN_DIR"

"$PYTHON" scripts/natural_evidence_v1/score_actual_prefix_reference_candidates.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --input-jsonl "$PLAN_INPUT_JSONL" \
  --manifest-json "$PLAN_MANIFEST_JSON" \
  --output-jsonl "$TOPK_JSONL" \
  --summary-json "$TOPK_SUMMARY_JSON" \
  --candidate-top-k "$CANDIDATE_TOP_K" \
  --batch-size "$SCORING_BATCH_SIZE" \
  --max-records "$MAX_RECORDS" \
  --stream-output \
  --progress-json "$TOPK_PROGRESS_JSON" \
  --progress-every "$PROGRESS_EVERY" \
  --require-cuda

"$PYTHON" scripts/natural_evidence_v1/build_expanded_actual_prefix_bucketized_candidates.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --candidate-jsonl "$TOPK_JSONL" \
  --output-dir "$BUCKETIZED_DIR" \
  --bucket-count "$BUCKET_COUNT" \
  --candidate-top-k "$CANDIDATE_TOP_K" \
  --min-arity "$MIN_ARITY" \
  --max-records "$MAX_RECORDS"

"$PYTHON" scripts/natural_evidence_v1/score_actual_prefix_suffix_compatibility.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --generated-outputs "$ORGANIC_GENERATED_OUTPUTS" \
  --bucketized-candidates-jsonl "$BUCKETIZED_JSONL" \
  --output-jsonl "$SCORING_JSONL" \
  --by-entry-csv "$SCORING_BY_ENTRY_CSV" \
  --summary-json "$SCORING_SUMMARY_JSON" \
  --progress-json "$SCORING_PROGRESS_JSON" \
  --suffix-window-tokens "$SUFFIX_WINDOW_TOKENS" \
  --max-candidates-per-bucket "$MAX_CANDIDATES_PER_BUCKET" \
  --max-records "$MAX_RECORDS" \
  --delta-nll-threshold "$DELTA_NLL_THRESHOLD" \
  --progress-every "$PROGRESS_EVERY" \
  --require-cuda

"$PYTHON" scripts/natural_evidence_v1/combine_variable_arity_density_inputs.py \
  --base-generated-outputs "$BASE_GENERATED_OUTPUTS" \
  --base-compatibility-by-entry-csv "$BASE_COMPATIBILITY_BY_ENTRY_CSV" \
  --organic-generated-outputs "$ORGANIC_GENERATED_OUTPUTS" \
  --organic-compatibility-by-entry-csv "$SCORING_BY_ENTRY_CSV" \
  --output-generated-outputs "$COMBINED_GENERATED_OUTPUTS" \
  --output-compatibility-by-entry-csv "$COMBINED_COMPATIBILITY_BY_ENTRY_CSV" \
  --summary-json "$COMBINED_SUMMARY_JSON"

"$PYTHON" scripts/natural_evidence_v1/audit_variable_arity_full_density.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --generated-outputs "$COMBINED_GENERATED_OUTPUTS" \
  --compatibility-by-entry-csv "$COMBINED_COMPATIBILITY_BY_ENTRY_CSV" \
  --output-dir "$DENSITY_DIR" \
  --bucket-count "$BUCKET_COUNT" \
  --min-arity "$MIN_ARITY" \
  --configured-min-members 2 \
  --token-count-mode "$TOKEN_COUNT_MODE"
