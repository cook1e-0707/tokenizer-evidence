#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-vaprenull
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-vaprenull-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-vaprenull-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"

RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_diagnostic_e2e_eval_844121}"
COMPATIBILITY_JSONL="${COMPATIBILITY_JSONL:-$EVAL_DIR/expanded_actual_prefix_suffix_compatibility_topk64_b4_cap2_v1/expanded_actual_prefix_suffix_compatibility.jsonl}"
COMPATIBILITY_BY_ENTRY_CSV="${COMPATIBILITY_BY_ENTRY_CSV:-$EVAL_DIR/expanded_actual_prefix_suffix_compatibility_topk64_b4_cap2_v1/expanded_actual_prefix_suffix_compatibility_by_entry.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_DIR/expanded_actual_prefix_variable_arity_pre_null_topk64_b4_cap2_v1}"
STATIC_VALIDATION_SUMMARY="${STATIC_VALIDATION_SUMMARY:-$OUTPUT_DIR/static_validation_summary_variable_arity_pre_null.json}"

BUCKET_COUNT="${BUCKET_COUNT:-4}"
MIN_ARITY="${MIN_ARITY:-2}"
WRONG_KEY_COUNT="${WRONG_KEY_COUNT:-4}"
QUERY_BUDGETS="${QUERY_BUDGETS:-64,128,256,512}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"
START_VARIABLE_ARITY_PRE_NULL="${START_VARIABLE_ARITY_PRE_NULL:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$OUTPUT_DIR"

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

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "config=$CONFIG"
echo "compatibility_jsonl=$COMPATIBILITY_JSONL"
echo "compatibility_by_entry_csv=$COMPATIBILITY_BY_ENTRY_CSV"
echo "output_dir=$OUTPUT_DIR"
echo "bucket_count=$BUCKET_COUNT"
echo "min_arity=$MIN_ARITY"
echo "wrong_key_count=$WRONG_KEY_COUNT"
echo "query_budgets=$QUERY_BUDGETS"
echo "dry_run_only=$DRY_RUN_ONLY"
echo "start_variable_arity_pre_null=$START_VARIABLE_ARITY_PRE_NULL"
echo "claim_note=variable-arity pre-null and variable-radix codec preflight only; no training, no E2E, no full FAR"

for required in "$COMPATIBILITY_JSONL" "$COMPATIBILITY_BY_ENTRY_CSV"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_VARIABLE_ARITY_PRE_NULL_INPUT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

"$PYTHON" scripts/natural_evidence_v1/validate_static.py \
  --config "$CONFIG" \
  --summary "$STATIC_VALIDATION_SUMMARY"

"$PYTHON" -m py_compile \
  scripts/natural_evidence_v1/variable_arity_pre_null.py \
  src/core/payload_codec.py

if [ "$DRY_RUN_ONLY" != "0" ] || [ "$START_VARIABLE_ARITY_PRE_NULL" != "1" ]; then
  echo "DRY_RUN_ONLY_OR_START_FLAG_NOT_SET: wrapper preflight completed; no variable-arity pre-null started"
  exit 0
fi

if [ -e "$OUTPUT_DIR/variable_arity_pre_null_summary.json" ] || \
   [ -e "$OUTPUT_DIR/variable_arity_pre_null_decodes.csv" ] || \
   [ -e "$OUTPUT_DIR/variable_radix_preflight.csv" ]; then
  echo "VARIABLE_ARITY_PRE_NULL_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" scripts/natural_evidence_v1/variable_arity_pre_null.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --compatibility-jsonl "$COMPATIBILITY_JSONL" \
  --compatibility-by-entry-csv "$COMPATIBILITY_BY_ENTRY_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --bucket-count "$BUCKET_COUNT" \
  --min-arity "$MIN_ARITY" \
  --wrong-key-count "$WRONG_KEY_COUNT" \
  --query-budgets "$QUERY_BUDGETS"
