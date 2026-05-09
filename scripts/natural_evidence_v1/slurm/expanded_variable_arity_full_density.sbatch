#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-vadens
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-vadens-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-vadens-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"

RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_diagnostic_e2e_eval_844121}"
GENERATED_OUTPUTS="${GENERATED_OUTPUTS:-$EVAL_DIR/qwen_diagnostic_generated_outputs.jsonl}"
COMPATIBILITY_BY_ENTRY_CSV="${COMPATIBILITY_BY_ENTRY_CSV:-$EVAL_DIR/expanded_actual_prefix_suffix_compatibility_topk64_b4_cap2_v1/expanded_actual_prefix_suffix_compatibility_by_entry.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_DIR/expanded_actual_prefix_variable_arity_full_density_topk64_b4_cap2_v1}"
STATIC_VALIDATION_SUMMARY="${STATIC_VALIDATION_SUMMARY:-$OUTPUT_DIR/static_validation_summary_full_density.json}"

BUCKET_COUNT="${BUCKET_COUNT:-4}"
MIN_ARITY="${MIN_ARITY:-2}"
CONFIGURED_MIN_MEMBERS="${CONFIGURED_MIN_MEMBERS:-2}"
TOKEN_COUNT_MODE="${TOKEN_COUNT_MODE:-tokenizer}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"
START_FULL_DENSITY_AUDIT="${START_FULL_DENSITY_AUDIT:-0}"

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

export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "config=$CONFIG"
echo "generated_outputs=$GENERATED_OUTPUTS"
echo "compatibility_by_entry_csv=$COMPATIBILITY_BY_ENTRY_CSV"
echo "output_dir=$OUTPUT_DIR"
echo "bucket_count=$BUCKET_COUNT"
echo "min_arity=$MIN_ARITY"
echo "configured_min_members=$CONFIGURED_MIN_MEMBERS"
echo "token_count_mode=$TOKEN_COUNT_MODE"
echo "dry_run_only=$DRY_RUN_ONLY"
echo "start_full_density_audit=$START_FULL_DENSITY_AUDIT"
echo "claim_note=full denominator variable-arity density audit only; no training, no E2E, no payload recovery"

for required in "$GENERATED_OUTPUTS" "$COMPATIBILITY_BY_ENTRY_CSV"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_FULL_DENSITY_INPUT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

"$PYTHON" scripts/natural_evidence_v1/validate_static.py \
  --config "$CONFIG" \
  --summary "$STATIC_VALIDATION_SUMMARY"

"$PYTHON" -m py_compile scripts/natural_evidence_v1/audit_variable_arity_full_density.py

if [ "$DRY_RUN_ONLY" != "0" ] || [ "$START_FULL_DENSITY_AUDIT" != "1" ]; then
  echo "DRY_RUN_ONLY_OR_START_FLAG_NOT_SET: wrapper preflight completed; no full-density audit started"
  exit 0
fi

if [ -e "$OUTPUT_DIR/variable_arity_full_density_summary.json" ] || \
   [ -e "$OUTPUT_DIR/variable_arity_full_density_by_slice.csv" ] || \
   [ -e "$OUTPUT_DIR/variable_arity_full_density_by_generated_row.csv" ]; then
  echo "VARIABLE_ARITY_FULL_DENSITY_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" scripts/natural_evidence_v1/audit_variable_arity_full_density.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --generated-outputs "$GENERATED_OUTPUTS" \
  --compatibility-by-entry-csv "$COMPATIBILITY_BY_ENTRY_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --bucket-count "$BUCKET_COUNT" \
  --min-arity "$MIN_ARITY" \
  --configured-min-members "$CONFIGURED_MIN_MEMBERS" \
  --token-count-mode "$TOKEN_COUNT_MODE"
