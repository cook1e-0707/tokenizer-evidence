#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-salvage
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-salvage-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-salvage-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e}"
EXPANSION_ROOT="${EXPANSION_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_diagnostic_e2e_eval_844121}"
GENERATED_OUTPUTS="${GENERATED_OUTPUTS:-$EVAL_DIR/qwen_diagnostic_generated_outputs.jsonl}"
BUCKET_OBSERVATIONS="${BUCKET_OBSERVATIONS:-$EVAL_DIR/qwen_diagnostic_bucket_observations.jsonl}"
BUCKET_BANK_ENTRIES="${BUCKET_BANK_ENTRIES:-$EXPANSION_ROOT/bucket_banks_4way_strict/qwen_bucket_bank_entries.jsonl}"
COMPATIBILITY_JSONL="${COMPATIBILITY_JSONL:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/probability_preserving_compatibility_repair/qwen_counterfactual_compatibility_with_probabilities.jsonl}"
COMPATIBILITY_BY_ENTRY_CSV="${COMPATIBILITY_BY_ENTRY_CSV:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/probability_preserving_compatibility_repair/compatibility_filtered_bank_dry_run_by_entry.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_DIR/actual_prefix_salvage_diagnostic_v1}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
QUERY_BUDGETS="${QUERY_BUDGETS:-64,128,256,512}"
LCP_THRESHOLDS="${LCP_THRESHOLDS:-0,1,4,8,16,32}"
MAX_EXAMPLES="${MAX_EXAMPLES:-80}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "config=$CONFIG"
echo "eval_dir=$EVAL_DIR"
echo "generated_outputs=$GENERATED_OUTPUTS"
echo "bucket_observations=$BUCKET_OBSERVATIONS"
echo "bucket_bank_entries=$BUCKET_BANK_ENTRIES"
echo "compatibility_jsonl=$COMPATIBILITY_JSONL"
echo "compatibility_by_entry_csv=$COMPATIBILITY_BY_ENTRY_CSV"
echo "output_dir=$OUTPUT_DIR"
echo "tokenizer_name=$TOKENIZER_NAME"
echo "query_budgets=$QUERY_BUDGETS"
echo "lcp_thresholds=$LCP_THRESHOLDS"
echo "claim_note=actual-prefix salvage diagnostic only; no model scoring, no training, not payload recovery"

for required in \
  "$GENERATED_OUTPUTS" \
  "$BUCKET_OBSERVATIONS" \
  "$BUCKET_BANK_ENTRIES" \
  "$COMPATIBILITY_JSONL" \
  "$COMPATIBILITY_BY_ENTRY_CSV"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

if [ -e "$OUTPUT_DIR/actual_prefix_salvage_summary.json" ]; then
  echo "ACTUAL_PREFIX_SALVAGE_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" scripts/natural_evidence_v1/salvage_actual_prefix_diagnostic.py \
  --config "$CONFIG" \
  --generated-outputs "$GENERATED_OUTPUTS" \
  --bucket-observations "$BUCKET_OBSERVATIONS" \
  --bucket-bank-entries "$BUCKET_BANK_ENTRIES" \
  --compatibility-jsonl "$COMPATIBILITY_JSONL" \
  --compatibility-by-entry-csv "$COMPATIBILITY_BY_ENTRY_CSV" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --bucket-count 4 \
  --query-budgets "$QUERY_BUDGETS" \
  --lcp-thresholds "$LCP_THRESHOLDS" \
  --output-dir "$OUTPUT_DIR" \
  --max-examples "$MAX_EXAMPLES"
