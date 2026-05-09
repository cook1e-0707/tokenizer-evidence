#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-align
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=pomplun
#SBATCH --account=cs_yinxin.wan
#SBATCH --qos=pomplun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-align-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-align-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e}"
EXPANSION_ROOT="${EXPANSION_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_diagnostic_e2e_eval_844121}"
GENERATED_OUTPUTS="${GENERATED_OUTPUTS:-$EVAL_DIR/qwen_diagnostic_generated_outputs.jsonl}"
BUCKET_OBSERVATIONS="${BUCKET_OBSERVATIONS:-$EVAL_DIR/qwen_diagnostic_bucket_observations.jsonl}"
DECODE_TRACE_CSV="${DECODE_TRACE_CSV:-$EVAL_DIR/qwen_diagnostic_decode_trace.csv}"
PROGRESS_JSON="${PROGRESS_JSON:-$EVAL_DIR/qwen_diagnostic_e2e_eval_progress.json}"
BUCKET_BANK_ENTRIES="${BUCKET_BANK_ENTRIES:-$EXPANSION_ROOT/bucket_banks_4way_strict/qwen_bucket_bank_entries.jsonl}"
COMPATIBILITY_JSONL="${COMPATIBILITY_JSONL:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/probability_preserving_compatibility_repair/qwen_counterfactual_compatibility_with_probabilities.jsonl}"
COMPATIBILITY_BY_ENTRY_CSV="${COMPATIBILITY_BY_ENTRY_CSV:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/probability_preserving_compatibility_repair/compatibility_filtered_bank_dry_run_by_entry.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_DIR/verifier_alignment_diagnosis_v1}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MAX_EXAMPLES="${MAX_EXAMPLES:-80}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "eval_dir=$EVAL_DIR"
echo "generated_outputs=$GENERATED_OUTPUTS"
echo "bucket_observations=$BUCKET_OBSERVATIONS"
echo "decode_trace_csv=$DECODE_TRACE_CSV"
echo "progress_json=$PROGRESS_JSON"
echo "bucket_bank_entries=$BUCKET_BANK_ENTRIES"
echo "compatibility_jsonl=$COMPATIBILITY_JSONL"
echo "compatibility_by_entry_csv=$COMPATIBILITY_BY_ENTRY_CSV"
echo "output_dir=$OUTPUT_DIR"
echo "tokenizer_name=$TOKENIZER_NAME"
echo "claim_note=verifier alignment diagnosis only; not payload recovery"

for required in \
  "$GENERATED_OUTPUTS" \
  "$BUCKET_OBSERVATIONS" \
  "$DECODE_TRACE_CSV" \
  "$PROGRESS_JSON" \
  "$BUCKET_BANK_ENTRIES" \
  "$COMPATIBILITY_JSONL" \
  "$COMPATIBILITY_BY_ENTRY_CSV"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

if [ -e "$OUTPUT_DIR/verifier_alignment_summary.json" ]; then
  echo "VERIFIER_ALIGNMENT_DIAGNOSIS_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" scripts/natural_evidence_v1/diagnose_verifier_alignment.py \
  --generated-outputs "$GENERATED_OUTPUTS" \
  --bucket-observations "$BUCKET_OBSERVATIONS" \
  --decode-trace-csv "$DECODE_TRACE_CSV" \
  --progress-json "$PROGRESS_JSON" \
  --bucket-bank-entries "$BUCKET_BANK_ENTRIES" \
  --compatibility-jsonl "$COMPATIBILITY_JSONL" \
  --compatibility-by-entry-csv "$COMPATIBILITY_BY_ENTRY_CSV" \
  --bucket-count 4 \
  --tokenizer-name "$TOKENIZER_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --max-examples "$MAX_EXAMPLES"
