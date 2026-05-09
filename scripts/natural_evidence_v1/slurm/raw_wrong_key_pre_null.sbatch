#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-prenull
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=pomplun
#SBATCH --account=cs_yinxin.wan
#SBATCH --qos=pomplun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-prenull-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-prenull-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
SOURCE_ROOT="${SOURCE_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192}"
EXPANSION_ROOT="${EXPANSION_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3}"
REFERENCE_OUTPUTS="${REFERENCE_OUTPUTS:-$SOURCE_ROOT/reference_outputs/qwen_reference_outputs.jsonl}"
COMPATIBILITY_JSONL="${COMPATIBILITY_JSONL:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/probability_preserving_compatibility_repair/qwen_counterfactual_compatibility_with_probabilities.jsonl}"
COMPATIBILITY_BY_ENTRY_CSV="${COMPATIBILITY_BY_ENTRY_CSV:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/probability_preserving_compatibility_repair/compatibility_filtered_bank_dry_run_by_entry.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/raw_wrong_key_pre_null}"
WRONG_KEY_COUNT="${WRONG_KEY_COUNT:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-50}"

cd "$REPO_HOME"
mkdir -p "$EXPANSION_ROOT/slurm"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "config=$CONFIG"
echo "reference_outputs=$REFERENCE_OUTPUTS"
echo "compatibility_jsonl=$COMPATIBILITY_JSONL"
echo "compatibility_by_entry_csv=$COMPATIBILITY_BY_ENTRY_CSV"
echo "output_dir=$OUTPUT_DIR"
echo "wrong_key_count=$WRONG_KEY_COUNT"
echo "claim_note=raw/wrong-key pre-null diagnostic, not full FAR and not payload recovery evidence"

if [ ! -s "$REFERENCE_OUTPUTS" ]; then
  echo "REFERENCE_OUTPUTS_MISSING_OR_EMPTY: $REFERENCE_OUTPUTS" >&2
  exit 2
fi
if [ ! -s "$COMPATIBILITY_JSONL" ]; then
  echo "COMPATIBILITY_JSONL_MISSING_OR_EMPTY: $COMPATIBILITY_JSONL" >&2
  exit 2
fi
if [ ! -s "$COMPATIBILITY_BY_ENTRY_CSV" ]; then
  echo "COMPATIBILITY_BY_ENTRY_CSV_MISSING_OR_EMPTY: $COMPATIBILITY_BY_ENTRY_CSV" >&2
  exit 2
fi
if [ -e "$OUTPUT_DIR/raw_wrong_key_pre_null_summary.json" ] || [ -e "$OUTPUT_DIR/raw_wrong_key_pre_null_decodes.csv" ]; then
  echo "RAW_WRONG_KEY_PRE_NULL_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" scripts/natural_evidence_v1/raw_wrong_key_pre_null.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --reference-outputs "$REFERENCE_OUTPUTS" \
  --compatibility-jsonl "$COMPATIBILITY_JSONL" \
  --compatibility-by-entry-csv "$COMPATIBILITY_BY_ENTRY_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --wrong-key-count "$WRONG_KEY_COUNT" \
  --max-examples "$MAX_EXAMPLES"
