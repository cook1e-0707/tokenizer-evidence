#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-density
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-density-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-density-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192}"
EXPANSION_ROOT="${EXPANSION_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3}"
BANK_ENTRIES="${BANK_ENTRIES:-$EXPANSION_ROOT/bucket_banks_4way_strict/qwen_bucket_bank_entries.jsonl}"
REFERENCE_OUTPUTS="${REFERENCE_OUTPUTS:-$REFERENCE_ROOT/reference_outputs/qwen_reference_outputs.jsonl}"
COMPATIBILITY_BY_ENTRY_CSV="${COMPATIBILITY_BY_ENTRY_CSV:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/probability_preserving_compatibility_repair/compatibility_filtered_bank_dry_run_by_entry.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/min1_compatible_density}"
HELDOUT_PROMPT_FAMILIES="${HELDOUT_PROMPT_FAMILIES:-PF4}"

cd "$REPO_HOME"
mkdir -p "$EXPANSION_ROOT/slurm"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "config=$CONFIG"
echo "entries=$BANK_ENTRIES"
echo "reference_outputs=$REFERENCE_OUTPUTS"
echo "compatibility_by_entry_csv=$COMPATIBILITY_BY_ENTRY_CSV"
echo "output_dir=$OUTPUT_DIR"
echo "heldout_prompt_families=$HELDOUT_PROMPT_FAMILIES"
echo "claim_note=compatible density diagnostic, not payload recovery"

if [ ! -s "$BANK_ENTRIES" ]; then
  echo "BANK_ENTRIES_MISSING_OR_EMPTY: $BANK_ENTRIES" >&2
  exit 2
fi
if [ ! -s "$REFERENCE_OUTPUTS" ]; then
  echo "REFERENCE_OUTPUTS_MISSING_OR_EMPTY: $REFERENCE_OUTPUTS" >&2
  exit 2
fi
if [ ! -s "$COMPATIBILITY_BY_ENTRY_CSV" ]; then
  echo "COMPATIBILITY_BY_ENTRY_CSV_MISSING_OR_EMPTY: $COMPATIBILITY_BY_ENTRY_CSV" >&2
  exit 2
fi
if [ -e "$OUTPUT_DIR/compatible_density_summary.json" ] || [ -e "$OUTPUT_DIR/compatible_density_by_split.csv" ]; then
  echo "COMPATIBLE_DENSITY_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" scripts/natural_evidence_v1/audit_compatible_density.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --entries "$BANK_ENTRIES" \
  --compatibility-by-entry-csv "$COMPATIBILITY_BY_ENTRY_CSV" \
  --reference-outputs "$REFERENCE_OUTPUTS" \
  --output-dir "$OUTPUT_DIR" \
  --heldout-prompt-families "$HELDOUT_PROMPT_FAMILIES"
