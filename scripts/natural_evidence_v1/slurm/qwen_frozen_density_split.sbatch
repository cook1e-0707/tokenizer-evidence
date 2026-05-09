#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-fdens
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-fdens-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-fdens-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192}"
EXPANSION_ROOT="${EXPANSION_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3}"
REFERENCE_OUTPUTS="${REFERENCE_OUTPUTS:-$REFERENCE_ROOT/reference_outputs/qwen_reference_outputs.jsonl}"
BANK_ENTRIES="${BANK_ENTRIES:-$EXPANSION_ROOT/bucket_banks_4way_strict/qwen_bucket_bank_entries.jsonl}"
COMPATIBILITY_BY_ENTRY_CSV="${COMPATIBILITY_BY_ENTRY_CSV:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/probability_preserving_compatibility_repair/compatibility_filtered_bank_dry_run_by_entry.csv}"
FREEZE_ID="${FREEZE_ID:-qwen_density_split_v1}"
HELDOUT_COUNT="${HELDOUT_COUNT:-2048}"
ORGANIC_COUNT="${ORGANIC_COUNT:-2048}"
SPLIT_SEED="${SPLIT_SEED:-20260505}"
FREEZE_OUTPUT_DIR="${FREEZE_OUTPUT_DIR:-$EXPANSION_ROOT/frozen_density_splits/$FREEZE_ID}"
DENSITY_OUTPUT_DIR="${DENSITY_OUTPUT_DIR:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/frozen_heldout_min1_compatible_density}"

cd "$REPO_HOME"
mkdir -p "$EXPANSION_ROOT/slurm"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "config=$CONFIG"
echo "reference_outputs=$REFERENCE_OUTPUTS"
echo "bank_entries=$BANK_ENTRIES"
echo "compatibility_by_entry_csv=$COMPATIBILITY_BY_ENTRY_CSV"
echo "freeze_id=$FREEZE_ID"
echo "heldout_count=$HELDOUT_COUNT"
echo "organic_count=$ORGANIC_COUNT"
echo "split_seed=$SPLIT_SEED"
echo "freeze_output_dir=$FREEZE_OUTPUT_DIR"
echo "density_output_dir=$DENSITY_OUTPUT_DIR"
echo "claim_note=frozen density inputs and held-out density diagnostic, not payload recovery"

if [ ! -s "$REFERENCE_OUTPUTS" ]; then
  echo "REFERENCE_OUTPUTS_MISSING_OR_EMPTY: $REFERENCE_OUTPUTS" >&2
  exit 2
fi
if [ ! -s "$BANK_ENTRIES" ]; then
  echo "BANK_ENTRIES_MISSING_OR_EMPTY: $BANK_ENTRIES" >&2
  exit 2
fi
if [ ! -s "$COMPATIBILITY_BY_ENTRY_CSV" ]; then
  echo "COMPATIBILITY_BY_ENTRY_CSV_MISSING_OR_EMPTY: $COMPATIBILITY_BY_ENTRY_CSV" >&2
  exit 2
fi
if [ -e "$FREEZE_OUTPUT_DIR/density_prompt_split_manifest.json" ]; then
  echo "FROZEN_DENSITY_SPLIT_EXISTS_REFUSING_OVERWRITE: $FREEZE_OUTPUT_DIR" >&2
  exit 74
fi
if [ -e "$DENSITY_OUTPUT_DIR/compatible_density_summary.json" ] || [ -e "$DENSITY_OUTPUT_DIR/compatible_density_by_split.csv" ]; then
  echo "FROZEN_HELDOUT_DENSITY_OUTPUT_EXISTS_REFUSING_OVERWRITE: $DENSITY_OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" scripts/natural_evidence_v1/freeze_density_prompt_splits.py \
  --config "$CONFIG" \
  --reference-outputs "$REFERENCE_OUTPUTS" \
  --output-dir "$FREEZE_OUTPUT_DIR" \
  --freeze-id "$FREEZE_ID" \
  --heldout-count "$HELDOUT_COUNT" \
  --organic-count "$ORGANIC_COUNT" \
  --seed "$SPLIT_SEED"

"$PYTHON" scripts/natural_evidence_v1/audit_compatible_density.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --entries "$BANK_ENTRIES" \
  --compatibility-by-entry-csv "$COMPATIBILITY_BY_ENTRY_CSV" \
  --reference-outputs "$FREEZE_OUTPUT_DIR/heldout_reference_outputs.jsonl" \
  --output-dir "$DENSITY_OUTPUT_DIR"
