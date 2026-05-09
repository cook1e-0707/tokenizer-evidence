#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-isuf
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-isuf-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-isuf-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
SOURCE_ROOT="${SOURCE_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192}"
EXPANSION_ROOT="${EXPANSION_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3}"
REFERENCE_OUTPUTS="${REFERENCE_OUTPUTS:-$SOURCE_ROOT/reference_outputs/qwen_reference_outputs.jsonl}"
CANDIDATE_JSONL="${CANDIDATE_JSONL:-$EXPANSION_ROOT/bucket_banks_4way_strict/qwen_counterfactual_candidates_from_bank_entries.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/invalid_suffix_review}"
SUFFIX_WINDOW_TOKENS="${SUFFIX_WINDOW_TOKENS:-16}"
MAX_EXAMPLES="${MAX_EXAMPLES:-50}"

cd "$REPO_HOME"
mkdir -p "$EXPANSION_ROOT/slurm"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "config=$CONFIG"
echo "reference_outputs=$REFERENCE_OUTPUTS"
echo "candidate_jsonl=$CANDIDATE_JSONL"
echo "output_dir=$OUTPUT_DIR"
echo "suffix_window_tokens=$SUFFIX_WINDOW_TOKENS"
echo "claim_note=invalid suffix diagnostic, not payload recovery"

if [ ! -s "$REFERENCE_OUTPUTS" ]; then
  echo "REFERENCE_OUTPUTS_MISSING_OR_EMPTY: $REFERENCE_OUTPUTS" >&2
  exit 2
fi
if [ ! -s "$CANDIDATE_JSONL" ]; then
  echo "CANDIDATE_JSONL_MISSING_OR_EMPTY: $CANDIDATE_JSONL" >&2
  exit 2
fi
if [ -e "$OUTPUT_DIR/invalid_suffix_review_summary.json" ] || [ -e "$OUTPUT_DIR/invalid_suffix_records.csv" ]; then
  echo "INVALID_SUFFIX_REVIEW_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" scripts/natural_evidence_v1/review_invalid_suffix_records.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --reference-outputs "$REFERENCE_OUTPUTS" \
  --candidate-jsonl "$CANDIDATE_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --suffix-window-tokens "$SUFFIX_WINDOW_TOKENS" \
  --max-examples "$MAX_EXAMPLES"
