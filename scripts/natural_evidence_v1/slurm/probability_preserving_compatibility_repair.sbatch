#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-ppcr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=pomplun
#SBATCH --account=cs_yinxin.wan
#SBATCH --qos=pomplun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-ppcr-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-ppcr-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
EXPANSION_ROOT="${EXPANSION_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3}"
BANK_ENTRIES="${BANK_ENTRIES:-$EXPANSION_ROOT/bucket_banks_4way_strict/qwen_bucket_bank_entries.jsonl}"
COMPATIBILITY_JSONL="${COMPATIBILITY_JSONL:-$EXPANSION_ROOT/bucket_banks_4way_strict/qwen_counterfactual_compatibility.jsonl}"
CANDIDATE_JSONL="${CANDIDATE_JSONL:-$EXPANSION_ROOT/reference_candidates/qwen_topk_candidates.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPANSION_ROOT/tables/qwen_4way_strict_expanded/probability_preserving_compatibility_repair}"
ANNOTATED_COMPATIBILITY_JSONL="${ANNOTATED_COMPATIBILITY_JSONL:-$OUTPUT_DIR/qwen_counterfactual_compatibility_with_probabilities.jsonl}"
JOIN_SUMMARY_JSON="${JOIN_SUMMARY_JSON:-$OUTPUT_DIR/probability_join_summary.json}"
RAW_OPPORTUNITY_SCALING_TARGET="${RAW_OPPORTUNITY_SCALING_TARGET:-24000}"
TARGET_ENTRIES="${TARGET_ENTRIES:-$RAW_OPPORTUNITY_SCALING_TARGET}"
BUCKET_COUNT="${BUCKET_COUNT:-4}"
MIN_COMPATIBLE_MEMBERS_PER_BUCKET="${MIN_COMPATIBLE_MEMBERS_PER_BUCKET:-2}"

cd "$REPO_HOME"
mkdir -p "$OUTPUT_DIR" "$EXPANSION_ROOT/slurm"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "config=$CONFIG"
echo "entries=$BANK_ENTRIES"
echo "compatibility_jsonl=$COMPATIBILITY_JSONL"
echo "candidate_jsonl=$CANDIDATE_JSONL"
echo "annotated_compatibility_jsonl=$ANNOTATED_COMPATIBILITY_JSONL"
echo "join_summary_json=$JOIN_SUMMARY_JSON"
echo "output_dir=$OUTPUT_DIR"
echo "target_entries=$TARGET_ENTRIES"
echo "target_note=raw opportunity scaling placeholder, not E2E training gate"
echo "bucket_count=$BUCKET_COUNT"
echo "min_compatible_members_per_bucket=$MIN_COMPATIBLE_MEMBERS_PER_BUCKET"

if [ ! -s "$BANK_ENTRIES" ]; then
  echo "BANK_ENTRIES_MISSING_OR_EMPTY: $BANK_ENTRIES" >&2
  exit 2
fi
if [ ! -s "$COMPATIBILITY_JSONL" ]; then
  echo "COMPATIBILITY_JSONL_MISSING_OR_EMPTY: $COMPATIBILITY_JSONL" >&2
  exit 2
fi
if [ ! -s "$CANDIDATE_JSONL" ]; then
  echo "CANDIDATE_JSONL_MISSING_OR_EMPTY: $CANDIDATE_JSONL" >&2
  exit 2
fi
if [ -e "$ANNOTATED_COMPATIBILITY_JSONL" ] || [ -e "$JOIN_SUMMARY_JSON" ] || [ -e "$OUTPUT_DIR/compatibility_filtered_bank_dry_run_summary.json" ]; then
  echo "PROBABILITY_PRESERVING_REPAIR_OUTPUT_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" scripts/natural_evidence_v1/join_compatibility_probabilities.py \
  --compatibility-jsonl "$COMPATIBILITY_JSONL" \
  --candidate-jsonl "$CANDIDATE_JSONL" \
  --output-jsonl "$ANNOTATED_COMPATIBILITY_JSONL" \
  --summary-json "$JOIN_SUMMARY_JSON" \
  --fail-on-missing

"$PYTHON" scripts/natural_evidence_v1/dry_run_compatibility_filtered_bank.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --entries "$BANK_ENTRIES" \
  --compatibility-jsonl "$ANNOTATED_COMPATIBILITY_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --target-entries "$TARGET_ENTRIES" \
  --bucket-count "$BUCKET_COUNT" \
  --min-compatible-members-per-bucket "$MIN_COMPATIBLE_MEMBERS_PER_BUCKET"
