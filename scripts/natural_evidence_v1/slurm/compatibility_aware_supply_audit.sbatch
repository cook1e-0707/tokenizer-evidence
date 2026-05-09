#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-casup
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-casup-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-casup-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"

RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_diagnostic_e2e_eval_844121}"
TOPK_DIR="${TOPK_DIR:-$EVAL_DIR/actual_prefix_reference_scoring_topk64_v1}"
SUPPLY_DIR="${SUPPLY_DIR:-$EVAL_DIR/compatibility_aware_supply_audit_topk64_v1}"

CANDIDATE_JSONL="${CANDIDATE_JSONL:-$TOPK_DIR/qwen_actual_prefix_topk_candidates.jsonl}"
STATIC_VALIDATION_SUMMARY="${STATIC_VALIDATION_SUMMARY:-$SUPPLY_DIR/static_validation_summary_compatibility_aware_supply.json}"
TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
GENERATED_OUTPUT_COUNT="${GENERATED_OUTPUT_COUNT:-14336}"
BUCKET_COUNTS="${BUCKET_COUNTS:-2,4}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-64}"
MIN_ARITY="${MIN_ARITY:-2}"
CONFIGURED_MIN_MEMBERS="${CONFIGURED_MIN_MEMBERS:-2}"
MIN_BUCKET_MASS="${MIN_BUCKET_MASS:-0.005}"
MAX_BUCKET_MASS_RATIO="${MAX_BUCKET_MASS_RATIO:-5.0}"
MIN_ENTROPY_FRACTION="${MIN_ENTROPY_FRACTION:-0.90}"
MAX_RECORDS="${MAX_RECORDS:-0}"
WRITE_POSITION_CSV="${WRITE_POSITION_CSV:-1}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"
START_COMPATIBILITY_AWARE_SUPPLY_AUDIT="${START_COMPATIBILITY_AWARE_SUPPLY_AUDIT:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$SUPPLY_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "candidate_jsonl=$CANDIDATE_JSONL"
echo "supply_dir=$SUPPLY_DIR"
echo "bucket_counts=$BUCKET_COUNTS"
echo "candidate_top_k=$CANDIDATE_TOP_K"
echo "generated_output_count=$GENERATED_OUTPUT_COUNT"
echo "dry_run_only=$DRY_RUN_ONLY"
echo "start_compatibility_aware_supply_audit=$START_COMPATIBILITY_AWARE_SUPPLY_AUDIT"
echo "claim_note=candidate-supply upper-bound diagnostic only; no model scoring, no training, no E2E, no payload recovery"

if [ ! -s "$CANDIDATE_JSONL" ]; then
  echo "CANDIDATE_JSONL_MISSING_OR_EMPTY: $CANDIDATE_JSONL" >&2
  exit 2
fi

"$PYTHON" scripts/natural_evidence_v1/validate_static.py \
  --config "$CONFIG" \
  --summary "$STATIC_VALIDATION_SUMMARY"

"$PYTHON" -m py_compile scripts/natural_evidence_v1/compatibility_aware_supply_audit.py

if [ "$DRY_RUN_ONLY" != "0" ] || [ "$START_COMPATIBILITY_AWARE_SUPPLY_AUDIT" != "1" ]; then
  echo "DRY_RUN_ONLY_OR_START_FLAG_NOT_SET: wrapper preflight completed; no supply audit started"
  exit 0
fi

if [ -e "$SUPPLY_DIR/compatibility_aware_supply_manifest.json" ]; then
  echo "COMPATIBILITY_AWARE_SUPPLY_OUTPUT_EXISTS_REFUSING_OVERWRITE: $SUPPLY_DIR" >&2
  exit 74
fi

POSITION_FLAG=()
if [ "$WRITE_POSITION_CSV" = "1" ]; then
  POSITION_FLAG=(--write-position-csv)
fi

"$PYTHON" scripts/natural_evidence_v1/compatibility_aware_supply_audit.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --candidate-jsonl "$CANDIDATE_JSONL" \
  --output-dir "$SUPPLY_DIR" \
  --generated-output-count "$GENERATED_OUTPUT_COUNT" \
  --bucket-counts "$BUCKET_COUNTS" \
  --candidate-top-k "$CANDIDATE_TOP_K" \
  --min-arity "$MIN_ARITY" \
  --configured-min-members "$CONFIGURED_MIN_MEMBERS" \
  --min-bucket-mass "$MIN_BUCKET_MASS" \
  --max-bucket-mass-ratio "$MAX_BUCKET_MASS_RATIO" \
  --min-entropy-fraction "$MIN_ENTROPY_FRACTION" \
  --max-records "$MAX_RECORDS" \
  "${POSITION_FLAG[@]}"
