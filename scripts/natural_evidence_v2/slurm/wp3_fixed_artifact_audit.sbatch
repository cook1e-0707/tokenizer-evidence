#!/bin/bash
#SBATCH --job-name=nat-ev-v2-wp3aud
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp3aud-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp3aud-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/wp3_configured_tokenizer_audit_${SLURM_JOB_ID:-manual}}"
CONFIG="${CONFIG:-configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml}"
DETECTOR_CONTRACT="${DETECTOR_CONTRACT:-results/natural_evidence_v2/status/wp3_detector_bank_scaffold_20260508_2153/wp3_detector_contract.json}"
BUCKET_BANK="${BUCKET_BANK:-results/natural_evidence_v2/status/wp3_detector_bank_scaffold_20260508_2153/two_way_bucket_bank_scaffold.json}"
SCAFFOLD_SUMMARY="${SCAFFOLD_SUMMARY:-results/natural_evidence_v2/status/wp3_detector_bank_scaffold_20260508_2153/wp3_detector_bank_scaffold_summary.json}"
RESPONSES_JSONL="${RESPONSES_JSONL:-}"
MASS_JSON="${MASS_JSON:-}"
MAX_RESPONSE_ROWS="${MAX_RESPONSE_ROWS:-0}"
TOKENIZER_BACKEND="${TOKENIZER_BACKEND:-}"
TOKENIZER_NAME="${TOKENIZER_NAME:-}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$OUTPUT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=natural_evidence_v2 WP3 fixed-artifact audit only; no training, no generation, no E2E, no FAR, no positive claim"
echo "config=$CONFIG"
echo "detector_contract=$DETECTOR_CONTRACT"
echo "bucket_bank=$BUCKET_BANK"
echo "scaffold_summary=$SCAFFOLD_SUMMARY"
echo "responses_jsonl=${RESPONSES_JSONL:-NONE}"
echo "mass_json=${MASS_JSON:-NONE}"
echo "output_dir=$OUTPUT_DIR"

for required in "$CONFIG" "$DETECTOR_CONTRACT" "$BUCKET_BANK" "$SCAFFOLD_SUMMARY"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

summary_json="$OUTPUT_DIR/wp3_fixed_artifact_audit_summary.json"
if [ -e "$summary_json" ]; then
  echo "WP3_AUDIT_SUMMARY_EXISTS_REFUSING_OVERWRITE: $summary_json" >&2
  exit 74
fi

args=(
  scripts/natural_evidence_v2/audit_wp3_fixed_artifacts.py
  --config "$CONFIG"
  --detector-contract "$DETECTOR_CONTRACT"
  --bucket-bank "$BUCKET_BANK"
  --scaffold-summary "$SCAFFOLD_SUMMARY"
  --output-dir "$OUTPUT_DIR"
  --max-response-rows "$MAX_RESPONSE_ROWS"
)

if [ -n "$RESPONSES_JSONL" ]; then
  if [ ! -s "$RESPONSES_JSONL" ]; then
    echo "RESPONSES_JSONL_MISSING_OR_EMPTY: $RESPONSES_JSONL" >&2
    exit 2
  fi
  args+=(--responses-jsonl "$RESPONSES_JSONL")
fi

if [ -n "$MASS_JSON" ]; then
  if [ ! -s "$MASS_JSON" ]; then
    echo "MASS_JSON_MISSING_OR_EMPTY: $MASS_JSON" >&2
    exit 2
  fi
  args+=(--mass-json "$MASS_JSON")
fi

if [ -n "$TOKENIZER_BACKEND" ]; then
  args+=(--tokenizer-backend "$TOKENIZER_BACKEND")
fi

if [ -n "$TOKENIZER_NAME" ]; then
  args+=(--tokenizer-name "$TOKENIZER_NAME")
fi

"$PYTHON" "${args[@]}"
