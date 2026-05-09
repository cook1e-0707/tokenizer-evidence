#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-survival
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-survival-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-survival-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot}"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-846699}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_natural_e2e_eval_846627_recovery}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$RUN_ROOT/data/variable_radix_frame_policy_dry_run_20260506_1848}"
ORACLE_DIR="${ORACLE_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_oracle_schedule_simulation}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_on_policy_survival}"
OBSERVATIONS_JSONL="${OBSERVATIONS_JSONL:-$EVAL_DIR/qwen_natural_e2e_bucket_observations.jsonl}"
ORACLE_SUMMARY_JSON="${ORACLE_SUMMARY_JSON:-$ORACLE_DIR/qwen_846699_oracle_schedule_simulation_summary.json}"
PAYLOAD_IDS="${PAYLOAD_IDS:-P0421,P1729}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MAX_EXAMPLES="${MAX_EXAMPLES:-240}"
MAX_TOKEN_TEXTS_PER_BUCKET="${MAX_TOKEN_TEXTS_PER_BUCKET:-4}"
FORCE="${FORCE:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$OUTPUT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=on-policy survival by slot/source only; no training, no E2E generation, no payload recovery claim"
echo "source_job_id=$SOURCE_JOB_ID"
echo "eval_dir=$EVAL_DIR"
echo "observations_jsonl=$OBSERVATIONS_JSONL"
echo "train_data_dir=$TRAIN_DATA_DIR"
echo "oracle_summary_json=$ORACLE_SUMMARY_JSON"
echo "output_dir=$OUTPUT_DIR"
echo "payload_ids=$PAYLOAD_IDS"
echo "tokenizer_name=$TOKENIZER_NAME"

for required in "$OBSERVATIONS_JSONL" "$ORACLE_SUMMARY_JSON"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

IFS=',' read -r -a payload_id_array <<< "$PAYLOAD_IDS"
for payload_id in "${payload_id_array[@]}"; do
  train_jsonl="$TRAIN_DATA_DIR/$payload_id/variable_radix_train.jsonl"
  if [ ! -s "$train_jsonl" ]; then
    echo "REQUIRED_TRAIN_JSONL_MISSING_OR_EMPTY: $train_jsonl" >&2
    exit 2
  fi
done

summary_json="$OUTPUT_DIR/qwen_846699_on_policy_survival_summary.json"
if [ -e "$summary_json" ] && [ "$FORCE" != "1" ]; then
  echo "QWEN_ON_POLICY_SURVIVAL_EXISTS_REFUSING_OVERWRITE: $summary_json" >&2
  exit 74
fi

args=(
  scripts/natural_evidence_v1/analyze_qwen_on_policy_survival.py
  --observations-jsonl "$OBSERVATIONS_JSONL"
  --train-data-dir "$TRAIN_DATA_DIR"
  --oracle-summary-json "$ORACLE_SUMMARY_JSON"
  --output-dir "$OUTPUT_DIR"
  --payload-ids "$PAYLOAD_IDS"
  --tokenizer-name "$TOKENIZER_NAME"
  --max-examples "$MAX_EXAMPLES"
  --max-token-texts-per-bucket "$MAX_TOKEN_TEXTS_PER_BUCKET"
)

if [ "$FORCE" = "1" ]; then
  args+=(--force)
fi

"$PYTHON" "${args[@]}"
