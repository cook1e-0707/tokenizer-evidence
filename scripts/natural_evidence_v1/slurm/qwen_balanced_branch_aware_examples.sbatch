#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-babr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-babr-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-babr-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_natural_e2e_eval_846627_recovery}"
GENERATED_JSONL="${GENERATED_JSONL:-$EVAL_DIR/qwen_natural_e2e_generated_outputs.jsonl}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$RUN_ROOT/data/variable_radix_frame_policy_dry_run_20260506_1848}"
BUCKETIZED_CANDIDATES_JSONL="${BUCKETIZED_CANDIDATES_JSONL:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/eval/qwen_diagnostic_e2e_eval_844121/expanded_actual_prefix_bucketized_topk64_b4_minarity2_v1/expanded_actual_prefix_bucketized_candidates.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_balanced_branch_aware_examples}"
PAYLOAD_IDS="${PAYLOAD_IDS:-P0421,P1729}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MATCH_POLICIES="${MATCH_POLICIES:-exact_full,suffix_32,suffix_16,suffix_8}"
MODEL_CONDITIONS="${MODEL_CONDITIONS:-protected_trained,task_only_lora,raw}"
DRIFT_REASONS="${DRIFT_REASONS:-compatible_non_target,observed_token_not_candidate_set,observed_bucket_not_compatible}"
PER_SLICE="${PER_SLICE:-8}"
MAX_ROWS="${MAX_ROWS:-768}"
FORCE="${FORCE:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$OUTPUT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=Balanced branch-aware example export only; no model scoring, no training, no generation, no E2E rerun, no payload recovery or FAR claim"
echo "generated_jsonl=$GENERATED_JSONL"
echo "train_data_dir=$TRAIN_DATA_DIR"
echo "bucketized_candidates_jsonl=$BUCKETIZED_CANDIDATES_JSONL"
echo "output_dir=$OUTPUT_DIR"
echo "payload_ids=$PAYLOAD_IDS"
echo "tokenizer_name=$TOKENIZER_NAME"
echo "match_policies=$MATCH_POLICIES"
echo "model_conditions=$MODEL_CONDITIONS"
echo "drift_reasons=$DRIFT_REASONS"
echo "per_slice=$PER_SLICE"
echo "max_rows=$MAX_ROWS"

for required in "$GENERATED_JSONL" "$BUCKETIZED_CANDIDATES_JSONL"; do
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

summary_json="$OUTPUT_DIR/balanced_branch_aware_example_export_summary.json"
if [ -e "$summary_json" ] && [ "$FORCE" != "1" ]; then
  echo "BALANCED_BRANCH_AWARE_EXPORT_EXISTS_REFUSING_OVERWRITE: $summary_json" >&2
  exit 74
fi

args=(
  scripts/natural_evidence_v1/export_balanced_branch_aware_examples.py
  --generated-jsonl "$GENERATED_JSONL"
  --train-data-dir "$TRAIN_DATA_DIR"
  --bucketized-candidates-jsonl "$BUCKETIZED_CANDIDATES_JSONL"
  --output-dir "$OUTPUT_DIR"
  --payload-ids "$PAYLOAD_IDS"
  --tokenizer-name "$TOKENIZER_NAME"
  --match-policies "$MATCH_POLICIES"
  --model-conditions "$MODEL_CONDITIONS"
  --drift-reasons "$DRIFT_REASONS"
  --per-slice "$PER_SLICE"
  --max-rows "$MAX_ROWS"
)

if [ "$FORCE" = "1" ]; then
  args+=(--force)
fi

"$PYTHON" "${args[@]}"
