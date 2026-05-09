#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-brscore
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-brscore-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-brscore-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot}"
BALANCED_PREP_DIR="${BALANCED_PREP_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_branch_aware_suffix_repair_preparation_balanced}"
BALANCED_DRY_RUN_DIR="${BALANCED_DRY_RUN_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_local_suffix_repair_dry_run_balanced}"
SCORING_PLAN_JSONL="${SCORING_PLAN_JSONL:-$BALANCED_PREP_DIR/branch_aware_compatibility_scoring_plan.jsonl}"
PREPARED_SUMMARY_JSON="${PREPARED_SUMMARY_JSON:-$BALANCED_PREP_DIR/branch_aware_compatibility_summary.json}"
DRY_RUN_ROWS_JSONL="${DRY_RUN_ROWS_JSONL:-$BALANCED_DRY_RUN_DIR/local_suffix_repair_dry_run_rows.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_branch_aware_compatibility_scored_balanced}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_RESPONSE_DELTA_PER_TOKEN="${MAX_RESPONSE_DELTA_PER_TOKEN:-0.5}"
MAX_SUFFIX_DELTA_PER_TOKEN="${MAX_SUFFIX_DELTA_PER_TOKEN:-1.0}"
MAX_ROWS="${MAX_ROWS:-0}"
FORCE="${FORCE:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$OUTPUT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=branch-aware compatibility proxy scoring only; no training, no generation, no E2E rerun, no payload recovery or FAR claim"
echo "model=$MODEL"
echo "tokenizer=$TOKENIZER"
echo "scoring_plan_jsonl=$SCORING_PLAN_JSONL"
echo "prepared_summary_json=$PREPARED_SUMMARY_JSON"
echo "dry_run_rows_jsonl=$DRY_RUN_ROWS_JSONL"
echo "output_dir=$OUTPUT_DIR"
echo "batch_size=$BATCH_SIZE"
echo "max_length=$MAX_LENGTH"
echo "max_response_delta_per_token=$MAX_RESPONSE_DELTA_PER_TOKEN"
echo "max_suffix_delta_per_token=$MAX_SUFFIX_DELTA_PER_TOKEN"
echo "max_rows=$MAX_ROWS"

for required in "$SCORING_PLAN_JSONL" "$PREPARED_SUMMARY_JSON" "$DRY_RUN_ROWS_JSONL"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

summary_json="$OUTPUT_DIR/branch_aware_compatibility_score_summary.json"
if [ -e "$summary_json" ] && [ "$FORCE" != "1" ]; then
  echo "BRANCH_AWARE_COMPATIBILITY_SCORE_EXISTS_REFUSING_OVERWRITE: $summary_json" >&2
  exit 74
fi

args=(
  scripts/natural_evidence_v1/score_branch_aware_compatibility.py
  --scoring-plan-jsonl "$SCORING_PLAN_JSONL"
  --dry-run-rows-jsonl "$DRY_RUN_ROWS_JSONL"
  --prepared-summary-json "$PREPARED_SUMMARY_JSON"
  --output-dir "$OUTPUT_DIR"
  --model-name "$MODEL"
  --tokenizer-name "$TOKENIZER"
  --batch-size "$BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --max-response-delta-per-token "$MAX_RESPONSE_DELTA_PER_TOKEN"
  --max-suffix-delta-per-token "$MAX_SUFFIX_DELTA_PER_TOKEN"
  --max-rows "$MAX_ROWS"
  --require-cuda
)

if [ "$FORCE" = "1" ]; then
  args+=(--force)
fi

"$PYTHON" "${args[@]}"
