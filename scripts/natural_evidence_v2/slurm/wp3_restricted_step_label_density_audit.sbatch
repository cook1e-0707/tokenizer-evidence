#!/bin/bash
#SBATCH --job-name=nat-ev-v2-wp3dens
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp3dens-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp3dens-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/wp3_restricted_step_label_density_audit_${SLURM_JOB_ID:-manual}}"
CONFIG="${CONFIG:-configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml}"
PROMPTS_JSONL="${PROMPTS_JSONL:-results/natural_evidence_v2/status/wp3_restricted_step_label_density_audit_plan_20260508_2055/restricted_step_label_density_audit_prompts.jsonl}"
POLICY_DIR="${POLICY_DIR:-results/natural_evidence_v2/status/wp3_restricted_step_label_policy_20260508_2049}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MAX_PROMPTS="${MAX_PROMPTS:-256}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-896}"
VALIDATE_PLAN_ONLY="${VALIDATE_PLAN_ONLY:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$OUTPUT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=natural_evidence_v2 WP3 restricted Step-label density audit only; no training, no E2E, no payload recovery, no FAR, no positive claim"
echo "config=$CONFIG"
echo "prompts_jsonl=$PROMPTS_JSONL"
echo "policy_dir=$POLICY_DIR"
echo "output_dir=$OUTPUT_DIR"
echo "model_name=$MODEL_NAME"
echo "tokenizer_name=$TOKENIZER_NAME"
echo "max_prompts=$MAX_PROMPTS"
echo "batch_size=$BATCH_SIZE"
echo "max_new_tokens=$MAX_NEW_TOKENS"

for required in "$CONFIG" "$PROMPTS_JSONL" "$POLICY_DIR/restricted_step_label_detector_contract.json" "$POLICY_DIR/restricted_step_label_bucket_bank.json"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

summary_json="$OUTPUT_DIR/restricted_step_label_density_audit_summary.json"
if [ "$VALIDATE_PLAN_ONLY" != "1" ] && [ -e "$summary_json" ]; then
  echo "WP3_RESTRICTED_DENSITY_SUMMARY_EXISTS_REFUSING_OVERWRITE: $summary_json" >&2
  exit 74
fi

args=(
  scripts/natural_evidence_v2/run_wp3_restricted_step_label_density_audit.py
  --config "$CONFIG"
  --prompts-jsonl "$PROMPTS_JSONL"
  --policy-dir "$POLICY_DIR"
  --model-name "$MODEL_NAME"
  --tokenizer-name "$TOKENIZER_NAME"
  --max-prompts "$MAX_PROMPTS"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
)

if [ "$VALIDATE_PLAN_ONLY" = "1" ]; then
  args+=(--validate-plan-only)
else
  args+=(--output-dir "$OUTPUT_DIR" --require-cuda)
fi

"$PYTHON" "${args[@]}"
