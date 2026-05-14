#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4tgap
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=pomplun
#SBATCH --account=cs_yinxin.wan
#SBATCH --qos=pomplun
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=30-00:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4tgap-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4tgap-%A_%a.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-857764}"
SELECTED_ARM="${SELECTED_ARM:-B_ceiling_lambda_0_5}"
PROTECTED_ADAPTER="${PROTECTED_ADAPTER:-$RUN_ROOT/status/r4_candidate_v3_pressure_relaxation_grid_${SOURCE_JOB_ID}/${SELECTED_ARM}/protected_micro_overfit_train/adapter}"
PROTECTED_ADAPTER_GAIN="${PROTECTED_ADAPTER_GAIN:-1.0}"
PROMPTS_JSONL="${PROMPTS_JSONL:-results/natural_evidence_v2/prompts/r4_candidate_v3_transfer_gap_repaired_dev_prompts_20260514_0710/dev_prompts.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/r4_candidate_v3_transfer_gap_repair_dev_diagnostic_${SLURM_ARRAY_JOB_ID:-manual}}"
VALIDATE_PLAN_ONLY="${VALIDATE_PLAN_ONLY:-0}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-}}"

cd "$REPO_HOME"

echo "claim_note=natural_evidence_v2 R4 candidate v3 transfer-gap repair small dev generation diagnostic; no training, no Qwen E2E rerun, no Llama, no same-family null, no sanitizer, no FAR, no payload diversity, no paper positive claim"
echo "source_job_id=$SOURCE_JOB_ID"
echo "selected_arm=$SELECTED_ARM"
echo "protected_adapter=$PROTECTED_ADAPTER"
echo "protected_adapter_gain=$PROTECTED_ADAPTER_GAIN"
echo "prompts_jsonl=$PROMPTS_JSONL"
echo "output_dir=$OUTPUT_DIR"
echo "validate_plan_only=$VALIDATE_PLAN_ONLY"

if [ "$VALIDATE_PLAN_ONLY" != "1" ] && [ ! -s "$PROTECTED_ADAPTER/adapter_config.json" ]; then
  echo "R4_TRANSFER_GAP_SELECTED_ADAPTER_MISSING: $PROTECTED_ADAPTER/adapter_config.json" >&2
  exit 2
fi

if [ -z "$SHARD_INDEX" ]; then
  if [ "$VALIDATE_PLAN_ONLY" != "1" ]; then
    echo "SHARD_INDEX_REQUIRED_FOR_NON_PLAN_RUN" >&2
    exit 2
  fi
  for shard in 0 1 2 3; do
    SHARD_INDEX="$shard" \
    PROTECTED_ADAPTER="$PROTECTED_ADAPTER" \
    PROTECTED_ADAPTER_GAIN="$PROTECTED_ADAPTER_GAIN" \
    PROMPTS_JSONL="$PROMPTS_JSONL" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    VALIDATE_PLAN_ONLY=1 \
      bash scripts/natural_evidence_v2/slurm/r4_candidate_v3_transfer_gap_repair_dev_diagnostic_h200.sbatch
  done
  echo "R4 transfer-gap repair dev diagnostic plan validation complete for shards 0..3"
  exit 0
fi

if [ "$SHARD_INDEX" -lt 0 ] || [ "$SHARD_INDEX" -gt 3 ]; then
  echo "INVALID_R4_TRANSFER_GAP_DEV_SHARD_INDEX: $SHARD_INDEX" >&2
  exit 2
fi

SHARD_INDEX="$SHARD_INDEX" \
PROTECTED_ADAPTER="$PROTECTED_ADAPTER" \
PROTECTED_ADAPTER_GAIN="$PROTECTED_ADAPTER_GAIN" \
PROMPTS_JSONL="$PROMPTS_JSONL" \
OUTPUT_DIR="$OUTPUT_DIR" \
VALIDATE_PLAN_ONLY="$VALIDATE_PLAN_ONLY" \
  bash scripts/natural_evidence_v2/slurm/r4_cover_natural_dev_diagnostic_h200.sbatch
