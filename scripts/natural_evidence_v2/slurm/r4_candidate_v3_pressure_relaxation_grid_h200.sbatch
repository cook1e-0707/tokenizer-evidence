#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4prg
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
#SBATCH --array=0-1%2
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4prg-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4prg-%A_%a.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
VALIDATE_PLAN_ONLY="${VALIDATE_PLAN_ONLY:-0}"

if [ "$VALIDATE_PLAN_ONLY" = "1" ] && [ -z "${SLURM_ARRAY_TASK_ID:-}" ] && [ -z "${PRESSURE_RELAXATION_ARM_INDEX:-}" ]; then
  echo "VALIDATE_PLAN_ONLY=1 without array task id: validating both fixed pressure-relaxation arms."
  PRESSURE_RELAXATION_ARM_INDEX=0 VALIDATE_PLAN_ONLY=1 bash "$0"
  PRESSURE_RELAXATION_ARM_INDEX=1 VALIDATE_PLAN_ONLY=1 bash "$0"
  echo "Pressure-relaxation grid plan-only validation complete for arms A and B."
  exit 0
fi

ARM_INDEX="${SLURM_ARRAY_TASK_ID:-${PRESSURE_RELAXATION_ARM_INDEX:-}}"
if [ -z "$ARM_INDEX" ]; then
  echo "PRESSURE_RELAXATION_ARM_INDEX_REQUIRED: set SLURM_ARRAY_TASK_ID or PRESSURE_RELAXATION_ARM_INDEX" >&2
  exit 2
fi

case "$ARM_INDEX" in
  0)
    ARM_LABEL="A_ceiling_lambda_1_0"
    TARGET_MASS_CEILING="${TARGET_MASS_CEILING:-0.50}"
    TARGET_MASS_CEILING_LAMBDA="${TARGET_MASS_CEILING_LAMBDA:-1.0}"
    ;;
  1)
    ARM_LABEL="B_ceiling_lambda_0_5"
    TARGET_MASS_CEILING="${TARGET_MASS_CEILING:-0.50}"
    TARGET_MASS_CEILING_LAMBDA="${TARGET_MASS_CEILING_LAMBDA:-0.5}"
    ;;
  *)
    echo "UNKNOWN_PRESSURE_RELAXATION_ARM_INDEX: $ARM_INDEX" >&2
    exit 2
    ;;
esac

export REPO_HOME
export RUN_ROOT
export VALIDATE_PLAN_ONLY
export TARGET_MASS_FLOOR="${TARGET_MASS_FLOOR:-0.20}"
export TARGET_MASS_FLOOR_LAMBDA="${TARGET_MASS_FLOOR_LAMBDA:-5.0}"
export TARGET_MASS_CEILING
export TARGET_MASS_CEILING_LAMBDA
export STRATUM_WEIGHTING_MODE="${STRATUM_WEIGHTING_MODE:-r4_candidate_v3_failed_surface}"
export STRATUM_WEIGHT_MAX="${STRATUM_WEIGHT_MAX:-3.0}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/r4_candidate_v3_pressure_relaxation_grid_${SLURM_ARRAY_JOB_ID:-manual}/$ARM_LABEL}"

echo "pressure_relaxation_arm_index=$ARM_INDEX"
echo "pressure_relaxation_arm_label=$ARM_LABEL"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-manual}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-manual}"
echo "target_mass_ceiling=$TARGET_MASS_CEILING"
echo "target_mass_ceiling_lambda=$TARGET_MASS_CEILING_LAMBDA"
echo "output_dir=$OUTPUT_DIR"
echo "claim_note=natural_evidence_v2 R4 candidate v3 fixed two-arm pressure-relaxation grid; teacher-forced micro-overfit training/scoring only; no generation, no Qwen E2E rerun, no Llama, no same-family null, no sanitizer, no FAR, no payload diversity, no paper positive claim"

cd "$REPO_HOME"
bash scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
