#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4ctl
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
#SBATCH --array=0-71%4
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4ctl-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4ctl-%A_%a.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
ROUTE_CONFIG="${ROUTE_CONFIG:-configs/natural_evidence_v2/r4_positive_selectivity_pressure_controller_route.yaml}"
SCORE_ROWS="${SCORE_ROWS:-results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_v3_20260513/r4_prefix_native_surface_probe_rows_v3.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/r4_positive_selectivity_pressure_controller_score_${SLURM_ARRAY_JOB_ID:-manual}}"
PROTECTED_ADAPTER="${PROTECTED_ADAPTER:-$RUN_ROOT/status/r4_candidate_v3_pressure_relaxation_grid_857764/B_ceiling_lambda_0_5/protected_micro_overfit_train/adapter}"
TASK_ONLY_ADAPTER="${TASK_ONLY_ADAPTER:-$RUN_ROOT/status/wp5_r2_teacher_forced_train_and_score_851481/task_only_train/adapter}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_ROWS="${MAX_ROWS:-8192}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
VALIDATE_PLAN_ONLY="${VALIDATE_PLAN_ONLY:-0}"
ALLOW_PRESSURE_CONTROLLER_SCORING="${ALLOW_PRESSURE_CONTROLLER_SCORING:-0}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"
GRID_INDEX="${SLURM_ARRAY_TASK_ID:-${GRID_INDEX:-0}}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3" ]; then
    PYTHON_BIN="/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3"
  elif [ -x "$REPO_HOME/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_HOME/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm"

set +u
if [ -f /etc/profile ]; then
  # shellcheck disable=SC1091
  . /etc/profile
fi
if [ -f "$VENV_PATH/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi
set -u

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\r\n' <"$HF_TOKEN_FILE")"
fi

BONUS_VALUES=(0.25 0.50 0.75 1.00 1.25 1.50)
PENALTY_VALUES=(0.00 0.25)
MAX_TARGET_MASS_VALUES=(0.25 0.35 0.45)
MAX_KL_VALUES=(0.05 0.10)
GRID_SIZE=$(( ${#BONUS_VALUES[@]} * ${#PENALTY_VALUES[@]} * ${#MAX_TARGET_MASS_VALUES[@]} * ${#MAX_KL_VALUES[@]} ))

if [ "$GRID_INDEX" -lt 0 ] || [ "$GRID_INDEX" -ge "$GRID_SIZE" ]; then
  echo "INVALID_PRESSURE_CONTROLLER_GRID_INDEX: $GRID_INDEX of $GRID_SIZE" >&2
  exit 2
fi

kl_index=$((GRID_INDEX % ${#MAX_KL_VALUES[@]}))
mass_index=$(((GRID_INDEX / ${#MAX_KL_VALUES[@]}) % ${#MAX_TARGET_MASS_VALUES[@]}))
penalty_index=$(((GRID_INDEX / (${#MAX_KL_VALUES[@]} * ${#MAX_TARGET_MASS_VALUES[@]})) % ${#PENALTY_VALUES[@]}))
bonus_index=$(((GRID_INDEX / (${#MAX_KL_VALUES[@]} * ${#MAX_TARGET_MASS_VALUES[@]} * ${#PENALTY_VALUES[@]})) % ${#BONUS_VALUES[@]}))

CONTROLLER_BONUS_NATS="${BONUS_VALUES[$bonus_index]}"
CONTROLLER_PENALTY_NATS="${PENALTY_VALUES[$penalty_index]}"
CONTROLLER_MAX_TARGET_MASS="${MAX_TARGET_MASS_VALUES[$mass_index]}"
CONTROLLER_MAX_KL_BUDGET="${MAX_KL_VALUES[$kl_index]}"
GRID_ID="$(printf 'grid_%02d' "$GRID_INDEX")"
GRID_OUTPUT_DIR="$OUTPUT_DIR/$GRID_ID"
export GRID_ID
export GRID_INDEX
export GRID_OUTPUT_DIR

echo "claim_note=R4 pressure-controller teacher-forced scoring wrapper plan; no generation, no training, no Llama, no FAR, no paper positive claim"
echo "route_config=$ROUTE_CONFIG"
echo "score_rows=$SCORE_ROWS"
echo "output_dir=$OUTPUT_DIR"
echo "grid_id=$GRID_ID"
echo "grid_index=$GRID_INDEX"
echo "grid_size=$GRID_SIZE"
echo "controller_bonus_nats=$CONTROLLER_BONUS_NATS"
echo "controller_penalty_nats=$CONTROLLER_PENALTY_NATS"
echo "controller_max_target_mass=$CONTROLLER_MAX_TARGET_MASS"
echo "controller_max_kl_budget=$CONTROLLER_MAX_KL_BUDGET"
echo "validate_plan_only=$VALIDATE_PLAN_ONLY"
echo "allow_pressure_controller_scoring=$ALLOW_PRESSURE_CONTROLLER_SCORING"
echo "slurm_policy=H200 pomplun cs_yinxin.wan time 30-00:00:00"
echo "python_bin=$PYTHON_BIN"

for required in "$ROUTE_CONFIG" "$SCORE_ROWS" \
  "scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_controller_route.py" \
  "scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

"$PYTHON_BIN" scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_controller_route.py \
  --config "$ROUTE_CONFIG" \
  --output-dir "$GRID_OUTPUT_DIR/route_validation"

"$PYTHON_BIN" -m py_compile \
  scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py \
  scripts/natural_evidence_v2/r4_prefix_native_soft_logit_controller.py \
  scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_controller_route.py

if [ "$VALIDATE_PLAN_ONLY" = "1" ]; then
  "$PYTHON_BIN" scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py \
    --score-rows "$SCORE_ROWS" \
    --output-dir "$GRID_OUTPUT_DIR/scorer_dry_run" \
    --protected-adapter "$PROTECTED_ADAPTER" \
    --task-only-adapter "$TASK_ONLY_ADAPTER" \
    --controller-mode additive \
    --controller-bonus-nats "$CONTROLLER_BONUS_NATS" \
    --controller-penalty-nats "$CONTROLLER_PENALTY_NATS" \
    --controller-max-target-mass "$CONTROLLER_MAX_TARGET_MASS" \
    --controller-max-kl-budget "$CONTROLLER_MAX_KL_BUDGET" \
    --controller-condition-set pressure_controls \
    --max-rows "$MAX_ROWS" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --dry-run

  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import os

out = Path(os.environ["GRID_OUTPUT_DIR"])
dry = json.loads((out / "scorer_dry_run/r4_teacher_forced_surface_mass_summary.json").read_text(encoding="utf-8"))
summary = {
    "status": "PASS_R4_PRESSURE_CONTROLLER_SCORING_WRAPPER_PLAN_ONLY",
    "full_mode_enabled": False,
    "slurm_submission_started": False,
    "model_scoring_started": False,
    "generation_started": False,
    "training_started": False,
    "grid_id": os.environ["GRID_ID"],
    "grid_index": int(os.environ["GRID_INDEX"]),
    "controller_config": dry["controller_config"],
    "condition_plan": dry["condition_plan"],
    "route_validator_status": json.loads((out / "route_validation/pressure_controller_route_plan_validation_summary.json").read_text(encoding="utf-8"))["status"],
}
(out / "wrapper_plan_only_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  echo "R4 pressure-controller scoring wrapper plan-only validation passed: $GRID_OUTPUT_DIR"
  exit 0
fi

if [ "$ALLOW_PRESSURE_CONTROLLER_SCORING" != "1" ]; then
  echo "ALLOW_PRESSURE_CONTROLLER_SCORING_REQUIRED_FOR_FULL_MODE" >&2
  exit 2
fi

for required in "$PROTECTED_ADAPTER/adapter_config.json" "$TASK_ONLY_ADAPTER/adapter_config.json"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ADAPTER_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

if [ -e "$GRID_OUTPUT_DIR/r4_teacher_forced_surface_mass_summary.json" ] || \
   [ -e "$GRID_OUTPUT_DIR/r4_teacher_forced_surface_mass_rows.jsonl" ]; then
  echo "R4_PRESSURE_CONTROLLER_GRID_OUTPUT_EXISTS_REFUSING_OVERWRITE: $GRID_OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON_BIN" scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py \
  --score-rows "$SCORE_ROWS" \
  --output-dir "$GRID_OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --protected-adapter "$PROTECTED_ADAPTER" \
  --task-only-adapter "$TASK_ONLY_ADAPTER" \
  --controller-mode additive \
  --controller-bonus-nats "$CONTROLLER_BONUS_NATS" \
  --controller-penalty-nats "$CONTROLLER_PENALTY_NATS" \
  --controller-max-target-mass "$CONTROLLER_MAX_TARGET_MASS" \
  --controller-max-kl-budget "$CONTROLLER_MAX_KL_BUDGET" \
  --controller-condition-set pressure_controls \
  --max-rows "$MAX_ROWS" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --require-cuda

echo "R4 pressure-controller teacher-forced scoring complete: $GRID_OUTPUT_DIR"
