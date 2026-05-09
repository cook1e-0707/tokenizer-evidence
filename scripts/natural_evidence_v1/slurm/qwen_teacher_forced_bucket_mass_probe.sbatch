#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-tfprob
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-tfprob-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-tfprob-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$RUN_ROOT/data/variable_radix_frame_policy_dry_run_20260506_1848}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUN_ROOT/training}"
SURVIVAL_DIR="${SURVIVAL_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_on_policy_survival}"
LIFT_DIR="${LIFT_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_protected_vs_task_only_lift}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_teacher_forced_bucket_mass_probe}"
SURVIVAL_SUMMARY_JSON="${SURVIVAL_SUMMARY_JSON:-$SURVIVAL_DIR/qwen_846699_on_policy_survival_summary.json}"
LIFT_SUMMARY_JSON="${LIFT_SUMMARY_JSON:-$LIFT_DIR/qwen_846699_protected_vs_task_only_lift_summary.json}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen2.5-7B-Instruct}"
PAYLOADS="${PAYLOADS:-P0421,P1729}"
SEEDS="${SEEDS:-17,23}"
TRAINING_JOB_ID="${TRAINING_JOB_ID:-846585}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_ROWS_PER_PAYLOAD="${MAX_ROWS_PER_PAYLOAD:-0}"
FORCE="${FORCE:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$OUTPUT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=teacher-forced bucket-mass probe only; no training, no generation, no E2E rerun, no payload recovery claim"
echo "model=$MODEL"
echo "tokenizer=$TOKENIZER"
echo "payloads=$PAYLOADS"
echo "seeds=$SEEDS"
echo "training_job_id=$TRAINING_JOB_ID"
echo "train_data_dir=$TRAIN_DATA_DIR"
echo "checkpoint_root=$CHECKPOINT_ROOT"
echo "survival_summary_json=$SURVIVAL_SUMMARY_JSON"
echo "lift_summary_json=$LIFT_SUMMARY_JSON"
echo "output_dir=$OUTPUT_DIR"
echo "batch_size=$BATCH_SIZE"
echo "max_length=$MAX_LENGTH"
echo "max_rows_per_payload=$MAX_ROWS_PER_PAYLOAD"

for required in "$SURVIVAL_SUMMARY_JSON" "$LIFT_SUMMARY_JSON"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

IFS=',' read -r -a payload_array <<< "$PAYLOADS"
IFS=',' read -r -a seed_array <<< "$SEEDS"
for payload_id in "${payload_array[@]}"; do
  train_jsonl="$TRAIN_DATA_DIR/$payload_id/variable_radix_train.jsonl"
  train_contract_json="$TRAIN_DATA_DIR/$payload_id/variable_radix_train_contract.json"
  for required in "$train_jsonl" "$train_contract_json"; do
    if [ ! -s "$required" ]; then
      echo "REQUIRED_TRAIN_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
      exit 2
    fi
  done
  for seed in "${seed_array[@]}"; do
    for arm in qwen_protected qwen_task_only_lora; do
      adapter="$CHECKPOINT_ROOT/${arm}_${payload_id}_seed${seed}_${TRAINING_JOB_ID}/checkpoints/natural_bucket_lora_last/adapter_model.safetensors"
      if [ ! -s "$adapter" ]; then
        echo "REQUIRED_ADAPTER_MISSING_OR_EMPTY: $adapter" >&2
        exit 2
      fi
    done
  done
done

summary_json="$OUTPUT_DIR/qwen_846699_teacher_forced_bucket_mass_probe_summary.json"
if [ -e "$summary_json" ] && [ "$FORCE" != "1" ]; then
  echo "QWEN_TEACHER_FORCED_PROBE_EXISTS_REFUSING_OVERWRITE: $summary_json" >&2
  exit 74
fi

args=(
  scripts/natural_evidence_v1/probe_qwen_teacher_forced_bucket_mass.py
  --train-data-dir "$TRAIN_DATA_DIR"
  --survival-summary-json "$SURVIVAL_SUMMARY_JSON"
  --lift-summary-json "$LIFT_SUMMARY_JSON"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --training-job-id "$TRAINING_JOB_ID"
  --output-dir "$OUTPUT_DIR"
  --model-name "$MODEL"
  --tokenizer-name "$TOKENIZER"
  --payload-ids "$PAYLOADS"
  --seeds "$SEEDS"
  --batch-size "$BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --max-rows-per-payload "$MAX_ROWS_PER_PAYLOAD"
  --require-cuda
)

if [ "$FORCE" = "1" ]; then
  args+=(--force)
fi

"$PYTHON" "${args[@]}"
