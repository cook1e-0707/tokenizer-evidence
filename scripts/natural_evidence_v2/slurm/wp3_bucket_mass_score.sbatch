#!/bin/bash
#SBATCH --job-name=nat-ev-v2-wp3mass
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp3mass-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-wp3mass-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"

BUCKET_BANK="${BUCKET_BANK:-results/natural_evidence_v2/status/wp3_detector_bank_scaffold_repaired_20260508_2308/two_way_bucket_bank_scaffold.json}"
DETECTOR_CONTRACT="${DETECTOR_CONTRACT:-results/natural_evidence_v2/status/wp3_detector_bank_scaffold_repaired_20260508_2308/wp3_detector_contract.json}"
SCAFFOLD_SUMMARY="${SCAFFOLD_SUMMARY:-results/natural_evidence_v2/status/wp3_detector_bank_scaffold_repaired_20260508_2308/wp3_detector_bank_scaffold_summary.json}"
MASS_OUTPUT_DIR="${MASS_OUTPUT_DIR:-$RUN_ROOT/status/wp3_bucket_mass_score_${SLURM_JOB_ID:-manual}}"
AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-$RUN_ROOT/status/wp3_model_mass_audit_${SLURM_JOB_ID:-manual}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$MASS_OUTPUT_DIR" "$AUDIT_OUTPUT_DIR"

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

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=natural_evidence_v2 WP3 fixed-prefix model-mass scoring only; no training, no generation, no E2E, no FAR, no positive claim"
echo "bucket_bank=$BUCKET_BANK"
echo "detector_contract=$DETECTOR_CONTRACT"
echo "scaffold_summary=$SCAFFOLD_SUMMARY"
echo "mass_output_dir=$MASS_OUTPUT_DIR"
echo "audit_output_dir=$AUDIT_OUTPUT_DIR"
echo "model_name=$MODEL_NAME"
echo "tokenizer_name=$TOKENIZER_NAME"

for required in "$BUCKET_BANK" "$DETECTOR_CONTRACT" "$SCAFFOLD_SUMMARY"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

mass_summary="$MASS_OUTPUT_DIR/qwen_v2_wp3_bucket_mass_score_summary.json"
if [ -e "$mass_summary" ]; then
  echo "WP3_BUCKET_MASS_SCORE_EXISTS_REFUSING_OVERWRITE: $mass_summary" >&2
  exit 74
fi
audit_summary="$AUDIT_OUTPUT_DIR/wp3_fixed_artifact_audit_summary.json"
if [ -e "$audit_summary" ]; then
  echo "WP3_MODEL_MASS_AUDIT_EXISTS_REFUSING_OVERWRITE: $audit_summary" >&2
  exit 74
fi

"$PYTHON" -m py_compile \
  scripts/natural_evidence_v2/score_wp3_bucket_mass.py \
  scripts/natural_evidence_v2/audit_wp3_fixed_artifacts.py

"$PYTHON" scripts/natural_evidence_v2/score_wp3_bucket_mass.py \
  --bucket-bank "$BUCKET_BANK" \
  --output-dir "$MASS_OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --require-cuda

"$PYTHON" scripts/natural_evidence_v2/audit_wp3_fixed_artifacts.py \
  --detector-contract "$DETECTOR_CONTRACT" \
  --bucket-bank "$BUCKET_BANK" \
  --scaffold-summary "$SCAFFOLD_SUMMARY" \
  --mass-json "$MASS_OUTPUT_DIR/qwen_v2_wp3_bucket_mass_artifact.json" \
  --output-dir "$AUDIT_OUTPUT_DIR"
