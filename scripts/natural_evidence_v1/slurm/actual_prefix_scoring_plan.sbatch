#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-aplan
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-aplan-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e/slurm/nat-ev-qwen-aplan-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/diagnostic_high_risk_qwen_e2e}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_diagnostic_e2e_eval_844121}"
GENERATED_OUTPUTS="${GENERATED_OUTPUTS:-$EVAL_DIR/qwen_diagnostic_generated_outputs.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_DIR/actual_prefix_scoring_plan_v1}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MIN_RESPONSE_PREFIX_TOKENS="${MIN_RESPONSE_PREFIX_TOKENS:-1}"
MAX_GENERATED_ROWS="${MAX_GENERATED_ROWS:-0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-40}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "config=$CONFIG"
echo "generated_outputs=$GENERATED_OUTPUTS"
echo "output_dir=$OUTPUT_DIR"
echo "tokenizer_name=$TOKENIZER_NAME"
echo "min_response_prefix_tokens=$MIN_RESPONSE_PREFIX_TOKENS"
echo "max_generated_rows=$MAX_GENERATED_ROWS"
echo "claim_note=actual-prefix scoring plan only; no model scoring, no training, not payload recovery"

if [ ! -s "$GENERATED_OUTPUTS" ]; then
  echo "GENERATED_OUTPUTS_MISSING_OR_EMPTY: $GENERATED_OUTPUTS" >&2
  exit 2
fi
if [ -e "$OUTPUT_DIR/actual_prefix_scoring_manifest.json" ]; then
  echo "ACTUAL_PREFIX_SCORING_PLAN_EXISTS_REFUSING_OVERWRITE: $OUTPUT_DIR" >&2
  exit 74
fi

"$PYTHON" scripts/natural_evidence_v1/prepare_actual_prefix_scoring_plan.py \
  --config "$CONFIG" \
  --generated-outputs "$GENERATED_OUTPUTS" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --min-response-prefix-tokens "$MIN_RESPONSE_PREFIX_TOKENS" \
  --max-generated-rows "$MAX_GENERATED_ROWS" \
  --max-examples "$MAX_EXAMPLES"
