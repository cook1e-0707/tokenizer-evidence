#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-rtfprob
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-rtfprob-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-rtfprob-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot}"
DESIGN_DIR="${DESIGN_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_repaired_teacher_forced_target_mass_probe_design}"
SCORE_PLAN_JSONL="${SCORE_PLAN_JSONL:-$DESIGN_DIR/repaired_teacher_forced_target_mass_probe_scoring_plan.jsonl}"
DESIGN_SUMMARY_JSON="${DESIGN_SUMMARY_JSON:-$DESIGN_DIR/repaired_teacher_forced_target_mass_probe_design_summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_repaired_teacher_forced_target_mass_probe_scored}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_ROWS="${MAX_ROWS:-0}"
FORCE="${FORCE:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$OUTPUT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=repaired teacher-forced target-mass probe only; no training, no generation, no E2E rerun, no payload recovery or FAR claim"
echo "score_plan_jsonl=$SCORE_PLAN_JSONL"
echo "design_summary_json=$DESIGN_SUMMARY_JSON"
echo "output_dir=$OUTPUT_DIR"
echo "batch_size=$BATCH_SIZE"
echo "max_length=$MAX_LENGTH"
echo "max_rows=$MAX_ROWS"

for required in "$SCORE_PLAN_JSONL" "$DESIGN_SUMMARY_JSON"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

summary_json="$OUTPUT_DIR/repaired_teacher_forced_target_mass_probe_score_summary.json"
if [ -e "$summary_json" ] && [ "$FORCE" != "1" ]; then
  echo "REPAIRED_TEACHER_FORCED_TARGET_MASS_SCORE_EXISTS_REFUSING_OVERWRITE: $summary_json" >&2
  exit 74
fi

args=(
  scripts/natural_evidence_v1/score_repaired_teacher_forced_target_mass_probe.py
  --score-plan-jsonl "$SCORE_PLAN_JSONL"
  --design-summary-json "$DESIGN_SUMMARY_JSON"
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --max-rows "$MAX_ROWS"
  --require-cuda
)

if [ "$FORCE" = "1" ]; then
  args+=(--force)
fi

"$PYTHON" "${args[@]}"
