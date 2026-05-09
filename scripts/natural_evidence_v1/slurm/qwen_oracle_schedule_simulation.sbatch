#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-oracle
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-oracle-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-oracle-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot}"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-846699}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_natural_e2e_eval_846627_recovery}"
FRAME_REPLAY_DIR="${FRAME_REPLAY_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_frame_completion_replay}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_oracle_schedule_simulation}"
OBSERVATIONS_JSONL="${OBSERVATIONS_JSONL:-$EVAL_DIR/qwen_natural_e2e_bucket_observations.jsonl}"
DECODE_TRACE_CSV="${DECODE_TRACE_CSV:-$EVAL_DIR/qwen_natural_e2e_decode_trace.csv}"
FRAME_REPLAY_SUMMARY_JSON="${FRAME_REPLAY_SUMMARY_JSON:-$FRAME_REPLAY_DIR/qwen_846699_frame_completion_replay_summary.json}"
QUERY_BUDGETS="${QUERY_BUDGETS:-64,128,256,512}"
MAX_PROMPT_EXAMPLES="${MAX_PROMPT_EXAMPLES:-20}"
MAX_FRAME_BOUNDS="${MAX_FRAME_BOUNDS:-20}"
FORCE="${FORCE:-0}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$OUTPUT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=oracle schedule simulation only; no training, no E2E generation, no payload recovery claim"
echo "source_job_id=$SOURCE_JOB_ID"
echo "eval_dir=$EVAL_DIR"
echo "observations_jsonl=$OBSERVATIONS_JSONL"
echo "decode_trace_csv=$DECODE_TRACE_CSV"
echo "frame_replay_summary_json=$FRAME_REPLAY_SUMMARY_JSON"
echo "output_dir=$OUTPUT_DIR"
echo "query_budgets=$QUERY_BUDGETS"

for required in "$OBSERVATIONS_JSONL" "$DECODE_TRACE_CSV" "$FRAME_REPLAY_SUMMARY_JSON"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

summary_json="$OUTPUT_DIR/qwen_846699_oracle_schedule_simulation_summary.json"
if [ -e "$summary_json" ] && [ "$FORCE" != "1" ]; then
  echo "QWEN_ORACLE_SCHEDULE_SIMULATION_EXISTS_REFUSING_OVERWRITE: $summary_json" >&2
  exit 74
fi

args=(
  scripts/natural_evidence_v1/oracle_qwen_schedule_simulation.py
  --observations-jsonl "$OBSERVATIONS_JSONL"
  --decode-trace-csv "$DECODE_TRACE_CSV"
  --frame-replay-summary-json "$FRAME_REPLAY_SUMMARY_JSON"
  --output-dir "$OUTPUT_DIR"
  --query-budgets "$QUERY_BUDGETS"
  --max-prompt-examples "$MAX_PROMPT_EXAMPLES"
  --max-frame-bounds "$MAX_FRAME_BOUNDS"
)

if [ "$FORCE" = "1" ]; then
  args+=(--force)
fi

"$PYTHON" "${args[@]}"
