#!/bin/bash
#SBATCH --job-name=nat-ev-qwen-obsprov
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=DGXA100
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger_unlim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-obsprov-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/slurm/nat-ev-qwen-obsprov-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot}"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-846699}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval/qwen_natural_e2e_eval_846627_recovery}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/qwen_natural_e2e_eval_846699_recovery}"
OBSERVATIONS_JSONL="${OBSERVATIONS_JSONL:-$EVAL_DIR/qwen_natural_e2e_bucket_observations.jsonl}"
DECODE_TRACE_CSV="${DECODE_TRACE_CSV:-$EVAL_DIR/qwen_natural_e2e_decode_trace.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-$EVAL_DIR/qwen_natural_e2e_eval_summary.json}"
PROGRESS_JSON="${PROGRESS_JSON:-$EVAL_DIR/qwen_natural_e2e_eval_progress.json}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_DIR/observation_erasure_summary_846699.json}"
EXPECTED_OBSERVATION_COUNT="${EXPECTED_OBSERVATION_COUNT:-372216}"
EXPECTED_DECODE_ROW_COUNT="${EXPECTED_DECODE_ROW_COUNT:-120}"
FORCE="${FORCE:-0}"
PATH_EXPLANATION="${PATH_EXPLANATION:-846699 was the completed recovery evaluation for the failed 846627 attempt; the recovery wrapper intentionally reused the fresh output directory name qwen_natural_e2e_eval_846627_recovery, so this summary pins source_job_id, source path, hashes, row counts, and decode join keys explicitly.}"

cd "$REPO_HOME"
mkdir -p "$RUN_ROOT/slurm" "$OUTPUT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=artifact provenance normalization only; no training, no E2E generation, no payload recovery claim"
echo "source_job_id=$SOURCE_JOB_ID"
echo "eval_dir=$EVAL_DIR"
echo "observations_jsonl=$OBSERVATIONS_JSONL"
echo "decode_trace_csv=$DECODE_TRACE_CSV"
echo "summary_json=$SUMMARY_JSON"
echo "progress_json=$PROGRESS_JSON"
echo "output_json=$OUTPUT_JSON"

for required in "$OBSERVATIONS_JSONL" "$DECODE_TRACE_CSV" "$SUMMARY_JSON"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

if [ -e "$OUTPUT_JSON" ] && [ "$FORCE" != "1" ]; then
  echo "OBSERVATION_ERASURE_SUMMARY_EXISTS_REFUSING_OVERWRITE: $OUTPUT_JSON" >&2
  exit 74
fi

args=(
  scripts/natural_evidence_v1/summarize_qwen_natural_e2e_observations.py
  --observations-jsonl "$OBSERVATIONS_JSONL"
  --decode-trace-csv "$DECODE_TRACE_CSV"
  --summary-json "$SUMMARY_JSON"
  --output-json "$OUTPUT_JSON"
  --source-job-id "$SOURCE_JOB_ID"
  --source-remote-path "$OBSERVATIONS_JSONL"
  --path-explanation "$PATH_EXPLANATION"
  --expected-observation-count "$EXPECTED_OBSERVATION_COUNT"
  --expected-decode-row-count "$EXPECTED_DECODE_ROW_COUNT"
)

if [ -s "$PROGRESS_JSON" ]; then
  args+=(--progress-json "$PROGRESS_JSON")
fi
if [ "$FORCE" = "1" ]; then
  args+=(--force)
fi

"$PYTHON" "${args[@]}"
