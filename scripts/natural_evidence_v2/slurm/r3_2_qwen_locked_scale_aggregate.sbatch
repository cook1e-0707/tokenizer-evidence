#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r32agg
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=Intel6240,Intel6248,Intel6326
#SBATCH --account=pi_yinxin.wan
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r32agg-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r32agg-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
PYTHON="${PYTHON:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must point to the completed R3.2 shard-array output dir}"

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

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "python=$PYTHON"
echo "claim_note=natural_evidence_v2 R3.2 Qwen locked-scale aggregate only; no training, no Llama, no same-family null, no sanitizer, no FAR, no paper positive claim"
echo "qos=scavenger"
echo "output_dir=$OUTPUT_DIR"

for shard_index in $(seq 0 11); do
  shard_id="$(printf 'shard_%02d' "$shard_index")"
  required="$OUTPUT_DIR/shards/$shard_id/coordinate_majority_r3_2_shard/r3_2_shard_summary.json"
  if [ ! -s "$required" ]; then
    echo "R3_2_REQUIRED_SHARD_SUMMARY_MISSING: $required" >&2
    exit 2
  fi
done

"$PYTHON" scripts/natural_evidence_v2/aggregate_r3_2_locked_scale_shards.py \
  --output-dir "$OUTPUT_DIR" \
  --shards-dir "$OUTPUT_DIR/shards" \
  --prompt-manifest "$OUTPUT_DIR/precommit/r3_2_selected_prompt_manifest.json"

echo "R3.2 Qwen same-contract locked-scale aggregation complete: $OUTPUT_DIR"
