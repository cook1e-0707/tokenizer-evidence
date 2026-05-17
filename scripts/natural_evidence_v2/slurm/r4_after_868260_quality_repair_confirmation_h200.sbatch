#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4qfix
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4qfix-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4qfix-%A_%a.err

set -euo pipefail

export ROUTE_CONFIG="${ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_868260_quality_repair_confirmation_route.yaml}"
export ROUTE_VALIDATOR="${ROUTE_VALIDATOR:-scripts/natural_evidence_v2/validate_r4_after_868260_quality_repair_confirmation_route.py}"
export PLAN_ONLY="${PLAN_ONLY:-1}"
export PYTHON_BIN="${PYTHON_BIN:-}"

echo "claim_note=R4 after-868260 quality-repair confirmation route; plan-only until reviewed preflight passes; no paper claim"
echo "route_config=$ROUTE_CONFIG"
echo "route_validator=$ROUTE_VALIDATOR"
echo "plan_only=$PLAN_ONLY"
echo "python_bin=${PYTHON_BIN:-uv run python}"
echo "generation_started=false"
echo "training_started=false"
echo "llama_started=false"
echo "far_started=false"
echo "paper_claim_allowed=false"

if [[ "$PLAN_ONLY" != "1" ]]; then
  echo "ERROR: full generation mode is fail-closed for this wrapper until a separate reviewed submission route is recorded." >&2
  exit 2
fi

if [[ -n "$PYTHON_BIN" ]]; then
  "$PYTHON_BIN" "$ROUTE_VALIDATOR" --config "$ROUTE_CONFIG" --output-dir "${OUTPUT_DIR:-results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_wrapper_plan_smoke_20260517}"
else
  uv run python "$ROUTE_VALIDATOR" --config "$ROUTE_CONFIG" --output-dir "${OUTPUT_DIR:-results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_wrapper_plan_smoke_20260517}"
fi
