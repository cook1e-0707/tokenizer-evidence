#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4relCtlB
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
#SBATCH --array=0-23%4
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4relCtlB-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4relCtlB-%A_%a.err

set -euo pipefail

export ROUTE_CONFIG="${ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_867621_reliability_controller_safety_bound_route.yaml}"
export CONTROLLER_CONDITION_SET="${CONTROLLER_CONDITION_SET:-controller_only_controls}"
export SCORE_ROWS="${SCORE_ROWS:-results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516/reliability_surface_mass_rows.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_867621_reliability_controller_safety_bound_score_${SLURM_ARRAY_JOB_ID:-manual}}"
export ALLOW_PRESSURE_CONTROLLER_SCORING="${ALLOW_PRESSURE_CONTROLLER_SCORING:-1}"

exec bash scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
