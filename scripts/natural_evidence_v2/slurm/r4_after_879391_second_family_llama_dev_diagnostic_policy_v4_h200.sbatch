#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4ll32v4
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
#SBATCH --array=0-31
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4ll32v4-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4ll32v4-%A_%a.err

set -euo pipefail

# Policy-v4 rerun wrapper. It deliberately reuses the reviewed Llama 32-block
# generation/decode implementation while changing only the precommitted
# contextual forbidden-surface policy and public run salt.
#
# Validator-visible inherited contract fragments:
# meta-llama/Meta-Llama-3.1-8B-Instruct
# GENERATION_CONDITIONS=protected,raw
# MAX_SHARD_INDEX=31
# TASK_ONLY_ADAPTER=
# ROUTE_VALIDATION_EXTRA_ARGS --skip-allowlist-state-check
# r4_after_868016_controller_generation_h200.sbatch

export ROUTE_CONFIG="${ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_879391_second_family_llama_dev_diagnostic_policy_v4_rerun_route.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_879391_second_family_llama_dev_diagnostic_policy_v4_${SLURM_ARRAY_JOB_ID:-manual}/llama3_1_8b_instruct}"
export CONTEXTUAL_LITERAL_POLICY="${CONTEXTUAL_LITERAL_POLICY:-results/natural_evidence_v2/precommit/r4_after_879391_contextual_forbidden_policy_v4_20260527/contextual_forbidden_surface_policy_v4.json}"
export PUBLIC_RUN_SALT="${PUBLIC_RUN_SALT:-r4_after_879391_second_family_llama_dev_diagnostic_policy_v4}"

exec bash scripts/natural_evidence_v2/slurm/r4_after_877895_second_family_llama_dev_diagnostic_h200.sbatch
