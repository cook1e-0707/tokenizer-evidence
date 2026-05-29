#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4ll96v4
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
#SBATCH --array=0-95
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4ll96v4-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4ll96v4-%A_%a.err

set -euo pipefail

# Llama locked-scale wrapper. It reuses the reviewed Llama generation/decode
# path while changing only the reviewed locked-scale route config, locked-scale
# row bank, precommitted contextual forbidden policy v4, and public run salt.
#
# Validator-visible inherited contract fragments:
# meta-llama/Meta-Llama-3.1-8B-Instruct
# GENERATION_CONDITIONS=protected,raw
# MAX_SHARD_INDEX=95
# TASK_ONLY_ADAPTER=
# ROUTE_VALIDATION_EXTRA_ARGS --skip-allowlist-state-check
# r4_after_868016_controller_generation_h200.sbatch

export ROUTE_CONFIG="${ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_879555_domain_repaired_llama_locked_scale_generation_route.yaml}"
export ROUTE_VALIDATOR="${ROUTE_VALIDATOR:-scripts/natural_evidence_v2/validate_r4_after_879406_second_family_llama_locked_scale_generation_route.py}"
export OUTPUT_DIR="${OUTPUT_DIR:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_879555_domain_repaired_llama_locked_scale_policy_v4_${SLURM_ARRAY_JOB_ID:-manual}/llama3_1_8b_instruct}"
export SCORE_ROWS="${SCORE_ROWS:-results/natural_evidence_v2/status/r4_after_879555_domain_repaired_second_family_llama_locked_scale_row_bank_plan_20260528/row_allocation_rows.jsonl}"
export ALLOCATION_ROWS="${ALLOCATION_ROWS:-results/natural_evidence_v2/status/r4_after_879555_domain_repaired_second_family_llama_locked_scale_row_bank_plan_20260528/row_allocation_rows.jsonl}"
export TOKENIZER_REVIEW="${TOKENIZER_REVIEW:-results/natural_evidence_v2/status/r4_after_879555_domain_repaired_llama_locked_scale_tokenizer_preflight_880120_review/review_summary.json}"
export CONTEXTUAL_LITERAL_POLICY="${CONTEXTUAL_LITERAL_POLICY:-results/natural_evidence_v2/precommit/r4_after_879391_contextual_forbidden_policy_v4_20260527/contextual_forbidden_surface_policy_v4.json}"
export PUBLIC_RUN_SALT="${PUBLIC_RUN_SALT:-r4_after_879555_domain_repaired_llama_locked_scale_policy_v4}"
export MAX_SHARD_INDEX=95
export ROWS_PER_SHARD=1024
export GENERATION_CONDITIONS=protected,raw
export TASK_ONLY_ADAPTER=

exec bash scripts/natural_evidence_v2/slurm/r4_after_877895_second_family_llama_dev_diagnostic_h200.sbatch
