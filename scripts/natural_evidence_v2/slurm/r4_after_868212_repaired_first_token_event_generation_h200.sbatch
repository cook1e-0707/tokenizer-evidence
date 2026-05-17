#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4c16
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4c16-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4c16-%A_%a.err

set -euo pipefail

export ROUTE_CONFIG="${ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_868212_repaired_first_token_event_generation_route.yaml}"
export ROUTE_VALIDATOR="${ROUTE_VALIDATOR:-scripts/natural_evidence_v2/validate_r4_after_868212_repaired_first_token_event_generation_route.py}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}/status/r4_after_868212_repaired_first_token_event_generation_${SLURM_ARRAY_JOB_ID:-manual}}"
export SCORE_ROWS="${SCORE_ROWS:-results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516/reliability_surface_mass_rows.jsonl}"
export ALLOCATION_ROWS="${ALLOCATION_ROWS:-results/natural_evidence_v2/status/r4_after_868212_full16_quality_repair_plan_20260516/row_allocation_rows.jsonl}"
export CONTEXTUAL_LITERAL_POLICY="${CONTEXTUAL_LITERAL_POLICY:-results/natural_evidence_v2/status/r4_after_868212_full16_quality_repair_plan_20260516/contextual_literal_policy.json}"
export TOKENIZER_REVIEW="${TOKENIZER_REVIEW:-results/natural_evidence_v2/status/r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_867828/review_summary.json}"
export CONTROLLER_REVIEW="${CONTROLLER_REVIEW:-results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_controller_score_868114_review/coordinate_pivot_controller_review_summary.json}"
export SURFACE_BANK="${SURFACE_BANK:-results/natural_evidence_v2/precommit/r4_after_864832_coordinate_unique_surface_bank_20260516/surface_bank.json}"
export PRECOMMIT_DIR="${PRECOMMIT_DIR:-results/natural_evidence_v2/precommit/r4_after_868212_repaired_first_token_event_precommit_20260516}"
export CODEBOOK="${CODEBOOK:-$PRECOMMIT_DIR/codebook.json}"
export DECODER_SPEC="${DECODER_SPEC:-$PRECOMMIT_DIR/decoder_spec.json}"
export FULL_PHRASE_DECODER_SPEC="${FULL_PHRASE_DECODER_SPEC:-results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516/decoder_spec.json}"
export DECODER_ROUTE_CONFIG="${DECODER_ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_868212_repaired_first_token_event_decoder_route.yaml}"
export ROWS_PER_SHARD="${ROWS_PER_SHARD:-1024}"
export EXPECTED_SELECTED_COORDINATE_COUNT="${EXPECTED_SELECTED_COORDINATE_COUNT:-16}"

echo "claim_note=R4 after-868212 repaired full16 first-token event generation diagnostic; no paper claim"
echo "route_config=$ROUTE_CONFIG"
echo "route_validator=$ROUTE_VALIDATOR"
echo "score_rows=$SCORE_ROWS"
echo "allocation_rows=$ALLOCATION_ROWS"
echo "precommit_dir=$PRECOMMIT_DIR"
echo "full_phrase_decoder_spec=$FULL_PHRASE_DECODER_SPEC"
echo "rows_per_shard=$ROWS_PER_SHARD"
echo "expected_selected_coordinate_count=$EXPECTED_SELECTED_COORDINATE_COUNT"

bash scripts/natural_evidence_v2/slurm/r4_after_868016_controller_generation_h200.sbatch
