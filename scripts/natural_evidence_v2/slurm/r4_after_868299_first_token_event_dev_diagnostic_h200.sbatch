#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4dev
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
#SBATCH --array=0-31%4
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4dev-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4dev-%A_%a.err

set -euo pipefail

export ROUTE_CONFIG="${ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_868299_first_token_event_dev_diagnostic_route.yaml}"
export ROUTE_VALIDATOR="${ROUTE_VALIDATOR:-scripts/natural_evidence_v2/validate_r4_after_868299_first_token_event_dev_diagnostic_route.py}"
export PLAN_ONLY="${PLAN_ONLY:-1}"
export PYTHON_BIN="${PYTHON_BIN:-}"

echo "claim_note=R4 after-868299 first-token event 32-block dev diagnostic; no paper claim"
echo "route_config=$ROUTE_CONFIG"
echo "route_validator=$ROUTE_VALIDATOR"
echo "plan_only=$PLAN_ONLY"
echo "generation_started=false"
echo "training_started=false"
echo "llama_started=false"
echo "far_started=false"
echo "paper_claim_allowed=false"

if [[ "$PLAN_ONLY" != "1" ]]; then
  export ROUTE_CONFIG
  export ROUTE_VALIDATOR
  export OUTPUT_DIR="${OUTPUT_DIR:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_868299_first_token_event_dev_diagnostic_${SLURM_ARRAY_JOB_ID:-manual}}"
  export SCORE_ROWS="${SCORE_ROWS:-results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516/reliability_surface_mass_rows.jsonl}"
  export ALLOCATION_ROWS="${ALLOCATION_ROWS:-results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_plan_20260517/row_allocation_rows.jsonl}"
  export CONTEXTUAL_LITERAL_POLICY="${CONTEXTUAL_LITERAL_POLICY:-results/natural_evidence_v2/precommit/r4_after_868260_quality_gate_repair_package_20260517/contextual_forbidden_surface_policy_v2.json}"
  export DUPLICATE_SAFE_POLICY="${DUPLICATE_SAFE_POLICY:-configs/natural_evidence_v2/r4_first_token_event_duplicate_safe_generation_v2.yaml}"
  export PUBLIC_RUN_SALT="${PUBLIC_RUN_SALT:-r4_after_868299_first_token_event_dev_diagnostic_v1}"
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
  export MAX_SHARD_INDEX=31
  export KEY_ID_NOT_SECRET_KEY="${KEY_ID_NOT_SECRET_KEY:-r4_first_token_event_controller_key_v1}"
  export PAYLOAD_ID="${PAYLOAD_ID:-a55e}"
  export VALIDATE_PLAN_ONLY="${VALIDATE_PLAN_ONLY:-0}"
  echo "full_mode_delegates_to=scripts/natural_evidence_v2/slurm/r4_after_868016_controller_generation_h200.sbatch"
  echo "output_dir=$OUTPUT_DIR"
  echo "allocation_rows=$ALLOCATION_ROWS"
  echo "duplicate_safe_policy=$DUPLICATE_SAFE_POLICY"
  echo "contextual_literal_policy=$CONTEXTUAL_LITERAL_POLICY"
  exec bash scripts/natural_evidence_v2/slurm/r4_after_868016_controller_generation_h200.sbatch
fi

if [[ -n "$PYTHON_BIN" ]]; then
  "$PYTHON_BIN" "$ROUTE_VALIDATOR" --config "$ROUTE_CONFIG" --output-dir "${OUTPUT_DIR:-results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_wrapper_plan_smoke_20260517}"
else
  uv run python "$ROUTE_VALIDATOR" --config "$ROUTE_CONFIG" --output-dir "${OUTPUT_DIR:-results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_wrapper_plan_smoke_20260517}"
fi
