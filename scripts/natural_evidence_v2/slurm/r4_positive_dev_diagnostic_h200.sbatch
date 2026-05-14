#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4posdev
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4posdev-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4posdev-%A_%a.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
ROUTE_CONFIG="${ROUTE_CONFIG:-configs/natural_evidence_v2/r4_positive_dev_diagnostic_route.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/r4_positive_event_bank_dev_diagnostic_${SLURM_ARRAY_JOB_ID:-manual}}"
PRECOMMIT_DIR="${PRECOMMIT_DIR:-results/natural_evidence_v2/precommit/r4_positive_event_bank_precommit_20260514_1605}"
PROMPTS_JSONL="${PROMPTS_JSONL:-results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/dev_prompts.jsonl}"
SURFACE_BANK="${SURFACE_BANK:-$PRECOMMIT_DIR/surface_bank.json}"
CODEBOOK="${CODEBOOK:-$PRECOMMIT_DIR/codebook.json}"
DECODER_SPEC="${DECODER_SPEC:-$PRECOMMIT_DIR/decoder_spec.json}"
PRECOMMIT_MANIFEST="${PRECOMMIT_MANIFEST:-$PRECOMMIT_DIR/precommit_manifest.json}"
PROTECTED_ADAPTER="${PROTECTED_ADAPTER:-$RUN_ROOT/status/r4_candidate_v3_pressure_relaxation_grid_857764/B_ceiling_lambda_0_5/protected_micro_overfit_train/adapter}"
TASK_ONLY_ADAPTER="${TASK_ONLY_ADAPTER:-$RUN_ROOT/status/wp5_r2_teacher_forced_train_and_score_851481/task_only_train/adapter}"
PROTECTED_ADAPTER_GAIN="${PROTECTED_ADAPTER_GAIN:-1.0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
ALLOW_STATIC_DEV_KEYS="${ALLOW_STATIC_DEV_KEYS:-0}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"
VALIDATE_PLAN_ONLY="${VALIDATE_PLAN_ONLY:-0}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-}}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3" ]; then
    PYTHON_BIN="/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3"
  elif [ -x "$REPO_HOME/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_HOME/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

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

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\r\n' <"$HF_TOKEN_FILE")"
fi

echo "claim_note=R4 positive event-bank dev diagnostic wrapper; plan-only implemented; no paper claim"
echo "route_config=$ROUTE_CONFIG"
echo "output_dir=$OUTPUT_DIR"
echo "validate_plan_only=$VALIDATE_PLAN_ONLY"
echo "shard_index=${SHARD_INDEX:-unset}"
echo "slurm_policy=H200 pomplun cs_yinxin.wan time 30-00:00:00"
echo "python_bin=$PYTHON_BIN"
echo "precommit_dir=$PRECOMMIT_DIR"
echo "prompts_jsonl=$PROMPTS_JSONL"
echo "surface_bank=$SURFACE_BANK"
echo "codebook=$CODEBOOK"
echo "decoder_spec=$DECODER_SPEC"
echo "precommit_manifest=$PRECOMMIT_MANIFEST"
echo "protected_adapter=$PROTECTED_ADAPTER"
echo "task_only_adapter=$TASK_ONLY_ADAPTER"
echo "protected_adapter_gain=$PROTECTED_ADAPTER_GAIN"
echo "allow_static_dev_keys=$ALLOW_STATIC_DEV_KEYS"

if [ "$VALIDATE_PLAN_ONLY" != "1" ]; then
  if [ -z "$SHARD_INDEX" ]; then
    echo "SHARD_INDEX_REQUIRED_FOR_FULL_R4_POSITIVE_DEV_DIAGNOSTIC" >&2
    exit 2
  fi
  if [ "$SHARD_INDEX" -lt 0 ] || [ "$SHARD_INDEX" -gt 3 ]; then
    echo "INVALID_R4_POSITIVE_DEV_SHARD_INDEX: $SHARD_INDEX" >&2
    exit 2
  fi
  expected_start=$((SHARD_INDEX * 512))
  expected_end=$((expected_start + 511))
  shard_id="$(printf 'shard_%02d' "$SHARD_INDEX")"
  shard_dir="$OUTPUT_DIR/shards/$shard_id"
  plan_validation_dir="$shard_dir/plan_validation"
else
  expected_start=0
  expected_end=511
  shard_id="plan_only"
  shard_dir="$OUTPUT_DIR"
  plan_validation_dir="$OUTPUT_DIR/plan_validation"
fi

export PLAN_VALIDATION_DIR="$plan_validation_dir"
mkdir -p "$plan_validation_dir"

"$PYTHON_BIN" scripts/natural_evidence_v2/validate_r4_positive_dev_diagnostic_route.py \
  --config "$ROUTE_CONFIG" \
  --output-dir "$plan_validation_dir"

"$PYTHON_BIN" -m py_compile \
  scripts/natural_evidence_v2/extract_r4_positive_phrase_events.py \
  scripts/natural_evidence_v2/r4_keyed_correlation_decoder.py \
  scripts/natural_evidence_v2/decode_r4_positive_keyed_correlation.py \
  scripts/natural_evidence_v2/generate_r4_cover_natural_outputs.py \
  scripts/natural_evidence_v2/validate_r4_positive_dev_diagnostic_route.py

"$PYTHON_BIN" scripts/natural_evidence_v2/extract_r4_positive_phrase_events.py \
  --text "1. Ask a focused question. Next action: use a calm tone and compare with a known case." \
  --scrub-mode all \
  --output-jsonl "$plan_validation_dir/extractor_smoke_events.jsonl"

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import os
output_dir = Path(os.environ["PLAN_VALIDATION_DIR"])
events = output_dir / "extractor_smoke_events.jsonl"
rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 3:
    raise SystemExit(f"expected 3 extractor smoke events, got {len(rows)}")
summary = {
    "status": "PASS_R4_POSITIVE_DEV_DIAGNOSTIC_WRAPPER_PLAN_ONLY",
    "full_mode_enabled": False,
    "slurm_submission_started": False,
    "generation_started": False,
    "training_started": False,
    "extractor_smoke_event_count": len(rows),
}
(output_dir / "wrapper_plan_only_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if [ "$VALIDATE_PLAN_ONLY" = "1" ]; then
  echo "R4 positive dev diagnostic wrapper plan-only validation passed"
  exit 0
fi

for required in "$PROMPTS_JSONL" "$SURFACE_BANK" "$CODEBOOK" "$DECODER_SPEC" "$PRECOMMIT_MANIFEST" \
  "scripts/natural_evidence_v2/generate_r4_cover_natural_outputs.py" \
  "scripts/natural_evidence_v2/decode_r4_positive_keyed_correlation.py"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ARTIFACT_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

for required in "$PROTECTED_ADAPTER/adapter_config.json" "$TASK_ONLY_ADAPTER/adapter_config.json"; do
  if [ ! -s "$required" ]; then
    echo "REQUIRED_ADAPTER_MISSING_OR_EMPTY: $required" >&2
    exit 2
  fi
done

if [ -e "$shard_dir/r4_generated_outputs.jsonl" ] || \
   [ -e "$shard_dir/decode_all/decode_summary.json" ] || \
   [ -e "$shard_dir/decode_none/decode_summary.json" ]; then
  echo "R4_POSITIVE_DEV_SHARD_OUTPUT_EXISTS_REFUSING_OVERWRITE: $shard_dir" >&2
  exit 74
fi

decode_key_args=()
if [ "$ALLOW_STATIC_DEV_KEYS" = "1" ]; then
  decode_key_args+=(--allow-static-dev-keys)
fi

mkdir -p "$shard_dir"

"$PYTHON_BIN" scripts/natural_evidence_v2/generate_r4_cover_natural_outputs.py \
  --prompts-jsonl "$PROMPTS_JSONL" \
  --output-dir "$shard_dir" \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --protected-adapter "$PROTECTED_ADAPTER" \
  --task-only-adapter "$TASK_ONLY_ADAPTER" \
  --protected-adapter-gain "$PROTECTED_ADAPTER_GAIN" \
  --split dev \
  --max-prompts 512 \
  --prompt-file-row-start "$expected_start" \
  --prompt-file-row-end "$expected_end" \
  --replicate-group-id "$shard_id" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --require-cuda

"$PYTHON_BIN" scripts/natural_evidence_v2/decode_r4_positive_keyed_correlation.py \
  --generated-outputs "$shard_dir/r4_generated_outputs.jsonl" \
  --surface-bank "$SURFACE_BANK" \
  --codebook "$CODEBOOK" \
  --decoder-spec "$DECODER_SPEC" \
  --precommit-manifest "$PRECOMMIT_MANIFEST" \
  --output-dir "$shard_dir/decode_all" \
  --format-scrub all \
  --prompts-per-block 64 \
  --include-protected-controls \
  "${decode_key_args[@]}"

"$PYTHON_BIN" scripts/natural_evidence_v2/decode_r4_positive_keyed_correlation.py \
  --generated-outputs "$shard_dir/r4_generated_outputs.jsonl" \
  --surface-bank "$SURFACE_BANK" \
  --codebook "$CODEBOOK" \
  --decoder-spec "$DECODER_SPEC" \
  --precommit-manifest "$PRECOMMIT_MANIFEST" \
  --output-dir "$shard_dir/decode_none" \
  --format-scrub none \
  --prompts-per-block 64 \
  --include-protected-controls \
  "${decode_key_args[@]}"

echo "R4 positive dev diagnostic shard complete: $shard_id $shard_dir"
