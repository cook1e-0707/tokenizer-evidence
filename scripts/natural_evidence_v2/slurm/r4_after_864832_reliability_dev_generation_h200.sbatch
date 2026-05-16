#!/bin/bash
#SBATCH --job-name=nat-ev-v2-r4relgen
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
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4relgen-%A_%a.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/slurm/nat-ev-v2-r4relgen-%A_%a.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
if [ ! -d "$REPO_HOME" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  CANDIDATE_REPO_HOME="$(cd "$SCRIPT_DIR/../../.." && pwd)"
  if [ -d "$CANDIDATE_REPO_HOME/.git" ]; then
    REPO_HOME="$CANDIDATE_REPO_HOME"
  fi
fi
RUN_ROOT="${RUN_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot}"
ROUTE_CONFIG="${ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_864832_reliability_dev_generation_route.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/status/r4_after_864832_reliability_dev_generation_${SLURM_ARRAY_JOB_ID:-manual}}"
PROMPTS_JSONL="${PROMPTS_JSONL:-results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/dev_prompts.jsonl}"
SURFACE_BANK="${SURFACE_BANK:-results/natural_evidence_v2/precommit/r4_after_864832_coordinate_unique_surface_bank_20260516/surface_bank.json}"
PRECOMMIT_DIR="${PRECOMMIT_DIR:-results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516}"
CODEBOOK="${CODEBOOK:-$PRECOMMIT_DIR/codebook.json}"
DECODER_SPEC="${DECODER_SPEC:-$PRECOMMIT_DIR/decoder_spec.json}"
DECODER_ROUTE_CONFIG="${DECODER_ROUTE_CONFIG:-configs/natural_evidence_v2/r4_after_864832_reliability_codebook_oracle_route.yaml}"
PROTECTED_ADAPTER="${PROTECTED_ADAPTER:-$RUN_ROOT/status/r4_candidate_v3_micro_overfit_864761/protected_micro_overfit_train/adapter}"
TASK_ONLY_ADAPTER="${TASK_ONLY_ADAPTER:-$RUN_ROOT/status/wp5_r2_teacher_forced_train_and_score_851481/task_only_train/adapter}"
PROTECTED_ADAPTER_GAIN="${PROTECTED_ADAPTER_GAIN:-1.0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
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
if [ "$VALIDATE_PLAN_ONLY" = "1" ]; then
  mkdir -p "$OUTPUT_DIR"
else
  mkdir -p "$RUN_ROOT/slurm"
fi

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

echo "claim_note=R4 after-864832 reliability-codebook dev generation diagnostic; no paper claim"
echo "route_config=$ROUTE_CONFIG"
echo "output_dir=$OUTPUT_DIR"
echo "validate_plan_only=$VALIDATE_PLAN_ONLY"
echo "shard_index=${SHARD_INDEX:-unset}"
echo "slurm_policy=H200 pomplun cs_yinxin.wan time 30-00:00:00"
echo "surface_bank=$SURFACE_BANK"
echo "codebook=$CODEBOOK"
echo "decoder_spec=$DECODER_SPEC"
echo "decoder_route_config=$DECODER_ROUTE_CONFIG"
echo "protected_adapter=$PROTECTED_ADAPTER"
echo "task_only_adapter=$TASK_ONLY_ADAPTER"
echo "protected_adapter_gain=$PROTECTED_ADAPTER_GAIN"

if [ "$VALIDATE_PLAN_ONLY" != "1" ]; then
  if [ -z "$SHARD_INDEX" ]; then
    echo "SHARD_INDEX_REQUIRED_FOR_R4_RELIABILITY_DEV_GENERATION" >&2
    exit 2
  fi
  if [ "$SHARD_INDEX" -lt 0 ] || [ "$SHARD_INDEX" -gt 3 ]; then
    echo "INVALID_R4_RELIABILITY_DEV_SHARD_INDEX: $SHARD_INDEX" >&2
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
export SURFACE_BANK
export CODEBOOK
export DECODER_ROUTE_CONFIG
mkdir -p "$plan_validation_dir"

VALIDATION_EXTRA_ARGS=""
if [ "$VALIDATE_PLAN_ONLY" != "1" ]; then
  VALIDATION_EXTRA_ARGS="--allow-submission-enabled-entry"
fi

"$PYTHON_BIN" scripts/natural_evidence_v2/validate_r4_after_864832_reliability_dev_generation_route.py \
  --config "$ROUTE_CONFIG" \
  --output-dir "$plan_validation_dir" \
  $VALIDATION_EXTRA_ARGS

"$PYTHON_BIN" -m py_compile \
  scripts/natural_evidence_v2/decode_r4_after_864832_reliability_codebook.py \
  scripts/natural_evidence_v2/generate_r4_cover_natural_outputs.py \
  scripts/natural_evidence_v2/validate_r4_after_864832_reliability_dev_generation_route.py

"$PYTHON_BIN" scripts/natural_evidence_v2/generate_r4_cover_natural_outputs.py \
  --prompts-jsonl "$PROMPTS_JSONL" \
  --output-dir "$plan_validation_dir/generation_plan" \
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
  --validate-plan-only

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import os
import re
import yaml

plan_dir = Path(os.environ["PLAN_VALIDATION_DIR"])
surface_bank = json.loads(Path(os.environ["SURFACE_BANK"]).read_text(encoding="utf-8"))
codebook = json.loads(Path(os.environ["CODEBOOK"]).read_text(encoding="utf-8"))
route_config = yaml.safe_load(Path(os.environ["DECODER_ROUTE_CONFIG"]).read_text(encoding="utf-8"))
expected = [int(bit) for bit in route_config["expected_codeword_bits"]]
selected = {int(item) for item in codebook["selected_coordinates"]}

def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()

entries = surface_bank["entries"]
by_phrase = {}
for entry in entries:
    phrase = norm(str(entry["canonical_lemma_or_phrase"]))
    by_phrase.setdefault(phrase, []).append(entry)

chosen = []
for item in sorted(codebook["pair_to_bit_mapping"], key=lambda row: int(row["bit_index"])):
    bit = expected[int(item["bit_index"])]
    for coord in [int(value) for value in item["coordinates"]]:
        candidates = [
            entry for entry in entries
            if int(entry["coordinate_id"]) == coord
            and int(entry["polarity_or_code_symbol"]) == bit
        ]
        stable = []
        for entry in candidates:
            phrase = norm(str(entry["canonical_lemma_or_phrase"]))
            selected_alias_entries = [
                other for other in by_phrase.get(phrase, [])
                if int(other["coordinate_id"]) in selected
            ]
            if all(int(other["polarity_or_code_symbol"]) == bit for other in selected_alias_entries):
                stable.append(entry)
        chosen.append(stable[0] if stable else candidates[0])

phrases = [str(entry["canonical_lemma_or_phrase"]) for entry in chosen]
response_text = "For this answer, I will " + ". I will ".join(phrases) + "."
row = {
    "schema_name": "natural_evidence_v2_r4_generated_output_v1",
    "generation_condition": "protected",
    "generation_id": "plan_only_reliability_toy_positive",
    "prompt_id": "plan_only_prompt",
    "prompt_index": 0,
    "replicate_group_id": "plan_only",
    "response_text": response_text,
    "split": "dev",
}
(plan_dir / "toy_generated_outputs.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
(plan_dir / "toy_phrases.json").write_text(json.dumps({"phrases": phrases}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"$PYTHON_BIN" scripts/natural_evidence_v2/decode_r4_after_864832_reliability_codebook.py \
  --generated-outputs "$plan_validation_dir/toy_generated_outputs.jsonl" \
  --surface-bank "$SURFACE_BANK" \
  --codebook "$CODEBOOK" \
  --decoder-spec "$DECODER_SPEC" \
  --route-config "$DECODER_ROUTE_CONFIG" \
  --output-dir "$plan_validation_dir/toy_decode_all" \
  --format-scrub all \
  --prompts-per-block 64 \
  --include-protected-controls

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import os

plan_dir = Path(os.environ["PLAN_VALIDATION_DIR"])
summary = json.loads((plan_dir / "toy_decode_all/decode_summary.json").read_text(encoding="utf-8"))
arms = summary["summary_by_arm"]
if arms["protected"]["accepts"] != 1:
    raise SystemExit("toy protected reliability decode did not accept")
if arms["wrong_key"]["accepts"] != 0 or arms["wrong_payload"]["accepts"] != 0:
    raise SystemExit("toy reliability decoder controls accepted")
result = {
    "status": "PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_WRAPPER_PLAN_ONLY",
    "full_mode_enabled": False,
    "slurm_submission_started": False,
    "model_scoring_started": False,
    "generation_started": False,
    "training_started": False,
    "toy_decode_protected_accepts": arms["protected"]["accepts"],
    "toy_decode_wrong_key_accepts": arms["wrong_key"]["accepts"],
    "toy_decode_wrong_payload_accepts": arms["wrong_payload"]["accepts"],
}
(plan_dir / "wrapper_plan_only_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
PY

if [ "$VALIDATE_PLAN_ONLY" = "1" ]; then
  echo "R4 after-864832 reliability dev generation wrapper plan-only validation passed"
  exit 0
fi

for required in "$PROMPTS_JSONL" "$SURFACE_BANK" "$CODEBOOK" "$DECODER_SPEC" "$DECODER_ROUTE_CONFIG" \
  "scripts/natural_evidence_v2/generate_r4_cover_natural_outputs.py" \
  "scripts/natural_evidence_v2/decode_r4_after_864832_reliability_codebook.py"; do
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
  echo "R4_RELIABILITY_DEV_SHARD_OUTPUT_EXISTS_REFUSING_OVERWRITE: $shard_dir" >&2
  exit 74
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

"$PYTHON_BIN" scripts/natural_evidence_v2/decode_r4_after_864832_reliability_codebook.py \
  --generated-outputs "$shard_dir/r4_generated_outputs.jsonl" \
  --surface-bank "$SURFACE_BANK" \
  --codebook "$CODEBOOK" \
  --decoder-spec "$DECODER_SPEC" \
  --route-config "$DECODER_ROUTE_CONFIG" \
  --output-dir "$shard_dir/decode_all" \
  --format-scrub all \
  --prompts-per-block 64 \
  --include-protected-controls

"$PYTHON_BIN" scripts/natural_evidence_v2/decode_r4_after_864832_reliability_codebook.py \
  --generated-outputs "$shard_dir/r4_generated_outputs.jsonl" \
  --surface-bank "$SURFACE_BANK" \
  --codebook "$CODEBOOK" \
  --decoder-spec "$DECODER_SPEC" \
  --route-config "$DECODER_ROUTE_CONFIG" \
  --output-dir "$shard_dir/decode_none" \
  --format-scrub none \
  --prompts-per-block 64 \
  --include-protected-controls

echo "R4 after-864832 reliability dev generation shard complete: $shard_id $shard_dir"
