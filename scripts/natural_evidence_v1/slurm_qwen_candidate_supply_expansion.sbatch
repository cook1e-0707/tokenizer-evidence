#!/usr/bin/env bash
#SBATCH --job-name=nat-ev-qwen-candx
#SBATCH --partition=pomplun
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

SOURCE_ROOT="${SOURCE_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192}"
EXPANSION_ROOT="${EXPANSION_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2}"

CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-64}"
MAX_PREFIXES_PER_RESPONSE="${MAX_PREFIXES_PER_RESPONSE:-64}"
PREFIX_STRIDE="${PREFIX_STRIDE:-2}"
MIN_RESPONSE_PREFIX_TOKENS="${MIN_RESPONSE_PREFIX_TOKENS:-1}"
TARGET_ENTRIES="${TARGET_ENTRIES:-24000}"
BALANCE_MIN_BUCKET_MASS="${BALANCE_MIN_BUCKET_MASS:-0.005}"
MAX_BUCKET_MASS_RATIO="${MAX_BUCKET_MASS_RATIO:-5}"
MIN_BUCKET_ENTROPY_FRACTION="${MIN_BUCKET_ENTROPY_FRACTION:-0.90}"

REFERENCE_OUTPUTS="$SOURCE_ROOT/reference_outputs/qwen_reference_outputs.jsonl"
REFERENCE_CANDIDATES="$EXPANSION_ROOT/reference_candidates/qwen_topk_candidates.jsonl"
STRICT_BANK_DIR="$EXPANSION_ROOT/bucket_banks_4way_strict"
STATUS_DIR="$EXPANSION_ROOT/status"

mkdir -p "$EXPANSION_ROOT/slurm" \
  "$EXPANSION_ROOT/reference_candidates" \
  "$STRICT_BANK_DIR" \
  "$STATUS_DIR"

cd "$REPO_HOME"

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

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "config=$CONFIG"
echo "source_root=$SOURCE_ROOT"
echo "expansion_root=$EXPANSION_ROOT"
echo "reference_outputs=$REFERENCE_OUTPUTS"
echo "candidate_top_k=$CANDIDATE_TOP_K"
echo "max_prefixes_per_response=$MAX_PREFIXES_PER_RESPONSE"
echo "prefix_stride=$PREFIX_STRIDE"
echo "python=$(command -v python3)"

if [ ! -s "$REFERENCE_OUTPUTS" ]; then
  echo "Missing reference outputs: $REFERENCE_OUTPUTS" >&2
  exit 2
fi

python3 - <<'PY'
import os
import torch

print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("device count =", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA_NOT_AVAILABLE_ON_ALLOCATED_NODE")
print("device name =", torch.cuda.get_device_name(0))
PY

python3 scripts/natural_evidence_v1/validate_static.py \
  --config "$CONFIG" \
  --summary "$STATUS_DIR/static_validation_summary_qwen_candidate_expansion.json"

python3 scripts/natural_evidence_v1/score_reference_candidates.py \
  --config "$CONFIG" \
  --tokenizer-key qwen \
  --input-jsonl "$REFERENCE_OUTPUTS" \
  --output-jsonl "$REFERENCE_CANDIDATES" \
  --candidate-top-k "$CANDIDATE_TOP_K" \
  --max-prefixes-per-response "$MAX_PREFIXES_PER_RESPONSE" \
  --prefix-stride "$PREFIX_STRIDE" \
  --min-response-prefix-tokens "$MIN_RESPONSE_PREFIX_TOKENS" \
  --require-cuda

python3 scripts/natural_evidence_v1/build_bucket_bank.py \
  --config "$CONFIG" \
  --tokenizer-key qwen \
  --candidate-jsonl "$REFERENCE_CANDIDATES" \
  --output-dir "$STRICT_BANK_DIR" \
  --target-entries "$TARGET_ENTRIES" \
  --bucket-count 4 \
  --strict-balance-gate \
  --balance-min-bucket-mass "$BALANCE_MIN_BUCKET_MASS" \
  --max-bucket-mass-ratio "$MAX_BUCKET_MASS_RATIO" \
  --min-bucket-entropy-fraction "$MIN_BUCKET_ENTROPY_FRACTION"

python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EXPANSION_ROOT"])
manifest = root / "bucket_banks_4way_strict" / "qwen_bank_manifest.json"
payload = json.loads(manifest.read_text(encoding="utf-8"))
summary = {
    "schema_name": "qwen_candidate_supply_expansion_summary_v1",
    "expansion_root": str(root),
    "reference_candidates": str(root / "reference_candidates" / "qwen_topk_candidates.jsonl"),
    "strict_bank_manifest": str(manifest),
    "accepted_entries": payload.get("accepted_entries"),
    "target_bank_entries": payload.get("target_bank_entries"),
    "coverage_complete": payload.get("accepted_entries", 0) >= payload.get("target_bank_entries", 24000),
    "strict_balance_gate": payload.get("strict_balance_gate"),
    "claim_control": payload.get("claim_control", {}),
}
path = root / "status" / "qwen_candidate_supply_expansion_summary.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
