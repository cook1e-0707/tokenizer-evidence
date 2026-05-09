#!/usr/bin/env bash
#SBATCH --job-name=nat-ev-qwen-cf
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=pomplun
#SBATCH --account=cs_yinxin.wan
#SBATCH --qos=pomplun
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-cf-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-cf-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
HF_HOME="${HF_HOME:-/hpcstor6/scratch01/g/guanjie.lin001/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/hpcstor6/scratch01/g/guanjie.lin001/keys/hf_token}"

EXPANSION_ROOT="${EXPANSION_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3}"
SOURCE_ROOT="${SOURCE_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192}"

TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
SUFFIX_WINDOW_TOKENS="${SUFFIX_WINDOW_TOKENS:-16}"
TOKENS_PER_BUCKET="${TOKENS_PER_BUCKET:-2}"
BUCKET_COUNT="${BUCKET_COUNT:-4}"
MAX_RECORDS="${MAX_RECORDS:-24000}"
DELTA_NLL_THRESHOLD="${DELTA_NLL_THRESHOLD:-0.5}"

BANK_ENTRIES="$EXPANSION_ROOT/bucket_banks_4way_strict/qwen_bucket_bank_entries.jsonl"
REFERENCE_OUTPUTS="$SOURCE_ROOT/reference_outputs/qwen_reference_outputs.jsonl"
CF_INPUT="$EXPANSION_ROOT/bucket_banks_4way_strict/qwen_counterfactual_candidates_from_bank_entries.jsonl"
OUTPUT_JSONL="$EXPANSION_ROOT/bucket_banks_4way_strict/qwen_counterfactual_compatibility.jsonl"
TABLE_DIR="$EXPANSION_ROOT/tables/qwen_4way_strict_expanded"
OUTPUT_CSV="$TABLE_DIR/counterfactual_compatibility.csv"
SUMMARY_JSON="$TABLE_DIR/counterfactual_compatibility_summary.json"

mkdir -p "$EXPANSION_ROOT/slurm" "$TABLE_DIR"

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

export BANK_ENTRIES CF_INPUT OUTPUT_JSONL OUTPUT_CSV SUMMARY_JSON
export MAX_RECORDS TOKENS_PER_BUCKET BUCKET_COUNT DELTA_NLL_THRESHOLD

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "config=$CONFIG"
echo "expansion_root=$EXPANSION_ROOT"
echo "bank_entries=$BANK_ENTRIES"
echo "reference_outputs=$REFERENCE_OUTPUTS"
echo "cf_input=$CF_INPUT"
echo "output_jsonl=$OUTPUT_JSONL"
echo "output_csv=$OUTPUT_CSV"
echo "summary_json=$SUMMARY_JSON"
echo "suffix_window_tokens=$SUFFIX_WINDOW_TOKENS"
echo "tokens_per_bucket=$TOKENS_PER_BUCKET"
echo "max_records=$MAX_RECORDS"
echo "delta_nll_threshold=$DELTA_NLL_THRESHOLD"
echo "python=$(command -v python3)"

if [ ! -s "$BANK_ENTRIES" ]; then
  echo "Missing bank entries: $BANK_ENTRIES" >&2
  exit 2
fi
if [ ! -s "$REFERENCE_OUTPUTS" ]; then
  echo "Missing reference outputs: $REFERENCE_OUTPUTS" >&2
  exit 2
fi
if [ -s "$OUTPUT_JSONL" ] || [ -s "$OUTPUT_CSV" ] || [ -s "$SUMMARY_JSON" ]; then
  echo "Counterfactual outputs already exist; refusing overwrite" >&2
  exit 74
fi

python3 - <<'PY'
import json
import os
from pathlib import Path

entries_path = Path(os.environ["BANK_ENTRIES"])
output_path = Path(os.environ["CF_INPUT"])
max_records = int(os.environ["MAX_RECORDS"])
tokens_per_bucket = int(os.environ["TOKENS_PER_BUCKET"])
bucket_count = int(os.environ["BUCKET_COUNT"])

if output_path.exists() and output_path.stat().st_size > 0:
    existing_rows = sum(1 for _ in output_path.open("r", encoding="utf-8"))
    if existing_rows != max_records:
        raise SystemExit(f"EXISTING_CF_INPUT_ROW_COUNT_MISMATCH {existing_rows} != {max_records}")
    print(json.dumps({"cf_input_status": "reuse_existing", "rows": existing_rows}, sort_keys=True))
    raise SystemExit(0)

tmp_path = output_path.with_suffix(".jsonl.tmp")
rows = 0
with entries_path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as out:
    for line in src:
        if rows >= max_records:
            break
        entry = json.loads(line)
        buckets = entry.get("buckets", {})
        bucket_token_texts = entry.get("bucket_token_texts", {})
        candidates = []
        for bucket_id in sorted(buckets, key=lambda value: int(value)):
            token_ids = list(buckets.get(bucket_id, []))[:tokens_per_bucket]
            token_texts = list(bucket_token_texts.get(bucket_id, []))
            for index, token_id in enumerate(token_ids):
                candidates.append(
                    {
                        "token_id": int(token_id),
                        "text": token_texts[index] if index < len(token_texts) else "",
                        "bucket_id": bucket_id,
                        "rank": index,
                        "probability": "",
                    }
                )
        if len(candidates) < bucket_count:
            continue
        out.write(
            json.dumps(
                {
                    "schema_name": "natural_evidence_counterfactual_candidate_from_bank_entry_v1",
                    "protocol_id": entry.get("protocol_id", "natural_evidence_v1"),
                    "bank_entry_id": entry.get("bank_entry_id", ""),
                    "bank_id": entry.get("bank_id", ""),
                    "context_signature": entry.get("context_signature", ""),
                    "prompt_id": entry.get("prompt_id", ""),
                    "prefix_response_token_count": entry.get("prefix_response_token_count", 0),
                    "bucket_count": entry.get("bucket_count", bucket_count),
                    "candidates": candidates,
                    "fingerprint_claim": False,
                },
                sort_keys=True,
            )
            + "\n"
        )
        rows += 1

if rows != max_records:
    raise SystemExit(f"CF_INPUT_UNDER_TARGET {rows} != {max_records}")
tmp_path.replace(output_path)
print(json.dumps({"cf_input_status": "created", "rows": rows}, sort_keys=True))
PY

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

python3 scripts/natural_evidence_v1/score_counterfactual_compatibility.py \
  --config "$CONFIG" \
  --tokenizer-key "$TOKENIZER_KEY" \
  --reference-outputs "$REFERENCE_OUTPUTS" \
  --candidate-jsonl "$CF_INPUT" \
  --output-jsonl "$OUTPUT_JSONL" \
  --suffix-window-tokens "$SUFFIX_WINDOW_TOKENS" \
  --max-candidates-per-record "$((BUCKET_COUNT * TOKENS_PER_BUCKET))" \
  --max-records "$MAX_RECORDS" \
  --delta-nll-threshold "$DELTA_NLL_THRESHOLD" \
  --require-cuda

python3 - <<'PY'
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

output_jsonl = Path(os.environ["OUTPUT_JSONL"])
output_csv = Path(os.environ["OUTPUT_CSV"])
summary_json = Path(os.environ["SUMMARY_JSON"])
threshold = float(os.environ["DELTA_NLL_THRESHOLD"])

rows = [json.loads(line) for line in output_jsonl.open("r", encoding="utf-8")]
if not rows:
    raise SystemExit("NO_COUNTERFACTUAL_ROWS")

deltas = [float(row.get("delta_suffix_nll_per_token", row["delta_suffix_nll"])) for row in rows]
raw_deltas = [float(row.get("delta_suffix_nll_raw", row["delta_suffix_nll"])) for row in rows]
suffix_lengths = [int(row.get("suffix_window_length", 0)) for row in rows]
passed = [bool(row["compatibility_pass"]) for row in rows]
by_entry: dict[str, list[dict]] = defaultdict(list)
for row in rows:
    key = str(row.get("bank_entry_id") or f"{row.get('prompt_id')}:{row.get('prefix_response_token_count')}")
    by_entry[key].append(row)

entry_passes = {
    entry_id: all(bool(row["compatibility_pass"]) for row in entry_rows)
    for entry_id, entry_rows in by_entry.items()
}

summary = {
    "schema_name": "natural_evidence_counterfactual_compatibility_summary_v1",
    "output_jsonl": str(output_jsonl),
    "output_csv": str(output_csv),
    "scored_candidate_rows": len(rows),
    "scored_bank_entries": len(by_entry),
    "compatible_candidate_rows": sum(1 for value in passed if value),
    "compatibility_pass_rate": sum(1 for value in passed if value) / len(passed),
    "fully_compatible_entries": sum(1 for value in entry_passes.values() if value),
    "entry_pass_rate": sum(1 for value in entry_passes.values() if value) / len(entry_passes),
    "delta_nll_threshold": threshold,
    "delta_nll_threshold_units": "per_token",
    "delta_suffix_nll_mean": statistics.mean(deltas),
    "delta_suffix_nll_median": statistics.median(deltas),
    "delta_suffix_nll_p95": sorted(deltas)[int(0.95 * (len(deltas) - 1))],
    "delta_suffix_nll_max": max(deltas),
    "delta_suffix_nll_per_token_mean": statistics.mean(deltas),
    "delta_suffix_nll_per_token_median": statistics.median(deltas),
    "delta_suffix_nll_per_token_p95": sorted(deltas)[int(0.95 * (len(deltas) - 1))],
    "delta_suffix_nll_raw_mean": statistics.mean(raw_deltas),
    "delta_suffix_nll_raw_median": statistics.median(raw_deltas),
    "delta_suffix_nll_raw_p95": sorted(raw_deltas)[int(0.95 * (len(raw_deltas) - 1))],
    "suffix_window_length_min": min(suffix_lengths),
    "suffix_window_length_max": max(suffix_lengths),
    "suffix_window_length_mean": statistics.mean(suffix_lengths),
    "fingerprint_claim": False,
    "result_claim": "counterfactual_compatibility_quality_gate_not_payload_recovery",
}

output_csv.parent.mkdir(parents=True, exist_ok=True)
with output_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary))
    writer.writeheader()
    writer.writerow(summary)
summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
