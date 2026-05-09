#!/usr/bin/env bash
#SBATCH --job-name=nat-ev-qwen-bank
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
#SBATCH --partition=pomplun
#SBATCH --account=cs_yinxin.wan
#SBATCH --qos=pomplun
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-bank-%j.out
#SBATCH --error=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3/slurm/nat-ev-qwen-bank-%j.err

set -euo pipefail

REPO_HOME="${REPO_HOME:-$HOME/tokenizer-evidence}"
CONFIG="${CONFIG:-configs/natural_evidence_v1/pilot.yaml}"
VENV_PATH="${VENV_PATH:-/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312}"
EXPANSION_ROOT="${EXPANSION_ROOT:-/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/phase_a_clean_decode8192_qwen_prefixdense_stride2_sharded3}"

SHARD_COUNT="${SHARD_COUNT:-3}"
TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"
TARGET_ENTRIES="${TARGET_ENTRIES:-24000}"
BUCKET_COUNT="${BUCKET_COUNT:-4}"
BALANCE_MIN_BUCKET_MASS="${BALANCE_MIN_BUCKET_MASS:-0.005}"
MAX_BUCKET_MASS_RATIO="${MAX_BUCKET_MASS_RATIO:-5}"
MIN_BUCKET_ENTROPY_FRACTION="${MIN_BUCKET_ENTROPY_FRACTION:-0.90}"

SHARD_DIR="$EXPANSION_ROOT/reference_candidates/shards"
PROGRESS_DIR="$EXPANSION_ROOT/status/shards"
MERGED_CANDIDATES="$EXPANSION_ROOT/reference_candidates/qwen_topk_candidates.jsonl"
STRICT_BANK_DIR="$EXPANSION_ROOT/bucket_banks_4way_strict"
STATUS_DIR="$EXPANSION_ROOT/status"
SLURM_DIR="$EXPANSION_ROOT/slurm"

mkdir -p "$EXPANSION_ROOT/reference_candidates" \
  "$STRICT_BANK_DIR" \
  "$STATUS_DIR" \
  "$SLURM_DIR"

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

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo=$REPO_HOME"
echo "config=$CONFIG"
echo "expansion_root=$EXPANSION_ROOT"
echo "merged_candidates=$MERGED_CANDIDATES"
echo "strict_bank_dir=$STRICT_BANK_DIR"
echo "target_entries=$TARGET_ENTRIES"
echo "bucket_count=$BUCKET_COUNT"
echo "python=$(command -v python3)"

MERGE_SUMMARY_JSON="$STATUS_DIR/qwen_shard_merge_summary_${SLURM_JOB_ID:-manual}.json"
BUILD_SUMMARY_JSON="$STATUS_DIR/qwen_shard_merge_strict_bank_slurm_summary_${SLURM_JOB_ID:-manual}.json"
export EXPANSION_ROOT SHARD_COUNT MERGE_SUMMARY_JSON MERGED_CANDIDATES STRICT_BANK_DIR BUILD_SUMMARY_JSON

python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EXPANSION_ROOT"])
shard_count = int(os.environ["SHARD_COUNT"])
shard_dir = root / "reference_candidates" / "shards"
progress_dir = root / "status" / "shards"
merged = root / "reference_candidates" / "qwen_topk_candidates.jsonl"
summary_path = Path(os.environ["MERGE_SUMMARY_JSON"])

expected_total = 0
shards = []
for index in range(shard_count):
    progress = progress_dir / f"qwen_candidate_shard_{index}_of_{shard_count}_progress.json"
    if not progress.is_file():
        raise SystemExit(f"MISSING_PROGRESS_JSON {progress}")
    payload = json.loads(progress.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise SystemExit(f"SHARD_NOT_COMPLETE {index}: {payload.get('status')}")
    if payload.get("processed_records") != payload.get("total_records"):
        raise SystemExit(
            f"SHARD_RECORD_MISMATCH {index}: "
            f"{payload.get('processed_records')} != {payload.get('total_records')}"
        )
    shard_path = shard_dir / f"qwen_topk_candidates_shard_{index}_of_{shard_count}.jsonl"
    if not shard_path.is_file() or shard_path.stat().st_size <= 0:
        raise SystemExit(f"MISSING_SHARD_JSONL {shard_path}")
    rows = sum(1 for _ in shard_path.open("r", encoding="utf-8"))
    if rows != int(payload.get("output_rows")):
        raise SystemExit(f"SHARD_ROW_MISMATCH {index}: {rows} != {payload.get('output_rows')}")
    shards.append({"index": index, "path": str(shard_path), "rows": rows})
    expected_total += rows

if merged.exists() and merged.stat().st_size > 0:
    merged_rows = sum(1 for _ in merged.open("r", encoding="utf-8"))
    if merged_rows != expected_total:
        raise SystemExit(f"MERGED_EXISTS_WITH_WRONG_ROWS {merged_rows} != {expected_total}")
    status = "reuse_existing"
else:
    tmp = merged.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for shard in shards:
            with Path(shard["path"]).open("r", encoding="utf-8") as src:
                for line in src:
                    out.write(line)
    merged_rows = sum(1 for _ in tmp.open("r", encoding="utf-8"))
    if merged_rows != expected_total:
        raise SystemExit(f"TMP_MERGE_ROW_MISMATCH {merged_rows} != {expected_total}")
    tmp.replace(merged)
    status = "created"

summary = {
    "schema_name": "qwen_candidate_shard_merge_summary_v1",
    "merge_status": status,
    "merged_candidates": str(merged),
    "merged_rows": merged_rows,
    "shards": shards,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY

ENTRIES_PATH="$STRICT_BANK_DIR/qwen_bucket_bank_entries.jsonl"
MANIFEST_PATH="$STRICT_BANK_DIR/qwen_bank_manifest.json"
COVERAGE_PATH="$STRICT_BANK_DIR/qwen_bucket_bank_coverage.csv"
REJECTIONS_PATH="$STRICT_BANK_DIR/qwen_bucket_bank_rejections.csv"
export ENTRIES_PATH MANIFEST_PATH COVERAGE_PATH REJECTIONS_PATH TARGET_ENTRIES

if [ -s "$ENTRIES_PATH" ] || [ -s "$MANIFEST_PATH" ] || [ -s "$COVERAGE_PATH" ] || [ -s "$REJECTIONS_PATH" ]; then
  python3 - <<'PY'
import json
import os
from pathlib import Path

entries = Path(os.environ["ENTRIES_PATH"])
manifest = Path(os.environ["MANIFEST_PATH"])
coverage = Path(os.environ["COVERAGE_PATH"])
rejections = Path(os.environ["REJECTIONS_PATH"])
target_entries = int(os.environ["TARGET_ENTRIES"])
if not (entries.is_file() and manifest.is_file() and coverage.is_file() and rejections.is_file()):
    raise SystemExit("PARTIAL_STRICT_BANK_OUTPUT_EXISTS_REFUSING_OVERWRITE")
entry_rows = sum(1 for _ in entries.open("r", encoding="utf-8"))
payload = json.loads(manifest.read_text(encoding="utf-8"))
if entry_rows != target_entries or int(payload.get("accepted_entries", -1)) != target_entries:
    raise SystemExit(
        "STRICT_BANK_EXISTS_BUT_DOES_NOT_MATCH_TARGET "
        f"rows={entry_rows} accepted={payload.get('accepted_entries')} target={target_entries}"
    )
print(
    json.dumps(
        {
            "strict_bank_status": "reuse_existing",
            "entries": str(entries),
            "entry_rows": entry_rows,
            "manifest": str(manifest),
        },
        sort_keys=True,
    )
)
PY
else
  python3 scripts/natural_evidence_v1/build_bucket_bank.py \
    --config "$CONFIG" \
    --tokenizer-key "$TOKENIZER_KEY" \
    --candidate-jsonl "$MERGED_CANDIDATES" \
    --output-dir "$STRICT_BANK_DIR" \
    --target-entries "$TARGET_ENTRIES" \
    --bucket-count "$BUCKET_COUNT" \
    --strict-balance-gate \
    --balance-min-bucket-mass "$BALANCE_MIN_BUCKET_MASS" \
    --max-bucket-mass-ratio "$MAX_BUCKET_MASS_RATIO" \
    --min-bucket-entropy-fraction "$MIN_BUCKET_ENTROPY_FRACTION"
fi

python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["EXPANSION_ROOT"])
merged = Path(os.environ["MERGED_CANDIDATES"])
bank_dir = Path(os.environ["STRICT_BANK_DIR"])
entries = Path(os.environ["ENTRIES_PATH"])
manifest = Path(os.environ["MANIFEST_PATH"])
coverage = Path(os.environ["COVERAGE_PATH"])
rejections = Path(os.environ["REJECTIONS_PATH"])
summary_path = Path(os.environ["BUILD_SUMMARY_JSON"])

payload = json.loads(manifest.read_text(encoding="utf-8"))
coverage_row = {}
with coverage.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
    if rows:
        coverage_row = rows[0]

summary = {
    "schema_name": "qwen_shard_merge_strict_bank_slurm_summary_v1",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "expansion_root": str(root),
    "merged_candidates": str(merged),
    "merged_candidate_rows": sum(1 for _ in merged.open("r", encoding="utf-8")),
    "strict_bank_dir": str(bank_dir),
    "entries_path": str(entries),
    "entries_rows": sum(1 for _ in entries.open("r", encoding="utf-8")),
    "manifest_path": str(manifest),
    "coverage_path": str(coverage),
    "rejections_path": str(rejections),
    "accepted_entries": payload.get("accepted_entries"),
    "target_bank_entries": payload.get("target_bank_entries"),
    "coverage_complete": coverage_row.get("coverage_complete", ""),
    "strict_balance_gate": payload.get("strict_balance_gate"),
    "bucket_count": payload.get("bucket_count"),
    "claim_control": payload.get("claim_control", {}),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
