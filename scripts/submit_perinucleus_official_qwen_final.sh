#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${MANIFEST:-manifests/baseline_perinucleus_official_qwen_final/eval_manifest.json}"
REGISTRY="${REGISTRY:-manifests/baseline_perinucleus_official_qwen_final/eval_job_registry.jsonl}"

python3 scripts/submit_slurm.py \
  --manifest "$MANIFEST" \
  --registry "$REGISTRY" \
  --all-pending \
  "$@"
