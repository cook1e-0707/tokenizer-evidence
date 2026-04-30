# Official Perinucleus Qwen Final Protocol

Generated at: `20260430T035120397576Z`
Decision: `PERINUCLEUS_OFFICIAL_QWEN_FINAL_DRY_RUN_READY: review manifest before submission.`

This is a dry-run package for the final protocol. It renders the manifest and case table only; it does not submit jobs.

## Frozen Candidate

- Arm: `all_linear_r64_fp64_e80`
- Adapter: `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_qwen_capacity_sweep/runs/manual_20260429T182756Z/arms/all_linear_r64_fp64_e80/adapter_final`
- Fingerprints file: `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_qwen_capacity_sweep/runs/manual_20260429T182756Z/generated/fingerprints_64/fingerprints-perinucleus-Qwen-Qwen2.5-7B-Instruct-nucleus_threshold-0.8-response_length-1-use_chat_template-True.json`
- Exact accuracy: `1.0`
- Utility sanity: `0.6191832009104691` vs base `0.6035317339934293`

## Matrix

- Cases: `48`
- Payloads: `['U00', 'U03', 'U12', 'U15']`
- Seeds: `[17, 23, 29]`
- Query budgets: `[1, 3, 5, 10]`

## Execution

Prepare or inspect jobs without submission:

```bash
python3 scripts/submit_slurm.py \
  --manifest manifests/baseline_perinucleus_official_qwen_final/eval_manifest.json \
  --registry manifests/baseline_perinucleus_official_qwen_final/eval_job_registry.jsonl \
  --all-pending
```

Submit only after reviewing the rendered scripts:

```bash
# First run one smoke case.
python3 scripts/submit_slurm.py \
  --manifest manifests/baseline_perinucleus_official_qwen_final/eval_manifest.json \
  --registry manifests/baseline_perinucleus_official_qwen_final/eval_job_registry.jsonl \
  --manifest-id perinucleus-official-q1-u00-s17 \
  --submit --force

# If the smoke case completes, submit the remaining pending cases.
python3 scripts/submit_slurm.py \
  --manifest manifests/baseline_perinucleus_official_qwen_final/eval_manifest.json \
  --registry manifests/baseline_perinucleus_official_qwen_final/eval_job_registry.jsonl \
  --all-pending --submit --force
```

## Outputs

- Dry-run summary: `results/processed/paper_stats/baseline_perinucleus_official_qwen_final_package_dry_run.json`
- Case table: `results/tables/baseline_perinucleus_official_qwen_final_cases.csv`
- Manifest: `manifests/baseline_perinucleus_official_qwen_final/eval_manifest.json`

After jobs finish, aggregate final artifacts with:

```bash
python3 scripts/build_perinucleus_official_final_artifacts.py
```
