# R4 After 867621 Reliability Tokenizer Preflight Submission

status:
`SUBMITTED_SINGLE_TOKENIZER_ONLY_H200_SLURM_JOB_ALLOWLIST_DISABLED_AFTER_SUBMISSION`

```text
job_id: 867828
job_name: nat-ev-v2-r4relTok
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
time: 30-00:00:00
wrapper: scripts/natural_evidence_v2/slurm/r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_h200.sbatch
score_rows: results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516/reliability_surface_mass_rows.jsonl
```

Scope:

```text
Tokenizer-only. No model forward, no scoring, no training, no generation, no
Llama, no same-family null, no sanitizer, no FAR, no payload-diversity claim,
and no paper-facing claim.
```

Allowlist was disabled immediately after submission. Remote post-submit
allowlist safety passed with zero enabled entries.
