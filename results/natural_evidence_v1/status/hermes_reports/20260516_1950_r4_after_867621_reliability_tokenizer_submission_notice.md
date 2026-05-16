# Hermes Submission Notice: R4 After 867621 Reliability Tokenizer Preflight

phase:
`V2_R4_AFTER_867621_RELIABILITY_TOKENIZER_PREFLIGHT_ROUTE_VALIDATED_NO_SUBMIT`

remote preflight:
`PASS_R4_AFTER_867621_RELIABILITY_TOKENIZER_REMOTE_PREFLIGHT_REPAIRED_NO_SUBMIT`

planned action:

```text
Enable exactly one allowlist entry:
v2_r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_h200

Submit exactly one H200/pomplun tokenizer-only Slurm job:
scripts/natural_evidence_v2/slurm/r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_h200.sbatch

Immediately disable the allowlist entry after sbatch returns.
```

scope:

```text
tokenizer-only actual Qwen boundary preflight
model forward: false
teacher-forced scoring: false
generation: false
training: false
Llama/same-family/sanitizer/FAR/paper claim: false
```

preflight facts:

```text
local/remote hashes match: true
remote route validation: PASS
remote zero-enabled allowlist safety: PASS
active Chimera jobs before submission: none
rows: 4096
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
time: 30-00:00:00
```
