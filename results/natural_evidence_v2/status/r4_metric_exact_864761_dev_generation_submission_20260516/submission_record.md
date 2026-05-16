# R4 metric-exact 864761 dev generation submission

Status: `SUBMITTED_R4_METRIC_EXACT_864761_DEV_GENERATION_H200_ARRAY_RUNNING`

- Job id: `864832`
- Job name: `nat-ev-v2-r4megd`
- Array: `0-3%4`
- Partition/QOS/account: `pomplun / pomplun / cs_yinxin.wan`
- GPU: `h200`
- Time limit: `30-00:00:00`
- Allowlist entry: `v2_r4_candidate_v3_metric_exact_864761_dev_diagnostic_h200`
- Source adapter job: `864761`
- Post-submit allowlist safety: local `PASS`, remote `PASS`

Scope: small Qwen dev generation/decode diagnostic only. No training, Llama, same-family null, sanitizer, FAR, payload diversity, or paper-facing claim.

Current squeue:

```text
JOBID NAME STATE TIME NODES NODELIST(REASON)
864832_3 nat-ev-v2-r4megd RUNNING 0:27 1 chimera21
864832_2 nat-ev-v2-r4megd RUNNING 0:27 1 chimera21
864832_1 nat-ev-v2-r4megd RUNNING 0:27 1 chimera21
864832_0 nat-ev-v2-r4megd RUNNING 0:28 1 chimera21
```
