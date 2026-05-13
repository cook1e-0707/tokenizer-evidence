# Hermes sync: R4 dev diagnostic submitted

Phase: `V2_R4_DEV_DIAGNOSTIC_H200_ARRAY_853691_SUBMITTED_MONITOR_ONLY`

Codex submitted exactly one reviewed R4 dev diagnostic Slurm array job after
the required preflights passed.

Submission:

- job id: `853691`;
- job name: `nat-ev-v2-r4dev`;
- partition/QOS/account: `pomplun` / `pomplun` / `cs_yinxin.wan`;
- GPU: `h200`;
- array: `0-3%4`;
- contract: same-contract `a55e`;
- prompt split: dev only;
- prompts: `2048`;
- blocks: `32`;
- primary decode: `format_scrub=all`.

Control-plane checks:

- local zero-enabled allowlist safety before enablement: PASS;
- remote zero-enabled allowlist safety before enablement: PASS;
- local/remote hash preflight: PASS;
- single-enabled local allowlist preflight: PASS;
- single-enabled remote allowlist preflight: PASS;
- post-submit local allowlist safety: PASS with zero enabled entries;
- post-submit remote allowlist safety: PASS with zero enabled entries.

Current next allowed action:

Monitor Slurm array job `853691`. After completion, sync artifacts and review
R4 dev diagnostic gates before any further route unlock.

Still not unlocked:

- training;
- Llama;
- same-family null;
- sanitizer;
- FAR aggregation;
- payload-diversity claims;
- paper-facing positive claims.
