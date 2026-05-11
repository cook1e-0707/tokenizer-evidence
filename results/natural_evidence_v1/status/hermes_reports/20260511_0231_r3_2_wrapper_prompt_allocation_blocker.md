# R3.2 Wrapper Prompt Allocation Blocker

Route phase:

```text
V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED
```

Controlling action reviewed:

```text
Prepare Route R3.2 Qwen locked-scale package/wrapper review only: payloads
P00/P01/P02/P03, seeds 17/23/29, 8 blocks per cell, arms
protected/raw/task_only/wrong_key/wrong_payload, primary budget 64 with 16/32
diagnostics.
```

Blocker:

The R3.2 wrapper cannot be safely implemented or precommitted from the current
package record because the prompt allocation policy is not fixed.

R3.2 requires:

```text
4 payloads * 3 seeds * 8 blocks * 64 prompts = 6144 prompt responses per arm
```

The current package review fixes payloads, seeds, block count, arms, budgets,
and gates, but does not specify whether the 12 payload/seed cells must use
disjoint prompt windows, reusable prompt windows with seed-separated decoding,
or another locked prompt allocation. The apparent reviewed prompt source has
`2560` rows, which is insufficient for a fully disjoint 6144-row allocation
without another source or reuse rule.

The disabled allowlist placeholder references:

```text
scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
```

That wrapper path is still a reserved future path and does not exist. Creating
it now would require choosing an unrecorded prompt allocation policy, which
would violate the v2 precommitment rules.

Decision:

```text
BLOCK_R3_2_WRAPPER_UNTIL_PROMPT_ALLOCATION_POLICY_RECORDED
```

Required next safe action:

Record an R3.2 prompt allocation decision before wrapper implementation:

```text
payload/seed cell prompt-window policy
prompt source path(s)
row ranges or deterministic reuse rule
per-cell block mapping
selected prompt manifest hash policy
overwrite refusal surfaces
```

No Slurm job was submitted. No training, generation, Qwen E2E rerun, Llama,
same-family null, sanitizer benchmark, FAR aggregation, or paper-facing
positive claim was started.
