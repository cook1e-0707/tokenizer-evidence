# R3.2 Repaired Prompt Split Allowlist Recheck and Route: 2026-05-11 20:32Z

## Scope

This is the recovery record for the 2026-05-11 20:32Z Hermes tick. The tick's
Codex worker failed before taking a project action because the `codex` CLI was
not found on the non-interactive PATH.

No Slurm job was submitted. No allowlist entry was enabled. No generation,
Qwen E2E rerun, training, Llama, same-family null, sanitizer benchmark, FAR
aggregation, or paper-facing positive claim was started.

Machine-readable records:

```text
results/natural_evidence_v2/status/r3_2_allowlist_recheck_repaired_prompt_split_20260511_2032.json
results/natural_evidence_v2/status/r3_2_repaired_prompt_split_single_job_route_20260511_2032.json
```

## Hermes Codex Worker Fix

The Hermes Codex worker failed because `find_codex_binary()` only knew older
hard-coded VS Code extension paths. The currently installed Codex binary is:

```text
/Users/guanjie/.vscode/extensions/openai.chatgpt-26.506.31421-darwin-arm64/bin/macos-aarch64/codex
```

`scripts/natural_evidence_v1/hermes_supervision_tick.py` now resolves Codex in
this order:

1. `HERMES_CODEX_BIN` or `CODEX_BIN` if explicitly configured;
2. `shutil.which("codex")`;
3. common local install paths;
4. the newest executable matching
   `~/.vscode/extensions/openai.chatgpt-*/bin/macos-aarch64/codex`.

Validation:

```text
uv run python -m py_compile scripts/natural_evidence_v1/hermes_supervision_tick.py
uv run python - <<'PY'
from scripts.natural_evidence_v1.hermes_supervision_tick import find_codex_binary
print(find_codex_binary())
PY
```

The resolver returned the current VS Code Codex binary above.

## Allowlist Recheck Result

`scripts/natural_evidence_v2/check_allowlist_safety.py --require-zero-enabled`
passed under the repaired R3.2 prompt split contract.

Observed safety state:

```text
enabled_entry_count = 0
enabled_entries = []
forbidden_enabled_entries = []
unknown_enabled_entries = []
r3_2_entry_disabled = true
training_allowed = false
llama_allowed = false
same_family_null_allowed = false
sanitizer_allowed = false
far_aggregation_allowed = false
paper_claim_allowed = false
```

The checked allowlist hash was:

```text
20b589d09a78a6df234d90b67790af80bde8c463fb7fc84423800e3ed403208f
```

The reviewed wrapper hash was:

```text
637579367eb045d02c5518b130be39a4231797a84b19b2db7385ec2bec9d9737
```

## Recorded Single-Job Route

Route id:

```text
V2_R3_2D_REPAIRED_PROMPT_SPLIT_SINGLE_JOB_ROUTE_RECORDED_NO_SUBMIT_THIS_TICK
```

For a later notified submission tick only, the permitted route is:

1. Confirm this route record and the repaired prompt split implementation.
2. Confirm the R3.2 allowlist safety recheck remains current or rerun it if
   any controlling artifact changed.
3. Send the required Telegram and email pre-notice.
4. Enable exactly one allowlist entry:
   `v2_r3_2_qwen_locked_scale_eval`.
5. Submit exactly one Slurm job:
   `sbatch scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch`.
6. Disable the allowlist entry immediately after `sbatch` returns.
7. Record the submission and stop.

This route does not authorize any other Slurm job, local Chimera CPU/GPU work,
training, Llama work, same-family nulls, sanitizer benchmark, FAR aggregation,
or paper-facing positive claim.

## Current Tick Stop

This tick stops after fixing the Hermes Codex CLI resolver and recording the
allowlist recheck plus route. Do not submit another R3.2 Slurm job in this tick.

## Status

```text
PASS_R3_2_ALLOWLIST_RECHECK_REPAIRED_PROMPT_SPLIT_AND_ROUTE_RECORDED_NO_SLURM
```
