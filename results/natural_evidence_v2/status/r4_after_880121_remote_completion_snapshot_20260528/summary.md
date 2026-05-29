# R4 after-880121 Remote Completion Snapshot

This is a lightweight, read-only aggregate snapshot for job `880121`. It records
remote completion evidence without downloading or committing raw generation
artifacts.

## Route

```text
phase: V2_R4_AFTER_880121_LLAMA_LOCKED_SCALE_COMPLETED_STRONG_READONLY_AGGREGATE_REVIEW_PENDING
job_id: 880121
job_name: nat-ev-v2-r4ll96v4
model: meta-llama/Meta-Llama-3.1-8B-Instruct
route: R4 after-879555 domain-repaired second-family Llama locked-scale generation
remote output:
  /hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_879555_domain_repaired_llama_locked_scale_policy_v4_880121/llama3_1_8b_instruct
```

## Read-Only Aggregate

```text
complete shards: 96/96
missing artifacts: 0
generated rows: 196608
unique response hashes: 196608
global duplicate extra rows: 0
max duplicate group size: 1
trace binding invalid rows: 0/196608
```

Decode summary from the remote artifact scan:

```text
protected strict accepts: 96/96
protected accepts ignoring quality: 96/96
raw accepts: 0/96
task-only accepts: 0/96
wrong-key accepts: 0/96
wrong-payload accepts: 0/96
protected duplicate response hash count: 0
protected forbidden public surface count: 0
raw duplicate response hash count: 0
raw forbidden public surface count: 0
```

The `task_only` line is a decode-control bucket for this route. `880121` did not
include a separate task-only generation arm.

## Boundary

This is a strong completed locked-scale diagnostic artifact, but it is not yet a
formal adopted review result. The next step is to sync the reviewed artifacts and
run the official locked-scale generation review/adoption script.

Still not allowed from this snapshot alone:

```text
paper-facing positive claim
Llama locked-scale transfer claim
text-only phrase-decoder success claim
full FAR claim
sanitizer robustness claim
payload diversity claim
HMAC/signed trace-provenance claim
```

Trace binding in this route is consistency/hash binding. The current route did
not use a secret HMAC signing key (`BINDING_HMAC_SECRET=false`).

One transient Hugging Face `HEAD config.json` timeout was observed in shard 26
logs. The shard still completed and all expected artifacts were present.
