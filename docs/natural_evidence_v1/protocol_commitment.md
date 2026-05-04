# Protocol Commitment

`natural_evidence_v1` uses a commit-then-reveal audit protocol. This document is
part of the execution contract, not a paper result.

## Commitment

Before any audit transcript is collected, the owner must commit to:

```text
commitment =
  H(
    protocol_id ||
    protocol_version ||
    audit_key_commitment ||
    payload_commitment ||
    bucket_policy_commitment ||
    query_budget ||
    probe_selection_rule ||
    model_arm_set ||
    decoder_config
  )
```

The audit key, payload, bucket policy, query budget, probe selection rule, model
arms, and decoder settings are fixed by this commitment. They must not be chosen
after observing transcripts.

## Transcript Freeze

The auditor collects black-box transcripts under the committed probe selection
rule. Before revealing the audit key or payload, the auditor records transcript
hashes:

```text
transcript_commitment =
  H(protocol_id || model_arm_id || probe_ids || output_texts || generation_config)
```

Only after this transcript commitment is fixed may the owner reveal the audit
key, payload, and public verifier configuration.

## Verification

After reveal, the verifier checks:

1. the revealed audit key, payload, and policy match the pre-audit commitment;
2. the transcript text matches the transcript commitment;
3. eligible positions are reconstructed from observed prefixes only;
4. observed tokens are mapped to bucket ids under the committed policy;
5. payload recovery is accepted only when the decoded payload equals the
   committed payload.

Post-hoc key search is disallowed. If multiple keys, payloads, probe sets, or
query budgets are evaluated, the report must either pre-register them in the
commitment or apply family-wise false-accept accounting.

## Null Protocols

The same committed verifier must be run on:

- raw exact-model transcripts,
- task-only LoRA transcripts,
- wrong-key transcripts,
- wrong-payload transcripts,
- same-family raw near-null transcripts,
- organic and non-owner prompt transcripts.

These nulls are part of the FAR protocol. A successful protected-model recovery
without these nulls is not paper-ready evidence.
