# natural_evidence_v2 Protocol Contract

Status: draft contract required before any v2 training.

## Scope

`natural_evidence_v2_controlled_micro_slots` replaces the v1 passive
opportunity/global-frame/strict-token-index route. The goal is not an explicit
evidence block. The goal is controlled-natural owner probes that induce ordinary
answers with dense, naturally located micro-slots, such as sentence openers,
step openers, discourse markers, hedges, transitions, and function-word
alternatives.

## Precommitment

Before transcript generation, the owner must fix:

```text
precommit = H(protocol_id || audit_key || payload || prompt_set || slot_policy || bucket_policy || decoder || query_budget || eval_split)
```

The committed record must include:

- `protocol_id`;
- audit key id and key commitment;
- payload id and payload bytes;
- prompt set id and prompt family ids;
- train/dev/eval/organic-null split ids;
- slot policy id;
- bucket policy id;
- decoder id and checksum policy;
- query budget list;
- decode threshold;
- allowed number of keys, payloads, thresholds, and retries.

## Anti Post-Hoc Rules

The following are forbidden:

- choosing a key after seeing transcripts;
- choosing or changing payload after seeing transcripts;
- selecting slots after seeing transcripts;
- changing the bucket policy after seeing transcripts;
- changing decode threshold after seeing transcripts;
- using dev failures to pick a test key;
- trying multiple hidden keys/payloads and reporting only the best;
- increasing the query budget after inspecting transcript quality.

Any exploratory run that violates these rules must be labeled diagnostic only
and excluded from FAR or ownership claims.

## Split Discipline

The eval split is locked before transcript generation. Dev results may tune
prompt families, slot policy, bucket policy, loss settings, and thresholds only
before the final eval lock. Once final eval is locked, no test-key, test-payload,
slot, threshold, or prompt selection is allowed.

Organic-null prompts must be disjoint from owner-probe prompts and must not be
used to tune the protected decoder threshold after final lock.

## Prompt Surface Rules

v2 prompts and outputs must not contain explicit evidence surface text:

```text
FIELD=
SECTION=
TOPIC=
PAYLOAD
CERT
EVIDENCE
CARRIER
OWNER
fingerprint
watermark
bucket
```

The verifier may use the committed slot and bucket policy, but public text must
remain ordinary natural output rather than a structured evidence block.

## v2 Gate Order

No step may skip the previous gate:

1. v1 negative diagnostic is frozen.
2. protocol contract and claim guardrails exist.
3. controlled-natural prompt families are generated and audited.
4. micro-slot detector and 2-way bucket bank pass density and mass gates.
5. prompt-local small payload contract passes oracle substitution at 100%.
6. teacher-forced target-mass gate passes for protected vs base and task-only.
7. Qwen v2 proof-of-life E2E runs with protected/raw/task-only/wrong-key/wrong-payload.
8. Only after Qwen positive recovery and null rejection may Llama or same-family nulls start.
9. Only after Qwen and Llama positive recovery may sanitizer benchmarks start.

## Initial Positive-E2E Gates

Training may not proceed until:

- average micro-slots per response is at least `16`;
- prompt coverage is at least `80%`;
- 2-way bucket mass ratio is at most `5`;
- min bucket mass is at least `0.005`;
- unstable token rate is `0`;
- forbidden surface rate is `0`;
- decoder oracle substitution accept rate is `100%`.

Free-generation E2E may not proceed until teacher-forced scoring satisfies:

- protected target bucket mass minus base is at least `+0.15`;
- protected target bucket mass minus task-only is at least `+0.10`;
- target bucket rank-1 rate is at least `70%`;
- median target margin is positive;
- task-only target bucket mass minus base is not materially positive.

## Claim Boundaries

Before Qwen v2 proof-of-life passes, no positive natural-output claim is allowed.
Before Llama also passes, no cross-family generality claim is allowed. Before
sanitizer tests pass, no robustness claim is allowed. Null rejection is not full
FAR unless the positive channel itself recovers under the same locked protocol.
