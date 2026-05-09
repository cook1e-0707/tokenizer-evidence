# End-To-End Audit Plan

Opportunity-bank audit is a pre-training gate. It is not the core paper result.
The core result must be natural-output payload recovery under a committed audit
protocol.

## Phase 0: Protocol Freeze

Before training:

- freeze `protocol_commitment.md`,
- freeze the bucket policy and reference-model versions,
- freeze payloads, query budgets, probe-selection rules, and model arms,
- define multiple-testing accounting for any repeated keys, payloads, budgets,
  or prompt families.

## Phase 1: Clean Opportunity Banks

After clean Phase A completes, rebuild banks with the latest policy:

- Qwen 4-way and 8-way,
- Llama 4-way and 8-way,
- held-out owner probes,
- organic null prompts.

Report:

- accepted entries,
- prompt coverage,
- eligible positions per 100 tokens,
- min bucket mass,
- bucket mass ratio,
- bucket entropy fraction,
- effective bits per response,
- reference-output contamination checks.

The 24,000 raw-entry value is only a static opportunity scaling placeholder. It
is not a training-start gate and must not be treated as parity with Scalable
Fingerprinting's implanted fingerprint identities. A raw static bank can be
large while the compatibility-aware usable bank is small; that gap is a method
diagnostic and a possible paper contribution, not a reason to label raw entries
as evidence.

If 8-way remains imbalanced, 4-way becomes the primary pilot configuration and
8-way is moved to ablation.

## Phase 2: Compatibility And Reconstructability

Run counterfactual suffix compatibility and on-policy reconstructability before
protected training.

Minimum required reports:

- `delta_suffix_nll_raw`, `delta_suffix_nll_per_token`, and suffix-window length,
- compatibility-aware min1, min2/multi-member, and probability-gated counts,
- effective compatible bits per response,
- protected/raw transcript reconstructability status when transcripts exist,
- raw pre-training accidental observation/null report,
- wrong-key and wrong-payload null protocol.

Do not build static banks and then use raw count as the success criterion.
Preferred construction order is compatibility-aware: score candidate
compatibility, filter by function/semantic class where available, partition
compatible candidates into buckets, then audit bucket mass and entropy after
compatibility filtering.

Run a sweep before selecting the final capacity point:

- `delta_nll_threshold` in 0.5, 1.0, 1.5, 2.0,
- `bucket_count` in 2, 4, 8,
- `candidate_top_k` in 64, 128, 256.

The downstream choice must be made from payload recovery, utility/naturalness,
symbol errors/erasures, and false-accept behavior, not from raw bank count.

## Phase 3: Qwen End-To-End Pilot

Do not wait for 24,000 compatibility-aware entries. Once the 4-way bank-side
viability gate passes, and held-out density plus raw/wrong-key pre-null checks
are not high risk, run a paper-facing Qwen viability pilot:

- Qwen protected,
- Qwen raw,
- Qwen task-only LoRA,
- wrong key,
- wrong payload,
- payloads = 2,
- seeds = 2,
- train prompts >= 8192,
- eval owner probes >= 2048,
- organic null prompts >= 2048,
- query budgets = 8, 16, 32, 64, 128.

Bank-side Qwen viability gate:

- 4-way min1-compatible entries >= 1,500,
- fully compatible min2 entries >= 200,
- held-out eligible density >= 0.5 positions per 100 generated tokens,
- effective compatible bits per response >= 1.0,
- raw and wrong-key pre-null behavior not high risk.

This pilot is allowed to use min1-compatible entries for payload recovery.
Multi-member compatible entries are still required for the bucket-mass versus
fixed-representative ablation, but they are not a reason to block the first
end-to-end recovery test when min1 capacity is sufficient.

After expert review on 2026-05-05, the pilot gate is split into two explicit
levels:

- Paper-ready Qwen gate: still requires effective compatible bits per response
  >= 1.0, held-out density >= 0.5 per 100 generated tokens, and raw/wrong-key
  pre-null behavior that is not high risk.
- Diagnostic high-risk Qwen gate: may run a proof-of-life pilot when min1 >=
  1,500, min2 >= 200, held-out density >= 0.3, effective compatible bits per
  response >= 0.3, raw/wrong-key pre-null is not obviously high risk, and
  invalid suffix records are explained or excluded.

The diagnostic high-risk pilot is not paper-facing and cannot support claims of
natural-output success. It is Qwen-only, 4-way min1-compatible primary with a
2-way fallback if available, and uses query budgets 64, 128, 256, and 512. It
must include Qwen protected, raw Qwen, Qwen task-only LoRA, wrong-key, and
wrong-payload arms, and must emit full transcript, eligible positions, bucket
observations, erasures, symbol errors, and decode trace. It must not start Llama,
8-way main, or the full matrix.

Training objective:

- non-eligible positions: task CE,
- eligible positions: bucket-mass loss,
- exact-token CE masked at eligible positions unless the target bucket contains
  the observed token.

## Phase 4: Llama Replication

If Qwen recovers payloads under null controls, run the same pilot for Llama:

- Llama protected,
- Llama raw,
- Llama task-only LoRA,
- wrong key,
- wrong payload.

Failure is informative and should be reported as tokenizer/model-family
analysis, not hidden.

## Phase 5: Near-Null Evaluation

Evaluate untrained same-family and cross-family models without protected
training:

- raw Qwen 3B,
- raw Qwen 7B,
- raw Qwen 14B or closest available Qwen size,
- raw Llama variants if available,
- raw unrelated 7B-family model if available.

Purpose: show the committed verifier does not accept untrained same-family
models.

## Phase 6: Sanitizer Cost

Compare natural evidence against the old structured carrier-slot baseline under:

- no attack,
- generic paraphrase,
- style normalization,
- compression/summarization,
- low-temperature regeneration or deterministic rewrite,
- suffix truncation,
- public surface scrub,
- oracle keyed sanitizer after key reveal.

The claim is not unconditional robustness. The target claim is that natural
evidence is not removed by simple visible-structure deletion and requires broader
rewriting or key-aware sanitization.

## Phase 7: Paper-Ready Criteria

Paper-ready evidence requires:

- protected Qwen recovery,
- protected Llama recovery,
- raw exact-model rejection,
- task-only LoRA rejection,
- wrong-key and wrong-payload rejection,
- same-family raw rejection,
- utility and naturalness within reported bounds,
- sanitizer benchmark,
- precommitted key protocol,
- no claim that opportunity entries are fingerprints.
