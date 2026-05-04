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

If 8-way remains imbalanced, 4-way becomes the primary pilot configuration and
8-way is moved to ablation.

## Phase 2: Compatibility And Reconstructability

Run counterfactual suffix compatibility and on-policy reconstructability before
protected training.

Minimum required reports:

- `delta_suffix_nll` pass rate,
- protected/raw transcript reconstructability status when transcripts exist,
- raw pre-training accidental observation/null report,
- wrong-key and wrong-payload null protocol.

## Phase 3: Qwen End-To-End Pilot

Do not wait for every bank audit to be perfect. Once the 4-way bank passes basic
quality gates, run a paper-facing Qwen pilot:

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
