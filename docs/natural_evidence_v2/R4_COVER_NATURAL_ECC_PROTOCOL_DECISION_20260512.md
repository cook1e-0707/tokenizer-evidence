# R4 cover-natural ECC protocol decision

Date: 2026-05-12

## Canonical Phase

`V2_R4_COVER_NATURAL_ECC_ARTIFACT_ONLY_PLANNING_NO_SLURM`

## Base Negative Artifact

R4 starts from the recorded R3.2 negative locked-scale artifact:

`853524`

The artifact is preserved as a failed R3.2 same-contract locked-scale result.
It may be used for failure taxonomy, oracle upper-bound audits, and
protocol-repair planning only.

## 853524 Reuse Policy

`853524` must not be reclassified as a positive result. The following actions
are explicitly disallowed:

- changing thresholds after inspecting `853524` transcripts;
- changing slots after inspecting `853524` transcripts and applying them back
  to the same run;
- changing payload, key, bucket policy, or decoder policy and treating the
  same run as positive;
- rerunning the same R3.2 contract as the next action;
- claiming payload diversity, full FAR, Llama transfer, sanitizer robustness,
  or paper-facing success from `853524`.

Allowed uses:

- failure taxonomy;
- artifact-only oracle recoverability audits;
- forbidden-surface matcher semantics audit;
- structural leakage audit;
- plan-only R4 protocol construction.

## Research Goal

R4 tests a stronger target than R3.2:

Tokenizer-aligned evidence should be embedded in natural, task-meaningful text
without depending on explicit public structure labels, fixed line or step
indices, public technical literals, or a fragile full-coordinate slot frame.

The initial R4 target is not certified paraphrase robustness. The initial target
is robustness to deterministic format scrub, public-surface scrub, and
whitespace/punctuation normalization under a precommitted keyed decoder.

## Security Assumptions

| Item | Assumption |
|---|---|
| Attacker sees output | The attacker can see the final output text. |
| Attacker can do trivial processing | The attacker can delete headings, numbering, bullets, step labels, repeated templates, and public technical terms. |
| Attacker does not know secret key | The attacker does not know the secret key, private codebook, or protected contract mapping. |
| Detector has key | The detector can use the secret key, precommitted surface bank, and precommitted decoder. |
| Initial robustness target | Deterministic format scrub, public-surface scrub, and whitespace/punctuation normalization. |
| Not yet claimed | No claim is made for full human paraphrase, semantic rewrite, or model-assisted paraphrase robustness. |
| Ethical/application assumption | The system is for authorized provenance, audit, and controlled traceability. |

## R4 Design Constraints

R4 forbids evidence dependence on:

- fixed `Step N` labels;
- fixed 16-line output;
- explicit slot labels;
- public literals such as `bucket`, `fingerprint`, `watermark`, `payload`,
  `key`, `coordinate`, `decoder`, or `hidden signal`;
- repeated labels;
- line number or step index as the primary coordinate source.

R4 uses:

- natural paragraph, sentence, or ordinary bullet text;
- phrase-level, lemma-normalized, multi-token semantic surfaces;
- erasure-aware weighted decoding;
- format-scrub reporting as a primary gate;
- raw, task-only, wrong-key, and wrong-payload null controls.

## Gate-Controlled Status

Current permissions:

```text
generation_allowed: false
slurm_allowed: false
training_allowed: false
llama_allowed: false
same_family_null_allowed: false
sanitizer_benchmark_allowed: false
far_aggregation_allowed: false
paper_claim_allowed: false
```

These actions are not permanently forbidden. They remain locked until R4
artifact-only audits, plan-only validation, route review, allowlist review, and
notification requirements pass.

## Immediate Allowed Work

Only artifact-only work is allowed in this phase:

- oracle recoverability audit;
- forbidden-surface matcher semantics audit;
- structural leakage audit;
- cover-natural prompt-bank construction;
- surface-bank and codebook construction;
- decoder implementation;
- plan-only validation.

No Slurm submission is authorized by this decision.
