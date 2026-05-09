# WP3 Micro-Slot Detector And 2-Way Bucket Policy

Status: `WP3_POLICY_DESIGN_RECORDED_NOT_IMPLEMENTED_NOT_AUDITED`

This is an artifact-only design record for the next v2 gate. It does not
implement a detector, audit tokenizer stability, score bucket mass, generate
model transcripts, train a model, run E2E, aggregate FAR, or make a positive
paper claim.

## Controlling Inputs

- `docs/natural_evidence_v2/PROTOCOL_CONTRACT.md`
- `docs/natural_evidence_v2/CLAIM_GUARDRAILS.md`
- `configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml`
- `results/natural_evidence_v2/prompts/wp2_controlled_natural_prompt_family_scaffold_20260508_2123/split_manifest.json`
- `docs/specs/stage3_method_core_smoke_spec.md`

## Detector Contract

Detector id:

```text
qwen_v2_wp3_micro_slot_detector_v0_artifact_design
```

The detector is response-local. It must run only on a fixed response text and
the associated WP2 prompt row metadata. It must not choose slots after seeing
payload, key, decoder failures, protected-vs-null outcomes, or transcript
quality.

The later implementation should emit one row per candidate micro-slot with:

```text
schema_name
protocol_id
prompt_set_id
prompt_id
split
family_id
response_id
response_sha256
slot_policy_id
slot_index
slot_type
anchor_kind
span_start
span_end
surface_text
left_context
right_context
candidate_bank_id
bucket_policy_id
bucket_id
eligibility_status
rejection_reason
```

Slot order is the stable response order `(span_start, span_end, slot_type)`.
`slot_index` is assigned after filtering rejected candidates. A future active
precommit must fix the slot policy id before transcript generation.

## Eligible Slot Types

The detector may consider only the slot types already listed in the v2 pilot
config:

| Slot type | Intended anchor |
|---|---|
| `sentence_opener` | first lexical candidate after a sentence boundary |
| `bullet_or_step_opener` | first lexical candidate after a bullet, numbered step, or checklist marker |
| `discourse_marker` | low-content marker at sentence, step, or clause start |
| `optional_hedge` | optional frequency, uncertainty, or softening word |
| `transition_word` | natural transition at the start of a sentence, step, or clause |
| `function_word_alternative` | common conjunction, preposition, determiner, or adverb alternative |

The detector must reject:

- content nouns and domain-specific nouns;
- rare or stylistically loud words;
- punctuation-only candidates;
- markdown-heavy candidates;
- invisible or control whitespace;
- multi-token candidates under the configured tokenizer;
- candidates whose detokenized text is unstable;
- any candidate containing a public forbidden surface term from the v2 config.

## Density Accounting

The later artifact-only audit should report:

```text
total_responses
responses_with_any_slot
prompt_coverage
average_micro_slots_per_response
median_micro_slots_per_response
slot_counts_by_family
slot_counts_by_type
rejected_candidate_counts_by_reason
forbidden_surface_rate
unstable_token_rate
```

The WP3 density gate is not passed by this design. A later audit must show:

```text
average_micro_slots_per_response >= 16
prompt_coverage >= 0.80
forbidden_surface_rate == 0
unstable_token_rate == 0
```

## 2-Way Bucket Policy

Bucket policy id:

```text
qwen_v2_wp3_two_way_bucket_policy_v0_artifact_design
```

Every active bucket bank must be a two-way bank with bucket ids `0` and `1`.
Four-way and eight-way banks are disallowed for the primary v2 route.

Bucket-bank entries are natural next-token measurable opportunity/catalog
entries. They are not payload recovery, not FAR evidence, and not paper-facing
positive claims.

For each candidate bank:

- both buckets must be non-empty;
- members must be disjoint;
- members must belong to the same slot type and comparable grammar role;
- casing must be fixed before transcript generation;
- tokenization must be exactly one token under the configured tokenizer;
- detokenization must round-trip exactly;
- normalized duplicate forms and token collisions are rejected;
- assignment to bucket `0` or `1` must be explicit in the artifact, not chosen
  after transcripts;
- the policy must record the candidate source, version, tokenizer id, and audit
  checksum.

The later mass audit must report:

```text
bucket_count == 2
min_bucket_mass >= 0.005
max_bucket_mass_ratio <= 5.0
unstable_token_rate == 0
```

This design does not evaluate those mass gates.

## Candidate Bank Families

The later implementation should start with small explicit banks for:

- sentence and step openers;
- discourse and transition markers;
- optional hedges;
- common function-word alternatives.

Each bank should prefer ordinary short words that can appear naturally in the
WP2 response shapes. The bank should be pruned by tokenizer audit before any
mass scoring or transcript generation. If pruning causes an empty bucket, the
bank fails closed.

## Gate Effect

This record advances WP3 from "unspecified" to "designed". It does not unlock
WP4, training, Qwen E2E, Llama, same-family nulls, sanitizer benchmarks, FAR
aggregation, or positive claims.

Next safe action:

```text
Implement an artifact-only WP3 detector and two-way bucket-bank audit scaffold
from this design. If any Chimera CPU/GPU work is needed, use Slurm. Do not
generate transcripts, train, run E2E, or make positive claims.
```

Follow-up status 2026-05-08T21:53Z: the scaffold was recorded at
`results/natural_evidence_v2/status/wp3_detector_bank_scaffold_20260508_2153/`.
The next safe action is fixed-artifact WP3 tokenizer/density/mass auditing only;
WP4, training, E2E, Llama, same-family nulls, sanitizer benchmarks, FAR
aggregation, and positive claims remain locked.

Follow-up status 2026-05-08T22:54Z: the configured-tokenizer audit from Slurm job
`850228` found five multi-token surfaces under `Qwen/Qwen2.5-7B-Instruct`:
`moreover`, `further`, `generally`, `therefore`, and `meanwhile`. The repaired
scaffold at
`results/natural_evidence_v2/status/wp3_detector_bank_scaffold_repaired_20260508_2308/`
removes or replaces those surfaces. Chimera Slurm job `850242` completed `0:0`
and the configured-tokenizer audit passed with `35/35` single-token surfaces and
`unstable_token_rate=0.0`. Density and model-mass gates remain unevaluated, so
WP4, training, E2E, Llama, same-family nulls, sanitizer benchmarks, FAR
aggregation, and positive claims remain locked.
