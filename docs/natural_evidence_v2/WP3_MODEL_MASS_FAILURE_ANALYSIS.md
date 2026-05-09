# WP3 Model-Mass Failure Analysis

Status: `WP3_MODEL_MASS_AUDIT_FAIL_NEEDS_BUCKET_CONTEXT_REPAIR`

This document records the result of the first fixed-prefix base Qwen model-mass
audit for `natural_evidence_v2_controlled_micro_slots`. It is artifact-only
analysis. It does not authorize training, generation, Qwen E2E, Llama,
same-family nulls, sanitizer benchmarks, FAR aggregation, or paper-facing
positive claims.

## Inputs

- Bucket scaffold:
  `results/natural_evidence_v2/status/wp3_detector_bank_scaffold_repaired_20260508_2308/two_way_bucket_bank_scaffold.json`
- Template density preflight:
  `results/natural_evidence_v2/status/wp3_template_density_audit_balanced_850278/`
- Model-mass Slurm job:
  `850288` (`nat-ev-v2-wp3mass`)
- Mass score artifact:
  `results/natural_evidence_v2/status/wp3_bucket_mass_score_850288/qwen_v2_wp3_bucket_mass_artifact.json`
- Mass audit:
  `results/natural_evidence_v2/status/wp3_model_mass_audit_850288/mass_audit.json`

## Result

Job `850288` completed successfully (`0:0`) and scored 21 fixed-prefix contexts
across 7 repaired two-way banks under base `Qwen/Qwen2.5-7B-Instruct`.

The configured mass gate failed:

```text
mass_gate_status=FAIL
wp4_allowed=false
```

All banks failed `min_bucket_mass >= 0.005` under full-vocabulary next-token
probability:

| Bank | Min full-vocab mass | Ratio | Candidate-normalized ratio |
|---|---:|---:|---:|
| sentence_opener_sequence_v0 | 4.052e-09 | 1.03 | 1.34 |
| step_opener_action_v0 | 7.574e-09 | 18.97 | 3.40 |
| discourse_marker_additive_v0 | 4.583e-09 | 6.59 | 3.60 |
| optional_hedge_frequency_v0 | 3.311e-07 | 4.94 | 5.75 |
| transition_word_plain_v0 | 4.893e-08 | 2.31 | 3.15 |
| function_word_conjunction_v0 | 8.530e-09 | 4.74 | 4.26 |
| function_word_preposition_v0 | 3.435e-07 | 5.11 | 2.08 |

## Interpretation

The current scaffold has two positive properties:

- configured Qwen tokenizer stability passes for `35/35` candidate surfaces;
- template fixed responses show dense detector opportunities across F1/F2/F3/F4
  with average `30.25` slots per response.

The blocker is different: the current fixed-prefix contexts and bucket surfaces
do not give the bucket words meaningful raw next-token probability mass under
base Qwen.

Candidate-normalized balance is not the configured gate. It shows that, when
the model is forced to choose only among the candidate words, several banks are
not wildly imbalanced. But the absolute full-vocabulary mass is too small,
which means these words are not naturally likely next-token events in the tested
prefix contexts.

## Likely Causes

1. **Context mismatch**: the fixed scoring contexts are generic and may not put
   the model at the actual micro-slot boundary implied by controlled answers.

2. **Casing mismatch**: the repaired bank uses lowercase surfaces, while natural
   sentence openers and transitions may often be capitalized at true sentence
   starts.

3. **Slot detector looseness**: the current detector counts any occurrence of a
   bank word, not only structurally valid sentence/step/transition anchors.
   Template density can therefore look strong even when next-token mass at true
   structural slots is weak.

4. **Surface-set narrowness**: some bucket words may be tokenizer-stable but not
   plausible in the same grammatical context.

5. **Mass-gate definition risk**: a full-vocabulary `0.005` minimum may be too
   strict for some low-content micro-slots unless the prefix is made much more
   slot-specific. This gate should not be relaxed until context-specific scoring
   is run.

## Repair Plan

### R1: Context-Specific Prefix Extraction

Build fixed scoring prefixes from the same template response rows used by the
density preflight:

```text
template response text -> detected candidate span -> prefix_before_candidate
```

Score each bank at actual detected micro-slot prefixes, not generic hand-written
prefixes. This tests whether the low full-vocab mass is a context-construction
artifact.

### R2: Casing Audit

Run tokenizer-stability and mass scoring for lowercase and sentence-case
variants separately:

```text
first vs First
also vs Also
then vs Then
however vs However
```

Do not mix casing variants in one bucket until tokenizer stability and mass are
audited.

### R3: Structural Detector Tightening

Add detector reason codes that distinguish:

- sentence/step opener;
- clause-internal conjunction;
- ordinary content occurrence;
- optional hedge at a valid modifier position.

Density should be reported both before and after structural filtering.

### R4: Bucket Surface Repair

If context-specific mass is still low, replace or remove banks whose full-vocab
mass remains below threshold. Prefer banks where both:

```text
min full-vocab mass >= 0.005
mass ratio <= 5
```

are plausible under real micro-slot prefixes.

### R5: Gate Decision

Only after R1/R2 should the team decide whether the `0.005` full-vocab minimum
is the right pre-training gate. Do not redefine it based only on the generic
context failure from job `850288`.

## Next Allowed Action

Prepare an artifact-only context-specific mass scoring plan from the balanced
template response detections. If model scoring is needed, submit it through
Chimera Slurm. Do not run scoring on a Chimera login node.
