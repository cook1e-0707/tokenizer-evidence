# WP3 Step-Local Expansion Review

## Scope

This review covers the artifact-only step-local sentence-case action-verb
expansion derived from the passing `850384` seed and scored by Slurm job
`850398`.

The run is context-mass scoring only. It is not training, generation, E2E,
payload recovery, FAR, or a paper-facing claim.

## Inputs

Expansion plan:

```text
results/natural_evidence_v2/status/wp3_step_local_expansion_plan_20260508_2038/
```

The plan contains:

```text
score_plan_rows=72
candidate_bank_count=24
prefix_families=step_label, numbered_list, dash_bullet
source_seed=step_opener_action_sentence_case_v1 from job 850384
```

Structural density feasibility:

```text
qwen_v2_wp3_step_local_density_feasibility.json
```

This records that a step-opener-only policy needs a sixteen-step/list response
or additional non-step slots to meet the `>=16` average micro-slot density gate.

## Slurm History

Job `850394` failed during tokenizer-only validation:

```text
Inspect -> [9726, 987]
```

The scorer/wrapper were repaired so tokenizer-invalid rows are written to an
invalid-tokenization artifact and skipped from mass gates.

Replacement job:

```text
job_id=850398
state=COMPLETED
exit_code=0:0
```

Synced outputs:

```text
results/natural_evidence_v2/status/wp3_step_local_expansion_mass_score_850398/
```

## Result

```text
score_plan_rows=72
context_score_rows=63
invalid_tokenization_rows=9
mass_rows=21
mass_gate_status=FAIL
wp4_allowed=false
```

Invalid tokenizer rows are all from the same bank family:

```text
step_local_*_inspect_test_adjust_update_v1
surface Inspect is not one Qwen next token
```

Those rows were skipped and did not contribute to mass gates.

## Passing Banks

Two step-label banks passed the configured base-Qwen mass gate:

| Candidate bank | Prefix family | Min mass | Ratio | Side 0 | Side 1 |
|---|---|---:|---:|---|---|
| `step_local_step_label_seed_check_review_choose_make_v1` | `Step N: ` | `0.0100467710` | `1.8203` | Check, Review | Choose, Make |
| `step_local_step_label_start_begin_create_set_v1` | `Step N: ` | `0.0071791444` | `3.8920` | Start, Begin | Create, Set |

The strongest and most stable passing seed remains:

```text
Step N: [Check/Review] vs [Choose/Make]
```

The `Start/Begin` vs `Create/Set` bank passes but is less balanced across
individual step numbers.

## Failed But Informative Banks

Several step-label banks are near or partially useful but fail the configured
gate:

- `Compare/Measure` vs `Record/Track`: ratio passes, but min mass is only
  `0.0012624`.
- `Prepare/Gather` vs `Plan/Schedule`: one side is high, but ratio is slightly
  above gate and min mass is `0.0011875`.
- `Verify/Confirm` vs `Select/Choose`: min mass `0.0020763`, ratio `7.0081`.

Numbered-list and dash-bullet prefixes consistently have much lower absolute
mass than `Step N: ` and are not good primary candidates for the next phase.

## Interpretation

The step-local direction is now supported by a stronger WP3 mass signal:

- `Step N: ` is the best current prefix family.
- Sentence-case action verbs are much better than lowercase action verbs.
- At least two 2-way banks pass the configured full-vocabulary mass gate under
  base Qwen.
- The current policy still does not pass WP3 overall because density is only
  structurally feasible, not model-output audited, and only two banks are
  currently safe.

## Decision

Do not start WP4 or training.

Promote these two banks as primary step-local candidates for the next
artifact-only policy-selection pass:

```text
step_local_step_label_seed_check_review_choose_make_v1
step_local_step_label_start_begin_create_set_v1
```

Drop `Inspect/Test` vs `Adjust/Update` unless tokenization is redesigned,
because `Inspect` is not a single Qwen next-token candidate.

## Next Allowed Action

Build a restricted step-label policy artifact using only passing banks and
evaluate structural density options:

1. `Step N: ` only;
2. 8-step plus additional non-step slots;
3. 16-step checklist/list prompt family.

Before any training or WP4 payload contract, the project still needs a real
model-output density audit for the restricted policy and a prompt-local payload
oracle substitution.

## Still Forbidden

- no training
- no generation of protected transcripts for E2E
- no Qwen proof-of-life E2E
- no Llama
- no same-family null
- no sanitizer benchmark
- no FAR aggregation
- no natural-output success claim
- no payload recovery claim
- no full FAR claim
