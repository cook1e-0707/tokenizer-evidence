# R4 prefix-slot contract addendum after 853815

Date: 2026-05-13

## Scope

This is an artifact-only design addendum to
`docs/natural_evidence_v2/R4_SURFACE_BANK_PREFIX_REPAIR_DESIGN_AFTER_853815_20260513.md`.
It does not construct a new candidate bank, run generation, score models,
submit Slurm, train, run Llama, aggregate FAR, benchmark sanitizers, or make
paper-facing positive claims.

The compact state already records the higher-level repair direction. This
addendum narrows the later builder contract so the next repaired surface bank
can be statically checked before any scorer route decision.

## Slot Contract

Every repaired R4 row should record the following fields before static
validation:

- `prefix_family_id`: stable id for the lead-in family;
- `prefix_text`: exact prefix ending immediately before the measured span;
- `slot_type`: syntactic category shared by target and other alternatives;
- `bit_side`: binary side represented by the row;
- `surface_text`: measured continuation surface;
- `measured_span_start`: character offset equal to `len(prefix_text)`;
- `measured_span_text`: exact string to be teacher-forced after the prefix.

For a coordinate to be eligible for later scoring, target and other alternatives
must share the same `prefix_family_id`, `slot_type`, coordinate id, and measured
span start convention. The alternatives may differ in surface text and bit side,
but not in the local syntactic slot being scored.

## Rejection Rules

The later static validator should reject a coordinate before any compute route
if any of these artifact-only checks fail:

1. either binary side is missing;
2. target and other alternatives use different slot types;
3. measured span starts anywhere other than immediately after `prefix_text`;
4. target and other first tokens overlap under the recorded tokenizer view;
5. `surface_text` contains public technical literals or explicit evidence-block
   structure;
6. the row depends on line index, fixed step labels, or a hidden answer field.

These rules are intentionally static. They do not authorize model scoring or a
new Slurm submission.

## Next Allowed Action

Within the current supervisor constraint, the next action remains repair-design
only. Construction of the repaired candidate bank and prefix-row static
validation should wait for a tick whose controlling next action explicitly
allows artifact construction.
