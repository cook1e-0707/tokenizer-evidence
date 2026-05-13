# R4 surface-bank and prefix-shape repair design after 853815

Date: 2026-05-13

## Scope

This is an artifact-only repair design following the failed R4 teacher-forced
surface-mass gate for job `853815`. It does not train, generate, submit Slurm,
score models, run Llama, aggregate FAR, benchmark sanitizers, or make
paper-facing positive claims.

Controlling evidence:

- diagnosis:
  `results/natural_evidence_v2/status/r4_surface_mass_failure_diagnosis_after_853815/surface_mass_failure_diagnosis_report.md`;
- scored rows:
  `results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_score_853815/r4_teacher_forced_surface_mass_rows.jsonl`;
- binary repair candidate:
  `results/natural_evidence_v2/status/r4_binary_surface_bank_repair_plan_20260513/candidate_binary_surface_bank.json`.

## Repair Target

The next R4 target-construction repair should stop treating target phrases as
free-floating answer content. The `853815` diagnosis shows that the existing
phrase cylinders are nearly zero-mass under the current prefixes even after the
two-sided bank repair. The repair target is therefore prefix-native surface
cylinders: each target or other phrase must be a natural, high-prior
continuation of the exact local prefix shape used in the teacher-forced probe.

## Design Rules

1. Prefixes must end immediately before the measured surface span.
2. Each coordinate side must use the same syntactic slot across target and
   other surfaces, such as verb phrase vs verb phrase or discourse transition
   vs discourse transition.
3. Target and other surfaces must have no first-token overlap within a scored
   coordinate.
4. Surfaces should be short ordinary continuations, preferably two to four
   tokens after Qwen tokenization, not rare verb-object clauses selected only
   for semantic distinctness.
5. Every coordinate must keep both binary sides populated before any scorer
   wrapper is considered.
6. The repaired bank must remain R4-cover-natural: no fixed step labels, no
   public technical literals, no explicit evidence block, and no line-index
   coordinate dependence.

## Candidate Prefix Shapes

The next artifact-only builder should construct candidate rows from a small
closed family of ordinary lead-ins whose continuation slot is narrow:

- intent lead-in: `For this update, I will `;
- recommendation lead-in: `The best next step is to `;
- clarification lead-in: `To keep this clear, we should `;
- follow-up lead-in: `Before moving on, please `;

These are design examples, not locked surfaces. The important constraint is the
shape: the measured surface begins at the next token after the lead-in, so
teacher-forced scoring measures the probability of the bank entry itself rather
than the probability of discovering the phrase later in an unconstrained answer.

## Required Artifact-Only Preflight Before Compute

Before any future Slurm scorer route decision, create a new candidate-bank
artifact and a local validation summary that checks only static properties:

- coordinate count and entries per coordinate;
- both bit sides present for every coordinate;
- target/other first-token overlap rate is `0`;
- forbidden public-surface literal rate is `0`;
- prefix family ids and exact prefix text are recorded;
- every row records the expected measured span start after the prefix.

No model scoring, generation, training, or Slurm submission is authorized by
this design.

## Gate Status

R4 remains blocked at the teacher-forced surface-mass gate. The next allowed
action is only artifact construction for this repaired target/prefix design and
static local validation. Any Qwen model scoring requires a separate reviewed
route decision after that artifact exists.
