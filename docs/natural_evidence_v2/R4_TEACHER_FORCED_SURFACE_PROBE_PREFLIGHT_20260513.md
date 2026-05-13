# R4 teacher-forced surface probe preflight

Date: 2026-05-13

## Decision

Job `853691` completed cleanly, but the R4 dev diagnostic failed as a positive
channel: protected accepts were `0/32` under both no-scrub and
`format_scrub=all`, while null arms remained clean.

The next step is not another generation run. The next step is an artifact-only
teacher-forced surface target-mass preflight for the R4 cover-natural phrase
bank.

## Purpose

The probe asks whether the existing Qwen protected adapter assigns higher
next-token mass to the R4 target phrase surfaces than base Qwen and task-only
LoRA at the same committed cover-natural prefixes.

This is a precondition for any future R4 generation or locked-scale test. If the
protected adapter does not show target surface mass lift under teacher forcing,
free generation should not be rerun.

## Scope

- Contract: same-contract `a55e`.
- Prompt split: R4 dev only.
- Score rows: `8192` rows from `256` prompts x `32` coordinates.
- Conditions to score later: base, protected, task-only.
- No generation.
- No training.
- No Llama.
- No same-family null.
- No sanitizer.
- No FAR aggregation.
- No paper-facing claim.

## Gate Before Any Further Generation

Future R4 generation remains blocked unless a Slurm-scored teacher-forced probe
meets:

| Gate | Required |
|---|---:|
| protected target surface mass lift vs base | `>= +0.15` |
| protected target surface mass lift vs task-only | `>= +0.10` |
| protected target surface rank-1 rate | `>= 0.70` |
| protected median target margin | `> 0` |
| task-only target surface lift vs base | `< +0.05` |

## Artifacts

The preflight builder writes:

- `r4_surface_teacher_forced_probe_rows.jsonl`
- `r4_surface_teacher_forced_probe_coordinate_coverage.csv`
- `r4_surface_teacher_forced_probe_plan_summary.json`
- `r4_surface_teacher_forced_probe_plan.md`

The scorer script supports dry-run locally and future Slurm scoring only:

- `scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py`

This document does not authorize a new Slurm submission by itself.
