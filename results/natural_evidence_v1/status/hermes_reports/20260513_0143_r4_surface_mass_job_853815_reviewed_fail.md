# Hermes / Codex sync: R4 surface-mass job 853815 reviewed

timestamp_utc: 2026-05-13T01:43:53Z

phase:
`V2_R4_SURFACE_BANK_REPAIR_DIAGNOSIS_AFTER_853815_FAIL`

job:

- job id: `853815`
- job name: `nat-ev-v2-r4tfm`
- state: `COMPLETED`
- elapsed: `00:04:39`
- exit code: `0:0`
- node: `chimera21`

review:

- review doc:
  `results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_score_853815_review/r4_surface_mass_score_853815_review.md`
- review summary:
  `results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_score_853815_review/r4_surface_mass_score_853815_review_summary.json`

gate_result:

`FAIL`

key_numbers:

- protected target surface mass lift vs base: `-0.0000864096` (required `>= +0.15`)
- protected target surface mass lift vs task-only: `-0.0002997293` (required `>= +0.10`)
- protected target surface rank-1 rate: `0.4375` (required `>= 0.70`)
- protected median target margin: `-0.0000096318` (required `> 0`)

interpretation:

This is not a Slurm/provider failure. The binary repair candidate fixed the
formal two-sided surface-bank issue, but it did not create a trainable surface
channel under the existing protected adapter. Target phrase-surface masses are
near zero across all arms.

next_allowed_action:

Artifact-only R4 surface-bank / prefix-shape / target construction diagnosis
only. Do not submit another scoring job, do not run generation, do not train,
and do not unlock Llama/FAR/sanitizer/paper claims until a new repair plan is
reviewed.
