# R4 controller-only job 863274 reviewed

phase:
V2_R4_CONTROLLER_ONLY_SCORE_863274_REVIEWED_FAIL_NO_GENERATION

summary:
```text
Job 863274 completed all 72/72 H200/pomplun array tasks with ExitCode 0:0.
All 72 summary artifacts were synced and reviewed.

Review outcome:
- controlled-base basic gate passes: 0/72
- overall selective gate passes: 0/72
- wrong-key basic gate passes: 0/72
- wrong-payload basic gate passes: 0/72
- best controlled lift vs base: +0.0154036601
- best controlled rank1: 0.498046875
- best controlled median margin: -0.0001098111

Interpretation: the controller-only repair fixed the wrong-control contamination from 859672, but positive controller pressure is far too weak for the R4 teacher-forced surface-mass gate. This does not unlock generation.
```

artifacts:
```text
docs/natural_evidence_v2/R4_CONTROLLER_ONLY_SCORE_863274_REVIEW_20260515.md
results/natural_evidence_v2/status/r4_controller_only_score_863274_review/
results/natural_evidence_v2/status/r4_controller_only_failure_diagnosis_863274_20260515/
```

next_allowed_action:
Artifact-only failure diagnosis and repair-route planning for a stronger and still selective pressure mechanism. No Slurm/generation/training/Llama/null/sanitizer/FAR/paper-claim action until a new reviewed route records prerequisites and control-plane checks.
