# R4 safety-bound controller job 864117 reviewed

phase:
V2_R4_CONTROLLER_ONLY_SAFETY_BOUND_SCORE_864117_REVIEWED_FAIL_NO_GENERATION

summary:
```text
Job 864117 completed all 24/24 H200/pomplun array tasks with ExitCode 0:0.
All 24 summary artifacts were synced and reviewed.

Review outcome:
- controlled-base basic gate passes: 0/24
- overall selective gate passes: 0/24
- wrong-key basic gate passes: 0/24
- wrong-payload basic gate passes: 0/24
- best controlled lift vs base: +0.0269583198
- best controlled rank1: 0.6015625
- best controlled median margin: +0.0033881384

Interpretation: wrong controls remain clean and positive pressure improved over 863274, but the safety-bound additive controller is still far below the teacher-forced selective gate. This does not unlock generation.
```

artifacts:
```text
docs/natural_evidence_v2/R4_CONTROLLER_ONLY_SAFETY_BOUND_SCORE_864117_REVIEW_20260516.md
results/natural_evidence_v2/status/r4_controller_only_safety_bound_score_864117_review/
results/natural_evidence_v2/status/r4_controller_only_safety_bound_failure_diagnosis_864117_20260516/
```

next_allowed_action:
Artifact-only pivot planning only. No Slurm/generation/training/Llama/null/sanitizer/FAR/paper-claim action until a new reviewed route records prerequisites and control-plane checks.
