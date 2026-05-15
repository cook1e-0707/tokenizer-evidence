# Hermes Sync: R4 Pressure Controller 859672 Reviewed

Phase:

```text
V2_R4_PRESSURE_CONTROLLER_SCORE_859672_REVIEWED_WRONG_CONTROLS_FAIL_NO_GENERATION
```

Blocker:

```text
BLOCK_R4_PRESSURE_CONTROLLER_WRONG_CONTROL_MAPPING_SCORER_DIAGNOSIS_NEXT
```

Summary:

- Job `859672` completed all `72/72` H200/pomplun array tasks with exit code `0:0`.
- Protected basic teacher-forced gate passed `72/72`.
- Overall selective gate passed `0/72`.
- Wrong-key controlled basic gate passed `72/72`.
- Wrong-payload controlled basic gate passed `72/72`.
- This is a selectivity-control semantics failure, not an infrastructure failure and not a positive result.

Key diagnosis:

```text
Wrong-control arms still load the protected adapter while the scorer measures committed target ids.
Wrong-payload uses complement controller ids and wrong-key uses coordinate_hash_v1, but committed target mass remains high under wrong controls.
```

Reviewed artifacts:

```text
docs/natural_evidence_v2/R4_PRESSURE_CONTROLLER_SCORE_859672_REVIEW_20260515.md
results/natural_evidence_v2/status/r4_pressure_controller_score_859672_review/
results/natural_evidence_v2/status/r4_pressure_controller_wrong_control_diagnosis_859672_20260515/
docs/natural_evidence_v2/CURRENT_STATE.md
```

Next allowed action:

```text
Artifact-only wrong-control mapping/scorer semantics repair planning.
```

Not unlocked:

```text
new H200 scoring resubmission
generation
training
Qwen E2E
Llama
same-family null
sanitizer
FAR
payload-diversity claim
paper-facing positive claim
```

No allowlist entry should be enabled from this state.

