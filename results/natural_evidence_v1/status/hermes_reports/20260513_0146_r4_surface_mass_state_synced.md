# Hermes sync: R4 surface-mass state updated

phase:
V2_R4_SURFACE_BANK_REPAIR_DIAGNOSIS_AFTER_853815_FAIL

summary:
Job 853815 (`nat-ev-v2-r4tfm`) completed successfully at the Slurm level and was reviewed. The R4 teacher-forced surface-mass gate failed.

Key results:
- scored rows: 24,576
- probe rows: 8,192
- base mean target mass: 0.0001302390
- protected mean target mass: 0.0000438295
- task-only mean target mass: 0.0003435588
- protected-vs-base lift: -0.0000864096
- protected-vs-task-only lift: -0.0002997293
- rank1 rate remained 0.4375 for base/protected/task-only

Interpretation:
The binary phrase-surface repair fixed the formal two-sided surface-bank issue, but it did not create a trainable target-mass channel under the existing protected adapter. This is not a Slurm/provider failure.

Sync status:
- TG/email Hermes notification for the 853815 review was sent.
- Local state was synced to Chimera `~/tokenizer-evidence`.
- GitHub `main` was updated with commit `22cff2d`.

next_allowed_action:
Artifact-only R4 surface-bank/prefix-shape/target-construction diagnosis only. Do not submit generation, training, Qwen E2E, Llama, same-family null, sanitizer, FAR aggregation, or paper-claim jobs from this state.
