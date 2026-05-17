# Hermes Sync: R4 After-868260 Full-Mode Wrapper Review

phase:
`V2_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_FULL_MODE_WRAPPER_VALIDATED_NO_SUBMIT`

summary:
```text
Codex implemented and locally reviewed the full-mode wrapper path for the
R4 after-868260 quality-repair confirmation route.

No Slurm job was submitted. No generation/model-forward/training/Llama/FAR/
sanitizer/paper-claim action was started.

Validated:
- py_compile for generator, decoders, trace-binding verifier, and route validator.
- pytest route/helper set: 16 passed.
- PLAN_ONLY=1 confirmation wrapper smoke: PASS.
- PLAN_ONLY=0 VALIDATE_PLAN_ONLY=1 delegation smoke: PASS.

New full-mode behavior:
- confirmation wrapper delegates to r4_after_868016_controller_generation_h200.sbatch.
- duplicate-safe generation policy v2 is wired into generator args.
- contextual forbidden policy v2 is wired into first-token decode.
- first-token event trace binding is verified after generation and before decode.
```

recorded:
```text
results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_full_mode_wrapper_review_20260517/
docs/natural_evidence_v2/CURRENT_STATE.md
results/natural_evidence_v1/status/gate_status.json
results/natural_evidence_v2/status/gate_status.json
```

next_allowed_action:
```text
Run fresh local/remote hash preflight for the updated full-mode wrapper package.
After that, a single-submission preflight may enable exactly
v2_r4_after_868260_quality_repair_confirmation_h200 and must disable it
immediately after sbatch returns.
```

not_started:
```text
Slurm submission
generation
model forward
training
Llama
same-family null
sanitizer
FAR aggregation
paper-facing positive claim
```
