# natural_evidence_v2

`natural_evidence_v2_controlled_micro_slots` is the replacement primary route
after the v1 negative diagnostic.

The route is:

```text
controlled-natural owner probes
-> dense natural micro-slots
-> 2-way tokenizer-aligned bucket policy
-> prompt-local small payload frames
-> teacher-forced bucket-margin gate
-> Qwen proof-of-life E2E only after the gate passes
```

This namespace is planning/scaffold only until the protocol, prompt families,
slot policy, bucket policy, payload contract, and teacher-forced target-mass gate
are built and audited.

Current files:

- `PROTOCOL_CONTRACT.md`: commit-then-reveal and anti post-hoc rules.
- `CLAIM_GUARDRAILS.md`: allowed and forbidden claims.
- `configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml`: initial Qwen
  controlled micro-slot pilot config.
- `WP3_MICRO_SLOT_DETECTOR_BUCKET_POLICY.md`: artifact-only WP3 micro-slot
  detector and 2-way bucket policy design record. The design is not yet an
  implemented detector or passed gate.
- `scripts/natural_evidence_v2/build_wp2_prompt_scaffold.py`: deterministic
  WP2 prompt split builder and public prompt-text forbidden-surface audit.
- `scripts/natural_evidence_v2/build_wp3_detector_bank_scaffold.py`:
  artifact-only WP3 detector contract and two-way bucket-bank scaffold builder.
- `scripts/natural_evidence_v2/audit_wp3_fixed_artifacts.py`: artifact-only
  WP3 fixed-artifact audit entrypoint for tokenizer stability, density
  accounting, and fixed model-mass artifacts.
- `scripts/natural_evidence_v2/build_wp3_context_mass_plan.py`: artifact-only
  builder for context-specific WP3 mass scoring plans from balanced template
  detections. It records lowercase and sentence-case bucket variants separately
  and does not run model scoring.
- `results/natural_evidence_v2/prompts/wp2_controlled_natural_prompt_family_scaffold_20260508_2123/`:
  WP2 prompt family split artifacts at the configured train/dev/eval/null
  counts, with `PASS_FORBIDDEN_SURFACE_AUDIT`.
- `results/natural_evidence_v2/status/wp3_micro_slot_policy_design_20260508_2140/`:
  WP3 design summary JSON. Density and mass gates remain `NOT_EVALUATED`.
- `results/natural_evidence_v2/status/wp3_detector_bank_scaffold_20260508_2153/`:
  WP3 detector/bucket-bank scaffold. Density, tokenizer stability, and mass
  gates remain `NOT_EVALUATED`.
- `results/natural_evidence_v2/status/wp3_fixed_artifact_audit_20260508_2223/`:
  WP3 audit implementation dry-run on the recorded scaffold with a mock tokenizer.
  This is not a configured-tokenizer gate result; density and mass gates remain
  `NOT_EVALUATED`.
- `results/natural_evidence_v2/status/wp3_configured_tokenizer_audit_20260508_2238/`:
  WP3 configured-tokenizer audit attempt on the fixed bucket-bank scaffold.
  The configured Qwen tokenizer was selected, but the local Hugging Face backend
  is blocked by unavailable `transformers`; density and mass gates remain
  `NOT_EVALUATED`.
- `results/natural_evidence_v2/status/wp3_detector_bank_scaffold_repaired_20260508_2308/`:
  repaired WP3 detector/bucket-bank scaffold after removing or replacing the
  configured-tokenizer multi-token surfaces found by job `850228`.
- `results/natural_evidence_v2/status/wp3_configured_tokenizer_audit_850242/`:
  configured-tokenizer Slurm audit on the repaired scaffold. Tokenizer stability
  passed with `unstable_token_rate=0.0`; density and mass remained
  `NOT_EVALUATED`.
- `results/natural_evidence_v2/status/wp3_template_density_responses_20260508_2321/`:
  template-only fixed responses for WP3 density preflight. These are not model
  generations.
- `results/natural_evidence_v2/status/wp3_template_density_responses_balanced_20260508_2331/`:
  balanced template-only fixed responses with 64 rows per WP2 family. These are
  not model generations.
- `results/natural_evidence_v2/status/wp3_template_density_audit_850276/`:
  configured-tokenizer Slurm density preflight on the template fixed responses.
  The review records `TEMPLATE_PREFLIGHT_PASS`; this is not a model-output
  density gate. Fixed model-mass artifacts are still missing.
- `results/natural_evidence_v2/status/wp3_bucket_mass_score_850288/` and
  `results/natural_evidence_v2/status/wp3_model_mass_audit_850288/`: Slurm
  fixed-prefix model-mass scoring and audit. The mass gate failed and WP4
  remains locked.
- `results/natural_evidence_v2/status/wp3_context_mass_plan_20260508_2324/`:
  artifact-only context-specific mass scoring plan from balanced template
  detections. It is not scored and does not change any gate.
- `results/natural_evidence_v2/status/gate_status.json`: v2 machine-readable
  gate status.

No training, model transcript generation, E2E, Llama, same-family null,
sanitizer benchmark, FAR aggregation, or positive paper claim is allowed at
this stage.
