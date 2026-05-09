# v2 WP3 density and model-mass progress report

phase:
V2_WP3_TEMPLATE_DENSITY_PREFLIGHT_PASS_MODEL_MASS_JOB_PENDING

actions:
- added template-response source handling to the WP3 audit script so template
  responses cannot be mistaken for real fixed model-output density;
- created balanced template fixed-response rows across all four v2 prompt
  families;
- ran Slurm job `850278` for balanced template density preflight;
- added fixed-prefix base Qwen bucket-mass scorer and Slurm wrapper;
- submitted Slurm job `850288` for WP3 model-mass scoring and audit.

balanced template density:
- job: `850278`
- state: `COMPLETED 0:0`
- `template_preflight_only=true`
- `density_gate_status=TEMPLATE_PREFLIGHT_PASS`
- `total_responses=256`
- family balance: F1=`64`, F2=`64`, F3=`64`, F4=`64`
- `prompt_coverage=1.0`
- `average_micro_slots_per_response=30.25`
- `median_micro_slots_per_response=31.5`
- `candidate_micro_slot_rows=7744`
- `wp4_allowed=false`

model-mass job:
- job: `850288`
- name: `nat-ev-v2-wp3mass`
- state at submission check: `PENDING(Resources)`
- scope: fixed-prefix next-token bucket-mass scoring under base Qwen plus mass
  audit;
- no generation, training, E2E, FAR, or paper claim.

known provenance note:
The remote output directory for job `850278` is named
`wp3_template_density_audit_balanced_850277` because the manual `OUTPUT_DIR`
value contained `850277`; the Slurm job id is `850278`. Local synchronized
artifacts are stored under
`results/natural_evidence_v2/status/wp3_template_density_audit_balanced_850278/`.

next_allowed_action:
Monitor job `850288`; when complete, sync and review its mass score artifact and
mass audit outputs.

forbidden_actions_confirmed:
No training, generation, Qwen E2E, Llama, same-family null, sanitizer, FAR, or
positive paper claim was started.
