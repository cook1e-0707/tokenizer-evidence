# R4 After-864832 Two-Sided Controller-Only Route

Status: `ROUTE_RECORDED_NO_SUBMIT`

The two-sided surface-mass job `865252` and adapter-gain sweep `865289` both failed the teacher-forced gate. Scalar adapter amplification is therefore not the immediate repair.

This route tests a narrower question:

```text
Can a provider-side keyed soft logit controller make the two-sided cover-natural bank measurable in teacher-forced scoring, without relying on the protected adapter?
```

Scope:

- Qwen-only teacher-forced scoring.
- Controller-only conditions: `base`, `task_only`, `controlled_base`, `wrong_key_controlled_base`, `wrong_payload_controlled_base`.
- Same 8192 two-sided cover-bank rows.
- No generation, no training, no Llama, no sanitizer, no FAR, no payload diversity, no paper-facing claims.

The route is an array over the precommitted controller grid in `configs/natural_evidence_v2/r4_after_864832_two_sided_controller_only_route.yaml`. It can only be submitted after route validation, wrapper plan-only smoke, remote hash preflight, zero-enabled allowlist safety, Hermes notification, and exactly-one allowlist enablement.

