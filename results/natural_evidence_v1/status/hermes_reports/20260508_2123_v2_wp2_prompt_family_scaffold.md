# v2 WP2 prompt family scaffold

Hermes notification succeeded before this Codex worker started:

```text
results/natural_evidence_v1/status/hermes_reports/20260508_2123_scheduled_tick_notification.json
status=SENT_ALL_REQUIRED_CHANNELS
```

Action taken:

- added `scripts/natural_evidence_v2/build_wp2_prompt_scaffold.py`;
- wrote deterministic v2 prompt split artifacts at configured split counts;
- audited public prompt text for the v2 forbidden-surface terms;
- updated v2 gate status and the Hermes-readable state notes.

Output directory:

```text
results/natural_evidence_v2/prompts/wp2_controlled_natural_prompt_family_scaffold_20260508_2123/
```

Split counts:

```text
train=4096
dev=1024
eval=2048
organic_null=2048
total=9216
```

Audit:

```text
status=PASS_FORBIDDEN_SURFACE_AUDIT
forbidden_surface_rate=0.0
violation_count=0
duplicate_prompt_text_count=0
empty_prompt_count=0
```

Validation:

```text
python3 -m py_compile scripts/natural_evidence_v2/build_wp2_prompt_scaffold.py
python3 scripts/natural_evidence_v2/build_wp2_prompt_scaffold.py --output-dir results/natural_evidence_v2/prompts/wp2_controlled_natural_prompt_family_scaffold_20260508_2123 --use-config-split-sizes
wc -l results/natural_evidence_v2/prompts/wp2_controlled_natural_prompt_family_scaffold_20260508_2123/qwen_v2_*_prompts.jsonl
```

Claim control:

- no training;
- no model calls;
- no model transcript generation;
- no Qwen E2E rerun;
- no Llama;
- no same-family null;
- no sanitizer benchmark;
- no FAR aggregation;
- no paper-facing positive claim.

Next allowed action:

WP3 artifact-only micro-slot detector and 2-way bucket policy design. Training,
model transcript generation, E2E, FAR aggregation, and positive paper claims
remain forbidden.
