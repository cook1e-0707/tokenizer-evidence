# R4 After-864117 Metric-Exact Objective Pivot

Date: 2026-05-16

## Decision

The scalar additive controller line is exhausted for the current candidate-v3 surface channel unless a new controller design is recorded first.

The next canonical route is artifact-only metric-exact objective repair planning:

```text
V2_R4_AFTER_864117_METRIC_EXACT_OBJECTIVE_PIVOT_RECORDED_ARTIFACT_ONLY
```

This does not unlock Slurm, model scoring, generation, training, Llama, null/FAR, sanitizer, payload diversity, or paper-facing claims.

## Evidence

Job `859672` showed apparent protected success, but wrong-key and wrong-payload controller arms also passed because the protected adapter still supplied committed-target pressure. That run was reviewed as a selectivity-control semantics failure.

Job `863274` removed protected adapter loading from controller arms. Wrong controls became clean, but positive controlled-base pressure was too weak:

```text
controlled basic gate pass: 0/72
overall selective gate pass: 0/72
best controlled lift vs base: +0.0154036601
best rank1: 0.498046875
```

Job `864117` increased pressure within a reviewed safety-bound controller envelope. Wrong controls stayed clean, but positive pressure still failed:

```text
controlled basic gate pass: 0/24
overall selective gate pass: 0/24
best controlled lift vs base: +0.0269583198
best rank1: 0.6015625
best median margin: +0.0033881384
```

The best 864117 grid still requires roughly `1.7191` additional target-logit odds nats to reach the `+0.15` lift target. Another scalar grid without a new design would mostly retest the same failure mode.

## Next Route

The selected route is metric-exact objective repair. The next work is code review and plan-only validation for an objective that directly targets the teacher-forced gate:

```text
target first-token probability mass
target-vs-other margin
rank1 among target/other alternatives
stratum coverage
```

Required contract:

```text
disabled by default
no behavior change when flags are disabled
protected arm only receives target pressure
task-only cannot receive target pressure
uses exact prefix-native target_first_token_ids
hard-fails target/other token overlap
uses reviewed stratum weights only
does not mine generated transcripts
```

## Future Compute Prerequisites

Any future training Slurm route must first record:

```text
objective code review
toy-logit tests
training wrapper plan-only pass
local zero-enabled allowlist pass
remote hash preflight pass
Hermes TG/email notification
exactly one enabled allowlist entry
immediate allowlist disablement after sbatch
```

If these prerequisites pass, the future allowlist entry is:

```text
v2_r4_candidate_v3_micro_overfit_h200
```

using H200/pomplun/cs_yinxin with max wall time.

## Current Stop Line

No Slurm job is submitted by this decision. No generation or paper-facing positive claim is allowed from this state.
