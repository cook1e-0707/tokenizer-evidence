# WP3 NVIDIA-Assisted Repair Context-Mass Review

## Scope

This review covers Slurm job `850384`, which scored the artifact-only
NVIDIA-assisted WP3 repair context-mass plan with base
`Qwen/Qwen2.5-7B-Instruct`.

The NVIDIA models were used only as design assistants:

```text
qwen/qwen3.5-397b-a17b
z-ai/glm-5.1
```

They are not gates, not ownership verifiers, and not evidence of recovery.
Final validation remains base-Qwen tokenizer/logit scoring through Chimera
Slurm.

## Inputs

Design proposals:

```text
results/natural_evidence_v2/status/nvidia_assisted_context_repair_20260508_2021/
```

Proposal-derived score plan:

```text
results/natural_evidence_v2/status/wp3_nvidia_repair_context_mass_plan_20260508_2028/
```

The plan contained `8` rows over four repaired candidate banks. Two GLM
suggestions were dropped before scoring because their prefix/surface joins were
unsafe or ill-formed.

## Slurm Result

```text
job_id=850384
job_name=nat-ev-v2-wp3ctxm
partition=DGXA100
state=COMPLETED
exit_code=0:0
context_score_rows=8
mass_rows=4
mass_gate_status=FAIL
```

Unlike job `850372`, this run did not fail on tokenizer prefix-boundary
retokenization. The repaired plan produced context-score, mass, audit, and
Slurm log artifacts:

```text
results/natural_evidence_v2/status/wp3_nvidia_repair_context_mass_score_850384/
```

## Mass Results

Configured gate:

```text
min_bucket_mass >= 0.005
max_bucket_mass_ratio <= 5.0
```

| Candidate bank | Variant | Min mass | Ratio | Status |
|---|---:|---:|---:|---|
| `step_opener_action_sentence_case_v1` | sentence_case | `0.0057856489` | `2.5349` | PASS |
| `transition_word_plain_sentence_case_v1` | sentence_case | `0.0006255802` | `1.6183` | FAIL |
| `discourse_marker_additive_internal_v1` | lowercase | `0.0000854874` | `1.8443` | FAIL |
| `step_opener_action_v1` | lowercase | `0.0000434897` | `2.2325` | FAIL |

The only passing repaired bank is:

```text
step_opener_action_sentence_case_v1
side0 = [Check, Review]
side1 = [Choose, Make]
prefixes = [Step 1: , - ]
```

The strongest individual context was:

```text
prefix = "Step 1: "
side0 mass = 0.0112965
side1 mass = 0.0285816
candidate-normalized ratio = 0.283 / 0.717
```

## Interpretation

This is a useful positive signal for the v2 direction, but it is not enough to
advance to WP4 or training.

What improved:

- The repaired plan avoided the tokenizer-boundary crash seen in `850372`.
- Strong-model design assistance found one context/surface family that passes
  the configured base-Qwen mass gate.
- The passing case is natural and easy to locate: sentence-case action verbs
  after step/list boundaries.

What remains blocked:

- Only one candidate bank passed; WP3 still lacks a complete high-density
  micro-slot policy.
- Lowercase action verbs after the same prefix fail absolute full-vocabulary
  mass.
- Transition and additive markers are balanced but too low in absolute mass.
- No detector-density audit has been rerun for the restricted passing bank.
- No payload contract, oracle substitution, teacher-forced gate, training, or
  E2E recovery is allowed from this result alone.

## Decision

Do not proceed to WP4 or training.

Promote `step_opener_action_sentence_case_v1` as the current primary repair
seed and use it to design a narrow step-local candidate policy. The next safe
action is artifact-only expansion around sentence-case action verbs after
step/list prefixes, followed by tokenizer stability, density, and base-Qwen
mass scoring through Chimera Slurm.

## Next Repair Direction

Focus on contexts like:

```text
Step 1:
Step 2:
-
1.
```

Prioritize sentence-case action-verb surfaces with natural checklist semantics.
Candidate expansion should search for pairs whose full-vocabulary mass is high
and whose ratio remains at most `5`.

Avoid treating this as evidence recovery. It is only a repaired WP3 mass signal.

## Still Forbidden

- no training
- no generation of protected transcripts
- no Qwen E2E
- no Llama
- no same-family null
- no sanitizer benchmark
- no FAR aggregation
- no natural-output success claim
- no payload recovery claim
- no full FAR claim
