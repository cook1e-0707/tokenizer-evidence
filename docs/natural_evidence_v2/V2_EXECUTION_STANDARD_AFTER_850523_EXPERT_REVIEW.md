# V2 Execution Standard After 850523 Expert Review

Status: execution standard for `natural_evidence_v2_controlled_micro_slots`.

Recorded: 2026-05-09T03:19:20Z

## Scope

This document records the expert decision after reviewing the v1 negative
diagnostic and the v2 WP3 restricted Step-label results through job `850523`.
It is a control-plane standard for future Codex/Hermes actions. It does not
start training, submit Slurm, run E2E, aggregate FAR, or make a paper-facing
positive claim.

## Current Decision

`natural_evidence_v1` remains frozen as a negative diagnostic. The v2
controlled-natural micro-slot route is the active route, but v2 is still in
WP3. Job `850523` is a close fail, not a pass.

The current bottleneck is not just one response failing to put `Step N:` labels
on separate lines. V2 has not yet shown a high-quality, trainable, decodable
natural-output evidence channel. Before WP4 or training, WP3 must show:

- a frozen detector contract;
- reliable model-output structural observability;
- high-mass 2-way bucket banks;
- manual naturalness review;
- no forbidden public surface;
- no post-hoc reclassification of failed density artifacts.

## V1 Interpretation

Job `846699` completed and produced the decisive v1 negative diagnostic:

```text
generated outputs = 18,432
bucket observations = 372,216
decode rows = 120
protected accepts = 0
null accepts = 0
decode rows with insufficient_symbols = 120 / 120
compatible variable-radix digit rows = 1,885 / 372,216
dominant erasure reason = observed_token_not_in_variable_radix_bucket_set
```

V1 failed because passive opportunity mining, global repeated frames, and
strict token-index anchors did not form a recoverable channel under
free-generation evaluation. This is not a proof that natural tokenizer-aligned
evidence is impossible. It is a protocol failure for the v1 route.

## 850523 Interpretation

Job `850523` ran base-Qwen model-output density only:

```text
total_responses = 256
complete_step_label_response_count = 255
complete_step_label_response_rate = 0.99609375
mean_detected_structural_slots_per_response = 15.94140625
median_detected_structural_slots_per_response = 16.0
detected_slot_rows = 4,081
forbidden_public_surface_rate = 0.0
raw_bank_surface_exact_hit_rate = 0.07718696397941681
wp4_allowed = false
```

The only structural failure came from `strict_compact_step_label_lines`: the
model included `Step 1:` through `Step 16:` inline in one paragraph, while the
scoring implementation accepted only line-start anchors. The artifact remains
a strict density fail.

## Detector Contract Standard

The primary v2 route uses a strict line-start `Step N:` detector unless a future
protocol revision explicitly replaces it before scoring.

Accepted slot labels:

```text
line-start Step N: labels after optional whitespace, optional markdown bullet,
and optional markdown emphasis
```

Rejected for the current primary gate:

```text
sentence-start inline Step N: labels inside one paragraph
```

The 850523 failure must not be reclassified as pass after observing the
transcript. A detector contract change is allowed only as a new protocol
revision with a new density audit.

## WP3-R1: Detector Contract Repair Standard

Preferred repair: keep the strict line-start protocol and strengthen prompt
wording so the model writes exactly 16 separate lines.

Required dev density gate:

```text
dev outputs >= 512
complete_step_label_response_rate >= 0.995
mean_detected_slots_per_response >= 15.9
forbidden_public_surface_rate = 0
```

Required eval density gate:

```text
eval outputs >= 2,048
complete_step_label_response_rate >= 0.995
oracle_prompt_local_frame_completion_rate >= 0.95
forbidden_public_surface_rate = 0
```

`850523` does not satisfy these gates. The repaired 850523 prompt plan is only a
reviewed candidate plan until a fresh Slurm density audit passes.

## WP3-R2: High-Mass 2-Way Bank Search Standard

The current strongest bank is balanced but weak:

```text
bank = step_label_recombined_create_develop_vs_choose_make_v1
bucket_0 = [Create, Develop]
bucket_1 = [Choose, Make]
min_bucket_mass = 0.0125512375
mass_ratio = 1.0047399181
```

This can remain an ablation or candidate, but it is not sufficient as the only
training bank under the new standard because its absolute bucket mass is below
the pilot threshold.

Pilot bank gate:

```text
single-token surfaces = 100%
min_bucket_mass >= 0.03
combined bank mass >= 0.10 preferred
mass_ratio <= 3
forbidden surface rate = 0
top-k rank within top 32 preferred
prompt-family coverage >= 90%
manual naturalness = pass
```

Paper-ready target:

```text
min_bucket_mass >= 0.05
```

Search classes allowed for WP3-R2:

- step-index-specific opener buckets;
- light adverb modifier buckets;
- action-verb semantic buckets.

Four-way banks are ablation only. Eight-way banks remain forbidden for the
primary route.

## WP3-R3: Manual Naturalness Review Standard

The 850523 output includes 32 manual naturalness examples. They must be reviewed
before WP4 or training.

Allowed labels:

```text
PASS
BORDERLINE
FAIL_FORMAT_OR_TEMPLATE_ARTIFACT
FAIL_FORBIDDEN_SURFACE
FAIL_SEMANTIC_COHERENCE
```

Gate:

```text
PASS + BORDERLINE >= 90%
FAIL_FORBIDDEN_SURFACE = 0
FAIL_FORMAT_OR_TEMPLATE_ARTIFACT = 0 for obvious coding artifacts
```

## WP4 Entry Standard

WP4 remains forbidden until WP3-R1, WP3-R2, and WP3-R3 pass.

When allowed, WP4 must use a prompt-local payload contract rather than the v1
global repeated-frame policy.

Initial payload contract:

```text
8-bit payload + 8-bit checksum
16 Step-label slots x 2-way buckets
```

Required oracle gates before training:

```text
decoder oracle substitution accept = 100%
wrong-payload oracle reject = 100%
wrong-key oracle reject = 100%
```

## WP5 Teacher-Forced Training Gate Standard

Training remains forbidden until WP4 oracle gates pass.

When training is allowed, the first Qwen training gate must compare:

- base Qwen score-only;
- Qwen protected;
- Qwen task-only LoRA.

Required teacher-forced gate before any E2E:

```text
protected target mass - base >= +0.15
protected target mass - task-only >= +0.10
target bucket rank-1 rate >= 70%
median target margin > 0
task-only target mass - base must not be significantly positive
```

If this gate fails, E2E remains forbidden.

## WP6 Qwen E2E Entry Standard

Qwen E2E remains forbidden until WP5 passes.

Required five arms:

```text
qwen_protected
qwen_raw
qwen_task_only_lora
wrong_key
wrong_payload
```

Required budgets:

```text
[8, 16, 32, 64]
```

Required outputs:

```text
slot_detection_rate
target_bucket_hit_rate
prompt_local_frame_completion_rate
payload_recovery_rate
raw_accept_rate
task_only_accept_rate
wrong_key_accept_rate
wrong_payload_accept_rate
naturalness_examples
decode_trace
```

Proof-of-life pass gate:

```text
protected payload recovery @64 >= 80% over payload/seed cells
raw accepts = 0
task-only accepts = 0
wrong-key accepts = 0
wrong-payload accepts = 0
slot detection rate >= 95%
target bucket hit rate >= 40% preferred, >=25% minimum
forbidden surface rate = 0
```

## Forbidden Until Further Expert Review

- no WP4 before WP3-R1/R2/R3 pass;
- no training before WP4 oracle gates pass;
- no Qwen E2E before WP5 passes;
- no Llama;
- no same-family null;
- no sanitizer benchmark;
- no full FAR aggregation;
- no manuscript positive-claim rewrite;
- no natural-output success claim;
- no payload recovery claim;
- no cross-family generality claim;
- no robustness claim;
- no superiority claim over Scalable or Perinucleus;
- no `24,000 fingerprints` wording;
- do not describe bucket-bank entries as fingerprints.

## Immediate Execution Standard

The current next action remains review-only for the repaired 850523 strict
density plan. A new Slurm density audit may be submitted only after explicit
approval and must use a fresh output directory. No other CPU/GPU work may run on
Chimera except through Slurm.
