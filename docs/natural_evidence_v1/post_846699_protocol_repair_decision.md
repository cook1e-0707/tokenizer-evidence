# Post-846699 Protocol Repair Decision

## Scope

This document records the protocol decision after the completed Qwen
natural-output five-arm evaluation job `846699` and the seven requested
artifact-only diagnostics. It is not a paper-facing result, not payload
recovery, and not FAR evidence.

The controlling local constraint is the Stage 4 real-integration rule: keep the
execution contract stable, avoid speculative architecture expansion, and do not
start large-scale training from an unresolved diagnostic state. For
`natural_evidence_v1`, this is stricter: no new Qwen training, no Qwen E2E
rerun, no Llama, no same-family null, no sanitizer benchmark, and no positive
paper claim until the repaired protocol passes new preflight gates.

## Inputs

| Artifact | Status | Key evidence |
|---|---|---|
| 846699 five-arm eval | COMPLETE_NEGATIVE | `protected_accept_count=0`, `null_accept_count=0`, 120/120 decode rows `insufficient_symbols` |
| Provenance normalization | PASS | `source_job_id=846699`, row counts and hashes pinned, `provenance_mismatches=[]` |
| Frame completion replay | COMPLETE | observed complete frames `0`; scheduled complete frames under no-erasure `5370` |
| Oracle schedule simulation | COMPLETE | no prompt subset completes a frame from observed survived digits; max iid complete-frame probability about `5.79e-54` |
| On-policy survival | COMPLETE | compatible hits `1885/372216` = `0.005064`; target hits `299/143160` = `0.002089` |
| Protected-vs-task-only lift | COMPLETE_MIXED | protected target-hit rate `0.002392` vs task-only `0.001851`; protected compatible-hit rate lower than task-only |
| Teacher-forced bucket-mass probe | COMPLETE_WEAK_LIFT | protected target candidate mass `0.410354`; base `0.406997`; task-only `0.405440` |
| Decoder oracle substitution | PASS_ORACLE_ONLY | protected oracle accepts `16/16`; wrong-payload accepts `0/16`; eligible-position mismatches `0` |

## Decision

The current `prompt_id + absolute token_index + repeat_payload variable-radix
frame` protocol is not acceptable as the main natural-output protocol.

The next repair target is **anchor/survival alignment plus stronger learned
target-bucket mass**, not more training under the same contract and not a decoder
rewrite.

Reason:

- The decoder and variable-radix frame schedule can recover if committed target
  digits are present.
- The completed transcript never observes a complete frame.
- The observed compatible-symbol survival rate is below 1%.
- The protected LoRA only weakly shifts teacher-forced target mass.
- Concrete examples show absolute token-index drift: the committed position
  expects a local lexical choice, while free generation at the same index emits
  a different discourse token.

Therefore the project should move from:

```text
observe token at precommitted prompt_id/token_index
```

to:

```text
reconstruct prefix-conditioned eligible events from the generated text
```

and from:

```text
complete one all-digits variable-radix frame
```

to an erasure-aware protocol that can use sparse known-coordinate observations
once event survival is measurable.

## What Not To Do

- Do not rerun Qwen E2E with the current strict token-index contract.
- Do not increase LoRA steps as the next action.
- Do not launch Llama.
- Do not launch same-family nulls or sanitizer benchmark.
- Do not push 8-way as a mainline repair.
- Do not rewrite positive manuscript claims.
- Do not report null accepts from 846699 as full FAR.
- Do not treat decoder-oracle wrong-key accepts as FAR; that oracle bypasses
  wrong-key bucketization by construction.

## Hypothesis Disposition

| Hypothesis | Disposition | Evidence |
|---|---|---|
| Provider/model failure | Unsupported | Job 846699 completed and produced all expected eval artifacts |
| Payload codec arithmetic bug | Unlikely | Synthetic and decoder-oracle target streams decode |
| Evaluator/frame impossible under no-erasure | Rejected | decoder oracle protected accepts `16/16`; no-erasure frame count `5370` |
| Schedule-only repair can recover 846699 transcript | Rejected | no prompt subset completes a frame from observed survived digits |
| More training steps are the immediate fix | Rejected for now | teacher-forced protected mass lift is only `+0.003357` over base |
| Strict token-index anchor is final-protocol viable | Rejected | observed erasures dominated by `observed_token_not_in_variable_radix_bucket_set` |
| Natural-output channel is impossible | Not shown | Current result only falsifies this strict-anchor/frame contract |

## Repair Candidate Assessment

### A. Strict Token-Index Anchor

Status: reject as final protocol.

It is useful as a negative diagnostic because it provides a sharply measurable
failure mode, but it is not robust to natural free-generation drift. The
verifier should not depend on an absolute token index selected from the
reference/train artifact.

### B. Frame-Aware Prompt Bundles

Status: keep as a diagnostic baseline, not the primary repair.

Frame-aware scheduling could make frame observability explicit, but with current
protected compatible survival around `0.00475`, requiring all digits of a
27-36-slot frame to survive is still effectively impossible. This may become
useful only after event survival improves by orders of magnitude or after a
sparse code is added.

### C. Prompt-Local Dense Frame

Status: not recommended.

Current prompt-level density is about 6-7 committed positions per response,
while a frame needs roughly 27-36 digits. Forcing one response to carry a full
frame would likely push the method back toward unnatural or structured output.

### D. Prefix-Conditioned Observed-Text Eligible Selector

Status: primary repair direction.

The verifier should scan the generated transcript left-to-right and re-identify
eligible events from the observed prefix, rather than assuming the train-time
absolute token index remains aligned. The selector must be precommitted before
generation:

```text
selector_id
bucket_policy_id
audit_key_id
payload_id
query_budget
thresholds
allowed prompt split
decode rule
```

The selector must emit known-coordinate observations, not post-hoc mined
opportunities.

### E. Branch-Aware / Regenerated-Suffix Training Data

Status: required training-data repair before any new E2E.

Suffix-preserving compatibility is a useful diagnostic, but it is too strict as
the whole naturalness proxy. If the evidence token changes, the suffix may need
to branch naturally. The repaired data path should compare:

- suffix-preserving compatibility;
- branch-aware short continuation compatibility;
- regenerated/local-suffix repaired training examples.

### F. Sparse Coordinate-Level Erasure Code

Status: preferred coding repair after event survival is measurable.

The current all-digits frame requires complete frames. Natural-output evidence
is sparse and stochastic. A better abstraction is:

```text
survived observation = known coordinate + radix digit
decoder = collect enough coordinates across responses
```

This can be implemented with an erasure-tolerant coordinate schedule or
rateless-style code. It requires new precommit and null accounting, so it should
be introduced through synthetic and artifact-only preflights before any
training.

## Repair Plan

### Phase R0: Freeze This Decision

Status: current action.

Outputs:

- this decision document;
- state update marking `qwen_post_846699_protocol_repair_decision` complete;
- next allowed action set to prefix-conditioned selector replay.

Gate:

- no training or E2E started.

### Phase R1: Prefix-Conditioned Selector Replay

Goal: test whether observed generated text contains recoverable eligible events
when the verifier is allowed to re-identify prefix-conditioned events rather
than using absolute token indices.

Inputs:

- completed 846699 generated outputs and prompt metadata;
- committed selector/bucket policy;
- actual-prefix candidate/bucket artifacts;
- payload and query-budget commitments.

Outputs:

```text
prefix_conditioned_event_replay_summary.json
prefix_conditioned_event_replay_by_condition.csv
prefix_conditioned_event_replay_examples.jsonl
prefix_conditioned_event_replay_rejections.csv
```

Required metrics:

- rediscovered eligible events per 100 generated tokens;
- compatible bucket hit rate;
- target bucket hit rate for protected arms;
- raw/task-only/wrong-payload accidental hit rates;
- drift reason table;
- examples of matched and rejected events;
- number of coordinates available under each query budget.

Decision rule:

- If rediscovered events remain below a usable density, repair candidate supply
  and branch-aware compatibility before training.
- If rediscovered compatible events exist but target hits remain near raw, fix
  training signal before training.
- If rediscovered events are dense and target-specific, proceed to sparse-code
  preflight.

### Phase R2: Branch-Aware Compatibility And Local Suffix Repair

Goal: reduce teacher-forcing conflict and expand natural compatible choices.

Run only as scored diagnostics or data-construction dry-runs. Any CPU/GPU work
on Chimera must use Slurm.

Outputs:

```text
branch_aware_compatibility_summary.json
branch_aware_compatibility_by_token_class.csv
regenerated_suffix_repair_manifest.json
regenerated_suffix_repair_examples.jsonl
```

Required metrics:

- suffix-preserving pass rate;
- branch-aware pass rate;
- disagreement table;
- token class/radix breakdown;
- local naturalness/coherence score;
- invalid suffix reason table.

Decision rule:

- If branch-aware compatibility greatly exceeds suffix-preserving compatibility,
  build the next bank from branch-aware/local-suffix repaired data.
- If both remain low, the candidate supply and bucket construction are still the
  bottleneck.

### Phase R3: Training Objective Repair Preflight

Goal: prove that the protected objective can produce a large teacher-forced
target-bucket mass lift before any free-generation E2E run.

Candidate fixes:

- stronger bucket-mass loss weight;
- evidence-token CE masking;
- weaker CE on the short post-evidence suffix;
- regenerated suffix rows;
- balanced examples by radix/token class;
- explicit task-only matched control.

Minimum pre-E2E teacher-forced gate:

| Metric | Provisional threshold |
|---|---:|
| protected minus base target candidate mass | >= `+0.05` |
| protected minus task-only target candidate mass | >= `+0.05` |
| protected target rank-1 lift over task-only | >= `+0.05` |
| slice coverage | positive in most payload/seed/radix/token-class slices |

These thresholds are provisional and should be tightened after the selector
replay gives realistic denominators.

### Phase R4: Sparse Coordinate Code Preflight

Goal: replace all-digits frame completion with an erasure-aware known-coordinate
decoder if R1 shows enough events.

Outputs:

```text
sparse_coordinate_code_preflight_summary.json
sparse_coordinate_code_decode_trace.csv
sparse_coordinate_code_null_trace.csv
```

Required tests:

- synthetic round trip;
- wrong-payload rejection;
- wrong-key rejection under real bucketization, not oracle target digits;
- incomplete-coordinate behavior;
- family-wise multiple-testing accounting.

Decision rule:

- If sparse-code preflight cannot recover from realistic event counts, do not
  train.
- If sparse-code preflight passes and R3 teacher-forced mass passes, prepare a
  new Qwen proof-of-life launch review.

### Phase R5: New Qwen Proof-Of-Life Gate

A new Qwen proof-of-life training launch can be reconsidered only if all of the
following pass:

- prefix-conditioned selector replay finds nontrivial eligible event density;
- protected target-specific hits separate from raw/task-only in replay or a
  controlled teacher-forced probe;
- branch-aware/local-suffix data path is documented and audited;
- teacher-forced target-mass lift clears the R3 gate;
- sparse-code or frame-aware decoder preflight passes;
- wrong-key and wrong-payload pre-null pass under the repaired protocol;
- protocol commitment fixes selector, key, payload, thresholds, query budget,
  and prompt split before generation;
- allowlist contains exactly the one intended job command.

Until then, Qwen E2E remains blocked.

## Immediate Next Allowed Action

Implement the Phase R1 prefix-conditioned selector replay as an artifact-only
diagnostic. If it requires Chimera access to large generated outputs, submit it
as a Slurm job. Do not run Python directly on Chimera login nodes.

The R1 implementation should not train, generate, or rerun E2E. It should only
read existing generated transcripts, precommitted bucket/candidate artifacts,
and write replay summaries.

## Forbidden Claims Remain

- natural-output success;
- payload recovery;
- full FAR;
- cross-family generality;
- robustness or sanitizer resistance;
- stealth guarantee;
- superiority over Scalable or Perinucleus.
