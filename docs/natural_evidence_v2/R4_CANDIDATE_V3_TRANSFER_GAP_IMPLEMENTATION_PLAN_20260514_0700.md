# R4 Candidate v3 Transfer-Gap Implementation Plan

Timestamp UTC: 2026-05-14T07:00:00Z

## Standing Execution Rule

The user has granted standing authorization for Codex and Hermes to continue
the reviewed route automatically once its prerequisite gates pass. Gate
controlled actions are not permanently forbidden. They remain conditionally
executable after the matching route decision, preflight evidence, allowlist
safety, Hermes TG/email notification, and single-submission controls pass.

This rule does not waive safety checks:

- no unreviewed Slurm submission;
- no direct login-node tokenizer/model execution on Chimera;
- H200/pomplun/cs_yinxin.wan for GPU work unless a later route explicitly
  supersedes it;
- zero-enabled allowlist before route enablement;
- exactly one enabled allowlist entry for a reviewed submission;
- immediate allowlist disablement after `sbatch`;
- no paper-facing claim without a recorded claim gate.

## Source Failure

Job `857795` is a clean free-generation diagnostic failure, not a wrapper or
Slurm failure.

- protected accepts under `format_scrub=all`: `0/32`;
- protected accepts under no scrub: `0/32`;
- raw/task-only/wrong-key/wrong-payload accepts: `0/32` each;
- trained prefix-native context hit rate: `0.0` for both checked contexts;
- max protected-vs-raw structural AUC: `0.9661500453948975`;
- protected mean support under `format_scrub=all`: `18.0`, versus raw/task-only
  `11.75` / `11.25`.

The teacher-forced arm B from `857764` passed, but free generation did not enter
the same prefix-native context distribution. The current bottleneck is therefore
teacher-forced-to-free-generation transfer, not target-mass pressure in the
controlled prefix scoring setup.

## Repair Scope

The next implementation package must repair four surfaces together.

### 1. Prefix-Context Elicitation

Future prompts must make natural action-recommendation contexts likely without
exposing a fixed template such as `Step N:` or a single repeated global lead-in.

Allowed repair direction:

- request normal task-useful answers that include concrete next actions;
- avoid exact fixed strings as public instructions;
- use a frozen prefix-family policy for scoring and decoding;
- cap repeated lead-in frequency in generated outputs.

The scoring context must remain aligned with the generation context. A future
route may only score prefixes that are precommitted in the repair package, not
prefixes mined from locked generated outputs.

### 2. Free-Generation Surface Polarity

The repair must distinguish generic action-verb frequency from committed
codeword support. Future diagnostics must report:

- protected/raw/task-only phrase hit rates by surface family;
- target-side versus other-side phrase support;
- per-coordinate support and erasure;
- support after `format_scrub=all`.

Passing teacher-forced target mass alone is not enough to unlock a positive
generation claim.

### 3. Forbidden Matcher Semantics

The matcher must separate technical public-surface leakage from ordinary domain
language.

Hard technical literals remain forbidden:

- `fingerprint`;
- `watermark`;
- `payload`;
- `secret key`;
- technical `bucket`;
- technical `coordinate`;
- `decoder`;
- `hidden signal`.

Ordinary words such as physical `bucket` or the verb `coordinate` must be
reported separately as contextual matches. They cannot rescue a failed positive
gate, but they also should not be conflated with public protocol leakage.

### 4. Structural Length Leakage

Future generation routes must control structural observables before positive
claims:

- no `Step N` labels;
- no fixed slot count;
- no visible coordinate labels;
- no repeated global lead-in template;
- matched output constraints across protected/raw/task-only arms;
- primary decoding under `format_scrub=all`;
- protected-vs-raw shallow structural AUC gate `<= 0.60`.

## Required Artifact-Only Work Before Compute

The next Codex/Hermes tick should proceed automatically with artifact-only
implementation planning and validation. It should not wait for another manual
approval on this same route.

Required artifacts:

1. A repaired prompt/scaffold config that elicits natural next-action contexts
   without fixed public templates.
2. A frozen prefix-family scoring policy aligned to that scaffold.
3. A forbidden matcher policy that emits technical and contextual counts
   separately.
4. A structural leakage validation plan with the same shallow features used in
   the `857795` review.
5. Local plan-only validation that does not load tokenizers, models, CUDA, or
   adapters.
6. A disabled-by-default future allowlist entry name if a later H200 route is
   needed.

## Future Conditional Compute Route

After the artifact-only repair package passes review, Codex/Hermes may prepare
and submit the next H200 route without asking for another user approval, subject
to the standard gates.

The first future compute route should be small and diagnostic:

- Qwen only;
- dev prompts only;
- same-contract `a55e`;
- H200/pomplun/cs_yinxin.wan;
- format-scrub decoding is primary;
- raw/task-only/wrong-key/wrong-payload controls preserved;
- no Llama, same-family null, sanitizer, FAR, payload diversity, or paper
  claim.

If that future route fails, record failure attribution and continue the route
tree automatically from the recorded blocker. Do not stop merely to ask whether
to proceed if the next allowed action is already clear.
