# R4 Candidate v3 Transfer-Gap Repair Route

Timestamp UTC: 2026-05-14T06:58:00Z

## Decision

Record an artifact-only repair route for the pressure-relaxation-B free
generation failure from job `857795`. No compute is unlocked by this decision.

The failed dev diagnostic showed:

- protected accepts: `0/32` with `format_scrub=all`;
- protected accepts: `0/32` with no scrub;
- raw/task-only/wrong-key/wrong-payload accepts: `0/32` each;
- trained prefix-native context hit rate: `0.0` for both checked prefixes;
- max protected-vs-raw shallow structural AUC: `0.9661500453948975`.

The next route must repair the teacher-forced-to-free-generation transfer gap
before any further training, generation, Qwen E2E rerun, Llama, same-family
null, sanitizer, FAR aggregation, payload-diversity claim, or paper-facing
positive claim.

## Required Repair Surface

The next implementation plan must address all four failure modes together:

1. Prefix-context elicitation: prompts or cover-natural scaffolds must make the
   trained prefix-native contexts likely enough in free generation to evaluate
   the same context distribution that passed teacher-forced scoring.
2. Free-generation surface polarity: the protected arm must be designed to
   increase committed-codeword surface support rather than only generic action
   verb frequency.
3. Forbidden matcher semantics: `bucket` and `coordinate` must be separated
   into natural word-boundary literals versus true protocol-surface leakage
   before any blocker or pass/fail interpretation uses the count.
4. Structural length leakage: protected/raw/task-only output constraints must
   remove the current length and line-length signature before a positive
   generation diagnostic can be trusted.

## Gate

Before any compute route can be proposed, a follow-up artifact must specify:

- exact prompt or scaffold changes for prefix-context elicitation;
- the corresponding teacher-forced scoring surface to keep train/eval context
  distributions aligned;
- the forbidden matcher rule and its expected false-positive handling;
- structural controls and the shallow-feature gate to rerun;
- local plan-only validation commands that do not load models or tokenizers;
- a disabled-by-default allowlist entry name, if a later Slurm route is needed.

## Still Blocked

This route does not allow:

- Slurm submission;
- training;
- generation;
- Qwen E2E rerun;
- Llama;
- same-family null;
- sanitizer benchmark;
- FAR aggregation;
- payload-diversity claim;
- paper-facing positive claim.

## Next Allowed Action

Artifact-only implementation planning or route review for the four-part repair
above. If the repair cannot be specified unambiguously, write a blocker report
and stop.
