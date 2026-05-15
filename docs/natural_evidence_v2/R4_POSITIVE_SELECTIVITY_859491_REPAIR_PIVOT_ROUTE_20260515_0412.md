# R4 Positive Selectivity 859491 Repair / Pivot Route

Timestamp UTC: `2026-05-15T04:12:00Z`

## Decision

Current phase:
`V2_R4_POSITIVE_SELECTIVITY_859491_REPAIR_PIVOT_ROUTE_RECORDED_NO_COMPUTE`.

Current blocker:
`BLOCK_R4_POSITIVE_SELECTIVITY_PRESSURE_PIVOT_ARTIFACT_PACKAGE_NEXT`.

Job `859491` is frozen as a failed diagnostic. It completed cleanly on
H200/pomplun, but protected recovery stayed at `0/32` under the primary
`format_scrub=all` decode and also `0/32` under no-scrub decode. Raw,
task-only, wrong-key, and wrong-payload controls were also `0/32`, so the
failure is not a null-safety failure. It is a missing protected-selective
channel.

Do not resubmit the `859491` route unchanged.

## Triggering Evidence

The reviewed `859491` run showed:

- generated outputs: `6144`;
- duplicate response hashes: `0`;
- duplicate condition/prompt rows: `0`;
- protected accepts: `0/32`;
- raw accepts: `0/32`;
- task-only accepts: `0/32`;
- wrong-key accepts: `0/32`;
- wrong-payload accepts: `0/32`;
- protected mean events/block: `9.875`;
- raw mean events/block: `9.375`;
- task-only mean events/block: `8.5625`;
- protected max keyed score: `16`;
- raw max keyed score: `23`.

The route did elicit support-window events, but those events are ordinary
task-language events. Raw and task-only outputs produce comparable support, and
raw has a higher observed max keyed score than protected. This is therefore not
a wrapper, Slurm, H200, or threshold-only issue.

## Route Interpretation

The sequence of R4 diagnostics now rules out the most direct lexical repair
line:

1. The phrase-only positive event bank failed with zero support in `859277`.
2. The support-window repair recovered support but was common across raw and
   task-only controls.
3. The selectivity prompt policy in `859491` increased support opportunities
   but still did not make support protected-selective.

The next route must not continue narrowing or prompting the same lexical
support-window contract as a standalone positive path. The project should pivot
back to a distributional pressure question:

```text
Can protected-specific pressure produce a key/payload-selective natural
continuation signal before free-generation compute is attempted again?
```

That question must be answered artifact-only first, using reviewed existing
teacher-forced and dev-diagnostic artifacts. Future compute is eligible for
review only after the new pressure/selectivity package records its exact
contract, static checks, null controls, and route gates.

## Allowed Reuse Of 859491

`859491` may be used only for:

- failure taxonomy;
- support/selectivity diagnostics;
- prompt-policy diagnosis;
- forbidden-surface matcher semantics diagnosis;
- wrapper/provenance audit;
- examples of why unchanged support-window prompt-policy reruns are not enough.

`859491` must not be used for:

- post-hoc phrase mining into a new locked bank;
- threshold tuning to relabel the run;
- key/payload remapping;
- decoder relaxation to reclassify the run as passing;
- paper-facing positive claims;
- unchanged route resubmission.

## Next Artifact-Only Work

Implement and statically validate a pressure/selectivity pivot package before
any new Slurm or generation route. The package should decide, without new
model runs, which future compute route is eligible:

1. a teacher-forced protected-pressure / controller simulation route;
2. a metric-exact protected objective route;
3. an explicit stop record if existing artifacts show no plausible selective
   pressure path.

Minimum artifact-only requirements:

- compare existing teacher-forced pressure evidence against free-generation
  failures (`857795`, `858019`, `859277`, `859491`);
- define what metric must improve before a future generation route is allowed;
- require wrong-key and wrong-payload rejection in the future route;
- require primary reporting under `format_scrub=all`;
- include a public-template leakage check so a new fixed opener does not
  replace the old `Step N` failure mode;
- keep H200/pomplun, allowlist, Hermes notification, and one-reviewed-submission
  governance for any later compute route.

## Not Unlocked

This route does not unlock:

- Slurm submission;
- free generation;
- model scoring;
- training;
- Llama;
- same-family null;
- sanitizer benchmark;
- FAR aggregation;
- payload-diversity work or claim;
- paper-facing positive claim.

Those actions remain conditionally allowed only after their recorded
prerequisite gates pass.
