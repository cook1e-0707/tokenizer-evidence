# R4 After-870987 Pre-FAR Null Expansion Route Decision - 2026-05-19

## Decision

The next canonical route after the `870210 + 870987` locked-scale pass is a
Qwen-only, same-contract, provider-side first-token event pre-FAR null expansion
package.

This route is not a full FAR claim. It is a preliminary null package for the
already-passed first-token event trace channel.

## Starting Evidence

The locked-scale aggregate passed:

```text
protected strict accepts: 94/96
raw accepts: 0/96
task-only accepts: 0/96
wrong-key accepts: 0/96
wrong-payload accepts: 0/96
global duplicate response hash extra rows: 0
trace binding invalid rows: 0
technical forbidden public surface count: 0
```

The full phrase decoder remains report-only and is not a success claim:

```text
full phrase protected accepts, format_scrub=all: 0/96
full phrase protected accepts, no scrub: 0/96
```

## Null Target

The pre-FAR package should expand the standard controls from 96 to 256
block-equivalent rows per arm:

```text
standard controls: raw, task_only, wrong_key, wrong_payload
existing block-equivalent per arm: 96
target block-equivalent per arm: 256
additional required per arm: 160
```

It must also prepare an organic-null prompt package:

```text
organic null target: 256 block-equivalent
organic null existing evidence: 0
```

## Scope

Allowed now:

```text
artifact-only route validation
artifact-only standard-control null row-bank planning
artifact-only organic-null prompt-bank planning
Hermes/Codex state synchronization
```

Not allowed by this decision:

```text
Slurm submission
generation
training
Llama
same-family null
sanitizer
FAR aggregation
payload diversity
text-only phrase decoder success claim
paper-facing positive claim
```

Route-controlled actions may proceed automatically after their recorded
preconditions pass; this decision does not require repeated user approval for
the same pre-FAR null expansion route.
