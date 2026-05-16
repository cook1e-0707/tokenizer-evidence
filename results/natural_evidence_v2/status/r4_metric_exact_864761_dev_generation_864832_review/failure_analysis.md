# R4 Metric-Exact 864761 Dev Generation 864832 Failure Analysis

## Observed Result

Job `864832` is an execution success and a method failure:

```text
Slurm state: COMPLETED
Exit code: 0:0
Generated outputs: 6144
Primary protected accepts under format_scrub=all: 0/32
Protected accepts under no scrub: 0/32
All null/control accepts: 0
```

The failure is not a cluster, wrapper, or null-control issue. The protected arm
does not produce recoverable cover-natural evidence.

## Evidence

The teacher-forced adapter pressure from job `864761` transferred into visible
generation behavior, but not into the precommitted R4 ECC surface channel:

```text
protected mean support, format_scrub=all: 0.75
protected median support, format_scrub=all: 0
protected max support, format_scrub=all: 8
required observed coordinates in decoder spec: 20
protected mean matched surfaces, format_scrub=all: 2.25
```

The generated protected text is dominated by repeated candidate-v3 pressure
phrases rather than the cover-natural surface bank:

```text
protected mean "Create a plan" count per output: 20.85
protected median "Create a plan" count per output: 17
protected max "Create a plan" count in one output: 114
protected mean "Prepare a" count per output: 15.04
protected duplicate response text hashes: 358
raw duplicate response text hashes: 0
task_only duplicate response text hashes: 0
```

The structural leakage audit also fails:

```text
max protected-vs-raw shallow feature AUC: 1.0
hard gate threshold: <= 0.60
```

## Cause Classification

The failure is a free-generation transfer gap:

```text
teacher-forced prefix-native target mass -> PASS in job 864761
free-generation cover-natural phrase evidence -> FAIL in job 864832
```

The adapter learned strong local logit pressure for the candidate-v3
first-token cylinders, but generation followed that pressure into repetitive
visible templates. Those templates are not the precommitted phrase-level ECC
surfaces used by the R4 decoder.

Raw and task-only outputs actually have more natural cover-bank support than
protected outputs, but they still do not accept:

```text
raw mean support, format_scrub=all: 11.75
task_only mean support, format_scrub=all: 11.25
protected mean support, format_scrub=all: 0.75
```

This indicates that the current protected adapter reduces natural surface
coverage instead of steering it into the keyed ECC channel.

## Safety / Claim Control

This artifact must remain a failed dev diagnostic:

```text
no paper-facing positive claim
no locked-scale unlock
no Llama unlock
no same-family null unlock
no sanitizer unlock
no FAR unlock
no payload-diversity claim
```

The source adapter caveat remains active: job `864761` trained by cycling a
512-row train artifact while scoring 8192 rows. It must not be described as
8192 unique train-row coverage.

## Next Allowed Action

Artifact-only repair or pivot route planning only. Any future compute route
must explicitly address the mismatch between teacher-forced prefix-native
pressure and free-generation cover-natural surface recovery, and it must pass
the usual allowlist, Hermes notification, remote hash preflight, and
single-submission controls before Slurm execution.
