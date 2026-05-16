# R4 After-868016 Controller Generation 868151 Repair Pivot

Current canonical outcome:

```text
job: 868151
status: FAIL_R4_AFTER_868016_CONTROLLER_GENERATION_DIAGNOSTIC_GATE
protected accepts, format_scrub=all: 0/4
null accepts, format_scrub=all: 0/4 for raw/task_only/wrong_key/wrong_payload
matched phrase-surface events: 0
forbidden public surfaces, format_scrub=all: 26
duplicate response hashes: 4422
```

This is a clean Slurm completion and a real method diagnostic failure. It is
not a control-plane failure and it is not a positive result.

## Pivot Decision

Do not rerun or scale the current first-step controller generation route.

The teacher-forced controller pass from `868114` established that a committed
token-id set can be given enough local probability mass under teacher forcing.
Job `868151` shows that this does not automatically transfer to phrase-level
natural generation: generated outputs contain no exact committed surface
phrases and produce no decoder matched events.

## Required Next Work

The next route is artifact-only repair planning. It must produce a reviewed
route before any new Slurm submission.

The repair must explicitly handle:

```text
1. full phrase-surface realization, not only first-token mass;
2. duplicate-output collapse from deterministic row-cylinder generation;
3. forbidden public surface leakage under format_scrub=all;
4. decoder contract clarity: whether evidence is phrase-level, token-level, or
   a precommitted hybrid, with no post-hoc rescue of 868151.
```

Allowed next actions:

```text
- artifact-only failure analysis;
- controller/decoder repair design;
- local toy tests for phrase-continuation controller helpers;
- local plan-only validation;
- Hermes/Codex state synchronization;
- GitHub synchronization.
```

Not allowed until a new reviewed route exists:

```text
- another generation Slurm submission;
- larger generation route;
- training;
- Llama;
- same-family null;
- sanitizer;
- FAR aggregation;
- payload diversity claim;
- paper-facing positive claim.
```

This is a gate-controlled "not yet allowed" state, not a permanent ban. Each
action may proceed automatically after its explicit route preconditions pass
and are recorded in `CURRENT_STATE.md`.
