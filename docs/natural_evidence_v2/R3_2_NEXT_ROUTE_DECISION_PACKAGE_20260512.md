# R3.2 Next Route Decision Package

Date: 2026-05-12

## Current State

The canonical Qwen v2 R3.2 H200 job `853430` completed, but the locked-scale
gate failed. The follow-up artifact-only attribution found that the nominal
`39/96` protected accepts collapse to `13/32` effective unique blocks because
12 shards reused only 4 deterministic prompt windows.

The repaired prompt-allocation preflight then showed that the current prompt
artifact cannot support a 96-block independent locked-scale package:

- current eval rows: `2,048`;
- required rows for 12 shards x 8 blocks x 64 prompts: `6,144`;
- current maximum unique package: `4` shards / `32` blocks;
- additional eval rows needed for 96 independent blocks: `4,096`.

## Route Options

### Option A: R3.2 Unique-32 Diagnostic

Use the 4 existing non-overlapping eval windows:

- rows `512-1023`;
- rows `1024-1535`;
- rows `1536-2047`;
- rows `2048-2559`.

This is feasible now as an artifact-defined route, but it is not a 96-block
locked-scale result. It can only answer whether the repaired aggregate and
window uniqueness controls behave correctly on the current prompt bank.

Allowed claim if completed later:

```text
Qwen v2 same-contract unique 32-block diagnostic result.
```

Forbidden claim:

```text
Qwen v2 96-block locked-scale stability.
```

### Option B: R3.2 Expanded-Prompt 96-Block Locked Scale

Expand the controlled-natural strict Step-label prompt bank before another
R3.2 submission:

- dev rows: at least `512`;
- eval rows: at least `6,144`;
- selected R3.2 windows: `12` distinct 512-row windows;
- no repeated prompt-window hashes;
- no repeated generated-output hashes after generation;
- no repeated decode-row hashes before aggregation.

This preserves the original R3.2 paper-readiness direction, but it requires a
new artifact-only prompt-bank expansion and density/preflight review before
Slurm submission.

Allowed claim if completed later and gate passes:

```text
Qwen v2 same-contract 96-block locked-scale stability under a precommitted
distinct-window prompt allocation.
```

Still forbidden:

```text
payload diversity, full FAR, Llama, same-family rejection, sanitizer
robustness, or paper-facing ownership success.
```

## Decision

The canonical R3 route should not submit another R3.2 job from the current
prompt artifact as a 96-block locked-scale run. The next artifact-only action
should prepare Option B: an expanded-prompt prompt-allocation plan with at least
6,144 eval rows and 12 distinct shard windows.

Option A remains available only as a short diagnostic route if the team decides
to debug the repaired aggregate/wrapper path before paying for another full
scale run. It must be labeled as 32-block diagnostic.

## Required Next Artifacts Before Any Slurm Submission

1. Expanded strict Step-label prompt plan with at least `6,144` eval rows.
2. Prompt audit with:
   - no duplicate prompt IDs;
   - no forbidden public surfaces in prompts;
   - `expected_structural_slots = 16` for every row;
   - balanced variant/topic distribution recorded.
3. Repaired R3.2 prompt allocation manifest with 12 distinct windows.
4. Local uniqueness preflight proving:
   - selected prompt hashes are all unique;
   - expected generated/decode output paths are fresh;
   - aggregate duplicate guard remains enabled.
5. Disabled allowlist entry and local/remote hash preflight.
6. Hermes TG/email notification before any eventual Slurm action.

## Current Next Allowed Action

Artifact-only expanded-prompt planning/implementation. No Slurm submission,
aggregation, rerun, Llama, FAR, sanitizer, or paper-facing claim is allowed
until the expanded prompt plan and repaired allocation preflight are reviewed.
