# R4 dev diagnostic route decision

Date: 2026-05-12

## Decision

The user approved the R4 cover-natural ECC route, the small Qwen dev
diagnostic design, the dev gates, and preparation of a Slurm wrapper/allowlist
entry.

This decision does not unlock training, Llama, same-family null, sanitizer,
FAR aggregation, payload-diversity claims, or paper-facing positive claims.

## Scope

Canonical phase after this decision:

`V2_R4_DEV_DIAGNOSTIC_WRAPPER_PREFLIGHT_APPROVED_NO_SUBMISSION_YET`

The approved dev diagnostic is:

- Qwen only;
- dev prompt bank only;
- same-contract `a55e`;
- 32 blocks;
- 64 prompts per block;
- 2048 total prompts;
- 4 shard-array tasks with 512 prompts per shard;
- arms: protected, raw, task-only, wrong-key, wrong-payload;
- primary reported decode uses `format_scrub=all`.

## Prompt-Bank Size Repair

The first R4 plan-only prompt bank had only `384` dev prompts. That is
insufficient for the approved `32 x 64 = 2048` prompt dev diagnostic without
duplicate prompt windows.

Therefore the R4 config is updated to build:

- `2048` dev prompts;
- `6144` locked prompts;
- disjoint dev/locked topic domains;
- no Step/slot structural instructions.

This is an artifact-only repair to satisfy the approved diagnostic scale. It
does not change the surface bank, codebook, decoder thresholds, payload, or key
based on locked outputs.

## Dev Gates

The proposed dev gates are:

| Gate | Required |
|---|---:|
| protected accepts, no scrub | `>=28/32` |
| protected accepts, `format_scrub=all` | `>=26/32` |
| raw accepts | `0/32` |
| task-only accepts | `0/32` |
| wrong-key accepts | `0/32` |
| wrong-payload accepts | `0/32` |
| forbidden public technical surface | `0` |
| duplicate generated-output hash | `0` |
| duplicate decode-row hash | `0` |
| shallow structural classifier AUC | `<=0.60` |

## Submission Preconditions

Before any Slurm submission:

- wrapper plan-only smoke must pass;
- allowlist safety must pass with zero enabled entries;
- local/remote file hashes must match;
- Hermes TG/email notification must be sent;
- exactly one reviewed R4 dev diagnostic allowlist entry may be enabled;
- the allowlist entry must be disabled immediately after `sbatch`;
- no active conflicting Chimera jobs may exist.

## Claim Policy

Even if the dev diagnostic passes, it remains a dev diagnostic. It does not
authorize paper-facing positive claims, payload diversity, full FAR, Llama, or
sanitizer robustness.
