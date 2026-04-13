# Baseline Protocol

Baseline logic is isolated under `src/baselines/`.

## Current state

- All baseline adapters are safe placeholders unless a real baseline implementation is integrated.
- No experimental path assumes that external baseline code is already installed.
- Placeholder adapters return explicit, non-crashing status objects so evaluation runs can still complete and report that a baseline is unavailable.

## Integration expectations

When integrating a real baseline:

1. Keep the adapter signature unchanged.
2. Put baseline-specific environment checks inside the adapter.
3. Avoid importing heavy baseline dependencies in lightweight tests.
4. Preserve the same summary schema so downstream aggregation remains stable.
