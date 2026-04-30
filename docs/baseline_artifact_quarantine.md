# Baseline Artifact Quarantine

Status: active as of 2026-04-30.

## Decision

The successful Scalable/Perinucleus baseline for paper-facing claims is:

- `results/processed/paper_stats/baseline_perinucleus_official_qwen_final_summary.json`
- `results/tables/baseline_perinucleus_official_qwen_final.csv`

That package is the Qwen-adapted official-code baseline and reports `48/48`
clean exact-verifier successes. It may enter the main external-baseline table
only with the label `Qwen-adapted official Scalable/Perinucleus baseline`.

The legacy `baseline_perinucleus` artifacts are not that baseline. They are
diagnostics from an earlier adapted/no-train path and must not be used for
Scalable Fingerprinting claims.

## Quarantined Artifacts

The diagnostic copies are preserved at:

- `results/processed/paper_stats/diagnostics/baseline_perinucleus_legacy_diagnostic_summary.json`
- `results/tables/diagnostics/baseline_perinucleus_legacy_diagnostic.csv`

The canonical legacy paths remain excluded from paper claims:

- `results/processed/paper_stats/baseline_perinucleus_summary.json`
- `results/tables/baseline_perinucleus.csv`

If these canonical legacy paths are regenerated, they must still be treated as
diagnostic-only unless a separate fidelity audit explicitly upgrades them.

## Allowed Uses

- Debugging old Perinucleus integration failures.
- Appendix discussion of rejected diagnostics, if needed.
- Negative evidence that a non-official or incomplete adaptation was not used
  for claims.

## Forbidden Uses

- Do not merge these rows into the official Perinucleus final result.
- Do not use them as evidence that Scalable Fingerprinting is weak.
- Do not place them in the main comparison table.
- Do not cite them as the external active ownership baseline.

## Paper-Facing Guardrail

The main paper may state clean success parity with the official Qwen-adapted
Perinucleus baseline. It may not state clean-success superiority over
Perinucleus unless a future, frozen, matched comparison artifact supports that
claim on a metric beyond clean exact verification.
