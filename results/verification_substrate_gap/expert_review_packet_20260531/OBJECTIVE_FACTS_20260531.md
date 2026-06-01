# Objective Facts For VSG Expert Review 2026-05-31

## Manuscript Snapshot

- Title: `The Verification Substrate Gap in Natural LLM Outputs`
- Compiled PDF: `manuscript/VSG_manuscript_snapshot_20260531.pdf`
- PDF build status: `PASS`
- PDF page count from LaTeX log: `32`
- PDF bytes from validation artifact: `741724`
- PDF sha256 from validation artifact: `81a119565a44b5c637380f3770f9ce38fe9266ff28c83d4c23b1e1531fcf3458`
- Active manuscript claim-scope lint: `PASS`
- Active manuscript claim-scope lint violations: `0`
- Active manuscript files checked by claim lint: `17`

## Core Evidence Values Stated In The Manuscript

- Qwen trace-bound protected accepts: `94/96`
- Llama trace-bound protected accepts: `96/96`
- Listed trace-bound controls: `0/96`
- Public final-text codeword recovered blocks: `0`
- Qwen P2 AUC: `0.554676`
- Llama P2 AUC: `0.63128`
- Guided rewrite/graft top-100 accepted source-mismatch rows:
  - Qwen raw: `100/100`
  - Qwen task-only: `100/100`
  - Llama raw: `100/100`
- Ownership scenario stress test:
  - scenario-method cells: `63`
  - supported cooperative trace-bound scenario count: `1`
  - supported public final-text-only portable scenario count: `0`

## Claim Boundaries In The Manuscript

Current manuscript scope:

- Trace-bound first-divergence results are provider-side diagnostics.
- Public final-text predicates recover `0` codeword blocks.
- Source-mismatch accepts are spoofing evidence.
- Source-mismatch accepts are not protected success.
- Source-mismatch accepts are not codeword recovery.

Claims not made:

- do not claim public text-only verification success;
- do not claim natural evidence success;
- do not claim phrase-decoder success;
- do not claim cryptographic provenance;
- do not claim sanitizer robustness;
- do not claim payload diversity;
- do not claim model-family general verification;
- do not claim ownership proof.

## Validation Artifacts

- Claim-scope lint report:
  `validation/claim_scope_lint_report.json`
- LaTeX build summary:
  `validation/latex_build_summary.json`
- LaTeX log scan:
  `validation/latex_log_scan.json`
- Git snapshot:
  `validation/git_snapshot.json`
- File manifest:
  `packet_manifest.json`

## Included Evidence Tables

```text
evidence/figure_data/trace_bound_accepts.csv
evidence/figure_data/public_text_verifier_baselines.csv
evidence/figure_data/template_leakage_summary.csv
evidence/figure_data/attack_ladder_summary.csv
evidence/figure_data/ownership_scenario_heatmap.csv
evidence/figure_data/claim_ledger.csv
```

## Included Visual Drafts

```text
evidence/visual_drafts/figure_1_verification_substrate_map.svg
evidence/visual_drafts/figure_2_first_divergence_diagnostic.svg
evidence/visual_drafts/figure_3_controllability_vs_observability.svg
evidence/visual_drafts/figure_4_public_predicate_attack_ladder.svg
evidence/visual_drafts/figure_5_ownership_scenario_heatmap.svg
evidence/visual_drafts/table_1_claim_ledger.csv
evidence/visual_drafts/table_2_historical_failure_chain.csv
```

## Excluded From This Packet

- No raw large-scale run directories.
- No private credentials or secret keys.
- No Overleaf remote operation.
- No expert-question list.
- No route recommendation document.
- No new experiment outputs beyond validation and packaging artifacts.
