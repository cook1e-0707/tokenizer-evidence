# VSG Paper Hardening Pass 2 2026-05-31

Status: `PASS_FIGURES_ATTACK_EXAMPLES_LICENSE_REPRO_HARDENED_ARTIFACT_ONLY`

This artifact records the second execution pass after the 2026-05-31 expert
reply.  It is artifact-only.  It starts no Slurm job, no generation, no model
scoring, no training, and no allowlist enablement.

## Expert Route Items Addressed

| Expert requirement | Pass 2 action | Status |
| --- | --- | --- |
| Figure regeneration and readability | Added local figure renderer and regenerated manuscript PNGs for Figures 1--5 from existing CSV evidence | Completed |
| Attack-ladder evidence strengthening | Added a compact source-mismatch guided-rewrite example review and manuscript appendix table | Completed |
| Attack naturalness caveat | Recorded that examples are report-only and no semantic naturalness gate is applied | Completed |
| Reproducibility and licensing | Added asset/license inventory and reproduction-command appendix | Completed |
| Prose-risk cleanup | Removed placeholder and visual-draft wording from active manuscript captions/tables | Completed for touched sections |

## New or Updated Scripts

| Script | Purpose |
| --- | --- |
| `scripts/verification_substrate_gap/render_vsg_manuscript_figures.py` | Renders publication-readable PNG Figures 1--5 from existing figure-data CSV files |
| `scripts/verification_substrate_gap/build_vsg_attack_examples_review.py` | Builds a compact attack-example review table from existing guided rewrite/graft examples |

## Manuscript Artifacts Updated

| Artifact | Change |
| --- | --- |
| `figures/figure_1_verification_substrate_map.png` | Re-rendered substrate map without placeholder labels |
| `figures/figure_2_first_divergence_diagnostic.png` | Re-rendered first-divergence schematic |
| `figures/figure_3_controllability_vs_observability.png` | Re-rendered trace-bound versus public final-text codeword panel |
| `figures/figure_4_public_predicate_attack_ladder.png` | Re-rendered attack ladder with top-100 accepted source-mismatch data |
| `figures/figure_5_ownership_scenario_heatmap.png` | Re-rendered scenario-method heatmap with readable labels |
| `appendix/attack_examples.tex` | Added representative source-mismatch guided rewrite/graft examples |
| `appendix/asset_licenses.tex` | Added asset/license inventory |
| `appendix/reproducibility_commands.tex` | Added local paper-asset reproduction commands |

## Generated Review Outputs

| Output | Status |
| --- | --- |
| `results/verification_substrate_gap/paper_manuscript_figures_20260531/` | Figure copies and manifest written |
| `results/verification_substrate_gap/paper_attack_examples_20260531/` | Attack-example CSV/Markdown/summary/manifest written |

## Claim Boundaries Preserved

- Figure 3 still separates provider-side trace-bound diagnostics from public
  final-text codeword recovery.
- Figure 4 shows source-mismatch spoofing evidence only.
- Attack examples are report-only and are not protected success.
- Attack examples are not codeword recovery.
- No semantic naturalness gate is claimed for attack examples.
- No Slurm, generation, model scoring, or training was run.
- No paper-facing public final-text verification claim is introduced.
- No ownership proof is introduced.

## Validation Commands

```text
python3 scripts/verification_substrate_gap/render_vsg_manuscript_figures.py
python3 scripts/verification_substrate_gap/build_vsg_attack_examples_review.py
python3 scripts/verification_substrate_gap/lint_claim_scope.py [active manuscript files]
latexmk -pdf -interaction=nonstopmode main.tex
rg -n "undefined|Citation .* undefined|LaTeX Warning: Reference|LaTeX Warning: Citation|Fatal|Emergency stop|! LaTeX Error|There were undefined" main.log
rg -n "placeholder|visual draft|claim lint|Do not claim" [active manuscript files]
```

Observed validation:

```text
claim-scope lint: PASS, 17 checked files, 0 violations
LaTeX build: PASS, main.pdf produced
LaTeX error/reference/citation log scan: PASS
active manuscript placeholder/prose-risk grep: PASS
new Slurm/generation/model scoring/training: false by script manifests
```

## Remaining Expert-Directed Work

The next pass should remain artifact-only and should focus on:

1. checking figure placement and page-level readability in the compiled PDF;
2. broad prose smoothing across introduction/results/discussion so the paper
   reads less like an internal project audit;
3. deciding whether the expert packet should be rebuilt from the hardened PDF
   snapshot.
