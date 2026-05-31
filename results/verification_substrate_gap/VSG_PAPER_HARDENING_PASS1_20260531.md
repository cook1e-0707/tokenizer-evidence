# VSG Paper Hardening Pass 1 2026-05-31

Status: `PASS_RELATED_WORK_AND_FORMAL_FRAMEWORK_HARDENED_ARTIFACT_ONLY`

This artifact records the first execution pass after the 2026-05-31 expert
reply.  It is artifact-only.  It starts no Slurm job, no generation, no model
scoring, no training, and no allowlist enablement.

## Expert Reply Decomposition

| Expert requirement | Concrete work item | Pass 1 status |
| --- | --- | --- |
| Related-work hardening | Add public watermark protocols, C2PA-style credentials, ZK proof systems, watermark learnability/spoofing, and substrate positioning | Completed in manuscript related work and extended related work |
| Formal framework upgrade | Add first-divergence reduction, public-predicate source-density spoofability, and substrate displacement principle | Completed in setup, first-divergence section, and formal appendix |
| Figure regeneration | Remove placeholder labels and make attack ladder/heatmap publication-readable | Deferred to next pass |
| Attack-ladder evidence strengthening | Add examples and naturalness caveat material | Deferred to next pass |
| Reproducibility and license inventory | Add asset license and command appendix material | Deferred to next pass |
| Prose-risk cleanup | Remove internal audit wording from academic prose and keep lint outputs in reproducibility context | Partially maintained; full prose sweep deferred |

## Manuscript Changes Completed

| File | Change |
| --- | --- |
| `main.tex` | Added `principle` theorem environment and included `appendix/formal_substrate_gap.tex` |
| `section_02_related_work.tex` | Expanded related work to cover SynthID-Text, publicly auditable watermark protocols, credential/proof substrates, learnability/removal/spoofing, and VSG positioning |
| `section_03_problem_setup.tex` | Added Substrate Displacement Principle with explicit scope limitation |
| `section_04_tokenizer_alignment.tex` | Added first-divergence reduction and public-predicate spoofability propositions; connected observed source-mismatch rates to expected rejection-sampling cost |
| `appendix/formal_substrate_gap.tex` | Added proofs and scope notes for the new formal statements |
| `appendix/extended_related_work.tex` | Rebuilt the extended related work as a substrate-positioning map with a method-family table |
| `references.bib` | Added references for SynthID-Text, Publicly-Detectable Watermarking, Puppy, PVMark, VOW, C2PA, zkLLM, ZKPROV, Watermarks in the Sand, watermark learnability, and DITTO |

## Claim Boundaries Preserved

- The manuscript continues to frame trace-bound evidence as provider-side
  diagnostic evidence only.
- Public final-text predicates remain observability and spoofing diagnostics.
- Source-mismatch accepts remain attack evidence, not protected success and not
  codeword recovery.
- The manuscript does not claim public text-only verification success.
- The manuscript does not claim natural evidence success.
- The manuscript does not claim phrase-decoder success.
- The manuscript does not claim cryptographic provenance.
- The manuscript does not claim sanitizer robustness.
- The manuscript does not claim payload diversity.
- The manuscript does not claim ownership proof.

## Validation Commands

```text
python3 scripts/verification_substrate_gap/lint_claim_scope.py \
  manuscripts/69db2644566dcc36c9da320e/main.tex \
  manuscripts/69db2644566dcc36c9da320e/section_01_introduction.tex \
  manuscripts/69db2644566dcc36c9da320e/section_02_related_work.tex \
  manuscripts/69db2644566dcc36c9da320e/section_03_problem_setup.tex \
  manuscripts/69db2644566dcc36c9da320e/section_04_tokenizer_alignment.tex \
  manuscripts/69db2644566dcc36c9da320e/section_05_bucket_level_injection.tex \
  manuscripts/69db2644566dcc36c9da320e/section_06_deterministic_verification.tex \
  manuscripts/69db2644566dcc36c9da320e/section_07_experiments.tex \
  manuscripts/69db2644566dcc36c9da320e/section_08_discussion_limitations.tex \
  manuscripts/69db2644566dcc36c9da320e/section_09_conclusion.tex \
  manuscripts/69db2644566dcc36c9da320e/appendix/proofs.tex \
  manuscripts/69db2644566dcc36c9da320e/appendix/formal_substrate_gap.tex \
  manuscripts/69db2644566dcc36c9da320e/appendix/extended_related_work.tex \
  manuscripts/69db2644566dcc36c9da320e/appendix/reproducibility.tex

latexmk -pdf -interaction=nonstopmode main.tex

rg -n "undefined|Citation .* undefined|LaTeX Warning: Reference|LaTeX Warning: Citation|Fatal|Emergency stop|! LaTeX Error|There were undefined" main.log
```

Observed validation:

```text
claim-scope lint: PASS, 14 checked files, 0 violations
LaTeX build: PASS, main.pdf produced
LaTeX log scan: PASS, no undefined citation/reference or LaTeX error matches
```

## Remaining Expert-Directed Work

The next pass should remain artifact-only and should address:

1. figure regeneration and readability;
2. attack-ladder example table and naturalness caveat material;
3. asset license appendix;
4. reproducibility commands appendix;
5. full prose-risk sweep to remove residual internal workflow language from
   main academic prose.
