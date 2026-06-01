# Objective Facts For VSG Expert Review 2026-06-01

## Manuscript Snapshot

- Title: `The Verification Substrate Gap in Natural LLM Outputs`
- Compiled PDF: `manuscript/VSG_manuscript_snapshot_20260601.pdf`
- PDF build status: `PASS`
- PDF page count from LaTeX log: `32`
- PDF bytes from validation artifact: `741313`
- PDF sha256 from validation artifact: `a64c984fac6503b20138805c8a9a323799f6feb1acfdcc1f7bb7310237f5a0fa`
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
  - supported trace-bound rows: `2`
  - supported public final-text rows: `0`

## 2026-06-01 Hardening Outputs

- Stronger public-text predicate local pilot:
  - codeword recovered blocks total: `0`
  - adopted locked evidence updated: `False`
- Attack naturalness proxy audit:
  - rows: `60`
  - proxy-readable rows: `0`
  - semantic naturalness claimed: `False`
- Reproducibility release inventory:
  - rows: `78`
  - missing files: `0`
  - requires anonymization/scope review: `18`
  - release-ready without review: `False`
- Ownership decision-rule audit:
  - rows: `63`
  - rule failures: `0`
  - supported public final-text rows: `0`
- Manuscript figure-quality audit:
  - figures checked: `5`
  - failed figure checks: `0`
  - failed data checks: `0`

## Claim Boundaries In The Manuscript

Current manuscript scope:

- Trace-bound first-divergence results are provider-side diagnostics.
- Public final-text predicates recover `0` codeword blocks.
- Source-mismatch accepts are spoofing evidence.
- Source-mismatch accepts are not protected success.
- Source-mismatch accepts are not codeword recovery.

Claims not established by current artifacts:

- public text-only verification success;
- natural evidence success;
- phrase-decoder success;
- cryptographic provenance;
- sanitizer robustness;
- payload diversity;
- model-family general verification;
- ownership proof.
