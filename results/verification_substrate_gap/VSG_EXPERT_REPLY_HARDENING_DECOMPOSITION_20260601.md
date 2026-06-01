# VSG Expert Reply Hardening Decomposition - 2026-06-01

Status: `PASS_EXPERT_REPLY_DECOMPOSED_ARTIFACT_ONLY_TEST_HARDENING_ADDED`

This record decomposes the latest expert reply into concrete work items and
records the artifact-only actions completed in this pass. No Slurm job,
generation, model scoring, training, or allowlist enablement was started.

## Expert Reply Parsed Into Stages

| Stage | Expert concern | Required action | Current status |
| --- | --- | --- | --- |
| 1 | VSG architecture is valid, but not submission-ready | Keep the claim as a substrate-gap manuscript, not a positive ownership/public-text verifier paper | Already reflected in the current manuscript and expert packet |
| 2 | Theory is too weak if it only states a trivial first-divergence lemma | Add first-divergence reduction, public predicate spoofability, and substrate displacement scope | Already present in `section_04_tokenizer_alignment.tex` and `appendix/formal_substrate_gap.tex` |
| 3 | Related work must face the strongest public-verification/protocol/proof work | Position publicly detectable watermarking, Puppy/PVMark/VOW-style protocols, C2PA, ZK proofs, and learnability/spoofing work by substrate | Already present in `section_02_related_work.tex` and `appendix/extended_related_work.tex` |
| 4 | Public predicate surrogate role must be central and limited | State that P2/P3-style predicates are diagnostic endpoints, not exhaustive verifier classes | Already reflected in manuscript wording and claim ledger |
| 5 | Guided rewrite/graft attack needs example-level audit and naturalness caveat | Include source-mismatch examples with score movement and explicit non-claim caveats | Already present in `appendix/attack_examples.tex` and `results/verification_substrate_gap/paper_attack_examples_20260531/` |
| 6 | Figures cannot remain placeholder drafts | Replace placeholder figures with rendered manuscript PNGs and remove internal placeholder labels | Already present in manuscript `figures/` and packet source; current text scan found no remaining placeholder language except NeurIPS style macros |
| 7 | Reproducibility/license plan is a blocker before submission | Record reproducibility commands and asset/license inventory | Already present in `appendix/reproducibility_commands.tex` and `appendix/asset_licenses.tex` |
| 8 | Handoff artifacts must remain objective and claim-safe | Make packet verification and handoff audit regression-testable | Completed in this pass |

## Work Completed In This Pass

Added regression tests:

```text
tests/verification_substrate_gap/test_vsg_expert_packet_verifiers.py
tests/verification_substrate_gap/test_vsg_manuscript_prose_hardening.py
```

The tests cover:

- current expert packet verifier passes;
- packet file count remains `60`;
- hashed file count remains `59`;
- claim-scope lint remains `PASS` with `0` violations;
- removing a required file causes verifier failure;
- setting `manifest_self_hash_excluded=false` causes verifier failure;
- current expert handoff audit passes with no failures;
- handoff audit continues to assert objective-only scope and no new experiments.
- active manuscript prose contains no internal placeholder/review phrases;
- expert-requested substrate-positioning references are both cited and defined;
- active manuscript uses rendered PNG figures rather than placeholder SVGs.

Updated manuscript prose:

```text
manuscripts/69db2644566dcc36c9da320e/appendix/reproducibility.tex
```

The reproducibility appendix no longer names an internal canonical phase or
claim-scope lint state. It describes the snapshot in terms of frozen artifacts,
recorded summaries, manifests, review tables, figure inputs, and prose-scope
checks. This is a local manuscript hardening commit only; it does not refresh
the delivered expert packet zip.

## Verification Run

```text
uv run pytest tests/verification_substrate_gap/test_vsg_manuscript_prose_hardening.py tests/verification_substrate_gap/test_vsg_expert_packet_verifiers.py tests/verification_substrate_gap/test_claim_scope_linter.py
```

Observed result:

```text
13 passed in 0.21s
```

Manuscript checks:

```text
python3 scripts/verification_substrate_gap/lint_claim_scope.py <17 active manuscript files>
latexmk -pdf -interaction=nonstopmode main.tex
rg -n "undefined|Citation .* undefined|LaTeX Warning: Reference|LaTeX Warning: Citation|Fatal|Emergency stop|! LaTeX Error|There were undefined|Overfull" main.log
```

Observed result:

```text
claim-scope lint: PASS, 17 files, 0 violations
LaTeX build: PASS, 32 pages
LaTeX log risk scan: no matches
local manuscript commit: 4d66568ec08325d1d81b5ce060fbfda302e3177d
local manuscript PDF sha256: 73d605183ae501f7555e91ccad4fb565fd43c7513714cf3c80936681941ed20b
```

Additional direct script checks:

```text
python3 scripts/verification_substrate_gap/verify_vsg_expert_review_packet.py
python3 scripts/verification_substrate_gap/audit_vsg_expert_handoff.py
```

Observed status:

```text
packet verifier: PASS
handoff audit: PASS
packet_total_file_count: 60
hashed_file_count: 59
claim_lint: PASS, 0 violations
handoff audit failures: 0
```

## Remaining Scope After This Pass

The current allowed route remains artifact-only manuscript/package hygiene.
The following are not unlocked by this pass:

- new Slurm submission;
- generation;
- model scoring;
- training;
- allowlist enablement;
- public text-only verification success claim;
- natural evidence success claim;
- ownership proof claim;
- cryptographic provenance claim.

The expert-suggested stronger public-predicate baselines and naturalness audits
remain future optional hardening candidates. They require a separate route
decision because they would go beyond the delivered expert packet snapshot.
