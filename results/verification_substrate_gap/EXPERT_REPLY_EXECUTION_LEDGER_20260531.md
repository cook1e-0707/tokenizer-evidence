# Expert Reply Execution Ledger 2026-05-31

Status: `PASS_EXPERT_REPLY_ANALYZED_DECOMPOSED_AND_EXECUTED_ARTIFACT_ONLY`

This ledger records how the 2026-05-31 expert reply was converted into
stepwise artifact work. It is a completion-audit artifact only. It starts no
Slurm job, no generation, no model scoring, no training, and no allowlist
enablement.

## Expert Reply Parsed Into Requirements

| Requirement | Concrete action derived from expert reply | Completion evidence |
| --- | --- | --- |
| Enter paper skeleton stage only as claim architecture, not full positive manuscript | Rewrite old manuscript around the VSG claim architecture and withhold positive ownership/public-text claims | Manuscript commit `64510b9`; PDF in `results/verification_substrate_gap/expert_review_packet_20260531/manuscript/VSG_manuscript_snapshot_20260531.pdf` |
| Freeze central VSG thesis and claim boundaries | Add claim ledger, forbidden claims, and claim-scope lint validation | `docs/paper_vsg/VSG_CLAIM_LEDGER_20260530.md`; `results/verification_substrate_gap/expert_review_packet_20260531/validation/claim_scope_lint_report.json` |
| Integrate figure/table assets that reflect the evidence map | Convert reviewed visual drafts into manuscript figure/table assets | Manuscript commit `d51f9cf`; figure files in `manuscript_source/figures/` inside the expert packet |
| Harden prose against claim drift | Run limited prose-risk review and make small wording edits | Manuscript commit `e375c51`; `VSG_PROSE_RISK_LINT_REVIEW_20260531.md` |
| Prepare expert-facing architecture review | Create architecture review describing section roles, evidence dependencies, and claim boundaries | Manuscript commit `87ef9cb`; `VSG_EXPERT_MANUSCRIPT_ARCHITECTURE_REVIEW_20260531.md` |
| Decide what state is sufficient for expert review | Create section-order/readiness gate artifact | Manuscript commit `0146795`; `VSG_SECTION_ORDER_VARIANT_PLAN_20260531.md` |
| Package objective review materials for expert inspection | Build a zip with PDF, source, figures, evidence tables, validation outputs, hashes, and usage instructions | `results/verification_substrate_gap/vsg_expert_review_packet_20260531.zip` |
| Make expert review scope explicit and objective | Add package-level review-scope file and Chinese external README | `EXPERT_REVIEW_SCOPE_20260531.md`; `vsg_expert_review_packet_20260531_README.txt` |

## Current Expert Packet Facts

| Item | Value |
| --- | --- |
| Zip path | `results/verification_substrate_gap/vsg_expert_review_packet_20260531.zip` |
| Zip SHA256 | `0c4d15c058960f2d242f8708be925ccf58c2e43fbf1d55cba6ce4f210ff6884f` |
| Packet manifest status | `PASS_PACKET_ASSEMBLED_ARTIFACT_ONLY_OBJECTIVE_FACTS` |
| Packet manifest file count | `60` |
| Expert review scope file included | `true` |
| New compute started | `false` |
| Slurm submitted | `false` |
| Generation started | `false` |
| Model scoring started | `false` |
| Training started | `false` |
| Paper claim allowed | `false` |
| Public final-text claim flag | `false` |

## Current Manuscript Facts

| Item | Value |
| --- | --- |
| Manuscript local HEAD | `64510b9 Polish VSG prose risk wording` |
| Latest full rewrite commit | `a02129f Rewrite manuscript around verification substrate gap` |
| Prose-risk hardening commit | `e375c51 Harden VSG manuscript prose-risk scope` |
| Architecture review commit | `87ef9cb Record VSG expert architecture review` |
| Section-order readiness commit | `0146795 Record VSG section-order review gate` |
| Formal/related-work hardening commit | `ccc4f39 Harden VSG related work and formal framing` |
| Figure/appendix hardening commit | `d51f9cf Harden VSG figures and appendices` |
| Final prose-risk polish commit | `64510b9 Polish VSG prose risk wording` |
| Manuscript worktree status at audit | clean |

## Claim Boundaries Preserved

The package and manuscript preserve the following boundaries:

- trace-bound first-divergence results are provider-side diagnostics;
- public final-text predicates are observability and spoofing diagnostics;
- accepted source-mismatch rows are spoofing evidence;
- accepted source-mismatch rows are not protected success;
- accepted source-mismatch rows are not codeword recovery;
- do not claim public text-only verification success;
- do not claim natural evidence success;
- do not claim phrase-decoder success;
- do not claim cryptographic provenance;
- do not claim sanitizer robustness;
- do not claim payload diversity;
- do not claim model-family general verification;
- do not claim ownership proof.

## Verification Commands Recorded During Execution

The following checks were run during the execution sequence:

```text
python3 scripts/verification_substrate_gap/lint_claim_scope.py ...
latexmk -pdf -interaction=nonstopmode main.tex
rg -n "undefined|Citation .* undefined|LaTeX Warning: Reference|LaTeX Warning: Citation|Fatal|Emergency stop|! LaTeX Error" main.log
unzip -t vsg_expert_review_packet_20260531.zip
shasum -a 256 vsg_expert_review_packet_20260531.zip
```

Observed state after packet rebuild:

```text
zip integrity: PASS
packet manifest status: PASS_PACKET_ASSEMBLED_ARTIFACT_ONLY_OBJECTIVE_FACTS
packet manifest file_count: 60
active manuscript claim-scope lint: PASS, 17 files, 0 violations
latest manuscript PDF sha256: 81a119565a44b5c637380f3770f9ce38fe9266ff28c83d4c23b1e1531fcf3458
zip sha256: 0c4d15c058960f2d242f8708be925ccf58c2e43fbf1d55cba6ce4f210ff6884f
```

## Completion Status

The expert reply has been:

1. analyzed into concrete requirements;
2. decomposed into stepwise artifact work;
3. executed through manuscript rewrite, claim hardening, architecture review,
   readiness gate, and expert packet assembly;
4. verified through claim lint, LaTeX build/log scan, manifest checks, zip
   integrity, and SHA256 recording.

No additional work is required to satisfy the requested artifact-only expert
review preparation state.
