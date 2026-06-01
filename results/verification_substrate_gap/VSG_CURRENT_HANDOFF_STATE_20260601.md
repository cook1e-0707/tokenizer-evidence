# VSG Current Handoff State - 2026-06-01

## Canonical Phase

```text
VSG_RELEASE_BOUNDARY_AUDIT_RECORDED_REVIEW_REQUIRED_NO_NEW_EXPERIMENTS
```

## Status

The Verification Substrate Gap paper package has been rewritten, hardened,
packaged, independently verified, handoff-audited, committed, and pushed to
GitHub. The latest expert response has now been parsed into paper-hardening
stages. The response validates the VSG architecture but does not make the
manuscript submission-ready.

This pass added regression tests for the expert packet verifier and handoff
audit, plus active-manuscript prose-risk tests, so that the objective review
package and current manuscript remain checkable after future artifact-only
edits.

This continuation also implemented stronger public final-text predicate
baselines and ran a local pilot on available non-adopted/historical text
artifacts. Adopted locked final-text JSONL rows are not currently local, so the
pilot does not update paper-facing adopted locked evidence and does not create
a public text-only verification claim.

This continuation additionally audited the existing adopted-locked guided
rewrite/graft attack examples with deterministic readability proxies. The
audit found `0/60` proxy-readable rows, so the attack remains evidence of
public-predicate optimizability only, not naturalness-preserving rewriting.

This continuation now also records a machine-checkable reproducibility release
inventory. The inventory identifies candidate manuscript, evidence, script,
config, and test artifacts for a future supplemental release, records hashes,
and flags release blockers such as private-path hits, key/HMAC-related terms,
and files not tracked by the selected git scopes. It does not publish files or
expand the claim boundary.

This continuation additionally audits the ownership scenario stress-test
decision rules. The audit confirms the 7 x 9 matrix is complete, has no rule
failures, restricts support to the cooperative trace-bundle rows, and has zero
supported public final-text rows.

This continuation also applies a local manuscript prose cleanup that replaces
remaining internal audit-style phrases with academic scope language. The local
LaTeX manuscript builds successfully after the cleanup; Overleaf push is still
not performed.

This continuation additionally audits active manuscript figure quality. The
audit checks the five rendered PNG figures for dimensions, nonblank rendered
content, render-manifest consistency, LaTeX references, caption scope terms,
and core data traceability. It passes with zero failed figure checks and zero
failed data checks.

This continuation now also refreshes the expert review packet to the
2026-06-01 hardened manuscript snapshot. The refreshed packet includes the
local manuscript commit `c10b3f1e73689d63ceb0a4b3b8ea980974df16c1`, the
current PDF SHA256
`a64c984fac6503b20138805c8a9a323799f6feb1acfdcc1f7bb7310237f5a0fa`,
the stronger public-text predicate pilot, attack naturalness proxy audit,
reproducibility release inventory, ownership decision-rule audit, and
manuscript figure-quality audit. Packet verifier, handoff audit, and zip
integrity checks all pass.

This state does not unlock new Slurm jobs, generation, model scoring, training,
or paper-facing positive claims.

This continuation additionally records an artifact-only public-supplement
release-boundary audit. It converts the reproducibility inventory into a
decision table identifying which artifacts are ready for a reviewed supplement,
which require redaction/scope/security review, which must be committed or
copied into a reviewed supplement bundle, and which internal handoff records
should remain outside the public supplement. No files were published or copied
as release material.

## Current Review Packet

- Zip:
  `results/verification_substrate_gap/vsg_expert_review_packet_20260601.zip`
- Zip SHA256:
  `82b4007525b3d213bc4920b6b4bd947a7de002fdcf2d9271cc5543a2c32418e8`
- External README:
  `results/verification_substrate_gap/vsg_expert_review_packet_20260601_README.txt`
- Packet manifest:
  `results/verification_substrate_gap/expert_review_packet_20260601/packet_manifest.json`
- Packet verifier:
  `scripts/verification_substrate_gap/verify_vsg_expert_review_packet_20260601.py`
- Handoff audit:
  `scripts/verification_substrate_gap/audit_vsg_expert_handoff_20260601.py`

## Verification Evidence

- Packet verifier status: `PASS`
- Handoff audit status: `PASS`
- Packet file count: `87`
- Hashed file count: `86`
- Manifest status: `PASS_PACKET_ASSEMBLED_ARTIFACT_ONLY_20260601_HARDENING_INCLUDED`
- Claim-scope lint: `PASS`, 17 files, 0 violations
- LaTeX log scan: `PASS`
- Overfull hbox warnings: `0`
- Expert packet and manuscript prose regression tests: `PASS`, 13 tests passed
- Stronger public-predicate regression tests: `PASS`, total targeted tests now
  `16` passed
- Attack naturalness proxy regression tests: `PASS`, total targeted tests now
  `19` passed
- Reproducibility release inventory regression tests: `PASS`, release inventory
  targeted tests `5` passed
- Ownership scenario decision-rule audit regression tests: `PASS`, decision-rule
  targeted tests `5` passed
- Manuscript academic scope-language regression tests: `PASS`, current targeted
  suite `30` passed
- Manuscript figure-quality audit regression tests: `PASS`, figure-quality
  targeted tests `3` passed
- Refreshed expert packet regression tests: `PASS`
- Full `tests/verification_substrate_gap` suite: `PASS`, `41` tests passed
- Release-boundary audit tests: `PASS`, targeted release/reproducibility
  tests `9` passed
- Manuscript PDF SHA256:
  `a64c984fac6503b20138805c8a9a323799f6feb1acfdcc1f7bb7310237f5a0fa`

## Local Manuscript Hardening After Packet Delivery

- Local manuscript commit:
  `c10b3f1e73689d63ceb0a4b3b8ea980974df16c1`
- Local manuscript PDF SHA256 after prose and attack-scope cleanup:
  `a64c984fac6503b20138805c8a9a323799f6feb1acfdcc1f7bb7310237f5a0fa`
- Prose-risk change:
  reproducibility appendix now describes frozen artifacts and recorded evidence
  without naming internal canonical phase or claim-lint state; active manuscript
  prose now avoids internal audit-style `do not claim` and `draft` language.
- Packet refresh:
  performed as `results/verification_substrate_gap/vsg_expert_review_packet_20260601.zip`.

## Refreshed Expert Review Packet - 2026-06-01

- Builder:
  `scripts/verification_substrate_gap/build_vsg_expert_review_packet_20260601.py`
- Verifier:
  `scripts/verification_substrate_gap/verify_vsg_expert_review_packet_20260601.py`
- Handoff audit:
  `scripts/verification_substrate_gap/audit_vsg_expert_handoff_20260601.py`
- Zip:
  `results/verification_substrate_gap/vsg_expert_review_packet_20260601.zip`
- Zip SHA256:
  `82b4007525b3d213bc4920b6b4bd947a7de002fdcf2d9271cc5543a2c32418e8`
- External Chinese README:
  `results/verification_substrate_gap/vsg_expert_review_packet_20260601_README.txt`
- Packet directory:
  `results/verification_substrate_gap/expert_review_packet_20260601/`
- Verification summary:
  `results/verification_substrate_gap/expert_review_packet_verification_20260601/packet_verification_summary.json`
- Handoff audit summary:
  `results/verification_substrate_gap/expert_review_packet_verification_20260601/handoff_audit_summary.json`
- Packet file count:
  `87`
- Hashed file count:
  `86`
- Packet manuscript head:
  `c10b3f1e73689d63ceb0a4b3b8ea980974df16c1`
- Packet PDF SHA256:
  `a64c984fac6503b20138805c8a9a323799f6feb1acfdcc1f7bb7310237f5a0fa`
- Included hardening outputs:
  stronger public-text predicate local pilot, attack naturalness proxy audit,
  reproducibility release inventory, ownership decision-rule audit,
  and manuscript figure-quality audit. Current handoff state and expert-reply
  decomposition remain outside the zip to avoid packet-hash self-reference.

## Local Stronger Public-Predicate Pilot

- Summary:
  `results/verification_substrate_gap/VSG_PUBLIC_TEXT_STRONGER_BASELINE_LOCAL_PILOT_20260601.md`
- Output directory:
  `results/verification_substrate_gap/public_text_verifier_stronger_local_pilot_20260601/`
- New variants:
  `P4_char_ngram_public_predicate`, `P5_word_trigram_public_predicate`,
  `P6_hybrid_char_word_public_predicate`
- Local sources:
  `qwen_dev_869348_local_text_probe`,
  `llama_historical_879555_local_text_probe`
- Codeword recovered blocks:
  `0`
- Claim scope:
  local pilot only; not adopted locked evidence; not public text-only
  verification success.

## Attack Naturalness Proxy Audit

- Summary:
  `results/verification_substrate_gap/VSG_ATTACK_NATURALNESS_PROXY_AUDIT_20260601.md`
- Output directory:
  `results/verification_substrate_gap/public_predicate_attack_naturalness_audit_20260601/`
- Input:
  `results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/surrogate_guided_rewrite_examples.csv`
- Rows:
  `60`
- Proxy-readable rows:
  `0`
- Proxy-failed rows:
  `60`
- Main failure counts:
  `does_not_end_with_sentence_punctuation=60`,
  `isolated_single_letter_fragment=39`,
  `known_broken_graft_marker=40`
- Claim scope:
  readability proxy only; not semantic naturalness; not protected success; not
  codeword recovery.

## Reproducibility Release Inventory

- Summary:
  `results/verification_substrate_gap/reproducibility_release_inventory_20260601/release_inventory_summary.json`
- Report:
  `results/verification_substrate_gap/reproducibility_release_inventory_20260601/release_inventory_report.md`
- CSV:
  `results/verification_substrate_gap/reproducibility_release_inventory_20260601/release_inventory.csv`
- Rows:
  `78`
- Existing files:
  `78`
- Missing files:
  `0`
- Existing files not tracked by selected git scopes:
  `31`
- Rows requiring anonymization/scope review:
  `18`
- Private path hits:
  `3`
- Secret-term hits:
  `1`
- Release-ready without review:
  `False`
- Claim scope:
  release inventory only; no publication, no generation, no model scoring, no
  public text-only verification claim, no ownership-proof claim.

## Reproducibility Release Boundary Audit

- Summary:
  `results/verification_substrate_gap/reproducibility_release_boundary_audit_20260601/release_boundary_summary.json`
- Report:
  `results/verification_substrate_gap/reproducibility_release_boundary_audit_20260601/release_boundary_report.md`
- CSV:
  `results/verification_substrate_gap/reproducibility_release_boundary_audit_20260601/release_boundary_decisions.csv`
- Rows:
  `78`
- Ready for reviewed public supplement:
  `39`
- Excluded from public supplement:
  `4`
- Pre-release review required:
  `35`
- Release blockers:
  `35`
- Decision counts:
  `ready_for_reviewed_public_supplement=39`,
  `stage_or_copy_to_supplement_before_release=21`,
  `scope_review_before_release=10`,
  `redact_or_summarize_before_release=3`,
  `security_review_before_release=1`,
  `exclude_from_public_supplement=4`
- Release-ready now:
  `False`
- Claim scope:
  release-boundary audit only; no publication, no file-copy release bundle, no
  Slurm, no generation, no model scoring, no training, no public text-only
  verification claim, no ownership-proof claim.

## Ownership Scenario Decision-Rule Audit

- Summary:
  `results/verification_substrate_gap/ownership_scenario_decision_rule_audit_20260601/decision_rule_audit_summary.json`
- Report:
  `results/verification_substrate_gap/ownership_scenario_decision_rule_audit_20260601/decision_rule_audit_report.md`
- Rows:
  `63`
- Scenarios:
  `7`
- Method families:
  `9`
- Rule failures:
  `0`
- Supported trace-bound rows:
  `2`
- Supported public final-text rows:
  `0`
- Supported trace-bound pairs:
  `S2_cooperative_provider_with_trace_bundle::provider_side_trace`,
  `S2_cooperative_provider_with_trace_bundle::first_divergence_diagnostic`
- Claim scope:
  stress-test rule audit only; no ownership proof, no public text-only
  verification success claim, no new compute.

## Manuscript Figure Quality Audit

- Summary:
  `results/verification_substrate_gap/manuscript_figure_quality_audit_20260601/figure_quality_summary.json`
- Report:
  `results/verification_substrate_gap/manuscript_figure_quality_audit_20260601/figure_quality_report.md`
- Figures checked:
  `5`
- Failed figure checks:
  `0`
- Data traceability checks:
  `5`
- Failed data traceability checks:
  `0`
- Core data checks:
  Qwen `94/96` and Llama `96/96` trace-bound protected counts present;
  public final-text recovered codeword blocks total is `0`;
  guided rewrite/graft top-100 source-mismatch accepts are `100/100` for all
  plotted groups;
  ownership heatmap contains `7 x 9 = 63` rows;
  supported public final-text rows are `0`.
- Claim scope:
  figure-quality audit only; no figure rerender, no Slurm, no generation, no
  model scoring, no public text-only verification claim, no ownership-proof
  claim.

## Git Sync Evidence

- Root repository branch: `main`
- Root repository commit at the pre-state-record sync check:
  `4e470810ed6a6741fa94206915a2ae0d0b59405b`
- `origin/main` commit at the pre-state-record sync check:
  `4e470810ed6a6741fa94206915a2ae0d0b59405b`
- Manuscript repository commit:
  `64510b9daf88deb2efd49a26c8046a023fa4904e`
- Overleaf push: not performed.

This file intentionally does not attempt to record the commit hash containing
itself, because that would be self-referential and unstable. Use `git log` for
the current state-record commit.

## Current Claim Scope

Current manuscript and review packet support only the VSG substrate-gap framing:

- trace-bound first-divergence results are provider-side diagnostics;
- public final-text predicates are observability and spoofing diagnostics;
- accepted source-mismatch rows are spoofing evidence;
- accepted source-mismatch rows are not protected success;
- accepted source-mismatch rows are not codeword recovery;
- public final-text codeword recovered blocks remain `0`.

The current artifacts do not claim:

- public text-only verification success;
- natural evidence success;
- phrase-decoder success;
- cryptographic provenance;
- sanitizer robustness;
- payload diversity;
- model-family general verification;
- ownership proof.

## Next Allowed Action

```text
Artifact-only manuscript/package hardening that does not alter the claim
boundary or start new experiments.
```

## Not Allowed Without New Expert/Human Route Decision

```text
new Slurm submission
new generation
new model scoring
new training
new public text-only verification success claim
new natural evidence success claim
new ownership proof claim
Overleaf push
```

## Notes

The root worktree still contains unrelated historical/generated modified and
untracked files. Those are not part of the current VSG handoff packet state.
