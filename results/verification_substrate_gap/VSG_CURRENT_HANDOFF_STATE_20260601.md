# VSG Current Handoff State - 2026-06-01

## Canonical Phase

```text
VSG_PUBLIC_SUPPLEMENT_COPY_REVIEW_PLAN_RECORDED_ARTIFACT_ONLY_NO_PUBLICATION
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

This continuation now also records a plan-only public-supplement staging map.
It maps every release-boundary row to a future supplement path, transform,
scope note, residual risk, and execution/review flag. It still performs no
supplement copying, no publication, no new experiments, and no claim expansion.

This continuation additionally records an artifact-only dry-run bundle
manifest. The manifest resolves the readiness audit into future bundle source
paths, target supplement paths, review artifacts, and remaining blockers. It
confirms all dry-run sources and review artifacts exist, but it still performs
no file copying, creates no public supplement, publishes nothing, and keeps the
release marked not ready.

This continuation now also records an artifact-only blocker checklist. It
splits the 35 remaining publication blockers into 21 copy-required rows and 14
human-review-required rows, with the evidence needed to close each blocker in a
future reviewed bundle pass. The checklist itself does not close blockers,
perform reviews, copy files, publish anything, or expand claim scope.

This continuation additionally records an artifact-only public-supplement
bundle construction preflight. The preflight converts the dry-run bundle
manifest and blocker checklist into candidate target paths, source hashes,
future copy-plan entries, and human-review holds. It confirms no source or
review-artifact files are missing and no candidate target paths collide, but it
does not copy files, create the candidate bundle, perform human review, publish
anything, or expand claim scope.

This continuation now also records an artifact-only public-supplement
copy/review plan. It converts the future copy plan into comments-only copy and
hash verification commands, and converts the human-review holds into a
reviewer-facing checklist. All review rows remain `pending_not_performed`;
the plan does not copy files, create the candidate bundle, perform human
review, publish anything, or expand claim scope.

This continuation additionally creates artifact-only review derivatives for
the supplement release blockers: redacted trace-bound CSVs with private path
fields removed, scope-note labels for source-mismatch/local-pilot/trace-bound
rows, and a config security review for key-related schema field names. These
are review derivatives only; no public supplement was created or published.

This continuation now records a public-supplement readiness re-audit using the
staging plan plus the review derivatives. All 14 derivative-required rows are
covered by review derivatives, with zero derivative-uncovered rows. The audit
still records 21 stage/copy rows and 14 human-review rows as publication
blockers, so no public supplement is release-ready yet.

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
- Full `tests/verification_substrate_gap` suite: `PASS`, `53` tests passed
- Release-boundary audit tests: `PASS`, targeted release/reproducibility
  tests `9` passed
- Release-staging plan tests: `PASS`, targeted staging tests `4` passed
- Public-supplement review-derivative tests: `PASS`, targeted derivative
  tests `4` passed
- Public-supplement readiness audit tests: `PASS`, targeted readiness tests
  `4` passed
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

## Public Supplement Release Staging Plan

- Summary:
  `results/verification_substrate_gap/reproducibility_release_staging_plan_20260601/release_staging_summary.json`
- Report:
  `results/verification_substrate_gap/reproducibility_release_staging_plan_20260601/release_staging_report.md`
- CSV:
  `results/verification_substrate_gap/reproducibility_release_staging_plan_20260601/release_staging_plan.csv`
- Rows:
  `78`
- Direct include candidates:
  `39`
- Stage/copy candidates:
  `21`
- Redacted derivative candidates:
  `3`
- Scope-note gated candidates:
  `10`
- Security-review gated candidates:
  `1`
- Excluded internal records:
  `4`
- Execution-required rows:
  `35`
- Manual-review-required rows:
  `14`
- Duplicate planned supplement targets:
  `0`
- Release-ready after plan:
  `False`
- Claim scope:
  staging plan only; no file copying, no public supplement creation, no
  publication, no Slurm, no generation, no model scoring, no training, no
  public text-only verification claim, no ownership-proof claim.

## Public Supplement Review Derivatives

- Summary:
  `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/review_derivatives_summary.json`
- Report:
  `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/review_derivatives_report.md`
- Manifest:
  `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/review_derivatives_manifest.json`
- Redacted trace CSVs:
  `3`
- Redacted trace rows:
  `1920`
- Dropped private-path field:
  `source_shard_dir`
- Private marker hits after redaction:
  `0`
- Scope notes:
  `10`
- Config security reviews:
  `1`
- Security field-name hits:
  `8`
- Literal secret-value hits:
  `0`
- Public supplement created:
  `False`
- Publication performed:
  `False`
- Release-ready after derivatives:
  `False`
- Claim scope:
  review derivatives only; redacted trace summaries remain provider-side
  diagnostic summaries, source-mismatch artifacts remain spoofing evidence
  only, local-pilot artifacts remain non-adopted/historical evidence only, and
  no public text-only verification or ownership-proof claim is made.

## Public Supplement Readiness Audit

- Summary:
  `results/verification_substrate_gap/public_supplement_readiness_audit_20260601/readiness_summary.json`
- Report:
  `results/verification_substrate_gap/public_supplement_readiness_audit_20260601/readiness_report.md`
- CSV:
  `results/verification_substrate_gap/public_supplement_readiness_audit_20260601/readiness_decisions.csv`
- Rows:
  `78`
- Direct include candidates:
  `39`
- Stage/copy still required:
  `21`
- Derivative-required rows:
  `14`
- Derivative-covered rows:
  `14`
- Derivative-uncovered rows:
  `0`
- Manual review still required:
  `14`
- Publication blockers:
  `35`
- Excluded internal records:
  `4`
- Release-ready now:
  `False`
- Claim scope:
  readiness audit only; no public supplement construction, no publication, no
  Slurm, no generation, no model scoring, no training, no public text-only
  verification claim, no ownership-proof claim.
- Verification:
  targeted readiness pytest `PASS` with `4` tests passed; full
  `tests/verification_substrate_gap` pytest `PASS` with `53` tests passed.

## Public Supplement Dry-Run Bundle Manifest

- Summary:
  `results/verification_substrate_gap/public_supplement_dry_run_manifest_20260601/dry_run_bundle_summary.json`
- Report:
  `results/verification_substrate_gap/public_supplement_dry_run_manifest_20260601/dry_run_bundle_report.md`
- CSV:
  `results/verification_substrate_gap/public_supplement_dry_run_manifest_20260601/dry_run_bundle_manifest.csv`
- Rows:
  `78`
- Dry-run bundle entries:
  `74`
- Excluded internal records:
  `4`
- Direct include entries:
  `39`
- Copy-required entries:
  `21`
- Redacted derivative entries:
  `3`
- Scope-note review entries:
  `10`
- Security-review entries:
  `1`
- Manual review required:
  `14`
- Publication blockers:
  `35`
- Missing dry-run sources:
  `0`
- Missing review artifacts:
  `0`
- Duplicate planned targets:
  `0`
- Release-ready after dry run:
  `False`
- Claim scope:
  dry-run manifest only; no public supplement construction, no publication, no
  Slurm, no generation, no model scoring, no training, no public text-only
  verification claim, no ownership-proof claim.
- Verification:
  targeted dry-run manifest pytest `PASS` with `4` tests passed; full
  `tests/verification_substrate_gap` pytest `PASS` with `57` tests passed.

## Public Supplement Blocker Checklist

- Summary:
  `results/verification_substrate_gap/public_supplement_blocker_checklist_20260601/blocker_checklist_summary.json`
- Report:
  `results/verification_substrate_gap/public_supplement_blocker_checklist_20260601/blocker_checklist_report.md`
- Full checklist:
  `results/verification_substrate_gap/public_supplement_blocker_checklist_20260601/blocker_checklist.csv`
- Copy-required checklist:
  `results/verification_substrate_gap/public_supplement_blocker_checklist_20260601/copy_required_checklist.csv`
- Human-review checklist:
  `results/verification_substrate_gap/public_supplement_blocker_checklist_20260601/human_review_checklist.csv`
- Publication blockers:
  `35`
- Copy-required rows:
  `21`
- Human-review-required rows:
  `14`
- Missing sources:
  `0`
- Missing review artifacts:
  `0`
- Unclassified blockers:
  `0`
- All blockers have resolution track:
  `True`
- Release-ready after checklist:
  `False`
- Claim scope:
  blocker checklist only; no public supplement construction, no publication,
  no Slurm, no generation, no model scoring, no training, no public text-only
  verification claim, no ownership-proof claim.
- Verification:
  targeted blocker-checklist pytest `PASS` with `3` tests passed; full
  `tests/verification_substrate_gap` pytest `PASS` with `60` tests passed.

## Public Supplement Bundle Construction Preflight

- Summary:
  `results/verification_substrate_gap/public_supplement_bundle_preflight_20260601/bundle_preflight_summary.json`
- Report:
  `results/verification_substrate_gap/public_supplement_bundle_preflight_20260601/bundle_preflight_report.md`
- Full preflight CSV:
  `results/verification_substrate_gap/public_supplement_bundle_preflight_20260601/bundle_construction_preflight.csv`
- Future copy plan:
  `results/verification_substrate_gap/public_supplement_bundle_preflight_20260601/future_copy_plan.csv`
- Human-review holds:
  `results/verification_substrate_gap/public_supplement_bundle_preflight_20260601/human_review_holds.csv`
- Rows:
  `78`
- Included entries:
  `74`
- Future copy-plan entries:
  `60`
- Human-review holds:
  `14`
- Excluded internal records:
  `4`
- Publication blockers:
  `35`
- Missing included sources:
  `0`
- Missing review artifacts:
  `0`
- Duplicate candidate targets:
  `0`
- Candidate bundle created:
  `False`
- Files copied:
  `False`
- Release-ready after preflight:
  `False`
- Claim scope:
  construction preflight only; no public supplement construction, no
  publication, no Slurm, no generation, no model scoring, no training, no
  public text-only verification claim, no ownership-proof claim.
- Verification:
  targeted bundle-preflight pytest `PASS` with `3` tests passed; full
  `tests/verification_substrate_gap` pytest `PASS` with `63` tests passed.

## Public Supplement Copy / Review Plan

- Summary:
  `results/verification_substrate_gap/public_supplement_copy_review_plan_20260601/copy_review_plan_summary.json`
- Report:
  `results/verification_substrate_gap/public_supplement_copy_review_plan_20260601/copy_review_plan_report.md`
- Copy-command dry run:
  `results/verification_substrate_gap/public_supplement_copy_review_plan_20260601/copy_command_dry_run.csv`
- Reviewer-facing checklist:
  `results/verification_substrate_gap/public_supplement_copy_review_plan_20260601/reviewer_facing_checklist.csv`
- Copy commands plan:
  `results/verification_substrate_gap/public_supplement_copy_review_plan_20260601/copy_commands_plan.txt`
- Copy commands:
  `60`
- Review checklist rows:
  `14`
- Redaction reviews:
  `3`
- Scope-note reviews:
  `10`
- Security reviews:
  `1`
- Missing copy sources:
  `0`
- Existing candidate targets:
  `0`
- Missing review artifacts:
  `0`
- Pending reviews:
  `14`
- Files copied:
  `False`
- Candidate bundle created:
  `False`
- Release-ready after plan:
  `False`
- Claim scope:
  copy/review plan only; no public supplement construction, no publication,
  no Slurm, no generation, no model scoring, no training, no public text-only
  verification claim, no ownership-proof claim.
- Verification:
  targeted copy-review pytest `PASS` with `3` tests passed; full
  `tests/verification_substrate_gap` pytest `PASS` with `66` tests passed.

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
