# VSG Expert Reply Hardening Decomposition - 2026-06-01

Status: `PASS_EXPERT_REPLY_DECOMPOSED_ARTIFACT_ONLY_HARDENING_IN_PROGRESS`

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

## Additional Work Completed In Later Continuations

Implemented stronger public final-text predicate baselines and ran a local
pilot on available non-adopted/historical text artifacts:

```text
scripts/verification_substrate_gap/evaluate_public_text_verifier.py
configs/verification_substrate_gap/public_text_verifier_stronger_local_pilot.yaml
tests/verification_substrate_gap/test_public_text_verifier_stronger_baselines.py
results/verification_substrate_gap/public_text_verifier_stronger_local_pilot_20260601/
```

Observed scope:

```text
codeword_recovered_blocks = 0
adopted_locked_evidence_updated = false
public_text_only_verification_claim_allowed = false
```

Audited the existing adopted-locked guided rewrite/graft attack examples with
deterministic readability proxies:

```text
scripts/verification_substrate_gap/audit_public_predicate_attack_naturalness.py
tests/verification_substrate_gap/test_public_predicate_attack_naturalness_audit.py
results/verification_substrate_gap/public_predicate_attack_naturalness_audit_20260601/
```

Observed scope:

```text
rows = 60
proxy_readable_rows = 0
semantic_naturalness_claimed = false
protected_success_claimed = false
codeword_recovery_claimed = false
```

Recorded a machine-checkable reproducibility release inventory:

```text
scripts/verification_substrate_gap/build_vsg_reproducibility_release_inventory.py
tests/verification_substrate_gap/test_vsg_reproducibility_release_inventory.py
results/verification_substrate_gap/reproducibility_release_inventory_20260601/
```

Observed scope:

```text
rows = 78
missing_files = 0
private_path_hits = 3
secret_term_hits = 1
release_ready_without_review = false
```

Audited the ownership scenario stress-test decision rules:

```text
scripts/verification_substrate_gap/audit_vsg_ownership_scenario_decision_rules.py
tests/verification_substrate_gap/test_vsg_ownership_scenario_decision_rules.py
results/verification_substrate_gap/ownership_scenario_decision_rule_audit_20260601/
```

Observed scope:

```text
rows = 63
scenarios = 7
method_families = 9
rule_failures = 0
supported_trace_bound_rows = 2
supported_public_final_text_rows = 0
```

Cleaned active manuscript prose to replace internal audit-style language with
academic scope statements:

```text
manuscripts/69db2644566dcc36c9da320e/section_01_introduction.tex
manuscripts/69db2644566dcc36c9da320e/section_02_related_work.tex
manuscripts/69db2644566dcc36c9da320e/section_03_problem_setup.tex
manuscripts/69db2644566dcc36c9da320e/section_05_bucket_level_injection.tex
manuscripts/69db2644566dcc36c9da320e/section_08_discussion_limitations.tex
manuscripts/69db2644566dcc36c9da320e/appendix/extended_related_work.tex
manuscripts/69db2644566dcc36c9da320e/appendix/asset_licenses.tex
```

Observed scope:

```text
local_manuscript_commit = c10b3f1e73689d63ceb0a4b3b8ea980974df16c1
local_pdf_sha256 = a64c984fac6503b20138805c8a9a323799f6feb1acfdcc1f7bb7310237f5a0fa
claim_scope_lint = PASS
latex_build = PASS
latex_log_risk_scan_matches = 0
overleaf_push_performed = false
full tests/verification_substrate_gap pytest = PASS, 37 passed
```

Audited active manuscript figure quality and data traceability:

```text
scripts/verification_substrate_gap/audit_vsg_manuscript_figure_quality.py
tests/verification_substrate_gap/test_vsg_manuscript_figure_quality_audit.py
results/verification_substrate_gap/manuscript_figure_quality_audit_20260601/
```

Observed scope:

```text
figures_checked = 5
failed_figure_checks = 0
data_traceability_checks = 5
failed_data_traceability_checks = 0
figure_3_trace_bound_counts = Qwen 94/96, Llama 96/96
figure_3_public_codeword_recovery = 0
figure_4_guided_rewrite_graft_top100 = 100/100 for all plotted groups
figure_5_matrix_shape = 7 x 9 = 63
figure_5_supported_public_final_text_rows = 0
```

Refreshed the expert review packet so the delivered packet now includes the
2026-06-01 hardened manuscript snapshot and the new hardening outputs:

```text
scripts/verification_substrate_gap/build_vsg_expert_review_packet_20260601.py
scripts/verification_substrate_gap/verify_vsg_expert_review_packet_20260601.py
scripts/verification_substrate_gap/audit_vsg_expert_handoff_20260601.py
tests/verification_substrate_gap/test_vsg_expert_packet_20260601.py
results/verification_substrate_gap/expert_review_packet_20260601/
results/verification_substrate_gap/vsg_expert_review_packet_20260601.zip
results/verification_substrate_gap/vsg_expert_review_packet_20260601_README.txt
results/verification_substrate_gap/expert_review_packet_verification_20260601/
```

Observed refreshed packet scope:

```text
packet_total_file_count = 87
hashed_file_count = 86
zip_sha256 = 82b4007525b3d213bc4920b6b4bd947a7de002fdcf2d9271cc5543a2c32418e8
packet_verifier = PASS
handoff_audit = PASS
zip_integrity = PASS
packet_manuscript_head = c10b3f1e73689d63ceb0a4b3b8ea980974df16c1
packet_pdf_sha256 = a64c984fac6503b20138805c8a9a323799f6feb1acfdcc1f7bb7310237f5a0fa
overleaf_push_performed = false
```

Recorded a public-supplement release-boundary audit from the reproducibility
release inventory:

```text
scripts/verification_substrate_gap/build_vsg_release_boundary_audit.py
tests/verification_substrate_gap/test_vsg_release_boundary_audit.py
results/verification_substrate_gap/reproducibility_release_boundary_audit_20260601/
```

Observed scope:

```text
rows = 78
ready_for_reviewed_public_supplement = 39
excluded_from_public_supplement = 4
pre_release_review_required = 35
release_blockers = 35
release_ready_now = false
publication_performed = false
new_slurm_started = false
generation_started = false
model_scoring_started = false
training_started = false
public_text_only_verification_claimed = false
ownership_proof_claimed = false
targeted release/reproducibility tests = PASS, 9 passed
```

Recorded a plan-only public-supplement staging map from the release-boundary
audit:

```text
scripts/verification_substrate_gap/build_vsg_release_staging_plan.py
tests/verification_substrate_gap/test_vsg_release_staging_plan.py
results/verification_substrate_gap/reproducibility_release_staging_plan_20260601/
```

Observed scope:

```text
rows = 78
direct_include_candidates = 39
stage_or_copy_candidates = 21
redacted_derivative_candidates = 3
scope_note_gated_candidates = 10
security_review_gated_candidates = 1
excluded_internal_records = 4
execution_required_rows = 35
manual_review_required_rows = 14
duplicate_planned_targets = 0
release_ready_after_plan = false
files_copied = false
public_supplement_created = false
publication_performed = false
new_slurm_started = false
generation_started = false
model_scoring_started = false
training_started = false
public_text_only_verification_claimed = false
ownership_proof_claimed = false
targeted release-staging tests = PASS, 4 passed
```

Created artifact-only public-supplement review derivatives from the staging
plan:

```text
scripts/verification_substrate_gap/build_vsg_public_supplement_review_derivatives.py
tests/verification_substrate_gap/test_vsg_public_supplement_review_derivatives.py
results/verification_substrate_gap/public_supplement_review_derivatives_20260601/
```

Observed scope:

```text
redacted_csv_written_count = 3
redacted_rows_total = 1920
dropped_private_path_field = source_shard_dir
private_marker_hits_after_redaction = 0
scope_note_count = 10
security_review_count = 1
security_field_name_hit_count = 8
security_secret_value_hit_count = 0
source_files_copied_without_transform = false
public_supplement_created = false
publication_performed = false
release_ready_after_derivatives = false
new_slurm_started = false
generation_started = false
model_scoring_started = false
training_started = false
public_text_only_verification_claimed = false
ownership_proof_claimed = false
targeted public-supplement review-derivative tests = PASS, 4 passed
full tests/verification_substrate_gap pytest = PASS, 49 passed
```

Recorded a public-supplement readiness re-audit from the staging plan and
review derivatives:

```text
scripts/verification_substrate_gap/build_vsg_public_supplement_readiness_audit.py
tests/verification_substrate_gap/test_vsg_public_supplement_readiness_audit.py
results/verification_substrate_gap/public_supplement_readiness_audit_20260601/
```

Observed scope:

```text
rows = 78
direct_include_candidates = 39
stage_or_copy_required = 21
derivative_required = 14
derivative_covered = 14
derivative_uncovered = 0
manual_review_required_after_derivatives = 14
publication_blockers = 35
excluded_internal_records = 4
release_ready_now = false
public_supplement_created = false
publication_performed = false
new_slurm_started = false
generation_started = false
model_scoring_started = false
training_started = false
public_text_only_verification_claimed = false
ownership_proof_claimed = false
targeted public-supplement readiness tests = PASS, 4 passed
full tests/verification_substrate_gap pytest = PASS, 53 passed
```

Recorded a public-supplement dry-run bundle manifest from the readiness audit:

```text
scripts/verification_substrate_gap/build_vsg_public_supplement_dry_run_manifest.py
tests/verification_substrate_gap/test_vsg_public_supplement_dry_run_manifest.py
results/verification_substrate_gap/public_supplement_dry_run_manifest_20260601/
```

Observed scope:

```text
rows = 78
dry_run_bundle_entries = 74
excluded_internal_records = 4
direct_include_entries = 39
copy_required_entries = 21
redacted_derivative_entries = 3
scope_note_review_entries = 10
security_review_entries = 1
manual_review_required = 14
publication_blockers = 35
missing_sources = 0
missing_review_artifacts = 0
duplicate_planned_targets = 0
release_ready_after_dry_run = false
dry_run_only = true
files_copied = false
public_supplement_created = false
publication_performed = false
new_slurm_started = false
generation_started = false
model_scoring_started = false
training_started = false
public_text_only_verification_claimed = false
ownership_proof_claimed = false
targeted public-supplement dry-run manifest tests = PASS, 4 passed
full tests/verification_substrate_gap pytest = PASS, 57 passed
```

Recorded a public-supplement blocker checklist from the dry-run manifest:

```text
scripts/verification_substrate_gap/build_vsg_public_supplement_blocker_checklist.py
tests/verification_substrate_gap/test_vsg_public_supplement_blocker_checklist.py
results/verification_substrate_gap/public_supplement_blocker_checklist_20260601/
```

Observed scope:

```text
publication_blockers = 35
copy_required = 21
human_review_required = 14
missing_sources = 0
missing_review_artifacts = 0
unclassified_blockers = 0
all_blockers_have_resolution_track = true
blockers_resolved = false
release_ready_after_checklist = false
artifact_only = true
files_copied = false
human_reviews_performed = false
public_supplement_created = false
publication_performed = false
new_slurm_started = false
generation_started = false
model_scoring_started = false
training_started = false
public_text_only_verification_claimed = false
ownership_proof_claimed = false
targeted public-supplement blocker checklist tests = PASS, 3 passed
full tests/verification_substrate_gap pytest = PASS, 60 passed
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

The next allowed route remains artifact-only manuscript/package hygiene and
release-boundary hardening. The current review derivatives reduce the private
path and scope-note blockers, but a future public supplement is still not
declared release-ready and has not been created or published. The current
readiness audit records 35 publication blockers remaining: 21 copy/commit rows
and 14 human-review rows.
