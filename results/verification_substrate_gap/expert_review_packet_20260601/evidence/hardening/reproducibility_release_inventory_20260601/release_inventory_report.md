# VSG Reproducibility Release Inventory

This artifact-only inventory identifies candidate files for a future
supplemental release. It does not publish the files and does not create
a public text-only verification or ownership-proof claim.

Status: `PASS_VSG_REPRODUCIBILITY_RELEASE_INVENTORY_RECORDED_REVIEW_REQUIRED`
Rows: `78`
Existing files: `78`
Missing files: `0`
Existing files not tracked by selected git scopes: `31`
Rows requiring anonymization/scope review: `18`
Private path hits: `3`
Secret-term hits: `1`
Release-ready without review: `False`

## Groups

| Group | Rows |
| --- | ---: |
| attack_naturalness_proxy_audit | 4 |
| figure_data | 8 |
| manuscript_figures | 5 |
| manuscript_source | 19 |
| ownership_stress_test | 2 |
| public_predicate_attack_ladder | 5 |
| public_text_verifier_baselines | 4 |
| reproducibility_code | 6 |
| reproducibility_config | 5 |
| reproducibility_tests | 5 |
| state_and_scope_records | 4 |
| stronger_public_predicate_local_pilot | 4 |
| substrate_matrix | 3 |
| trace_bound_corpus_summary | 4 |

## Required Follow-Up Before Public Release

- Resolve missing files or mark them intentionally out of scope.
- Scrub private cluster/local paths from release candidates.
- Review any files containing key/HMAC-related field names before release.
- Decide whether untracked-but-existing files should be committed, copied into a release bundle, or excluded.

This inventory is compatible with the current VSG claim boundary: trace-bound
results remain provider-side diagnostics; public final-text predicates remain
observability/spoofing diagnostics; source-mismatch accepts are not protected
success and not codeword recovery.
