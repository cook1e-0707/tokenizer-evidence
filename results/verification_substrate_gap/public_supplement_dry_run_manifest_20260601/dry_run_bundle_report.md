# VSG Public Supplement Dry-Run Bundle Manifest

This artifact-only manifest resolves the readiness audit into a future
bundle construction plan. It records source files, review artifacts,
target supplement paths, and remaining blockers. It does not copy files,
create a public supplement, publish artifacts, start compute, or expand
claim scope.

Status: `PASS_VSG_PUBLIC_SUPPLEMENT_DRY_RUN_MANIFEST_RECORDED_NOT_RELEASE_READY`
Rows: `78`
Dry-run bundle entries: `74`
Excluded internal records: `4`
Direct include entries: `39`
Copy-required entries: `21`
Redacted derivative entries: `3`
Scope-note review entries: `10`
Security-review entries: `1`
Manual-review-required rows: `14`
Publication blockers: `35`
Missing dry-run sources: `0`
Missing review artifacts: `0`
Duplicate planned targets: `0`
Release-ready after dry-run: `False`

## Bundle Actions

| Bundle action | Rows |
| --- | ---: |
| copy_source_to_bundle_after_review | 21 |
| direct_include_after_final_license_scope_review | 39 |
| exclude_internal_record | 4 |
| include_source_after_security_review | 1 |
| include_source_with_scope_note_after_human_review | 10 |
| use_redacted_derivative_after_human_review | 3 |

## Remaining Work Before Actual Bundle Construction

- Copy the 21 copy-required entries into a reviewed supplement layout.
- Human-review the 14 rows that depend on redaction, scope notes, or security review.
- Keep the 4 internal handoff records excluded.
- Preserve the non-claim guards: no public text-only verification success and no ownership proof.
