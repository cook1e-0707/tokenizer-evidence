# VSG Release Staging Plan

This artifact-only plan maps release-boundary decisions to a future
public-supplement layout. It does not copy files, create a supplement
bundle, publish artifacts, start compute, or expand claim scope.

Status: `PASS_VSG_RELEASE_STAGING_PLAN_RECORDED_PLAN_ONLY`
Rows: `78`
Direct include candidates: `39`
Stage/copy candidates: `21`
Redacted derivative candidates: `3`
Scope-note gated candidates: `10`
Security-review gated candidates: `1`
Execution-required rows: `35`
Manual-review-required rows: `14`
Excluded internal records: `4`
Duplicate planned targets: `0`
Release-ready after this plan: `False`

## Staging Decisions

| Staging decision | Rows |
| --- | ---: |
| direct_include_candidate | 39 |
| excluded_internal_record | 4 |
| redacted_derivative_candidate | 3 |
| scope_note_gated_candidate | 10 |
| security_review_gated_candidate | 1 |
| stage_or_copy_candidate | 21 |

## Required Before Bundle Construction

- Execute copy/commit decisions for stage-or-copy candidates.
- Create redacted derivatives for private-path trace summaries.
- Attach scope notes or derived summaries for source-mismatch and local-pilot artifacts.
- Complete security review of key/HMAC-related configuration field names.
- Keep internal handoff and state records outside the public supplement.
