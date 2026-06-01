# VSG Release Boundary Audit

This artifact-only audit converts the reproducibility release inventory
into a public-supplement boundary decision table. It does not publish
files, copy raw artifacts, start compute, or expand the manuscript claim
boundary.

Status: `PASS_VSG_RELEASE_BOUNDARY_AUDIT_RECORDED_REVIEW_REQUIRED`
Rows: `78`
Ready for reviewed public supplement: `39`
Excluded from public supplement: `4`
Pre-release review required: `35`
Release blockers: `35`
Release-ready now: `False`

## Boundary Decisions

| Decision | Rows |
| --- | ---: |
| exclude_from_public_supplement | 4 |
| ready_for_reviewed_public_supplement | 39 |
| redact_or_summarize_before_release | 3 |
| scope_review_before_release | 10 |
| security_review_before_release | 1 |
| stage_or_copy_to_supplement_before_release | 21 |

## Required Before Public Supplement Release

- Redact or summarize files with private path markers.
- Review files with key/HMAC-related field names.
- Decide whether untracked candidate files are committed, copied into a reviewed supplement bundle, or excluded.
- Keep internal handoff/state records outside the public supplement unless explicitly approved.

## Claim Scope

This audit preserves the current VSG claim boundary: trace-bound
first-divergence results remain provider-side diagnostics; public
final-text predicates remain observability/spoofing diagnostics;
source-mismatch accepts are not protected success and not codeword
recovery.
