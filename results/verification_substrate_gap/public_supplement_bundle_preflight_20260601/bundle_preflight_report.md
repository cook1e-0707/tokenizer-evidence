# VSG Public Supplement Bundle Construction Preflight

This artifact-only preflight converts the dry-run bundle manifest and
blocker checklist into a future bundle construction plan. It records
candidate target paths, source hashes, copy-plan entries, and human-review
holds. It does not copy files, create the candidate bundle, perform human
review, publish artifacts, start compute, or expand claim scope.

Status: `PASS_VSG_PUBLIC_SUPPLEMENT_BUNDLE_PREFLIGHT_RECORDED_ARTIFACT_ONLY`
Rows: `78`
Included entries: `74`
Future copy-plan entries: `60`
Human-review holds: `14`
Excluded internal records: `4`
Publication blockers: `35`
Missing included sources: `0`
Missing review artifacts: `0`
Duplicate candidate targets: `0`
Candidate bundle created: `False`
Files copied: `False`
Release-ready after preflight: `False`

## Preflight Classes

| Class | Rows |
| --- | ---: |
| copy_required_preflight | 21 |
| direct_include_final_scope_check | 39 |
| excluded_internal_record | 4 |
| human_review_hold | 14 |

## Construction Boundary

- The 60 future copy-plan entries are not copied by this preflight.
- The 14 human-review holds remain blocked until explicit review evidence exists.
- The 4 excluded internal records remain outside the candidate bundle.
- Public text-only verification success and ownership-proof claims remain disallowed.
