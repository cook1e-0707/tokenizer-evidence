# VSG Public Supplement Readiness Audit

This artifact-only audit checks whether the supplement staging blockers
are covered by review derivatives. It does not copy files into a public
supplement, publish artifacts, start compute, or expand claim scope.

Status: `PASS_VSG_PUBLIC_SUPPLEMENT_READINESS_AUDIT_RECORDED_REVIEW_REQUIRED`
Rows: `78`
Direct include candidates: `39`
Stage/copy still required: `21`
Derivative-required rows: `14`
Derivative-covered rows: `14`
Derivative-uncovered rows: `0`
Manual review still required: `14`
Publication blockers: `35`
Release-ready now: `False`

## Readiness Decisions

| Decision | Rows |
| --- | ---: |
| copy_or_commit_required_before_supplement_bundle | 21 |
| excluded_from_public_supplement | 4 |
| ready_for_final_license_scope_review | 39 |
| redacted_derivative_available_manual_review_required | 3 |
| scope_note_available_manual_review_required | 10 |
| security_review_available_manual_review_required | 1 |

## Remaining Work Before A Public Supplement

- Copy or commit the 21 stage/copy candidates into a reviewed bundle.
- Human-review the 14 derivative-covered rows before inclusion.
- Keep the 4 internal handoff records excluded from the public supplement.
- Do not claim public text-only verification success or ownership proof.
