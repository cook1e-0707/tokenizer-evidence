# VSG Public Supplement Review Decision Validation

This artifact-only validator checks future human review decision records.
It does not create approvals, perform human review, copy files, build a
candidate supplement, publish artifacts, start compute, or expand claim
scope.

Status: `PASS_VSG_PUBLIC_SUPPLEMENT_REVIEW_DECISIONS_VALIDATED_PENDING_ONLY`
Decision rows: `14`
Pending decisions: `14`
Approved decisions: `0`
Rejected decisions: `0`
Hold decisions: `0`
Invalid decisions: `0`
Release-ready after validation: `False`

## Boundary

- Pending template rows remain valid only when reviewer and approval fields stay empty.
- Approved rows require reviewer identity, UTC review timestamp, hash verification, assertion confirmation, claim-scope preservation, and validated approval gate.
- Rejected and hold rows require reviewer identity, UTC review timestamp, and hash-verification fields, but do not release artifacts.
- Public text-only verification success and ownership-proof claims remain disallowed.
