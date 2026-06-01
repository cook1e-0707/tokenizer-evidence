# VSG Public Supplement Human Review Packet

This packet is artifact-only. It lists pending review rows, hashes,
review artifacts, claim guards, and required reviewer assertions. It does
not approve any row, copy files, create a public supplement, publish
artifacts, start compute, or expand claim scope.

## PSR-001 - redaction_review

- Entry: `PSP-033`
- Blocker: `PSB-009`
- Artifact group: `trace_bound_corpus_summary`
- Source: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/evidence/trace_bound_controllability_redacted/combined_blocks.csv`
- Source SHA256: `e9aaea586d2e019f0a0a85c2f73f43868f577416e4c927d8c9aa804f70e364be`
- Planned supplement path: `evidence/trace_bound_controllability_redacted/combined_blocks.csv`
- Review artifact: `results/verification_substrate_gap/corpora/trace_bound_controllability/combined_blocks.csv`
- Review artifact SHA256: `35e0373d126df82b3f8024804435b1564b774f2245d292ee347329c7837c4938`
- Required evidence: human reviewer confirms redacted derivative removes private fields and preserves trace-bound-only claim scope
- Reviewer assertion required: approve redacted derivative; confirm private fields removed and trace-bound-only scope preserved
- Approval status: `pending_not_performed`
- Claim scope guard: provider-side trace-bound diagnostic summary only; not public text-only verification
- Redaction dropped fields: `source_shard_dir`
- Private marker hits after redaction: `0`

## PSR-002 - redaction_review

- Entry: `PSP-034`
- Blocker: `PSB-010`
- Artifact group: `trace_bound_corpus_summary`
- Source: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/evidence/trace_bound_controllability_redacted/qwen_blocks.csv`
- Source SHA256: `51c5f39d1aa8910a7f109e8a4443d65e3a16b08d113c7c27ba04a8ae93cd3105`
- Planned supplement path: `evidence/trace_bound_controllability_redacted/qwen_blocks.csv`
- Review artifact: `results/verification_substrate_gap/corpora/trace_bound_controllability/qwen_blocks.csv`
- Review artifact SHA256: `53ab77ddba1b2f0e77e0972bd5d511465e9bc4190e03f93785945d9958a53681`
- Required evidence: human reviewer confirms redacted derivative removes private fields and preserves trace-bound-only claim scope
- Reviewer assertion required: approve redacted derivative; confirm private fields removed and trace-bound-only scope preserved
- Approval status: `pending_not_performed`
- Claim scope guard: provider-side trace-bound diagnostic summary only; not public text-only verification
- Redaction dropped fields: `source_shard_dir`
- Private marker hits after redaction: `0`

## PSR-003 - redaction_review

- Entry: `PSP-035`
- Blocker: `PSB-011`
- Artifact group: `trace_bound_corpus_summary`
- Source: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/evidence/trace_bound_controllability_redacted/llama_blocks.csv`
- Source SHA256: `2a017df490422f42df115ae13312a27932d6bf04e303d1b9d36a2a3641f28d50`
- Planned supplement path: `evidence/trace_bound_controllability_redacted/llama_blocks.csv`
- Review artifact: `results/verification_substrate_gap/corpora/trace_bound_controllability/llama_blocks.csv`
- Review artifact SHA256: `d02bd77465b3ddd37c11b9afea9800a762c4cc2f5940fe712cd609bccde4112c`
- Required evidence: human reviewer confirms redacted derivative removes private fields and preserves trace-bound-only claim scope
- Reviewer assertion required: approve redacted derivative; confirm private fields removed and trace-bound-only scope preserved
- Approval status: `pending_not_performed`
- Claim scope guard: provider-side trace-bound diagnostic summary only; not public text-only verification
- Redaction dropped fields: `source_shard_dir`
- Private marker hits after redaction: `0`

## PSR-004 - scope_note_review

- Entry: `PSP-036`
- Blocker: `PSB-012`
- Artifact group: `trace_bound_corpus_summary`
- Source: `results/verification_substrate_gap/corpora/trace_bound_controllability/corpus_manifest.json`
- Source SHA256: `2c6da64f505f21d271210020c0abaa46d189d7c5cbcb5dd6923dbaec9ace6eae`
- Planned supplement path: `evidence/trace_bound_controllability/corpus_manifest.json`
- Review artifact: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv`
- Review artifact SHA256: `cea0848b33f227f580c4702db0a6f803ea553b9fcceaee7c3ad0165e11be7889`
- Required evidence: human reviewer approves source with scope note and confirms source-mismatch/non-claim wording
- Reviewer assertion required: approve scope note; confirm artifact remains source-mismatch or non-claim evidence only
- Approval status: `pending_not_performed`
- Claim scope guard: public text-only verification success; ownership proof; public final-text codeword recovery
- Allowed interpretation: provider-side trace-bound diagnostic summary only; not public text-only verification
- Forbidden claims: public text-only verification success; ownership proof; public final-text codeword recovery

## PSR-005 - scope_note_review

- Entry: `PSP-041`
- Blocker: `PSB-017`
- Artifact group: `public_predicate_attack_ladder`
- Source: `results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/surrogate_guided_rewrite_curve.csv`
- Source SHA256: `fcf9babeeb9a109ef1c5895c6aa06757a919ad897a6ab1f9eb333000f4cede0e`
- Planned supplement path: `evidence/public_predicate_attack_ladder_scope_limited/surrogate_guided_rewrite_curve.csv`
- Review artifact: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv`
- Review artifact SHA256: `cea0848b33f227f580c4702db0a6f803ea553b9fcceaee7c3ad0165e11be7889`
- Required evidence: human reviewer approves source with scope note and confirms source-mismatch/non-claim wording
- Reviewer assertion required: approve scope note; confirm artifact remains source-mismatch or non-claim evidence only
- Approval status: `pending_not_performed`
- Claim scope guard: public text-only verification success; ownership proof; protected success; codeword recovery; naturalness-preserving rewrite
- Allowed interpretation: source-mismatch spoofing evidence only; not protected success and not codeword recovery
- Forbidden claims: public text-only verification success; ownership proof; protected success; codeword recovery; naturalness-preserving rewrite

## PSR-006 - scope_note_review

- Entry: `PSP-042`
- Blocker: `PSB-018`
- Artifact group: `public_predicate_attack_ladder`
- Source: `results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/surrogate_guided_transform_summary.csv`
- Source SHA256: `afe93deb1cfdfb66c1326730f959e49446d4ccf4457640879de00b665bd83cc7`
- Planned supplement path: `evidence/public_predicate_attack_ladder_scope_limited/surrogate_guided_transform_summary.csv`
- Review artifact: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv`
- Review artifact SHA256: `cea0848b33f227f580c4702db0a6f803ea553b9fcceaee7c3ad0165e11be7889`
- Required evidence: human reviewer approves source with scope note and confirms source-mismatch/non-claim wording
- Reviewer assertion required: approve scope note; confirm artifact remains source-mismatch or non-claim evidence only
- Approval status: `pending_not_performed`
- Claim scope guard: public text-only verification success; ownership proof; protected success; codeword recovery; naturalness-preserving rewrite
- Allowed interpretation: source-mismatch spoofing evidence only; not protected success and not codeword recovery
- Forbidden claims: public text-only verification success; ownership proof; protected success; codeword recovery; naturalness-preserving rewrite

## PSR-007 - scope_note_review

- Entry: `PSP-043`
- Blocker: `PSB-019`
- Artifact group: `public_predicate_attack_ladder`
- Source: `results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/surrogate_guided_rewrite_examples.csv`
- Source SHA256: `e5e13d20aa816030e11324a57e9f78545dc083bdf752d60f88331e22c0d48ef0`
- Planned supplement path: `evidence/public_predicate_attack_ladder_scope_limited/surrogate_guided_rewrite_examples.csv`
- Review artifact: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv`
- Review artifact SHA256: `cea0848b33f227f580c4702db0a6f803ea553b9fcceaee7c3ad0165e11be7889`
- Required evidence: human reviewer approves source with scope note and confirms source-mismatch/non-claim wording
- Reviewer assertion required: approve scope note; confirm artifact remains source-mismatch or non-claim evidence only
- Approval status: `pending_not_performed`
- Claim scope guard: public text-only verification success; ownership proof; protected success; codeword recovery; naturalness-preserving rewrite
- Allowed interpretation: source-mismatch spoofing evidence only; not protected success and not codeword recovery
- Forbidden claims: public text-only verification success; ownership proof; protected success; codeword recovery; naturalness-preserving rewrite

## PSR-008 - scope_note_review

- Entry: `PSP-044`
- Blocker: `PSB-020`
- Artifact group: `public_predicate_attack_ladder`
- Source: `results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/surrogate_guided_rewrite_summary.json`
- Source SHA256: `a322d0493f8b6daedf5dcd57230cf0b6a92d2e610a1a2a4b726d177142d8966e`
- Planned supplement path: `evidence/public_predicate_attack_ladder_scope_limited/surrogate_guided_rewrite_summary.json`
- Review artifact: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv`
- Review artifact SHA256: `cea0848b33f227f580c4702db0a6f803ea553b9fcceaee7c3ad0165e11be7889`
- Required evidence: human reviewer approves source with scope note and confirms source-mismatch/non-claim wording
- Reviewer assertion required: approve scope note; confirm artifact remains source-mismatch or non-claim evidence only
- Approval status: `pending_not_performed`
- Claim scope guard: public text-only verification success; ownership proof; protected success; codeword recovery; naturalness-preserving rewrite
- Allowed interpretation: source-mismatch spoofing evidence only; not protected success and not codeword recovery
- Forbidden claims: public text-only verification success; ownership proof; protected success; codeword recovery; naturalness-preserving rewrite

## PSR-009 - scope_note_review

- Entry: `PSP-045`
- Blocker: `PSB-021`
- Artifact group: `public_predicate_attack_ladder`
- Source: `results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/surrogate_guided_rewrite_report.md`
- Source SHA256: `936dccd48337f0badad5307ee25eeb4616606d1cfe7df06e8d041148e2703c4a`
- Planned supplement path: `evidence/public_predicate_attack_ladder_scope_limited/surrogate_guided_rewrite_report.md`
- Review artifact: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv`
- Review artifact SHA256: `cea0848b33f227f580c4702db0a6f803ea553b9fcceaee7c3ad0165e11be7889`
- Required evidence: human reviewer approves source with scope note and confirms source-mismatch/non-claim wording
- Reviewer assertion required: approve scope note; confirm artifact remains source-mismatch or non-claim evidence only
- Approval status: `pending_not_performed`
- Claim scope guard: public text-only verification success; ownership proof; protected success; codeword recovery; naturalness-preserving rewrite
- Allowed interpretation: source-mismatch spoofing evidence only; not protected success and not codeword recovery
- Forbidden claims: public text-only verification success; ownership proof; protected success; codeword recovery; naturalness-preserving rewrite

## PSR-010 - scope_note_review

- Entry: `PSP-050`
- Blocker: `PSB-022`
- Artifact group: `stronger_public_predicate_local_pilot`
- Source: `results/verification_substrate_gap/public_text_verifier_stronger_local_pilot_20260601/public_text_verifier_results.csv`
- Source SHA256: `447401ad936f1164daecbbdd2192f9c139088cc8d41649349f78e1dfa035d03b`
- Planned supplement path: `evidence/local_pilots/stronger_public_predicate/public_text_verifier_results.csv`
- Review artifact: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv`
- Review artifact SHA256: `cea0848b33f227f580c4702db0a6f803ea553b9fcceaee7c3ad0165e11be7889`
- Required evidence: human reviewer approves source with scope note and confirms source-mismatch/non-claim wording
- Reviewer assertion required: approve scope note; confirm artifact remains source-mismatch or non-claim evidence only
- Approval status: `pending_not_performed`
- Claim scope guard: public text-only verification success; ownership proof; adopted locked evidence; paper-facing final-text claim
- Allowed interpretation: local non-adopted/historical pilot only; not adopted locked evidence
- Forbidden claims: public text-only verification success; ownership proof; adopted locked evidence; paper-facing final-text claim

## PSR-011 - scope_note_review

- Entry: `PSP-051`
- Blocker: `PSB-023`
- Artifact group: `stronger_public_predicate_local_pilot`
- Source: `results/verification_substrate_gap/public_text_verifier_stronger_local_pilot_20260601/public_text_verifier_block_scores.csv`
- Source SHA256: `7c475543f2b1a3a3c3646dd3e7ea9340367f0a419a69ce657b1116822cadc297`
- Planned supplement path: `evidence/local_pilots/stronger_public_predicate/public_text_verifier_block_scores.csv`
- Review artifact: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv`
- Review artifact SHA256: `cea0848b33f227f580c4702db0a6f803ea553b9fcceaee7c3ad0165e11be7889`
- Required evidence: human reviewer approves source with scope note and confirms source-mismatch/non-claim wording
- Reviewer assertion required: approve scope note; confirm artifact remains source-mismatch or non-claim evidence only
- Approval status: `pending_not_performed`
- Claim scope guard: public text-only verification success; ownership proof; adopted locked evidence; paper-facing final-text claim
- Allowed interpretation: local non-adopted/historical pilot only; not adopted locked evidence
- Forbidden claims: public text-only verification success; ownership proof; adopted locked evidence; paper-facing final-text claim

## PSR-012 - scope_note_review

- Entry: `PSP-052`
- Blocker: `PSB-024`
- Artifact group: `stronger_public_predicate_local_pilot`
- Source: `results/verification_substrate_gap/public_text_verifier_stronger_local_pilot_20260601/public_text_verifier_summary.json`
- Source SHA256: `1f0faea341defd8509e36d9597883ce5b6253f9ed439807a04e87823846cc1e0`
- Planned supplement path: `evidence/local_pilots/stronger_public_predicate/public_text_verifier_summary.json`
- Review artifact: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv`
- Review artifact SHA256: `cea0848b33f227f580c4702db0a6f803ea553b9fcceaee7c3ad0165e11be7889`
- Required evidence: human reviewer approves source with scope note and confirms source-mismatch/non-claim wording
- Reviewer assertion required: approve scope note; confirm artifact remains source-mismatch or non-claim evidence only
- Approval status: `pending_not_performed`
- Claim scope guard: public text-only verification success; ownership proof; adopted locked evidence; paper-facing final-text claim
- Allowed interpretation: local non-adopted/historical pilot only; not adopted locked evidence
- Forbidden claims: public text-only verification success; ownership proof; adopted locked evidence; paper-facing final-text claim

## PSR-013 - scope_note_review

- Entry: `PSP-053`
- Blocker: `PSB-025`
- Artifact group: `stronger_public_predicate_local_pilot`
- Source: `results/verification_substrate_gap/public_text_verifier_stronger_local_pilot_20260601/public_text_verifier_report.md`
- Source SHA256: `6cafeb929bac19cb63939a6d6a999a16c3bf0a33afac2e5247aef63a4eb95a60`
- Planned supplement path: `evidence/local_pilots/stronger_public_predicate/public_text_verifier_report.md`
- Review artifact: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/scope_notes.csv`
- Review artifact SHA256: `cea0848b33f227f580c4702db0a6f803ea553b9fcceaee7c3ad0165e11be7889`
- Required evidence: human reviewer approves source with scope note and confirms source-mismatch/non-claim wording
- Reviewer assertion required: approve scope note; confirm artifact remains source-mismatch or non-claim evidence only
- Approval status: `pending_not_performed`
- Claim scope guard: public text-only verification success; ownership proof; adopted locked evidence; paper-facing final-text claim
- Allowed interpretation: local non-adopted/historical pilot only; not adopted locked evidence
- Forbidden claims: public text-only verification success; ownership proof; adopted locked evidence; paper-facing final-text claim

## PSR-014 - security_review

- Entry: `PSP-067`
- Blocker: `PSB-033`
- Artifact group: `reproducibility_config`
- Source: `configs/verification_substrate_gap/text_only_observability.yaml`
- Source SHA256: `6c4e0dc072c272617547f9078cb8f23fa5a2fe8d0e256bb6a4643dd5654c545c`
- Planned supplement path: `configs/text_only_observability.yaml`
- Review artifact: `results/verification_substrate_gap/public_supplement_review_derivatives_20260601/security_review_text_only_observability.json`
- Review artifact SHA256: `42267af772a7ea51fdbd71882ee2b4e89e8815d80abff66e1d00f3bf7cb86bb1`
- Required evidence: human reviewer confirms security review has no secret values and field names are acceptable for release
- Reviewer assertion required: approve security review; confirm no secret values and acceptable schema/field-name exposure
- Approval status: `pending_not_performed`
- Claim scope guard: preserves VSG substrate-gap claim boundary
- Release recommendation: `schema_field_review_required_no_literal_secret_values_detected`
- Security field-name hits: `8`
- Security secret-value hits: `0`
