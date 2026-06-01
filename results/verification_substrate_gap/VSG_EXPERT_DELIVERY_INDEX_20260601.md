# VSG Expert Delivery Index - 2026-06-01

## Purpose

This index is the single entry point for the current Verification Substrate Gap
expert-review handoff. It lists the objective packet, verification artifacts,
and reproduction commands. It does not add questions, recommendations, new
experiments, or new claims.

## Canonical Handoff State

```text
VSG_EXPERT_REVIEW_PACKET_DELIVERED_WAITING_FOR_REVIEW_NO_NEW_EXPERIMENTS
```

State record:

```text
results/verification_substrate_gap/VSG_CURRENT_HANDOFF_STATE_20260601.md
results/verification_substrate_gap/VSG_CURRENT_HANDOFF_STATE_20260601.json
```

## Expert Packet

```text
results/verification_substrate_gap/vsg_expert_review_packet_20260531.zip
```

Zip SHA256:

```text
0c4d15c058960f2d242f8708be925ccf58c2e43fbf1d55cba6ce4f210ff6884f
```

External Chinese README:

```text
results/verification_substrate_gap/vsg_expert_review_packet_20260531_README.txt
```

## Packet Contents To Review

Inside the zip:

```text
manuscript/VSG_manuscript_snapshot_20260531.pdf
manuscript_source/
evidence/figure_data/
evidence/visual_drafts/
validation/
EXPERT_REVIEW_SCOPE_20260531.md
OBJECTIVE_FACTS_20260531.md
packet_manifest.json
```

## Verification Artifacts

```text
results/verification_substrate_gap/expert_review_packet_verification_20260531/packet_verification_report.md
results/verification_substrate_gap/expert_review_packet_verification_20260531/packet_verification_summary.json
results/verification_substrate_gap/expert_handoff_audit_20260531/handoff_audit_report.md
results/verification_substrate_gap/expert_handoff_audit_20260531/handoff_audit_summary.json
```

## Verification Commands

```bash
python3 scripts/verification_substrate_gap/verify_vsg_expert_review_packet.py
python3 scripts/verification_substrate_gap/audit_vsg_expert_handoff.py
unzip -t results/verification_substrate_gap/vsg_expert_review_packet_20260531.zip
shasum -a 256 results/verification_substrate_gap/vsg_expert_review_packet_20260531.zip
cat results/verification_substrate_gap/vsg_expert_review_packet_20260531.zip.sha256
```

Expected verification facts:

```text
packet verifier: PASS
handoff audit: PASS
packet total file count: 60
hashed file count: 59
claim-scope lint: PASS, 17 files, 0 violations
LaTeX log scan: PASS
overfull hbox warnings: 0
manuscript head: 64510b9daf88deb2efd49a26c8046a023fa4904e
```

## Claim Boundary

Current packet scope:

```text
trace-bound first-divergence results: provider-side diagnostics only
public final-text predicates: observability and spoofing diagnostics only
source-mismatch accepts: spoofing evidence only
public final-text codeword recovered blocks: 0
```

Claims not made:

```text
public text-only verification success
natural evidence success
phrase-decoder success
cryptographic provenance
sanitizer robustness
payload diversity
model-family general verification
ownership proof
```

## Current Gate

```text
Allowed:
  wait for expert review
  artifact-only manuscript/package hygiene that does not alter claim boundary

Not allowed without a new expert/human route decision:
  new Slurm submission
  new generation
  new model scoring
  new training
  new public text-only verification success claim
  new natural evidence success claim
  new ownership proof claim
  Overleaf push
```

## Git Note

This index intentionally does not record the commit hash that contains itself.
Use `git log --oneline -5` in the root repository for the latest state-record
commit.
