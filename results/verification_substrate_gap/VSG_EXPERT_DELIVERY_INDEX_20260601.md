# VSG Expert Delivery Index - 2026-06-01

## Purpose

This index is the single entry point for the current Verification Substrate Gap
expert-review handoff. It lists the objective packet, verification artifacts,
and reproduction commands. It does not add questions, recommendations, new
experiments, or new claims.

## Canonical Handoff State

```text
VSG_EXPERT_REVIEW_PACKET_20260601_REFRESHED_HARDENING_INCLUDED_NO_NEW_EXPERIMENTS
```

State record:

```text
results/verification_substrate_gap/VSG_CURRENT_HANDOFF_STATE_20260601.md
results/verification_substrate_gap/VSG_CURRENT_HANDOFF_STATE_20260601.json
```

## Expert Packet

```text
results/verification_substrate_gap/vsg_expert_review_packet_20260601.zip
```

Zip SHA256:

```text
82b4007525b3d213bc4920b6b4bd947a7de002fdcf2d9271cc5543a2c32418e8
```

External Chinese README:

```text
results/verification_substrate_gap/vsg_expert_review_packet_20260601_README.txt
```

## Packet Contents To Review

Inside the zip:

```text
manuscript/VSG_manuscript_snapshot_20260601.pdf
manuscript_source/
evidence/figure_data/
evidence/visual_drafts/
evidence/hardening/
validation/
EXPERT_REVIEW_SCOPE_20260601.md
OBJECTIVE_FACTS_20260601.md
HARDENING_STATUS_20260601.md
packet_manifest.json
```

## Verification Artifacts

```text
results/verification_substrate_gap/expert_review_packet_verification_20260601/packet_verification_report.md
results/verification_substrate_gap/expert_review_packet_verification_20260601/packet_verification_summary.json
results/verification_substrate_gap/expert_review_packet_verification_20260601/handoff_audit_report.md
results/verification_substrate_gap/expert_review_packet_verification_20260601/handoff_audit_summary.json
```

## Verification Commands

```bash
python3 scripts/verification_substrate_gap/verify_vsg_expert_review_packet_20260601.py
python3 scripts/verification_substrate_gap/audit_vsg_expert_handoff_20260601.py
unzip -t results/verification_substrate_gap/vsg_expert_review_packet_20260601.zip
shasum -a 256 results/verification_substrate_gap/vsg_expert_review_packet_20260601.zip
cat results/verification_substrate_gap/vsg_expert_review_packet_20260601.zip.sha256
```

Expected verification facts:

```text
packet verifier: PASS
handoff audit: PASS
packet total file count: 87
hashed file count: 86
claim-scope lint: PASS, 17 files, 0 violations
LaTeX log scan: PASS
overfull hbox warnings: 0
manuscript head: c10b3f1e73689d63ceb0a4b3b8ea980974df16c1
packet PDF sha256: a64c984fac6503b20138805c8a9a323799f6feb1acfdcc1f7bb7310237f5a0fa
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
