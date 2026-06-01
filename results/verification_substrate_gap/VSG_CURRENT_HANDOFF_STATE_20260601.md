# VSG Current Handoff State - 2026-06-01

## Canonical Phase

```text
VSG_EXPERT_REVIEW_PACKET_DELIVERED_WAITING_FOR_REVIEW_NO_NEW_EXPERIMENTS
```

## Status

The Verification Substrate Gap paper package has been rewritten, hardened,
packaged, independently verified, handoff-audited, committed, and pushed to
GitHub. The current active state is waiting for expert review of the objective
packet.

This state does not unlock new Slurm jobs, generation, model scoring, training,
or paper-facing positive claims.

## Current Review Packet

- Zip:
  `results/verification_substrate_gap/vsg_expert_review_packet_20260531.zip`
- Zip SHA256:
  `0c4d15c058960f2d242f8708be925ccf58c2e43fbf1d55cba6ce4f210ff6884f`
- External README:
  `results/verification_substrate_gap/vsg_expert_review_packet_20260531_README.txt`
- Packet manifest:
  `results/verification_substrate_gap/expert_review_packet_20260531/packet_manifest.json`
- Packet verifier:
  `scripts/verification_substrate_gap/verify_vsg_expert_review_packet.py`
- Handoff audit:
  `scripts/verification_substrate_gap/audit_vsg_expert_handoff.py`

## Verification Evidence

- Packet verifier status: `PASS`
- Handoff audit status: `PASS`
- Packet file count: `60`
- Hashed file count: `59`
- Manifest status: `PASS_PACKET_ASSEMBLED_ARTIFACT_ONLY_OBJECTIVE_FACTS`
- Claim-scope lint: `PASS`, 17 files, 0 violations
- LaTeX log scan: `PASS`
- Overfull hbox warnings: `0`
- Manuscript PDF SHA256:
  `81a119565a44b5c637380f3770f9ce38fe9266ff28c83d4c23b1e1531fcf3458`

## Git State

- Root repository branch: `main`
- Root repository commit:
  `bb5d2738f5360317d858cf5f63c823bb606bac2c`
- `origin/main` commit:
  `bb5d2738f5360317d858cf5f63c823bb606bac2c`
- Manuscript repository commit:
  `64510b9daf88deb2efd49a26c8046a023fa4904e`
- Overleaf push: not performed.

## Current Claim Scope

Current manuscript and review packet support only the VSG substrate-gap framing:

- trace-bound first-divergence results are provider-side diagnostics;
- public final-text predicates are observability and spoofing diagnostics;
- accepted source-mismatch rows are spoofing evidence;
- accepted source-mismatch rows are not protected success;
- accepted source-mismatch rows are not codeword recovery;
- public final-text codeword recovered blocks remain `0`.

The current artifacts do not claim:

- public text-only verification success;
- natural evidence success;
- phrase-decoder success;
- cryptographic provenance;
- sanitizer robustness;
- payload diversity;
- model-family general verification;
- ownership proof.

## Next Allowed Action

```text
Wait for expert review, or perform artifact-only manuscript/package hygiene
that does not alter the claim boundary or start new experiments.
```

## Not Allowed Without New Expert/Human Route Decision

```text
new Slurm submission
new generation
new model scoring
new training
new public text-only verification success claim
new natural evidence success claim
new ownership proof claim
Overleaf push
```

## Notes

The root worktree still contains unrelated historical/generated modified and
untracked files. Those are not part of the current VSG handoff packet state.
