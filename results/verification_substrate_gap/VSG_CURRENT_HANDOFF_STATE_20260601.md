# VSG Current Handoff State - 2026-06-01

## Canonical Phase

```text
VSG_MANUSCRIPT_SNAPSHOT_REVIEWED_ARCHITECTURE_VALIDATED_ARTIFACT_TEST_HARDENED_NO_SUBMIT
```

## Status

The Verification Substrate Gap paper package has been rewritten, hardened,
packaged, independently verified, handoff-audited, committed, and pushed to
GitHub. The latest expert response has now been parsed into paper-hardening
stages. The response validates the VSG architecture but does not make the
manuscript submission-ready.

This pass added regression tests for the expert packet verifier and handoff
audit, plus active-manuscript prose-risk tests, so that the objective review
package and current manuscript remain checkable after future artifact-only
edits.

This continuation also implemented stronger public final-text predicate
baselines and ran a local pilot on available non-adopted/historical text
artifacts. Adopted locked final-text JSONL rows are not currently local, so the
pilot does not update paper-facing adopted locked evidence and does not create
a public text-only verification claim.

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
- Expert packet and manuscript prose regression tests: `PASS`, 13 tests passed
- Stronger public-predicate regression tests: `PASS`, total targeted tests now
  `16` passed
- Manuscript PDF SHA256:
  `81a119565a44b5c637380f3770f9ce38fe9266ff28c83d4c23b1e1531fcf3458`

## Local Manuscript Hardening After Packet Delivery

- Local manuscript commit:
  `4d66568ec08325d1d81b5ce060fbfda302e3177d`
- Local manuscript PDF SHA256 after prose cleanup:
  `73d605183ae501f7555e91ccad4fb565fd43c7513714cf3c80936681941ed20b`
- Prose-risk change:
  reproducibility appendix now describes frozen artifacts and recorded evidence
  without naming internal canonical phase or claim-lint state.
- Packet refresh:
  not performed in this pass.

## Local Stronger Public-Predicate Pilot

- Summary:
  `results/verification_substrate_gap/VSG_PUBLIC_TEXT_STRONGER_BASELINE_LOCAL_PILOT_20260601.md`
- Output directory:
  `results/verification_substrate_gap/public_text_verifier_stronger_local_pilot_20260601/`
- New variants:
  `P4_char_ngram_public_predicate`, `P5_word_trigram_public_predicate`,
  `P6_hybrid_char_word_public_predicate`
- Local sources:
  `qwen_dev_869348_local_text_probe`,
  `llama_historical_879555_local_text_probe`
- Codeword recovered blocks:
  `0`
- Claim scope:
  local pilot only; not adopted locked evidence; not public text-only
  verification success.

## Git Sync Evidence

- Root repository branch: `main`
- Root repository commit at the pre-state-record sync check:
  `4e470810ed6a6741fa94206915a2ae0d0b59405b`
- `origin/main` commit at the pre-state-record sync check:
  `4e470810ed6a6741fa94206915a2ae0d0b59405b`
- Manuscript repository commit:
  `64510b9daf88deb2efd49a26c8046a023fa4904e`
- Overleaf push: not performed.

This file intentionally does not attempt to record the commit hash containing
itself, because that would be self-referential and unstable. Use `git log` for
the current state-record commit.

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
Artifact-only manuscript/package hardening that does not alter the claim
boundary or start new experiments.
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
