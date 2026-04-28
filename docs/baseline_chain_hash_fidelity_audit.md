# Chain&Hash Fidelity Audit

Date: 2026-04-28

This audit was completed before running any full Chain&Hash train/eval jobs. No
Chimera job was launched and existing Chain&Hash dry-run artifacts were not
overwritten.

## Official Source Check

The paper/OpenReview page for `Hey, That's My Model! Introducing Chain & Hash,
An LLM Fingerprinting Technique` states that code is released at:

```text
https://github.com/microsoft/Chain-Hash
```

Clone attempt:

```text
timestamp_utc=2026-04-28T22:17:24Z
repo=https://github.com/microsoft/Chain-Hash.git
Cloning into 'external_baselines/chain_hash_official'...
remote: Repository not found.
fatal: repository 'https://github.com/microsoft/Chain-Hash.git/' not found
```

Result: official code was not reachable from this environment. Therefore the
current local package cannot be classified as official code. It must be treated
as a reimplementation or proxy.

## Local Package State

Local package:

- `configs/experiment/baselines/chain_hash/package__baseline_chain_hash_qwen_v1.yaml`
- `configs/experiment/baselines/chain_hash/exp_train__qwen2_5_7b__baseline_chain_hash_v1.yaml`
- `configs/experiment/baselines/chain_hash/exp_eval__qwen2_5_7b__baseline_chain_hash_v1.yaml`
- `scripts/prepare_chain_hash_baseline.py`
- `scripts/build_chain_hash_baseline_artifacts.py`
- `src/baselines/chain_hash_adapter.py`
- `manifests/baseline_chain_hash/train_manifest.json`
- `manifests/baseline_chain_hash/eval_manifest.json`

Artifact state:

- Dry-run package exists.
- `train_manifest_entry_count = 12`.
- `eval_manifest_entry_count = 48`.
- `contracts_written = false` in the committed dry-run summary.
- `baseline_chain_hash_summary.json` reports `completed_count = 0`,
  `pending_count = 48`, and `paper_ready = false`.

## Component Fidelity

The current implementation has a useful core but is not yet faithful enough to
be called Chain&Hash in a main comparison table.

- Question/fingerprint prompt generation: present as a fixed `prompt_bank` and
  deterministic prompt template.
- Candidate answer set: present as a public 26-word candidate set.
- Cryptographic hash binding: partially present. The code hashes
  `(secret, payload, seed, query_index, key_text)` to select a candidate answer
  and stores `secret_hash`, `candidate_set_hash`, `prompt_bank_hash`, and
  `contract_hash`.
- Chain construction: missing. The local contract binds each prompt independently
  and does not implement a chained dependency between fingerprint items.
- Random padding: missing. The train examples do not include random padding as a
  robustness mechanism.
- Varied meta-prompt configurations: missing. The package uses one natural
  trigger-response prompt family and one prompt template.
- Robustness to output-style changes: not evaluated. The adapter reports
  `prompt_family_robustness_status = not_evaluated`.
- LoRA support: present in train config (`adapter_mode = lora`, `lora_r = 16`,
  `lora_alpha = 64`), but no completed run verifies it.
- Verification decision rule: present as exact first-word match over query
  budget with threshold `1.0`.
- False-claim/unforgeability logic: incomplete. `false_claim_score` is recorded
  as `0.0`, but there is no implemented false-claim protocol with wrong-secret
  or competing-owner trials.

## Fidelity Assessment

This is not an official Chain&Hash implementation because the announced
official repository is unreachable. It is also not yet a faithful
reimplementation because it omits several paper-critical mechanisms:

- chain construction,
- random padding,
- varied meta-prompt configurations,
- output-style robustness evaluation,
- false-claim/unforgeability evaluation.

The correct label is therefore:

```text
Chain&Hash-style proxy
```

It may be useful as an appendix diagnostic or as a development scaffold, but it
must not be used as a main-table external ownership baseline.

## Required Fixes To Promote Fidelity

To reach decision B, the local implementation must add and validate:

- a documented chain construction matching the paper's cryptographic binding;
- random padding in the training data generation path;
- varied meta-prompt configurations during training;
- style-change robustness evaluation during verification;
- wrong-secret or competing-owner false-claim trials;
- an anchor validation showing behavior consistent with the paper's reported
  setup, despite official code being unavailable.

Only after these are present should full Chain&Hash final train/eval be allowed.

## Decision

C. only Chain&Hash-style proxy can proceed; main table use forbidden.

Full Chain&Hash final Chimera jobs remain blocked. Do not launch them unless the
implementation is promoted to A or B by a subsequent fidelity audit.
