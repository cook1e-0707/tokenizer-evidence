# R4 Positive Full Generation/Decode Wrapper Implementation Review

Timestamp: `2026-05-14T18:18:08Z`

## Decision

Status: `PASS_R4_POSITIVE_FULL_GENERATION_DECODE_WRAPPER_LOCAL_REVIEW_NO_SUBMIT`

The R4 positive event-bank dev diagnostic wrapper no longer exits with the
old non-plan-only fail-closed marker. Full mode now has an explicit path:

1. select one of four `512`-prompt dev shards;
2. generate Qwen outputs for `protected`, `raw`, and `task_only`;
3. decode the generated outputs with the precommitted keyed phrase-event
   decoder under `format_scrub=all`;
4. decode the same outputs under `format_scrub=none`;
5. add `wrong_key` and `wrong_payload` decoder controls over protected
   transcripts.

This review does not submit Slurm and does not make a positive claim.

## Implemented Artifacts

- Wrapper: `scripts/natural_evidence_v2/slurm/r4_positive_dev_diagnostic_h200.sbatch`
- Decoder: `scripts/natural_evidence_v2/decode_r4_positive_keyed_correlation.py`
- Decoder tests: `tests/natural_evidence_v2/test_r4_positive_keyed_correlation_decode.py`
- Disabled allowlist entry: `v2_r4_positive_dev_diagnostic_h200`

## Validation

- `bash -n scripts/natural_evidence_v2/slurm/r4_positive_dev_diagnostic_h200.sbatch`: `PASS`
- Focused pytest:
  `tests/natural_evidence_v2/test_r4_positive_keyed_correlation_decode.py`
  `tests/natural_evidence_v2/test_r4_positive_phrase_event_extractor.py`
  `tests/natural_evidence_v2/test_r4_positive_dev_diagnostic_route.py`
  result: `13 passed`
- Local wrapper plan-only smoke:
  `results/natural_evidence_v2/status/r4_positive_dev_diagnostic_wrapper_plan_smoke_20260514_full_impl_local/plan_validation/wrapper_plan_only_summary.json`
- Full-mode missing-adapter smoke:
  status: expected fail at adapter precondition, not implementation-pending
  fail-closed marker.
- Static keyed-decoder fixture:
  `results/natural_evidence_v2/status/r4_positive_keyed_decoder_fixture_20260514/decode_all/decode_summary.json`
  protected `1/1`, wrong-key `0/1`, wrong-payload `0/1`.
- Local allowlist safety after adding disabled entry:
  `results/natural_evidence_v2/status/r4_positive_full_wrapper_allowlist_safety_20260514.json`
  status: `PASS`, enabled entries: `[]`.

## Hashes

| artifact | sha256 |
| --- | --- |
| `scripts/natural_evidence_v2/slurm/r4_positive_dev_diagnostic_h200.sbatch` | `2015fc1b077e409c086ab5667f309311b7b9615bfcd45b13a1d0efc170073095` |
| `scripts/natural_evidence_v2/decode_r4_positive_keyed_correlation.py` | `18aafd477d5ce1b63e45d87fd00d8555ef9ad2d3df50d0439f4fbf545159efe2` |
| `tests/natural_evidence_v2/test_r4_positive_keyed_correlation_decode.py` | `de441d5fc5f5e535850c22f054f78205d285f552c4a80bcc48b8caaa18f1785b` |
| `configs/natural_evidence_v2/run_allowlist.yaml` | `da7c85b1ea924d3434359ca3a79a083979bdac78195e4ec5a223e85cc5b526fc` |
| decoder fixture summary | `4e636308abc29d62061a2ee07efa3268e1c09b290710812993f79cc3916e6faa` |
| local wrapper plan-only summary | `30618eb609c3bdae26130a2f97960ced9f948980aef386541b776c5831a2f0ea` |
| local allowlist safety summary | `c2b88fc127fea187880022f5d36cb9a398e5d51356c6e5dac2fe65bd6663b1d5` |

## Current Boundary

The wrapper is locally implemented and reviewed. It is not yet a submission
route. Before Slurm submission, the route still needs remote sync, remote
plan-only validation with the same full wrapper, local/remote hash preflight,
zero-enabled allowlist safety, active-job preflight, Hermes TG/email
notification, exactly-one allowlist enablement, and immediate allowlist
disablement after `sbatch`.

