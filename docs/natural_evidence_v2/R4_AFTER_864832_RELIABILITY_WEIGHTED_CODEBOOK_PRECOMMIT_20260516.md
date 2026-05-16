# R4 After 864832 Reliability-Weighted Codebook Precommit

Status: `PRECOMMITTED_ARTIFACT_ONLY_NO_COMPUTE`

The reliability-weighted codebook candidate derived from reviewed `866147` dev
artifacts is frozen as an artifact-only precommit. This does not unlock
generation, training, Llama, same-family nulls, sanitizer, FAR, payload
diversity, or paper-facing claims.

## Inputs

```text
source review: results/natural_evidence_v2/status/r4_after_864832_two_sided_controller_safety_bound_score_866147_review/aggregate_summary.json
source attribution: results/natural_evidence_v2/status/r4_after_864832_two_sided_controller_safety_bound_failure_attribution_866147_20260516/failure_attribution_summary.json
source plan: results/natural_evidence_v2/status/r4_after_864832_reliability_weighted_codebook_plan_20260516/codebook_plan_summary.json
```

## Precommit

```text
precommit dir: results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516/
codebook: codebook.json
decoder spec: decoder_spec.json
manifest: precommit_manifest.json
selected coordinates: [6,22,10,26,1,17,3,19,15,31,8,24,4,20,7,23]
contract: 4 payload bits + 4 checksum bits, 2 coordinates per bit
decoder: pair-majority then checksum
primary scrub mode: all
```

## Next

The next allowed action is artifact-only decoder/oracle route planning for this
precommit. Any model scoring or generation still requires a separate reviewed
route decision, local/remote hash preflight, allowlist safety, Hermes
notification, exactly one H200 Slurm submission if compute is needed, and
immediate allowlist disablement after `sbatch`.
