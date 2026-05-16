# R4 After 864832 Transfer-Gap Repair Decision

Canonical phase:
`V2_R4_METRIC_EXACT_864761_DEV_GENERATION_864832_FAILED_REVIEWED_NO_DOWNSTREAM_UNLOCK`

Current blocker:
`BLOCK_R4_METRIC_EXACT_864761_DEV_GENERATION_TRANSFER_GAP_ARTIFACT_ONLY_REPAIR_OR_PIVOT_NEXT`

## Decision

Do not resubmit generation from the 864761 adapter, and do not start locked
scale, Llama, same-family null, sanitizer, FAR, payload diversity, or paper
claim work.

Job `864832` showed a clean execution and a clean negative result:

```text
protected accepts, format_scrub=all: 0/32
protected accepts, format_scrub=none: 0/32
raw/task-only/wrong-key/wrong-payload accepts: 0
protected mean support, format_scrub=all: 0.75
protected median support, format_scrub=all: 0
max protected-vs-raw shallow feature AUC: 1.0
```

The source job `864761` passed teacher-forced surface-mass scoring, but its
free generation collapsed into repetitive visible `Create/Prepare/Plan` text
instead of the precommitted cover-natural ECC phrase surfaces. This means the
current failure is a transfer-gap/objective-alignment failure, not a Slurm
failure and not a null-control failure.

The 864761 caveat remains active: the adapter trained by repeated-cycling a
512-row train artifact while scoring 8192 rows. It must not be described as an
8192 unique train-row coverage result.

## Next Artifact-Only Package

Prepare an artifact-only transfer-gap repair package with no model run and no
Slurm submission. It must answer these questions from existing artifacts only:

```text
1. Which candidate-v3 pressure phrases dominate protected generation?
2. Which precommitted R4 cover-natural ECC surfaces are missing?
3. Does protected generation reduce natural cover-bank support relative to raw/task-only?
4. Is the failure caused by decoder-bank mismatch, prompt scaffold mismatch,
   objective mismatch, or repetition collapse?
5. What exact future route would repair the mismatch without reusing
   864832 transcripts as post-hoc positive surfaces?
```

Required outputs:

```text
results/natural_evidence_v2/status/r4_after_864832_transfer_gap_repair_package_20260516/
  transfer_gap_repair_plan.md
  transfer_gap_repair_summary.json
  candidate_v3_pressure_phrase_audit.csv
  cover_bank_missing_surface_audit.csv
  protected_vs_raw_surface_support_audit.csv
```

## Guardrails

The repair package may use existing artifacts from jobs `864761` and `864832`,
but it must not:

```text
- add 864832-observed phrases to the locked surface bank as post-hoc positives;
- lower protected accept/support/margin gates;
- change locked-output thresholds after inspection;
- submit Slurm;
- run generation;
- start training;
- start Llama, same-family null, sanitizer, FAR, payload-diversity work, or
  paper-facing claims.
```

## Future Route Requirements

Any later compute route must explicitly align the optimized training objective
with the exact free-generation decoder surface bank. A future route must record
at least:

```text
- whether target surfaces are the precommitted cover-natural ECC surfaces or a
  new frozen bank;
- anti-repetition / naturalness controls;
- output-side structural leakage gate;
- primary `format_scrub=all` decode;
- raw/task-only/wrong-key/wrong-payload controls;
- allowlist entry;
- Hermes notification;
- remote hash preflight;
- exactly-one H200/pomplun submission policy;
- post-submit allowlist shutdown.
```
