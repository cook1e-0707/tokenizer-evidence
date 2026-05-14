# R4 Positive Selectivity-Repair Package Coverage Dry-Run

## Decision

Status: `FAIL_SUPPORT_WINDOW_DRY_RUN_NO_PROTECTED_ACCEPTS_NO_COMPUTE`.

The independently sourced selectivity package removes the previous null accept
problem, but it is too sparse on the existing failed `859277` outputs. This
does not reclassify `859277` and does not unlock compute or claims.

## Evidence

- protected accepts: `0/29` dry-run blocks with any support events;
- raw accepts: `0/31`;
- task-only accepts: `0/26`;
- wrong-key accepts: `0/29`;
- wrong-payload accepts: `0/29`;
- protected row support rate: `0.057`;
- raw row support rate: `0.052`;
- task-only row support rate: `0.042`;
- protected mean observed events per dry-run block: `4.45`;
- protected mean distinct coordinates per dry-run block: `1.79`.

The result is clean but under-supported. The old `859277` prompt/output policy
does not naturally elicit enough selective events from the new bank.

## Artifacts

- `results/natural_evidence_v2/status/r4_positive_selectivity_repair_package_coverage_dry_run_20260514_2202/coverage_summary.json`
- `results/natural_evidence_v2/status/r4_positive_selectivity_repair_package_coverage_dry_run_20260514_2202/coverage_report.md`
- `results/natural_evidence_v2/status/r4_positive_selectivity_repair_package_coverage_dry_run_20260514_2202/condition_coverage.csv`
- `results/natural_evidence_v2/status/r4_positive_selectivity_repair_package_coverage_dry_run_20260514_2202/per_block_dry_run_decode.csv`
- `results/natural_evidence_v2/status/r4_positive_selectivity_repair_package_coverage_dry_run_20260514_2202/event_family_counts.csv`

## Next Allowed Action

Artifact-only prompt-policy elicitation route design for the selectivity
package. The next route must define natural prompts that elicit
constraint/reasoning/context/quality events without public structural labels.

No Slurm submission, generation, model scoring, training, Llama,
same-family null, sanitizer, FAR aggregation, payload-diversity work, or
paper-facing claim is unlocked.
