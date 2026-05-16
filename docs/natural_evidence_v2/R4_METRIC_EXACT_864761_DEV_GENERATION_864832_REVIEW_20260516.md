# R4 Metric-Exact 864761 Dev Generation 864832 Review

Status: `FAIL_R4_METRIC_EXACT_864761_DEV_GENERATION_NO_PROTECTED_ACCEPTS_NO_DOWNSTREAM_UNLOCK`

Job `864832` completed cleanly on Chimera H200:

```text
array tasks: 864832_0..864832_3
state: COMPLETED
exit code: 0:0
node: chimera21
source adapter: 864761, gain 1.0
generated outputs: 6144 = 2048 protected + 2048 raw + 2048 task_only
```

Primary decode result:

```text
format_scrub=all protected accepts: 0/32
format_scrub=none protected accepts: 0/32
raw/task-only/wrong-key/wrong-payload accepts: 0 in all modes
```

The null/control arms are clean, but the positive channel failed. Under
`format_scrub=all`, protected mean support is only `0.75`, median support is
`0`, and mean matched surfaces are `2.25` per block. The decoder threshold is
not close.

The visible generation behavior is also a failure signal. Protected outputs are
strongly repetitive and dominated by the training-pressure verbs:

```text
protected mean "Create a plan" count per output: 20.85
protected median "Create a plan" count per output: 17
protected duplicate response text hashes: 358
raw duplicate response text hashes: 0
task_only duplicate response text hashes: 0
max protected-vs-raw shallow structural feature AUC: 1.0
```

Interpretation:

```text
864761 proved teacher-forced target-mass pressure can be created, but 864832
shows that pressure does not become recoverable cover-natural evidence in free
generation. The adapter drives visible repetitive Create/Prepare/Plan text
rather than the precommitted phrase-level ECC surface bank.
```

The 864761 caveat remains active: the adapter was trained by repeated-cycling a
512-row train artifact while scoring 8192 rows. This run is not paper-facing
evidence and does not unlock locked-scale, Llama, same-family null, sanitizer,
FAR, payload diversity, or paper claims.

Reviewed artifacts:

```text
results/natural_evidence_v2/status/r4_candidate_v3_metric_exact_864761_dev_diagnostic_864832/
results/natural_evidence_v2/status/r4_metric_exact_864761_dev_generation_864832_review/
```

Next allowed action:

```text
Artifact-only failure analysis and a reviewed repair or pivot route. No new
Slurm submission until the next route records prerequisites and control-plane
checks.
```
