# natural_evidence_v1 Negative Diagnostic Summary

Status: frozen negative diagnostic, not a positive natural-output result.

## Decision

`natural_evidence_v1` should no longer be advanced as the primary route. The
passive opportunity mining, global repeated-frame assignment, and strict
token-index anchoring protocol produced a completed and informative negative
diagnostic. The next research line is `natural_evidence_v2_controlled_micro_slots`.

This does not prove natural tokenizer-aligned evidence is impossible. It shows
that the v1 contract is not a viable end-to-end protocol under the tested
free-generation setting.

## Provenance

The primary completed evaluation is job `846699`.

Provenance-normalized summary:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_recovery/observation_erasure_summary_846699.json
```

This summary is clean for internal negative-diagnostic use:

- `status=PASS_EXPLAINED_RECOVERY_DIR_NAME`
- `source_job_id=846699`
- remote observation path uses the reused recovery directory name containing
  `846627`
- observation artifact hash is recorded
- decode trace artifact hash is recorded
- row counts and condition counts are recorded
- `provenance_mismatches=[]`

Do not cite the v1 negative result in paper-facing form without carrying this
provenance explanation forward.

## Completed Evaluation Facts

Job `846699` completed. It was not a provider failure, evaluator crash, or
payload codec arithmetic failure.

Observed artifacts:

| Item | Value |
|---|---:|
| generated outputs | 18,432 |
| bucket observations | 372,216 |
| decode rows | 120 |
| protected accepts | 0 |
| null accepts | 0 |
| accepted decode rows | 0 |
| decode rows with `insufficient_symbols` | 120 / 120 |
| compatible variable-radix digit rows | 1,885 / 372,216 |
| erasure rows | 370,331 / 372,216 |
| erasure rate | 0.994935736239173 |
| dominant erasure reason | `observed_token_not_in_variable_radix_bucket_set` |

The decoder oracle substitution diagnostic passed: when observed tokens are
replaced with committed target bucket digits, the current evaluator/frame
schedule can decode target digits. Therefore the primary v1 failure is not a
payload codec arithmetic bug.

## Bottleneck 1: Frame Observability

The v1 compiler used a global ordered position stream and repeated payload
frames across that stream. Evaluation generated one response per unique prompt,
then aggregated that prompt's committed positions. In practice, a prompt's
observed positions are scattered across many frames, while a frame requires
roughly 27-36 variable-radix digits to decode.

Frame replay results:

- `status=COMPLETE_REPLAY_NO_OBSERVED_COMPLETE_FRAMES_SCHEDULE_CAN_COMPLETE_WITH_NO_ERASURE`
- observed complete frames: `0`
- max observed slots per frame: `1`
- no-erasure scheduled complete frames: `5370`
- decode rows with scheduled complete frames under no-erasure: `120`

Oracle schedule simulation results:

- `status=COMPLETE_ORACLE_SCHEDULE_NO_OBSERVED_SUBSET_CAN_COMPLETE_FRAME`
- no prompt subset can complete a frame using actually survived digits
- schedule-only repair cannot recover the completed transcript

This makes `variable_radix_min_positions=500` an insufficient gate. It checks
total position count, not whether the observable query schedule can complete
any frame.

## Bottleneck 2: Symbol Survival

The v1 strict token-index anchor did not survive free generation. The dominant
failure is that the token at the committed index is not in the variable-radix
bucket set.

On-policy survival results:

- compatible hits: `1885 / 372216`
- compatible hit rate: `0.0050642637608270466`
- target hits: `299 / 143160`
- target hit rate: `0.0020885722268790164`
- bucket misses: `370331`
- token-index-out-of-response rows: `0`

Protected LoRA showed at most weak evidence of lift and did not create a stable
free-generation channel.

## Bottleneck 3: Weak Teacher-Forced Target-Mass Lift

The teacher-forced bucket-mass probe did not show the kind of protected-vs-base
or protected-vs-task-only separation needed before any E2E rerun.

Recorded summary:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_teacher_forced_bucket_mass_probe/qwen_846699_teacher_forced_bucket_mass_probe_summary.json
```

Known aggregate result from the project state:

- protected mean target candidate mass: `0.410354`
- base mean target candidate mass: `0.406997`
- task-only mean target candidate mass: `0.405440`
- protected minus base: `+0.003357`
- protected minus task-only: `+0.004914`
- target rank-1 rates are similarly close

This is far below the kind of margin needed for a reliable natural-output
channel. More v1 training steps are not the next scientific action.

## Branch-Aware Repair Interpretation

The branch-aware compatibility proxy found plausible examples but did not unlock
training:

- scored rows: `209`
- branch-aware proxy pass rows: `153`
- primary repaired target-mass probe candidates: `75`
- protected/control separation remained weak

These artifacts are useful for understanding drift and candidate design, but
they do not justify continuing the v1 passive/global-frame path.

## Final v1 Interpretation

The v1 route failed as a protocol:

- passive opportunity mining is too sparse and not controllable enough;
- global repeated-payload frames do not align with prompt-level observability;
- strict token-index anchors are too fragile under free generation;
- 4-way/8-way and variable-radix completion are too demanding for the measured
  survival rates;
- the protected LoRA did not learn a large teacher-forced target-bucket margin;
- no payload recovery was observed.

Preserve v1 as a negative diagnostic and an ablation asset. Do not run new v1
training, Qwen v1 E2E, Llama v1, same-family nulls, sanitizer benchmarks, FAR
aggregation, or paper-facing positive claims from this route.

## Forbidden Claims

- natural-output success
- payload recovery
- full FAR
- cross-family generality
- robustness to paraphrase or sanitizers
- stealth guarantee
- superiority over Scalable or Perinucleus
- `24,000 fingerprints` or any bucket-bank-as-fingerprint wording

## Next Route

Switch to `natural_evidence_v2_controlled_micro_slots`:

- controlled-natural owner probes;
- prompt-local, high-density, naturally located micro-slots;
- 2-way tokenizer-aligned buckets;
- bucket-margin supervision;
- prompt-local small payload frames;
- commit-then-reveal audit;
- raw, task-only, wrong-key, and wrong-payload controls before any claim.
