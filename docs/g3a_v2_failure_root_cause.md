# G3a-v2 Failure Root Cause

This report analyzes the four valid G3a-v2 method failures without rerunning training or evaluation.

## Accounting

- Failures: `B1_U03_s23, B1_U12_s23, B1_U15_s23, B4_U12_s23`
- Failure count by seed: `{'23': 4}`
- Failure count by payload: `{'U03': 1, 'U12': 2, 'U15': 1}`
- Failure count by variant: `{'B1': 3, 'B4': 1}`
- Wrong fields among failed slots: `{'TOPIC': 3, 'SECTION': 3}`

## Neighbor Comparisons

- B1 U03/U12/U15 seed 23 all pass at seeds 17 and 29 under the same block count and payload.
- B4 U12 seed 23 passes at seeds 17 and 29 under the same block count and payload.
- U12 seed 23 fails for B1 and B4 but passes for B2, so the pattern is not a pure payload-only failure.
- Total neighbor comparison rows written: `28`.

## Slot And Margin Evidence

The committed slot diagnostics identify wrong decoded buckets and generated representatives, but do not include logits or bucket logmasses. Therefore the requested per-slot margin fields are recorded as `missing` in `results/tables/g3a_v2_failure_slot_margin.csv`.

Available evidence shows valid semantic bucket substitutions rather than parser or RS failures: all four failures are valid completed runs with matching contracts, no erasures, and one symbol error each.

## Training Dynamics

The committed paper-facing artifacts include final aggregate training metrics, selected checkpoint metadata, and paths to raw training metrics. The diagnostic script checks both `train_metrics.jsonl` and `train_metrics.json`, records full curves if run on a machine where the scratch paths exist, and otherwise marks curves as `missing`.

Aggregate training metrics are not sufficient to explain the failures. B1 seed-23 cases share the same aggregate training metrics: `B1_U00_s23` passes, while `B1_U03_s23`, `B1_U12_s23`, and `B1_U15_s23` fail. Conversely, `B4_U12_s23` fails despite high aggregate target-bucket mass and positive aggregate slot margin. This points to seed-specific payload/bucket interactions at evaluation slots, not a confirmed global loss or aggregate-margin failure.

No per-checkpoint verifier replay or per-checkpoint generated text is available, so the current artifacts cannot determine whether an earlier checkpoint would have passed.

## Root-Cause Ranking

1. seed-specific optimization instability: All four method failures occur at seed 23; the specified same-payload same-block neighbors at seeds 17 and 29 pass. The pattern is not global seed collapse because some seed-23 neighbors pass. Confidence: `moderate`.
2. payload/bucket hardness: Within B1 seed 23, U00 passes while U03/U12/U15 fail under the same aggregate training metrics. U12 fails in B1 and B4 at seed 23 but passes in B2. Confidence: `moderate`.
3. insufficient target-vs-wrong bucket margin: Failures are slot/symbol bucket substitutions, but target-vs-wrong logmass margins were not saved. Aggregate training margins are not decisive: B4_U12_s23 has high positive aggregate slot_margin_min_final, and B1_U00_s23 passes despite sharing the negative B1 seed-23 aggregate margin. Confidence: `plausible_unmeasured`.
4. checkpoint drift: No per-checkpoint eval or saved per-checkpoint generated text is available to determine whether an earlier checkpoint passed. Confidence: `inconclusive`.
5. verifier/RS logic bug: Contract hashes match, failures decompose to symbol errors with no erasures, and nearest neighbors pass under the same verifier. Confidence: `unlikely`.

## V3 Instrumentation Required

- per-slot target_bucket_logmass at generation/evaluation time
- per-slot strongest wrong bucket id and logmass
- per-slot target bucket rank
- per-slot target token probability
- top-5 bucket logmasses per slot
- top-5 token probabilities per slot
- per-checkpoint verifier replay or saved generated_text per checkpoint
- explicit checkpoint step-to-epoch mapping in training_health.json
- per-slot L_margin curve; current training metrics only provide aggregate slot_margin_mean/min when raw train_metrics is accessible

## Conclusion

The strongest observed pattern is seed-specific optimization instability at seed 23, with possible insufficient target-vs-wrong bucket margin. However, the margin-level root cause cannot be confirmed because per-slot logits, bucket logmasses, ranks, and per-checkpoint verifier replay were not saved.

F. still inconclusive due to missing instrumentation
