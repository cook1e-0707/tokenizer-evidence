# R4 dev diagnostic 853691 failure attribution

Job `853691` completed cleanly, but the R4 dev gate failed.

## Main facts

- Protected accepts with `format_scrub=all`: `0/32`.
- Protected accepts with no scrub: `0/32`.
- Null accepts with `format_scrub=all`: `{'raw': 0, 'task_only': 0, 'wrong_key': 0, 'wrong_payload': 0}`.
- Protected median support: `16.0` observed coordinates.
- Protected median margin: `2.0`.
- Protected median correct observed coordinates: `10.0`.
- Protected payload matches: `0/32`.
- Forbidden public surface total across decode rows: `237`.
- Protected-vs-raw shallow structural AUC: `0.624`.

## Attribution

The run is not a Slurm/provider failure. It is a positive-channel failure: R4 output contains surface matches, but the matched coordinate polarities are not aligned with the protected `a55e` codeword. The current R4 surface bank/codebook was plan-only and not paired with an R4 teacher-forced or free-generation training objective, so this outcome is consistent with an untrained cover-natural surface channel.

## Next allowed action

Run artifact-only R4 repair planning: design a trainable cover-natural surface target path and an R4 teacher-forced target-mass probe before any further generation Slurm job. Do not launch locked-scale, Llama, sanitizer, FAR, same-family null, or paper-facing claims from this failed dev diagnostic.
