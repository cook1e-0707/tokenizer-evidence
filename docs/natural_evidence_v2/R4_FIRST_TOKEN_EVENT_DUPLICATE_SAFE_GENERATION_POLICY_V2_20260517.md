# R4 First-Token Event Duplicate-Safe Generation Policy V2

Status: `ARTIFACT_ONLY_POLICY_RECORDED_NO_SUBMIT`

Source diagnostic: `868260`.

This policy does not reclassify `868260` and does not unlock Slurm, generation, training, Llama, FAR, sanitizer, payload diversity, or paper-facing positive claims.

## Motivation

`868260` recovered the protected first-token event codeword in `4/4` protected blocks before quality filtering, while raw/task-only/wrong-key/wrong-payload controls stayed at `0/4`. It failed the strict quality gate because two protected blocks had exact duplicate response hashes, and one of those blocks also hit the old hard-forbid literal policy.

The global duplicate level was not a minor artifact:

- generated rows: `12288`
- unique exact response hashes: `4676`
- duplicate extra rows: `7612`
- max duplicate group size: `8`

Future R4 first-token event diagnostics must therefore prevent deterministic duplicate collapse without using decoder success or payload match as a selection signal.

## Precommitted Policy

Generation uses controlled sampling with:

- temperature: `0.45`
- top-p: `0.90`
- max duplicate retries: `3`
- retry selection: first nonduplicate exact response hash

The retry rule is blind to:

- decoder success
- payload match
- checksum match
- protected-vs-control arm identity beyond the common arm field used for the public sampling seed

The same retry policy applies to all arms.

Sampling seeds are derived from a public run salt and public row coordinates:

```text
HMAC(public_run_salt, arm, shard_id, block_id, prompt_id, attempt_index)
```

The protected secret key is not used for sampling seeds.

## Gates

Future quality-repair confirmation runs must keep:

- within-block duplicate response hash count: `0`
- global duplicate response hash count: `0`
- duplicate prompt/prefix pair count: `0`
- duplicate generation id count: `0`
- duplicate decode-row hash count: `0`

If all retry attempts remain duplicates, the row is retained with:

```text
row_quality_status = duplicate_exhausted
```

It is not silently replaced and remains a quality failure.

## Bias Control

Allowed rejection reason:

```text
exact duplicate response hash
```

Forbidden rejection reasons:

```text
decode failure
payload mismatch
checksum mismatch
wrong target bit
forbidden-surface failure
```

All attempts must be logged.
