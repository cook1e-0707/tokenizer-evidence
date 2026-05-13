# R4 dev diagnostic 853691 review

Job `853691` completed all 4 H200 shards with `ExitCode 0:0`.

## Gate summary

| Gate | Observed | Required | Status |
|---|---:|---:|---|
| `protected_accepts_no_scrub` | `0` | `>=28/32` | FAIL |
| `protected_accepts_format_scrub_all` | `0` | `>=26/32` | FAIL |
| `raw_accepts_all` | `0` | `0/32` | PASS |
| `task_only_accepts_all` | `0` | `0/32` | PASS |
| `wrong_key_accepts_all` | `0` | `0/32` | PASS |
| `wrong_payload_accepts_all` | `0` | `0/32` | PASS |
| `duplicate_generated_output_hashes` | `0` | `0` | PASS |
| `duplicate_generation_ids` | `0` | `0` | PASS |
| `duplicate_decode_rows` | `0` | `0` | PASS |
| `forbidden_public_surface_all` | `237` | `0` | FAIL |

## Decode summary: format_scrub=all

| Arm | Accepts | Payload matches | Support median | Margin median | Forbidden total |
|---|---:|---:|---:|---:|---:|
| `protected` | `0/32` | `0/32` | `16.0` | `2.0` | `27` |
| `raw` | `0/32` | `0/32` | `8.0` | `3.5` | `72` |
| `task_only` | `0/32` | `0/32` | `8.0` | `3.5` | `84` |
| `wrong_key` | `0/32` | `0/32` | `16.0` | `2.0` | `27` |
| `wrong_payload` | `0/32` | `0/32` | `16.0` | `2.0` | `27` |

## Decode summary: no scrub

| Arm | Accepts | Payload matches | Support median | Margin median | Forbidden total |
|---|---:|---:|---:|---:|---:|
| `protected` | `0/32` | `0/32` | `16.0` | `2.0` | `27` |
| `raw` | `0/32` | `0/32` | `8.0` | `3.5` | `72` |
| `task_only` | `0/32` | `0/32` | `8.0` | `3.5` | `84` |
| `wrong_key` | `0/32` | `0/32` | `16.0` | `2.0` | `27` |
| `wrong_payload` | `0/32` | `0/32` | `16.0` | `2.0` | `27` |

## Interpretation

This is a clean Slurm completion and a failed R4 dev diagnostic. Protected recovery is `0/32` under both `format_scrub=all` and no-scrub decode. Null arms remain clean, but the positive channel does not recover. The next allowed work is artifact-only failure attribution and R4 surface/training-target repair planning; no locked-scale, Llama, sanitizer, FAR, same-family null, or paper-facing claim is unlocked.
