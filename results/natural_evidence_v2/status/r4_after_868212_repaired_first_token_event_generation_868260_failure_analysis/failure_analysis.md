# R4 After-868212 Generation 868260 Failure Analysis

Status: `RECORDED_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_GENERATION_868260_FAILED_QUALITY_GATE_SIGNAL_PRESENT_NO_SUBMIT`

## Bottom Line

`868260` completed cleanly on H200, but it is **not** a passing result. The strict first-token event gate failed because quality filters rejected two protected blocks. Importantly, the evidence signal itself is present: all four protected blocks decoded the expected codeword before quality filtering.

```text
strict protected accepts: 2/4
protected accepts ignoring quality: 4/4
raw/task-only/wrong-key/wrong-payload accepts: {'raw': 0, 'task_only': 0, 'wrong_key': 0, 'wrong_payload': 0}
full-phrase protected accepts, format_scrub=all: 0
```

## Protected Block Failures

| block | strict | ignoring quality | decoded | duplicate | forbidden | min support | reason |
|---|---:|---:|---|---:|---:|---:|---|
| shard_00_block_00 | False | True | 10100101 | 1 | 0 | 8 | duplicate_response_hash |
| shard_01_block_00 | False | True | 10100101 | 2 | 1 | 5 | duplicate_response_hash,forbidden_public_surface |
| shard_02_block_00 | True | True | 10100101 | 0 | 0 | 7 |  |
| shard_03_block_00 | True | True | 10100101 | 0 | 0 | 9 |  |

## Quality Failure Details

- Duplicate quality failed in `shard_00` and `shard_01` protected blocks.
- The only contextual forbidden hit was `bucket` in an ordinary physical plumbing/home-maintenance output, but the current policy hard-forbids `bucket`, so the block is rejected.
- Global duplicates remain severe: `7612` duplicate extra rows, `4676` unique hashes out of `12288` rows, max duplicate group size `8`.

## Interpretation

This run should not be reclassified as positive. It shows that the first-token event signal can recover the committed codeword under the repaired full16 codebook, but the output quality/safety gates are not satisfied. The next route must be artifact-only repair/pivot planning: precommit a contextual forbidden-surface policy and a duplicate-safe generation/allocation policy before any new Slurm rerun.
