# R3.2 H200 853430 Failure Attribution And Repair Decision

Date: 2026-05-12

## Decision

Job `853430` is a completed but failed Qwen v2 R3.2 locked-scale diagnostic.
It must not be aggregated as a 96-block locked-scale package, and it must not
be used for paper-facing positive claims.

The immediate next route is artifact-only repair design. Do not submit another
R3.2 Slurm job until the prompt allocation policy, uniqueness checks, and
forbidden-surface matcher semantics are repaired and reviewed.

## What Failed

The Slurm array completed cleanly:

- `853430_0` through `853430_11` completed with exit code `0:0`;
- all expected shard summaries were present;
- null arms stayed clean at the controlling budget:
  raw/task-only/wrong-key/wrong-payload accepts were all `0/96`.

The locked-scale gate still failed:

- protected accepts @64: `39/96`, below the required `>=80/96`;
- min accepted-block support: `6`, below the required `>=16`;
- min accepted-block majority margin: `0`, below the required `>=3`;
- forbidden public surface count: `9`, above the required `0`.

## Primary Attribution

The nominal 12-shard package did not provide 12 independent prompt windows.
The current H200 wrapper computes:

```bash
window_index=$((SHARD_INDEX % 4))
expected_start=$((512 + window_index * 512))
expected_end=$((expected_start + 511))
```

As a result, the 12 shards collapse to 4 deterministic prompt windows repeated
3 times:

| Window | Shards | Prompt rows | Protected accepts |
|---|---|---:|---:|
| window_00 | shard_00, shard_04, shard_08 | 512-1023 | 8/8 |
| window_01 | shard_01, shard_05, shard_09 | 1024-1535 | 3/8 |
| window_02 | shard_02, shard_06, shard_10 | 1536-2047 | 1/8 |
| window_03 | shard_03, shard_07, shard_11 | 2048-2559 | 1/8 |

The generated output and decode-row hashes are identical within each repeated
group. Therefore the effective unique result is `13/32`, not `39/96`.

This is a control-plane/statistical-design failure, not a Llama result, not a
payload-diversity result, and not a Slurm/provider failure.

## Secondary Attribution

Later prompt windows are much weaker under the same contract and decoder:

| Window | Protected target hit | Resolved slot rate |
|---|---:|---:|
| window_00 | 0.759 | 0.842 |
| window_01 | 0.839 | 0.922 |
| window_02 | 0.561 | 0.701 |
| window_03 | 0.333 | 0.396 |

The failure is therefore also prompt-window dependent. The late windows need
prompt/topic attribution before another locked-scale submission.

## Forbidden Surface Attribution

The shard summaries report `forbidden_public_surface_count=9`, all from the
final prompt-window replicas. A response-text substring audit also found
ordinary-language matches such as `cert` inside `certain` and `owner` in normal
phrases. This does not clear the gate. It means the exact forbidden-surface
matcher must be audited before interpreting the count as deliberate public
surface leakage.

## Required Repairs Before Any New R3.2 Submission

1. Prompt allocation must be explicit and unique.

   The wrapper must not use `SHARD_INDEX % 4` for a 12-shard locked-scale route
   unless the route is explicitly declared as a 4-window deterministic replicate
   diagnostic. For a 96-block claim, the precommit must map each shard to a
   distinct selected prompt window.

   Repair design: the next reviewed route must move prompt allocation out of
   shell arithmetic and into the precommit manifest. The wrapper should read
   `prompt_file_row_start`, `prompt_file_row_end_inclusive`, and
   `window_jsonl_sha256` for the current `shard_id` from
   `precommit/r3_2_selected_prompt_manifest.json`; it should not recompute a
   window from `SHARD_INDEX`.

2. The preflight must verify prompt-window uniqueness.

   Before Slurm submission, a local plan-only preflight must output all shard
   prompt row ranges and selected prompt hashes, and it must hard-fail if any
   hash repeats in a route that claims independent blocks.

   Repair design: a 12-shard independent route must require
   `len(unique(window_jsonl_sha256)) == replicate_group_count == 12`. The
   preflight summary should also report block-level row ranges and
   `row_jsonl_sha256` values so the 96 blocks can be checked before submission.

3. The aggregate path must refuse duplicate prompt windows.

   The aggregate checker must hard-fail if it sees repeated generated-output or
   decode-row hashes unless the route explicitly declares deterministic
   replicate blocks and does not count them as independent evidence.

4. The prompt bank size must match the statistical claim.

   A 12-shard, 8-block-per-shard, 64-query-per-block locked scale needs 6,144
   unique prompt rows if every block is to be independent. If only 2,048 eval
   rows are available, the route must be redefined as a smaller 32-block unique
   package or the prompt bank must be expanded before rerun.

   Route decision for review: do not call the current 2,048-row eval split a
   96-independent-block package. The conservative repaired route is either
   `R3.2u`, a 4-shard/32-block unique package using rows `512-2559`, or a new
   prompt-bank expansion followed by a fresh frozen manifest before returning
   to a 12-shard/96-block route.

5. Forbidden-surface matching must be audited.

   The matcher must distinguish explicit protocol surfaces from ordinary
   substrings such as `cert` in `certain`. If the project still wants to forbid
   ordinary words such as `bucket`, that decision must be stated explicitly
   because it can fail natural checklist responses.

   Artifact audit result: the decode-path matcher currently treats non-`=`
   terms as whole-token markers, so the recorded gate failures in
   `forbidden_surface_decode_decision_hits.csv` are whole-word `bucket`
   matches, not `CERT`/`OWNER` substring matches. The separate substring audit
   is diagnostic only; it identifies why naive substring matching would be too
   broad. The still-open policy question is whether ordinary whole-word
   `bucket` should remain forbidden for natural checklist responses.

6. Late-window prompt/topic attribution must be recorded.

   Before rerun, compare the first-word distributions, prompt topics, target-hit
   survival, support, and majority margin for rows `1536-2559` against the
   stronger earlier windows.

## Next Allowed Action

Artifact-only implementation/review of the repaired prompt allocation preflight
and duplicate-window guard is allowed. Slurm submission remains blocked until
that repair is reviewed, the allowlist is safe, and a fresh Hermes/user
notification is sent.

Still blocked until gates pass:

- aggregate `853430` as a locked-scale success;
- submit a replacement R3.2 job;
- start Llama;
- run same-family nulls;
- run sanitizer or FAR aggregation;
- make paper-facing positive claims;
- claim payload diversity.
