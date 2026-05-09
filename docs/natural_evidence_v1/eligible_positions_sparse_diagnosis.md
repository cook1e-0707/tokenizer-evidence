# Eligible Positions Sparse Diagnosis

Read-only diagnosis for the post-846699 natural_evidence_v1 five-arm Qwen eval.
No code, generated run outputs, training, or provider APIs were modified.

## Expert Review Snapshot

Job 846699 answers a narrow question: whether the current variable-radix
train/eval/verifier contract can recover `P0421` or `P1729` from Qwen
natural-output generations after a 64-step proof-of-life LoRA run.

It did not recover:

| Item | Value | Interpretation |
|---|---:|---|
| Slurm status | COMPLETED 0:0 | The eval pipeline ran to completion; this is not a crash. |
| Generated outputs | 18432 | All five-arm scheduled generations completed. |
| Bucket observations | 372216 | The verifier inspected the committed token-index opportunities. |
| Decode rows | 120 | 5-arm decode matrix across payloads, seeds, and budgets was written. |
| Protected accepts | 0 | No protected payload recovery under the current protocol. |
| Null accepts | 0 | No null accept observed, but this is not full FAR. |
| Decode status | 120/120 `insufficient_symbols` | No row reached frame-complete variable-radix decoding. |
| Paper claim allowed | false | This remains diagnostic only. |

The scientific interpretation is not "Qwen/provider failure" and not "payload
codec arithmetic failure." The completed run exposes a contract mismatch:
train-time total eligible positions are high, but eval-time observations do not
complete any repeated payload frame.

Two bottlenecks are now separable:

1. **Frame observability bottleneck.** The compiler spreads one prompt's grouped
   positions across many frames, so a single generated response cannot complete a
   frame.
2. **Symbol survival bottleneck.** Even before frame completion, only 1885 of
   372216 observations produce compatible variable-radix digits.

Either bottleneck is sufficient to block recovery. The current result shows both.

## Short Answer

The `eligible_positions=7` eval surface is not a single config value and is not
written as a 7-item list in each train JSONL row. It emerges at eval time:

1. The source reference outputs contain `2048` unique prompts, each repeated
   exactly `7` times.
2. `compile_train_dataset.py` keeps one variable-radix compatible position for
   almost every source row.
3. `evaluate_qwen_natural_e2e.py` then generates one output per unique
   `prompt_id`, but groups all train rows with that `prompt_id` back together.

That converts the train artifact from "one eligible position per source row" to
"about seven eligible positions per eval prompt."

The unfavorable frame mapping is created by global variable-radix frame
assignment. The compiler lays the repeated payload over the whole ordered
position stream. Since the stream is ordered in blocks of 2048 prompts, the same
prompt's seven positions land about `2048 / 32 ~= 64` frames apart. A single
generated response can therefore contribute at most one digit to each of a few
widely separated frames, while each frame requires roughly `27-36` digits.

The immediate failure is a protocol/evaluator alignment issue, with parameter
mismatches that made it invisible in preflight. It is not evidence of a provider
failure and not a payload codec arithmetic bug.

## Active Config And Artifacts

The relevant config values are split across fixed-bank/static settings and the
active variable-radix E2E path. This distinction matters for expert review.

`configs/natural_evidence_v1/pilot.yaml` sets these bank/selector limits:

- `bucket_bank.max_evidence_positions_per_response: 4`
- `bucket_bank.min_spacing_tokens: 12`
- `selector.max_evidence_positions_per_response: 4`
- `selector.min_spacing_tokens: 12`

Relevant lines: `configs/natural_evidence_v1/pilot.yaml:96-110` and
`configs/natural_evidence_v1/pilot.yaml:201-205`.

The same config also contains fixed-radix decoder settings:

- `decoder.bucket_tuple_width: 3`
- `decoder.bucket_radix: 8`

Those fixed-radix fields are not the active decode mode for job 846699. The
846699 evaluator calls `_decode_observation_group(..., decoder_mode="variable_radix")`.
The variable-radix decoder consumes each observation's own `digit` and `radix`
fields. It does not decode 846699 rows as an 8-way fixed-radix carrier.

For wrong-key mapping in the active E2E path, `evaluate_qwen_natural_e2e.py`
reads `bucket_bank.compatibility_adjusted_capacity.qwen_e2e_viability_gate.bucket_count`,
which is currently `4`. Relevant config lines:
`configs/natural_evidence_v1/pilot.yaml:152-155`; evaluator line:
`scripts/natural_evidence_v1/evaluate_qwen_natural_e2e.py:549`.

I did not find a `train_data_dir` key in `pilot.yaml`. The active Qwen eval
wrapper supplies it through `TRAIN_DATA_DIR`, defaulting to:

`$RUN_ROOT/data/variable_radix_frame_policy_dry_run_20260506_1848`

Relevant lines: `scripts/natural_evidence_v1/slurm/qwen_natural_e2e_eval.sbatch:17-21`
and `scripts/natural_evidence_v1/slurm/qwen_natural_e2e_eval.sbatch:78-80`.

The local copy of that train-data artifact is:

`results/natural_evidence_v1/status/variable_radix_frame_policy_dry_run_20260506_1848/`

For both `P0421` and `P1729`, the contract reports:

- `example_count = 14336`
- `evidence_example_count = 14316`
- `total_eligible_positions = 14316`
- `variable_radix_frame_policy = repeat_payload`
- `variable_radix_frame_count = 448`
- `variable_radix_available_positions = 14336`
- `variable_radix_used_positions = 14316`
- `variable_radix_unused_tail_positions = 20`
- `variable_radix_min_positions = 500`
- `variable_radix_min_positions_satisfied = true`

The train JSONL structure is:

- `14316` rows have exactly one eligible position.
- `20` rows have zero eligible positions.
- Grouping by `prompt_id`, `2028` prompts have 7 positions and `20` prompts have
  6 positions.

The completed 846699 review artifacts are local here:

`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_recovery/`

The copied local files are:

- `qwen_natural_e2e_eval_summary.json`
- `qwen_natural_e2e_eval_progress.json`
- `qwen_natural_e2e_decode_trace.csv`
- `qwen_natural_e2e_eval_wrapper_preflight_846699.json`
- `observation_erasure_summary_846699.json`
- `slurm/nat-ev-qwen-nat-eval-846699.out`
- `slurm/nat-ev-qwen-nat-eval-846699.err`
- `slurm/nat-ev-qwen-obsprov-847630.out`
- `slurm/nat-ev-qwen-obsprov-847630.err`

The completed frame replay artifacts are local here:

`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_frame_completion_replay/`

The copied local files are:

- `qwen_846699_frame_completion_replay_summary.json`
- `qwen_846699_frame_completion_by_decode_row.csv`
- `qwen_846699_frame_completion_closest_frames.csv`
- `qwen_846699_frame_completion_slot_distribution.csv`
- `slurm/nat-ev-qwen-frame-847634.out`
- `slurm/nat-ev-qwen-frame-847634.err`

The completed oracle schedule simulation artifacts are local here:

`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_oracle_schedule_simulation/`

The copied local files are:

- `qwen_846699_oracle_schedule_simulation_summary.json`
- `qwen_846699_oracle_schedule_by_decode_row.csv`
- `qwen_846699_oracle_schedule_prompt_examples.csv`
- `qwen_846699_oracle_schedule_frame_bounds.csv`
- `slurm/nat-ev-qwen-oracle-847640.out`
- `slurm/nat-ev-qwen-oracle-847640.err`

The large observation JSONL was not copied locally. The erasure counts below were
computed from the remote observation file:

`/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/eval/qwen_natural_e2e_eval_846627_recovery/qwen_natural_e2e_bucket_observations.jsonl`

For strict reproducibility, Slurm job `847630` generated the local
`observation_erasure_summary_846699.json` from that remote file. The summary
records `source_job_id=846699`, source path job id candidate `846627`, file
hashes, row counts, condition counts, erasure reasons, and decode trace join
keys. Its provenance status is `PASS_EXPLAINED_RECOVERY_DIR_NAME`, with
`provenance_mismatches=[]`. The explanation is that 846699 was the completed
recovery evaluation for the failed 846627 attempt, and the recovery wrapper
intentionally reused the fresh output directory name
`qwen_natural_e2e_eval_846627_recovery`.

## Pipeline Trace

### 1. Bucket Bank Construction

`build_bucket_bank.py` does not decide the prompt-level count of 7. It turns
candidate records into per-prefix bucket entries:

- Filters candidates by stable token, rank, probability, and surface rules:
  `scripts/natural_evidence_v1/build_bucket_bank.py:111-150`.
- Assigns accepted candidates to keyed buckets:
  `scripts/natural_evidence_v1/build_bucket_bank.py:153-216`.
- Rejects entries without enough filtered candidates and serializes bucket maps:
  `scripts/natural_evidence_v1/build_bucket_bank.py:219-283`.

`opportunity_policy.py` is just the policy boundary wrapper around the same
`_entry_from_record` implementation:
`scripts/natural_evidence_v1/opportunity_policy.py:8-53`.

### 2. Train Data Compilation

The variable-radix compiler has these key steps:

- CLI enables `--encoding-mode variable_radix` and
  `--variable-radix-frame-policy repeat_payload`:
  `scripts/natural_evidence_v1/compile_train_dataset.py:37-47`.
- It selects spaced variable entries with `min_spacing_tokens` and
  `max_positions`:
  `scripts/natural_evidence_v1/compile_train_dataset.py:256-267`.
- It builds a global `ordered_positions` stream from selected entries:
  `scripts/natural_evidence_v1/compile_train_dataset.py:368-387`.
- It maps that global stream to payload frames:
  `scripts/natural_evidence_v1/compile_train_dataset.py:389-425`.
- It writes selected positions into each train row:
  `scripts/natural_evidence_v1/compile_train_dataset.py:427-473`.

The crucial artifact fact is that the reference outputs used by the contract are:

`results/natural_evidence_v1/status/expanded_actual_prefix_suffix_compatibility_845981/qwen_diagnostic_generated_outputs.jsonl`

They contain:

- `14336` rows
- `2048` unique prompts
- exactly `7` source rows per prompt
- conditions: `2048` raw, `6144` protected_trained, `6144` task_only_lora

The variable arity bank has `23774` compatible entries, usually `1-4` per
generated row. But the compatible positions are at `position_index` 0, 1, 2, or
3. With `min_spacing_tokens=12`, `_spaced_variable_entries` keeps only the first
one for each generated row. That is why the compiled train JSONL has about one
eligible position per source row, not up to four.

### 3. Payload Bytes To Frame Digits

The variable-radix codec consumes radices greedily until each byte has mixed-radix
capacity at least 256:

- `_byte_group_width`: `src/core/payload_codec.py:210-222`
- `encode_bytes_variable_radices`: `src/core/payload_codec.py:225-248`
- `decode_bytes_variable_radices`: `src/core/payload_codec.py:251-276`

For payloads like `P0421` and `P1729` (`5` bytes) and compatible arities mostly
2, 3, and 4, a full frame needs about `27-36` digits. The inspected contracts
show this full frame digit-count distribution:

- 32 digits: 3296 positions
- 33 digits: 3069 positions
- 31 digits: 2418 positions
- 30 digits: 2070 positions
- 34 digits: 1700 positions
- 29 digits: 725 positions
- 35 digits: 595 positions
- 36 digits: 360 positions
- 28 digits: 56 positions
- 27 digits: 27 positions

The frame count is `448` because `14316` usable positions divided by roughly 32
digits per full payload frame yields about 448 complete repeated frames.

### 4. Why Prompt Positions Map Across About 65 Frames

Frame assignment is global, not prompt-local. The compiler appends selected
positions in source-row order, then slices that stream into repeated payload
frames.

Example from `P0421` train data for `nat_prompt_000001`:

| payload digit | frame | frame digit | frame digits | token index |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 33 | 2 |
| 2048 | 64 | 14 | 33 | 2 |
| 4096 | 128 | 28 | 33 | 2 |
| 6144 | 192 | 27 | 33 | 2 |
| 8192 | 256 | 31 | 33 | 2 |
| 10240 | 321 | 3 | 32 | 2 |
| 12288 | 385 | 0 | 34 | 2 |

Prompt-local slot coverage across all prompts:

- slot 0: 2048 positions, 65 distinct frames
- slot 1: 2048 positions, 65 distinct frames
- slot 2: 2048 positions, 65 distinct frames
- slot 3: 2048 positions, 65 distinct frames
- slot 4: 2048 positions, 66 distinct frames
- slot 5: 2048 positions, 64 distinct frames
- slot 6: 2028 positions, 63 distinct frames

This is the source of the "one prompt has 7 positions, but each position slot
spans about 65 frames" behavior.

### 5. Observation And Decode

The five-arm eval wrapper changes the unit of observation:

- `_prompt_rows` selects unique `prompt_id` rows for generation:
  `scripts/natural_evidence_v1/evaluate_qwen_natural_e2e.py:262-280`.
- `_positions_by_prompt` groups all train-row positions with the same
  `prompt_id`:
  `scripts/natural_evidence_v1/evaluate_qwen_natural_e2e.py:283-290`.
- `_observe_outputs_variable_radix` emits one observation per grouped position:
  `scripts/natural_evidence_v1/evaluate_qwen_natural_e2e.py:338-407`.

That means one generated response for a prompt observes the prompt's 6-7 grouped
positions. Those positions are not one frame; they are digits from widely
separated frames.

The decoder then groups observations by `frame_index` and requires every
`frame_digit_index` in `0..frame_digit_count-1` before decoding that frame:

- frame grouping and expected count:
  `scripts/natural_evidence_v1/evaluate_diagnostic_e2e.py:512-523`
- incomplete-frame rejection:
  `scripts/natural_evidence_v1/evaluate_diagnostic_e2e.py:529-547`
- budget-row decode summary:
  `scripts/natural_evidence_v1/evaluate_diagnostic_e2e.py:656-719`

Job 846699 local summary artifacts show:

- `observation_count = 372216`
- erasures = `370331`
- observed symbols = `1885`
- erasure rate = `0.994935736239173`
- all erasures had reason `observed_token_not_in_variable_radix_bucket_set`
- all 120 decode rows have `decode_status=insufficient_symbols`
- all 120 decode rows have `usable_symbols=0`

Condition-level decode trace summary:

| Condition | Decode rows | Eligible positions | Observed symbols | Usable symbols | Observed / eligible |
|---|---:|---:|---:|---:|---:|
| raw | 8 | 13440 | 28 | 0 | 0.00208 |
| protected_trained | 16 | 26880 | 213 | 0 | 0.00792 |
| task_only_lora | 16 | 26880 | 70 | 0 | 0.00260 |
| wrong_key | 64 | 107520 | 852 | 0 | 0.00792 |
| wrong_payload | 16 | 26880 | 213 | 0 | 0.00792 |

The protected arm shows more compatible-token hits than raw/task-only, but the
absolute survival rate remains below 1%. Wrong-payload rows reuse the protected
observations with a wrong expected payload. Wrong-key rows are four wrong-key
decodes over protected generations, so their aggregate observed count is four
times the protected count. These null rows are useful sanity checks but cannot be
reported as full FAR because the protected channel itself did not recover.

Within individual decode groups, the observed non-erased symbols are spread
thinly. At query budget 512, the largest inspected group had 51 observed symbols,
but those 51 symbols landed in 51 different frames. The per-frame maximum was 1
observed symbol, while frame completion requires roughly 27-36 observed symbols
with the right frame-digit indices.

## What Experts Need To Decide

This result narrows the decision to protocol design, not implementation
plumbing. The key question is whether natural-output tokenizer-aligned evidence
should be made frame-completable under the actual black-box query schedule, or
whether the current strict token-index anchor is too brittle for the intended
claim.

### Supported conclusions

- The five-arm eval wrapper, checkpoint loading, generation loop, observation
  writer, decode trace writer, and summary writer all executed.
- The current variable-radix frame policy is not frame-completable under the
  prompt-level eval schedule.
- Total eligible positions and total repeated frames are insufficient quality
  gates by themselves.
- A preflight that only validates synthetic complete target streams is too weak.
- The current 64-step proof-of-life did not produce recoverable natural-output
  evidence.

### Unsupported conclusions

- Do not conclude full FAR. The summary explicitly marks `not_full_far=true`,
  and there is no positive protected recovery.
- Do not conclude provider/model impossibility. The run tested one short,
  high-risk training setup and one strict observation protocol.
- Do not conclude payload codec failure. The codec decodes complete synthetic
  variable-radix streams; the observed run never forms a complete frame.
- Do not claim natural-output success, cross-family generality, or sanitizer
  robustness.

### Open hypotheses

| Hypothesis | Current evidence | Status |
|---|---|---|
| Codec arithmetic bug | Synthetic target-stream preflight decoded complete streams; 846699 never reached complete frames | Unlikely |
| Eval grouping bug | `_prompt_rows` deduplicates prompts while `_positions_by_prompt` aggregates all repeated rows under that prompt | Plausible contract gap |
| Frame-policy mismatch | Prompt-local positions are spread across about 64-66 frames; max observed per frame is 1 | Strongly supported |
| Training too weak | Protected hit rate exceeds raw/task-only but remains below 1% | Plausible contributor |
| Strict token-index anchor too brittle | 370331/372216 observations erase by bucket-set miss | Strongly supported |
| Opportunity bank impossible | Offline density and variable-arity capacity passed earlier, but on-policy survival failed | Not decided |

### Decision points for the next design

1. **Eval unit.** Decide whether the committed observable unit is a unique
   prompt, an original source row, or a frame-aware bundle of prompts. The
   compiler and verifier must use the same unit.
2. **Frame placement.** Decide whether payload frames must be prompt-local,
   frame-bundle-local, or globally distributed. The current global distribution
   is not recoverable under prompt-level querying.
3. **Survival gate.** Decide the minimum acceptable held-out compatible-token hit
   rate per frame slot before training is allowed. A total-position gate is not
   enough.
4. **Training diagnostic.** Decide whether to run a teacher-forced bucket-mass
   probe before free generation. This would separate "LoRA did not learn bucket
   preference" from "free generation drift erases the anchor."
5. **Protocol commitment.** If frames are rescheduled or cross-frame harvesting
   is introduced, treat it as a new protocol requiring a fresh precommitment and
   new null analysis.

## Answers To The Requested Questions

### 1. Where is `eligible_positions` count determined?

There are two counts:

- Train-row count: determined in `compile_train_dataset.py` by
  `_spaced_variable_entries` and the variable-radix row loop. In the inspected
  artifact, this is usually 1 per source row because candidate `position_index`
  values are 0-3 and `min_spacing_tokens=12` keeps only the first compatible
  entry.
- Eval-prompt count: determined in `evaluate_qwen_natural_e2e.py` by grouping
  all train rows with the same `prompt_id`. Since the source reference outputs
  contain seven rows per prompt, the eval prompt gets about seven grouped
  positions.

So `eligible_positions=7` is an eval-time grouping artifact over train data, not
a direct value in `pilot.yaml`.

### 2. Where does each position get mapped to N frames?

The frame mapping happens in `compile_train_dataset.py`:

- `ordered_positions` is built globally in row order.
- `radices` is derived from `len(entry["compatible_bucket_ids"])`.
- `_variable_radix_frame_assignments` repeatedly encodes the whole payload over
  that global radix stream.
- The resulting `frame_index`, `frame_digit_index`, and `frame_digit_count` are
  stored in each eligible position.

Because row order is organized as 2048-prompt blocks and frames are about 32
digits wide, the same prompt's seven grouped positions land around 64 frames
apart.

### 3. Why is the ratio 7 positions to about 448 frames so unfavorable?

The current protocol mixes incompatible units:

- Payload frames are defined over the global ordered position stream.
- Eval queries are unique prompt-level generations.
- A prompt-level generation sees only the 6-7 positions grouped under that
  prompt.
- Those 6-7 positions are digits from 6-7 different frames, not a complete frame.

Therefore a successful response can only contribute sparse single digits across
many frames. To complete one frame, the eval needs many different prompts whose
positions happen to fall in the same frame and whose generated tokens all land
inside the configured variable-radix bucket sets. With the observed survival
rate near 0.5%, no frame gets close to completion.

### 4. Is this a parameter mismatch, code bug, or fundamental protocol design issue?

It is primarily a protocol/evaluator design issue, with contributing parameter
mismatches.

Parameter mismatches:

- `min_spacing_tokens=12` is incompatible with current variable entries at
  positions 0-3 if the goal is multiple positions per source row.
- `variable_radix_min_positions=500` checks total positions, not complete-frame
  coverage under the actual eval query schedule.
- The synthetic target-stream preflight validated the decoder on complete
  streams, not frame completion under prompt-grouped observations and realistic
  erasures.

Possible code bug or contract gap:

- `_prompt_rows` deduplicates by `prompt_id`, while `_positions_by_prompt`
  aggregates all positions for that prompt. If the intended eval unit was the
  original train/reference row, deduplication loses that row-level structure. If
  the intended eval unit was unique owner probes, the compiler should have
  framed positions in prompt-observable units instead.

Not the root cause:

- The bucket bank construction is doing deterministic candidate filtering and
  bucketization as designed.
- The variable-radix codec is behaving consistently with its greedy byte
  grouping rule.
- Reducing `frame_digit_count` would mask the mismatch rather than fix it.

### 5. Minimal viable fixes that do not reduce `frame_digit_count`

1. Add a frame-completion preflight gate before any new E2E run.

   The gate should replay the exact eval prompt selection and position grouping,
   then report per-budget complete-frame coverage, partial-frame distribution,
   and expected completion under measured survival. It should fail if no frame
   can complete under the committed query budgets. This does not reduce the
   security anchor; it prevents another run that can only produce partial frames.

2. Align frame construction with the eval observation unit.

   Build frames so all `frame_digit_count` positions for at least one complete
   frame are observable within the committed eval unit. Two conservative options:

   - Frame-local prompt packing: assign all digits of a frame to a deterministic
     bundle of prompts that the eval scheduler queries and decodes as one frame.
   - Dense per-response framing: generate and score enough eligible positions
     within a single response so one prompt can carry a complete 27-36 digit
     frame.

   Both keep `frame_digit_count` intact; they change where the digits are placed.

3. Make the eval scheduler frame-aware.

   Instead of taking the first N unique prompts and hoping frame slots complete,
   select prompts by committed `frame_index` coverage. A budget should guarantee
   that at least one full frame's scheduled positions are queried. This still
   needs survival gating, but it removes the current accidental scatter.

4. Raise observed-symbol survival before claiming recovery.

   Even with frame-aware scheduling, the current 99.49% erasure rate is too high.
   The next train/eval design should require a precommitted lower bound on
   bucket-hit survival per frame slot, measured on held-out prompts, before a
   recovery run. Repeating payload frames is useful only if at least one repeated
   frame has a plausible chance of completion.

5. Keep the decoder strict unless the protocol is explicitly changed.

   Do not "fix" this by aggregating arbitrary digits across different
   `frame_index` values after the fact. The current verifier intentionally
   treats frames independently. Cross-frame harvesting would be a new protocol
   and would need a new commitment, null analysis, and multiple-testing account.

## Artifact-Only Diagnostics To Run Before Another Training Job

These checks use existing 846699 artifacts and train contracts. They should
complete before any new LoRA training, Llama run, same-family null, or sanitizer
benchmark.

The expert review makes this a hard gate: no Qwen rerun, no Llama run, no
same-family null, no sanitizer benchmark, and no positive paper-facing claim
until these diagnostics are complete.

1. **Artifact provenance normalization.** COMPLETE.

   This ran first because the final 846699 summary lives in a recovery
   output directory named `qwen_natural_e2e_eval_846627_recovery`. That naming is
   explainable as a fresh recovery directory for failed job 846627, but the
   erasure/survival statistics should not depend on an implicit convention.

   Output:

   - `results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_recovery/observation_erasure_summary_846699.json`
   - `source_job_id=846699`
   - `source_path_job_id_candidates=["846627"]`
   - observation JSONL sha256
     `1594258ab37309d421751a80dddcad0b078e5e5c8f066bb018b71176a98e8213`
   - decode trace sha256
     `4d26c7ca8b9236151502bae454bacda6f28f00d2668e62586690987d52a8ad26`
   - observation row count `372216`
   - decode trace row count `120`
   - compatible variable-radix digit rows `1885`
   - erasure reason rows `370331`
   - `erasure_reason_counts={"observed_token_not_in_variable_radix_bucket_set": 370331}`
   - `provenance_mismatches=[]`

   The source provenance is reconciled. The erasure/survival statistics are no
   longer provisional on the `846627_recovery` path naming issue.

2. **Frame completion replay.** COMPLETE.

   Input: `variable_radix_train.jsonl` for each payload and the 846699
   observations.

   Output:

   - `qwen_846699_frame_completion_replay_summary.json`
   - `qwen_846699_frame_completion_by_decode_row.csv`
   - `qwen_846699_frame_completion_closest_frames.csv`
   - `qwen_846699_frame_completion_slot_distribution.csv`

   Key results:

   - replay status:
     `COMPLETE_REPLAY_NO_OBSERVED_COMPLETE_FRAMES_SCHEDULE_CAN_COMPLETE_WITH_NO_ERASURE`
   - decode rows replayed: `120`
   - observed complete frame count: `0`
   - decode rows with observed complete frames: `0`
   - max observed slots per frame: `1`
   - selected observation rows across cumulative decode budgets: `201600`
   - compatible digit rows across cumulative decode budgets: `1376`
   - digit rate across decode budgets: `0.006825396825396826`
   - scheduled complete frame count under no-erasure: `5370`
   - decode rows with scheduled complete frames under no-erasure: `120`
   - max scheduled slots per frame: `36`

   This refines the earlier frame-observability attribution. The current
   prompt-level schedule can complete frames if every scheduled committed token
   survives into the intended bucket. The actual observed transcript never
   completes a frame because compatible digits are too sparse and distributed
   with at most one observed slot per frame. The immediate blocker for observed
   recovery is therefore symbol survival / strict-anchor drift, while oracle
   scheduling still needs to quantify whether a better committed prompt subset
   changes the completion probability.
   - distribution of observed slots per frame
   - list of frames closest to completion

   Expected from current evidence: complete frames should be 0; max observed
   slots per frame should be 1 in the inspected worst case.

3. **Oracle schedule simulation.** COMPLETE.

   Run three oracle views:

   - **Oracle survival:** assume every committed observation lands in the correct
     bucket under the current schedule.
   - **Oracle scheduler:** choose the best prompt subset under budgets
     64/128/256/512 using the existing train contract.
   - **Oracle protected lift:** resample frame completion using the measured
     protected survival rate.

   Results:

   - `qwen_846699_oracle_schedule_simulation_summary.json`
   - `qwen_846699_oracle_schedule_by_decode_row.csv`
   - `qwen_846699_oracle_schedule_prompt_examples.csv`
   - `qwen_846699_oracle_schedule_frame_bounds.csv`
   - replay status:
     `COMPLETE_ORACLE_SCHEDULE_NO_OBSERVED_SUBSET_CAN_COMPLETE_FRAME`
   - decode rows replayed: `120`
   - greedy no-erasure scheduled-complete decode rows: `120`
   - greedy no-erasure scheduled complete frames: `5370`
   - observed complete frames under any prompt subset: `0`
   - decode rows with any observed-complete subset: `0`
   - max observed slots per frame under all prompts: `2`
   - minimum greedy prompts to complete any frame under no-erasure: `27`
   - median minimum greedy prompts to complete any frame under no-erasure: `27`
   - max iid complete-frame probability under selected observed survival:
     `5.791901064855407e-54`
   - max protected-lift iid complete-frame probability:
     `7.584907586228995e-60`

   Interpretation: schedule-only repair cannot recover the completed 846699
   transcript. The no-erasure schedule is frame-completable, but observed
   survived digits are so sparse that even an oracle prompt subset cannot
   complete a frame. This moves the immediate blocker to symbol survival,
   strict token-index anchoring, and potentially training objective alignment.

4. **On-policy survival by slot and row source.**

   Input: 846699 observations joined with train-row metadata.

   Output:

   - survival by `model_condition`
   - survival by payload, seed, frame digit index, token index, prompt slot, and
     source condition
   - examples of bucket-set misses with observed token text and candidate bucket
     token texts

   This separates "model generated a plausible natural alternative outside the
   bucket set" from "model failed to move toward the target bucket."

5. **Protected-vs-task-only lift.**

   Input: decode trace and observation rows.

   Output:

   - protected/raw and protected/task-only observed-symbol lift
   - lift by payload and seed
   - whether lift concentrates in a few prompts/tokens or is broadly distributed

   Current aggregate lift exists but is far too small: protected is about 0.79%
   observed/eligible versus raw about 0.21% and task-only about 0.26%.

6. **Teacher-forced bucket-mass probe.**

   Input: trained checkpoints and the committed train/eval positions. This is a
   scoring diagnostic, not free-generation evaluation.

   Output:

   - target bucket probability before/after LoRA at the exact evidence positions
   - mass assigned to compatible non-target buckets
   - comparison against task-only LoRA

   If teacher-forced target bucket mass is low, the training objective or
   implementation failed. If teacher-forced mass is high but free-generation
   survival remains low, the bottleneck is anchor drift/free-generation
   stability.

   Current status: complete. Slurm job `847652` wrote
   `results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_teacher_forced_bucket_mass_probe/`.
   Protected LoRA has only a small teacher-forced target-mass lift over base and
   task-only, which is not enough to explain away the sub-1% free-generation
   survival failure.

7. **Decoder oracle substitution.**

   Replace observation rows with the target bucket digit under the current frame
   schedule and run the decoder.

   This answers:

   - If every committed position emitted the correct bucket, could the current
     evaluator recover the payload?

   Decision rule:

   - If this fails, the frame/scheduler contract is wrong independent of
     training.
   - If this passes, the schedule is formally decodable and observed survival is
     the blocker.

   Current status: complete. Under committed target-digit substitution,
   protected rows decode `16/16`, wrong-payload rows decode `0/16`, and
   eligible-position mismatches are `0`. The current evaluator/frame schedule is
   formally decodable if target digits are observed.

## Protocol Repair Candidates After Diagnostics

Do not continue `repeat_payload` over a global ordered position stream as the
main protocol without a positive oracle diagnosis. It is currently mismatched to
prompt-level observation.

Candidate directions:

1. **Frame-aware prompt-bundle scheduling.**

   Precommit prompt bundles that jointly carry a complete frame. The evaluator
   queries and decodes the bundle as one frame. This is closest to the current
   variable-radix design, but still requires symbol survival to rise
   substantially.

2. **Prompt-local dense frame.**

   Put a complete frame in one response. This has simple observability but is not
   currently supported by the data: each prompt has about 6-7 positions, while a
   frame needs about 27-36 digits.

3. **Sparse coordinate-level erasure code or rateless-style recovery.**

   Treat each survived observation as a known-coordinate low-rate code symbol and
   recover from any sufficiently large set of survived coordinates. This may fit
   sparse natural-output evidence better than requiring all digits of a frame to
   survive. It is a new protocol and would require a fresh commitment, null
   analysis, and verifier design.

Strict token-index anchoring should not be treated as the final protocol. The
next protocol should move toward prefix-conditioned eligible events
reconstructed from generated text, plus branch-aware compatibility and
regenerated/local suffix repair diagnostics.

## Bottom Line

The five-arm eval used a train contract that had enough total variable-radix
positions, but not enough frame-completable positions under the actual
prompt-level observation protocol. The 7-position prompt surface is produced by
eval-time prompt grouping over seven repeated source rows per prompt. The
roughly 65-frame spread is produced by global repeated-payload framing over a
2048-prompt ordered stream.

The expert decision is to stop training and E2E reruns until artifact-only
diagnostics prove what can and cannot decode under the current contract. If
decoder oracle substitution or oracle schedule simulation fails, the
compiler/evaluator frame contract must be redesigned. If oracle decode succeeds
but teacher-forced or free-generation survival fails, the training objective,
anchor policy, and prefix-conditioned observation protocol must be repaired.

## Addendum: 846699 On-Policy Survival by Slot/Source

The next artifact-only diagnostic was run as Slurm job `847644`
(`nat-ev-qwen-survival`) on DGXA100/chimera12. It did not load model weights,
train, generate, rerun E2E, or make paper-facing claims. Output directory:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_on_policy_survival/
```

Key outputs:

```text
qwen_846699_on_policy_survival_summary.json
qwen_846699_on_policy_survival_by_slice.csv
qwen_846699_on_policy_survival_by_condition_payload_seed.csv
qwen_846699_on_policy_bucket_miss_examples.jsonl
qwen_846699_on_policy_compatible_hit_examples.jsonl
```

### Core Numbers

| Metric | Value | Interpretation |
|---|---:|---|
| observation_rows | 372,216 | Same denominator as 846699 bucket observations |
| compatible_hit_rows | 1,885 | Only rows producing a variable-radix digit |
| compatible_hit_rate | 0.005064 | About 0.506% compatible survival |
| target_comparable_rows | 143,160 | correct-key raw/protected/task-only rows |
| target_hit_rows | 299 | Correct-key rows whose observed digit equals the target digit |
| target_hit_rate | 0.002089 | About 0.209% target survival |
| bucket_miss_rows | 370,331 | Dominant erasure mode |
| token_index_out_of_response_rows | 0 | Not a truncation/short-output failure in this artifact |
| metadata_missing_rows | 0 | Train metadata join succeeded for all rows |
| oracle provenance mismatches | 0 | Observation hash/count is consistent with oracle summary |

The main update is that symbol survival is not only globally sparse; it is
sparse even after joining observations back to the committed variable-radix
train metadata. The observed token usually exists at the committed token index,
but it is outside the compatible variable-radix bucket set.

### Condition-Level View

| Condition | Rows | Compatible hits | Compatible rate | Target hits | Target rate |
|---|---:|---:|---:|---:|---:|
| raw | 28,632 | 172 | 0.006007 | 56 | 0.001956 |
| protected_trained | 57,264 | 272 | 0.004750 | 137 | 0.002392 |
| task_only_lora | 57,264 | 353 | 0.006164 | 106 | 0.001851 |
| wrong_key | 229,056 | 1,088 | 0.004750 | N/A | N/A |

This aggregate table should not be overinterpreted as final protected-vs-null
lift. Wrong-key rows duplicate protected generations under alternate keys, and
the next diagnostic must compare protected vs task-only by payload, seed, slot,
token class, source condition, target radix, and target bucket class.

### Example Failure Mode

One representative bucket-miss example:

```text
prompt_id=nat_prompt_000001
token_index=2
frame_index=385
frame_digit_index=0/34
target_bucket=0
target bucket token: " footwear"
compatible bucket token: " shoes"
observed token: " Check"
erasure_reason=observed_token_not_in_variable_radix_bucket_set
```

This is a concrete example of strict token-index anchor drift: the committed
position expects a local lexical choice such as footwear/shoes, while the
free-generated response at the same token index emits a different discourse
continuation token. This supports the expert's concern that strict absolute
token-index anchoring is not an acceptable final natural-output protocol.

### Immediate Decision

This diagnostic strengthens, rather than resolves, the bottleneck diagnosis:

- Do not start new training.
- Do not rerun Qwen E2E.
- Do not launch Llama, same-family null, or sanitizer benchmark.
- Do not write paper-facing positive claims.
- Protocol repair decision is recorded in
  `docs/natural_evidence_v1/post_846699_protocol_repair_decision.md`.
- Next step: artifact-only prefix-conditioned selector replay.

## Addendum: 846699 Decoder Oracle Substitution

The final artifact-only diagnostic in the expert-requested sequence was run
locally with:

```text
scripts/natural_evidence_v1/oracle_qwen_decoder_substitution.py
```

It did not load a model, train, generate, rerun E2E, or make paper-facing
claims. It used the completed 846699 decode trace and the committed
variable-radix train artifacts. Output directory:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_decoder_oracle_substitution/
```

Key outputs:

```text
qwen_846699_decoder_oracle_substitution_summary.json
qwen_846699_decoder_oracle_decode_trace.csv
qwen_846699_decoder_oracle_by_condition.csv
qwen_846699_decoder_oracle_observation_sample.jsonl
```

### Oracle Result

| Metric | Value | Interpretation |
|---|---:|---|
| decode rows | 120 | Mirrors the completed 846699 decode trace |
| protected oracle accepts | 16 / 16 | Positive rows recover under target-digit substitution |
| wrong-payload oracle accepts | 0 / 16 | Payload mismatch still rejects |
| eligible-position mismatches | 0 | Oracle schedule matches the eval trace denominators |
| decoded frame count | 5,370 | Same no-erasure frame-completion scale seen in replay |
| accepted frame count | 4,654 | Target frames recover when symbols are present |

Wrong-key rows also accept under this oracle because the diagnostic directly
substitutes post-bucket target digits and therefore bypasses wrong-key
bucketization. These rows are explicitly not FAR evidence.

### Interpretation

This closes off one hypothesis: the current variable-radix payload codec and
evaluator frame decoder can recover when the committed target digits are
available. Therefore the 846699 failure is not explained by decoder arithmetic
or by a frame schedule that is impossible under no erasure.

The remaining bottleneck is upstream:

- the learned target-bucket mass is only weakly shifted under teacher forcing;
- free generation almost never lands inside the committed bucket sets;
- strict absolute token-index anchoring causes local choice points to drift;
- all-digits frame completion is too brittle for sub-1% symbol survival.

### Immediate Decision

- Do not start new training.
- Do not rerun Qwen E2E.
- Do not launch Llama, same-family null, or sanitizer benchmark.
- Do not write paper-facing positive claims.
- Next step: protocol repair decision and anchor/survival repair plan.

## Addendum: 846699 Protected-vs-Task-Only Lift by Slice

The next artifact-only diagnostic was run as Slurm job `847649`
(`nat-ev-qwen-lift`) on DGXA100/chimera12. It did not load model weights beyond
tokenizer metadata, train, generate, rerun E2E, or make paper-facing claims.
Output directory:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_protected_vs_task_only_lift/
```

Key outputs:

```text
qwen_846699_protected_vs_task_only_lift_summary.json
qwen_846699_protected_vs_task_only_lift_by_slice.csv
qwen_846699_protected_vs_task_only_lift_extremes.jsonl
```

### Aggregate Lift

| Metric | Protected | Task-only | Protected - task-only | Interpretation |
|---|---:|---:|---:|---|
| rows | 57,264 | 57,264 | 0 | Balanced comparison |
| compatible-hit rate | 0.004750 | 0.006164 | -0.001415 | Task-only has more bucket hits |
| target-hit rate | 0.002392 | 0.001851 | +0.000541 | Protected has a small target-specific lift |
| target-hit rows | 137 | 106 | +31 | Absolute count is still tiny |

This is a mixed diagnostic. The protected LoRA shows a small target-specific
lift over task-only, but it does not increase generic compatible survival; in
aggregate it lowers compatible-hit rate. Most importantly, target survival is
still far below 1%, so this cannot support a new E2E rerun.

### Slice-Level Signal

Among slice rows with enough rows per arm:

| Direction | Target-hit lift slices | Compatible-hit lift slices |
|---|---:|---:|
| protected_higher | 82 | 75 |
| task_only_higher | 58 | 118 |
| tie | 188 | 135 |

Representative slice examples:

- `payload_seed=P1729|23`: protected target-hit rate `0.004191`, task-only
  `0.001467`, target delta `+0.002724`.
- `payload_seed=P1729|17`: protected target-hit rate `0.000489`, task-only
  `0.001956`, target delta `-0.001467`.
- `target_radix=2`: protected target-hit rate `0.002962`, task-only `0.001449`.
- `target_radix=4`: protected target-hit rate `0`, task-only `0.002681`.
- `target_bucket_token_class=function_word`: protected target-hit rate
  `0.005115`, task-only `0.002273`.
- `target_bucket_token_class=word`: protected target-hit rate `0.002123`,
  task-only `0.001892`, but task-only has higher compatible-hit rate.

The lift is therefore not broad enough to repair the failure. It is
payload/seed/radix/token-class dependent, and even favorable slices remain far
too sparse for complete frame decoding.

### Immediate Decision

- Do not start new training.
- Do not rerun Qwen E2E.
- Do not launch Llama, same-family null, or sanitizer benchmark.
- Do not write paper-facing positive claims.
- At this point in the diagnosis, the next artifact-only diagnostic was decoder
  oracle substitution.

## Addendum: Balanced Branch-Aware Proxy Scoring

After the R1 selector-contract analysis and balanced example export, a
Slurm-scored branch-aware compatibility proxy diagnostic was run as job
`848414` (`nat-ev-qwen-brscore`). It completed 0:0 in 00:00:55 and wrote:

`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_compatibility_scored_balanced/`

Status:

`COMPLETE_BRANCH_AWARE_COMPATIBILITY_MODEL_SCORED_PROXY_NOT_GENERATED`

The diagnostic scored `209` balanced rows using Qwen/Qwen2.5-7B-Instruct as a
reference-model NLL proxy. It did not generate branch continuations, regenerate
suffixes, train, rerun E2E, decode payload recovery, estimate FAR, or make a
paper-facing positive claim.

Aggregate:

| Metric | Value |
|---|---:|
| Response naturalness proxy pass | 155 / 209 (`0.7416`) |
| Suffix-preserving proxy pass | 169 / 209 (`0.8086`) |
| Branch-aware proxy pass | 153 / 209 (`0.7321`) |
| Mean response delta NLL/token | `0.318643` |
| Mean suffix delta NLL/token | `0.554276` |

By condition:

| Condition | Rows | Branch-aware proxy pass |
|---|---:|---:|
| protected_trained | 76 | 57 (`0.7500`) |
| raw | 74 | 52 (`0.7027`) |
| task_only_lora | 59 | 44 (`0.7458`) |

This refines, but does not resolve, the bottleneck. The result suggests that
many local target substitutions are plausible under a branch-aware/local-suffix
proxy, especially `compatible_non_target` rows. However, protected is
essentially tied with task-only, so the current artifact still lacks a
protected-specific signal. The next work should interpret these proxy scores by
slice and design a repaired training-target preflight; training and E2E remain
blocked.

## Addendum: Repaired-Target Candidate Preflight

The branch-aware scores were then interpreted by condition, drift reason, token
class, payload/seed, and protected-vs-control slices:

`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_score_interpretation/`

Status:

`COMPLETE_BRANCH_AWARE_SCORE_INTERPRETATION_REPAIRED_TARGET_PREFLIGHT_NOT_TRAINING`

The analysis produced:

| Category | Rows |
|---|---:|
| Primary repaired target-mass probe candidates | 75 |
| Secondary / ablation candidates | 78 |
| Rejected rows | 56 |

Primary candidate breakdown:

| Slice | Rows |
|---|---:|
| protected_trained | 39 |
| task_only_lora | 20 |
| raw | 16 |
| compatible_non_target | 58 |
| observed_token_not_candidate_set | 17 |
| word tokens | 74 |
| function_word tokens | 1 |

Interpretation:

- The best repairable slice is word-token `compatible_non_target` drift.
- `observed_bucket_not_compatible` remains secondary because it needs
  bucket-policy review before becoming a clean target.
- punctuation rows are ablation/control-only.
- primary candidates exist, but this is not a training gate because
  protected-vs-control separation is weak or low-N in the slice table.

Next allowed work is an artifact-only repaired teacher-forced target-mass probe
over the primary candidates. Training and E2E remain blocked.

## Addendum: Branch-Aware Compatibility And Local-Suffix Repair Preparation

The next artifact-only preparation step is complete and recorded in:

```text
docs/natural_evidence_v1/branch_aware_suffix_repair_preparation.md
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_suffix_repair_preparation/
```

Machine-readable outputs:

```text
branch_aware_compatibility_summary.json
branch_aware_compatibility_by_token_class.csv
branch_aware_compatibility_scoring_plan.jsonl
regenerated_suffix_repair_manifest.json
regenerated_suffix_repair_examples.jsonl
branch_aware_suffix_repair_readiness.csv
```

Status:

```text
COMPLETE_BRANCH_AWARE_SUFFIX_REPAIR_INPUTS_PREPARED_NOT_SCORED
```

Counts:

| Item | Value |
|---|---:|
| source R1 examples | 240 |
| planned branch-aware rows | 68 |
| regenerated/local-suffix repair examples | 68 |
| train metadata matches | 68 |
| `compatible_non_target` rows | 60 |
| `observed_token_not_candidate_set` rows | 8 |

This step prepared inputs only. It did not score branch-aware compatibility,
regenerate suffixes, train, generate, rerun E2E, decode payload recovery, or
estimate FAR. The prepared set is currently raw-only
(`model_condition_counts.raw=68`), so protected-vs-task-only branch-aware
comparison still requires a richer R1 replay example export or expanded example
selection.

Next allowed actions:

- run one Slurm-scored branch-aware compatibility diagnostic from the prepared
  scoring plan; or
- construct an artifact-only local-suffix repair dry-run from the prepared
  repair examples.

Training and E2E remain forbidden.

The artifact-only local-suffix repair dry-run is also complete:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_local_suffix_repair_dry_run/
```

Machine-readable outputs:

```text
local_suffix_repair_dry_run_summary.json
local_suffix_repair_dry_run_rows.jsonl
local_suffix_repair_dry_run_rows.csv
local_suffix_repair_dry_run_by_status.csv
local_suffix_repair_dry_run_by_condition.csv
local_suffix_repair_dry_run_by_drift_reason.csv
local_suffix_repair_dry_run_readiness.csv
local_suffix_repair_dry_run_examples.md
```

Status:

```text
COMPLETE_LOCAL_SUFFIX_REPAIR_DRY_RUN_NOT_GENERATED
```

Key counts:

| Item | Value |
|---|---:|
| repair examples | 68 |
| approximate text-substitution-ready rows | 36 |
| rows needing tokenizer-aligned or branch regeneration | 32 |

This confirms that a simple local text replacement is not enough. Nearly half
of the prepared rows cannot even locate the observed token text in the original
response text, and the rows that can be replaced are not naturalness-scored.
The input remains raw-only, so protected/task-only comparison remains blocked.
The next meaningful step is a Slurm-scored branch-aware compatibility diagnostic
or a richer protected/task-only example export before scoring.

The richer example export completed as Slurm job `848405`
(`nat-ev-qwen-babr`) with exit code `0:0` in `00:00:46`. It is artifact-only
and did not score a model, regenerate suffixes, train, generate, rerun E2E,
recover payload, or estimate FAR. It replaced the raw-only example set with a
balanced protected/task-only/raw example set before branch-aware compatibility
scoring.

Balanced export:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_balanced_branch_aware_examples/
```

| Condition | Rows |
|---|---:|
| protected_trained | 288 |
| task_only_lora | 288 |
| raw | 192 |

Balanced branch-aware/local-suffix preparation:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_suffix_repair_preparation_balanced/
```

| Item | Value |
|---|---:|
| planned branch-aware rows | 209 |
| regenerated/local-suffix repair examples | 209 |
| protected_trained rows | 76 |
| task_only_lora rows | 59 |
| raw rows | 74 |

Balanced local-suffix dry-run:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_local_suffix_repair_dry_run_balanced/
```

All `209/209` rows are text-substitution-ready in the generated transcript
response text. This is still not compatibility scoring or training-data
approval. The next meaningful step is a Slurm-scored branch-aware compatibility
diagnostic over the balanced scoring plan.

## Addendum: Selector Contract And Training-Target Preflight

The selector precommit/training-target preflight is complete and recorded in:

```text
docs/natural_evidence_v1/selector_contract_training_target_preflight.md
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_selector_contract_preflight/
```

Machine-readable outputs:

```text
selector_contract_training_target_preflight_summary.json
selector_precommit_contract_draft.json
selector_contract_precommit_fields.csv
branch_aware_training_target_preflight_plan.csv
selector_contract_training_target_preflight.md
```

Status:

```text
COMPLETE_SELECTOR_PRECOMMIT_DRAFT_BRANCH_AWARE_PREFLIGHT_PLAN_READY
```

The selector contract remains inactive:

```text
DRAFT_NOT_ACTIVE_BLOCKED_BY_R1_NO_PROTECTED_LIFT
```

The preflight makes the next blockers explicit:

| Gate | Status |
|---|---|
| r1_protected_lift_over_raw | FAIL_BLOCKER |
| r1_protected_lift_over_task_only | FAIL_BLOCKER |
| branch_aware_compatibility | NEEDS_RESULTS |
| regenerated_suffix_repair | NEEDS_RESULTS |
| teacher_forced_repaired_target_mass | NEEDS_RESULTS |
| sparse_coordinate_code | SYNTHETIC_PREFLIGHT_NEEDED |
| fresh_lockbox_or_locked_replay | NEEDS_RESULTS |

The next allowed action is artifact-only branch-aware compatibility plus
regenerated/local-suffix repair diagnostics under the draft selector contract.
Any Chimera CPU/GPU work must be submitted through Slurm.

## Addendum: 846699 Teacher-Forced Bucket-Mass Probe

The next artifact-only diagnostic was run as Slurm job `847652`
(`nat-ev-qwen-tfprob`) on DGXA100/chimera13. It loaded the base Qwen model and
the protected/task-only LoRA adapters from training job `846585`, but did not
train, generate, rerun E2E, or make paper-facing claims. Output directory:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_teacher_forced_bucket_mass_probe/
```

Key outputs:

```text
qwen_846699_teacher_forced_bucket_mass_probe_summary.json
qwen_846699_teacher_forced_bucket_mass_by_condition.csv
qwen_846699_teacher_forced_bucket_mass_by_slice.csv
qwen_846699_teacher_forced_bucket_mass_positions.jsonl
```

## Addendum: 846699 Phase R1 Prefix-Conditioned Selector Replay

Phase R1 was run as an artifact-only Slurm job:

```text
job_id=847879
job_name=nat-ev-qwen-pfxsel
state=COMPLETED
elapsed=00:00:37
exit_code=0:0
node=chimera12
```

Output directory:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_prefix_conditioned_selector_replay/
```

Files:

```text
prefix_conditioned_selector_replay_summary.json
prefix_conditioned_selector_replay_by_condition.csv
prefix_conditioned_selector_replay_aggregate_by_policy.csv
prefix_conditioned_selector_replay_rejections.csv
prefix_conditioned_selector_replay_examples.jsonl
```

This replay read existing generated transcripts, variable-radix train positions,
and expanded actual-prefix bucketized candidate artifacts. It did not train,
generate new outputs, rerun E2E, decode payload recovery, or estimate FAR.

Key provenance and denominators:

| Item | Value |
|---|---:|
| generated rows | 18,432 |
| considered generated/payload rows | 20,480 |
| required unique bank-entry ids | 3,025 |
| matched candidate rows | 3,025 |
| missing candidate rows | 0 |
| query budgets | 64, 128, 256, 512 |
| match policies | exact_full, suffix_32, suffix_16, suffix_8 |

Budget-512 aggregate results:

| Policy | Scheduled events | Prefix matched | Compatible hits | Target hits | Target coordinate count | Max target slots/frame |
|---|---:|---:|---:|---:|---:|---:|
| exact_full | 35,840 | 11,582 (0.3232) | 11,122 (0.3103) | 4,681 (0.1306) | 2,858 | 35 |
| suffix_32 | 35,840 | 11,601 (0.3237) | 11,139 (0.3108) | 4,692 (0.1309) | 2,863 | 35 |
| suffix_16 | 35,840 | 11,881 (0.3315) | 11,401 (0.3181) | 4,782 (0.1334) | 2,871 | 36 |
| suffix_8 | 35,840 | 12,652 (0.3530) | 12,109 (0.3379) | 5,080 (0.1417) | 2,916 | 40 |

The strongest warning is the by-condition split. At budget 512 under
`exact_full`, raw rows have target-hit rates around `0.384` to `0.386`, while
protected rows range from `0.000` to `0.037` and task-only rows range from about
`0.107` to `0.121`. Under `suffix_8`, raw remains around `0.386` to `0.387`,
protected ranges from about `0.001` to `0.054`, and task-only ranges from about
`0.127` to `0.135`.

Interpretation:

- Prefix-conditioned replay can rediscover many eligible raw actual-prefix
  events, so strict token-index anchoring was indeed discarding observable
  prefix-conditioned opportunities.
- The replay does not produce an ownership signal. Raw and task-only behavior
  are too strong relative to protected, and protected often underperforms
  task-only.
- Coordinate-level target hits are not payload recovery. They do not imply FAR,
  verifier acceptance, or successful natural-output evidence.
- The next repair decision must treat selector commitment, null calibration,
  multiple-testing risk, and training-target mismatch as first-class blockers
  before any new training is considered.

## Addendum: 846699 R1 Selector-Contract Repair Analysis

The R1 interpretation pass is complete and recorded in:

```text
docs/natural_evidence_v1/r1_selector_contract_repair_analysis.md
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_r1_selector_contract_analysis/
```

Machine-readable outputs:

```text
r1_selector_contract_repair_summary.json
r1_selector_contract_pairwise_lift.csv
r1_selector_contract_by_policy_budget.csv
r1_selector_contract_repair_analysis.md
```

Status:

```text
COMPLETE_R1_ANALYSIS_NO_PROTECTED_LIFT_OVER_NULLS
```

Across 64 protected-vs-null comparison slices:

| Comparison | Positive protected-lift rows |
|---|---:|
| protected vs raw | 0 / 64 |
| protected vs task-only | 0 / 64 |

Budget-512 mean target-hit rates:

| Policy | Protected | Raw | Task-only | Protected - raw | Protected - task-only |
|---|---:|---:|---:|---:|---:|
| exact_full | 0.020089 | 0.384905 | 0.113979 | -0.364816 | -0.093890 |
| suffix_16 | 0.021973 | 0.385882 | 0.118652 | -0.363909 | -0.096680 |
| suffix_32 | 0.020438 | 0.384905 | 0.114397 | -0.364467 | -0.093959 |
| suffix_8 | 0.030134 | 0.386440 | 0.130999 | -0.356306 | -0.100865 |

Interpretation:

- Prefix-conditioned selector replay recovers observable coordinates, but raw
  and task-only null behavior dominate protected behavior.
- R1 coordinate target hits cannot be interpreted as verifier accepts.
- Any direct replay verifier, new training, or E2E rerun is blocked until a
  precommitted selector and repaired training target pass artifact-only
  preflight.

### Aggregate Teacher-Forced Signal

| Metric | Base | Protected | Task-only | Interpretation |
|---|---:|---:|---:|---|
| position rows | 143,160 | 143,160 | 143,160 | Full committed-prefix probe |
| mean target candidate mass | 0.406997 | 0.410354 | 0.405440 | Protected is only slightly higher |
| target rank-1 rate | 0.410659 | 0.413488 | 0.408022 | Same small lift pattern |
| protected - base target mass | - | +0.003357 | - | Very small absolute effect |
| protected - task-only target mass | - | +0.004914 | - | Very small absolute effect |

This says the protected LoRA does learn a weak target-direction signal at
teacher-forced committed prefixes, but the effect is small. It is not consistent
with a robust learned channel that should survive free generation, and it does
not repair the already measured frame-completion and symbol-survival failures.

### Condition Examples

Representative per-condition rows:

- `P0421 seed17`: protected mean target candidate mass `0.413559`, task-only
  `0.406375`.
- `P0421 seed23`: protected `0.408936`, task-only `0.404734`.
- `P1729 seed17`: protected `0.408787`, task-only `0.405615`.
- `P1729 seed23`: protected `0.410134`, task-only `0.405036`.

The protected lift is directionally positive in aggregate, but each slice is
close to base/task-only rather than a decisive separation. Full-vocab target
mass is also tiny, so most probability mass remains outside the narrow target
bucket set.

### Immediate Decision

- Do not start new training.
- Do not rerun Qwen E2E.
- Do not launch Llama, same-family null, or sanitizer benchmark.
- Do not write paper-facing positive claims.
- Next artifact-only diagnostic: decoder oracle substitution.
