# R4 After-868016 Controller Generation 868151 Failure Analysis

Status: `FAIL_R4_AFTER_868016_CONTROLLER_GENERATION_DIAGNOSTIC_GATE`

Job `868151` completed cleanly on Chimera H200/pomplun:

```text
array tasks: 868151_0..868151_3
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
exit codes: 0:0 for all array tasks
generated rows: 9216
shards seen: 4/4
```

This is a model/protocol diagnostic failure, not a Slurm/runtime failure.

## Gate Result

Primary review:

```text
review dir:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868151_review/
protected accepts, format_scrub=all: 0/4
raw accepts, format_scrub=all: 0/4
task_only accepts, format_scrub=all: 0/4
wrong_key accepts, format_scrub=all: 0/4
wrong_payload accepts, format_scrub=all: 0/4
forbidden public surfaces, format_scrub=all: 26
duplicate response hashes: 4422
```

The null controls remain clean, but protected recovery is zero and the
forbidden/duplicate gates fail. This does not unlock larger generation,
training, Llama, sanitizer, FAR, payload diversity, or paper-facing claims.

## Direct Failure Evidence

Decoder artifacts show no phrase-level carrier events:

```text
decode_all matched_surface_events:
  shard_00: 0
  shard_01: 0
  shard_02: 0
  shard_03: 0
decode_none matched_surface_events:
  shard_00: 0
  shard_01: 0
  shard_02: 0
  shard_03: 0
```

Exact target surface phrase recovery also failed:

```text
target surface phrase contained in response:
  protected: 0/3072
  raw: 0/3072
  task_only: 0/3072
response starts with target surface:
  protected: 0/3072
  raw: 0/3072
  task_only: 0/3072
```

There is a weak first-word directional effect, but it is far below the
phrase-level evidence required by the decoder:

```text
target first word appears in first 120 chars:
  protected: 222/3072
  raw: 31/3072
  task_only: 33/3072
```

This means the teacher-forced first-token controller can shift local token
mass, but the current row-cylinder free generation path does not produce the
committed natural phrase surfaces. The signal does not survive into the
precommitted phrase-level decoder.

## Representative Protected Outputs

The protected controller often produces natural task-relevant responses, but
not the committed target surface:

```text
target surface:
  clarify the working notes for the team
output:
  A useful next action is to ensure everyone has clear roles and responsibilities...

target surface:
  prioritize the working notes for the team
output:
  One practical option is to ensure all volunteers have clear roles and responsibilities...

target surface:
  prepare the status update for the team
output:
  A simple follow-up is to check if everyone has received the latest schedule and instructions...
```

The outputs are semantically reasonable, but they are not evidence-bearing
under the locked phrase bank.

## Diagnosis

The failure mode is:

```text
teacher-forced first-token control passes
  -> free generation starts in the right broad semantic region only weakly
  -> full committed phrase surfaces are almost never emitted
  -> phrase-level decoder observes zero matched events
  -> protected accepts remain 0/4
```

The current first-step controller is therefore insufficient as a generation
mechanism for the phrase-level codebook. It should not be rerun or scaled in
the same form.

The duplicate response hash count is also material:

```text
duplicate response hashes: 4422 / 9216 rows
```

Because many row-cylinders share the same prompt/prefix pattern, deterministic
greedy generation collapses to repeated generic continuations. This makes the
current row-cylinder generation design too low-diversity for a robust natural
carrier test.

The forbidden public surface count is nonzero in protected and raw arms:

```text
protected forbidden count, format_scrub=all: 6
raw forbidden count, format_scrub=all: 8
task_only forbidden count, format_scrub=all: 0
wrong_key forbidden count, format_scrub=all: 6
wrong_payload forbidden count, format_scrub=all: 6
```

This remains a hard gate failure. It cannot be treated as a harmless side
effect because the R4 route requires the primary reported decode to pass under
format scrub and audited forbidden-surface policy.

## Control-Plane Status

The duplicate submission `868158` was cancelled immediately. The active
canonical job for this result is `868151`.

Post-review allowlist checks passed locally and remotely:

```text
local:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_post_review_allowlist_safety_local_20260516.json
remote:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_post_review_allowlist_safety_remote_20260516.json
enabled entries: []
```

## Next Route

The next allowed action is artifact-only repair/pivot planning. No additional
Slurm submission is allowed from this route until a new reviewed repair route
is recorded.

The repair target should address all three observed failures:

```text
1. first-token controller does not force full phrase-surface realization;
2. deterministic row-cylinder generation collapses to repeated generic outputs;
3. forbidden public surface remains nonzero under primary format_scrub=all decode.
```

Candidate repair directions must be evaluated artifact-only first:

```text
- phrase-continuation controller, not first-token-only controller;
- short constrained natural continuation window with explicit KL/naturalness cap;
- revised decoder that records first-token evidence separately from phrase evidence only if precommitted;
- prompt/prefix diversity repair to reduce duplicate outputs;
- forbidden-surface matcher/output audit repair before any new generation.
```

This failure remains a useful diagnostic, but it is not a natural-output
positive result.
