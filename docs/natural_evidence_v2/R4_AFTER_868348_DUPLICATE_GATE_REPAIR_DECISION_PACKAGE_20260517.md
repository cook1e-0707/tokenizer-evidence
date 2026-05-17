# R4 After-868348 Duplicate-Gate Repair Decision Package

## Current Phase

```text
V2_R4_AFTER_868299_DEV_DIAGNOSTIC_868348_FAILED_GLOBAL_DUPLICATE_GATE_SIGNAL_PASSING_NO_RERUN
```

## Controlling Facts

`868348` completed all 32 H200 shards and is a strong first-token event signal
diagnostic, but it is not a pass under the precommitted strict-quality gate.

```text
protected strict accepts: 32/32
protected ignoring-quality accepts: 32/32
raw accepts: 0/32
task_only accepts: 0/32
wrong_key accepts: 0/32
wrong_payload accepts: 0/32
trace binding invalid rows: 0 / 98304
protected forbidden public surface count: 0
protected duplicate response hash count: 0
global exact duplicate extra rows: 2
full phrase protected accepts, format_scrub=all: 0
```

The two exact duplicate groups are both confined to `task_only` rows:

```text
task_only duplicate group 1:
  prompt_id: r4_cover_dev_4409a3670c843c3b1383
  prefix_family_id: useful_habit
  shards: 10 and 31

task_only duplicate group 2:
  prompt_id: r4_cover_dev_54dcac7d434267b59ff1
  prefix_family_id: practical_option
  shards: 9 and 24
```

This means the active blocker is not first-token event recovery, null
separation, trace binding, or protected output quality. The active blocker is
the interaction between cyclic dev allocation reuse and the precommitted global
exact duplicate gate.

## Non-Negotiable Constraints

```text
do not reclassify 868348 as positive
do not adopt partial 868313 output
do not lower gates retroactively for 868348
do not make paper-facing positive claims
do not run another Slurm job before a reviewed repair route
do not unlock training, Llama, sanitizer, FAR, or payload diversity
```

## Repair Options

### Option A: Globally Unique Allocation Repair

Precommit a new dev allocation that avoids global prompt/prefix reuse across
all 32 blocks, then rerun the same first-token event dev diagnostic gate.

Advantages:

```text
keeps global exact duplicate gate unchanged
avoids post-hoc gate reinterpretation
directly addresses the observed failure source
cleaner for expert/reviewer audit
```

Risks:

```text
requires enough tokenizer-valid rows to cover 32768 unique row cylinders
may require a new allocation builder and preflight
may slightly change prompt/topic distribution
```

Minimum preflight:

```text
duplicate prompt_id: 0
duplicate prompt_text_hash: 0
duplicate prompt_prefix_pair globally: 0
duplicate shard/block/prompt tuple: 0
selected row count: 32768
per-shard rows: 1024
all tokenizer/controller/codebook artifacts unchanged
no generation before route review
```

### Option B: Future Duplicate-Gate Semantics Repair

Precommit a narrower future gate that treats duplicate control rows separately
from protected accepted-output duplicates, while still reporting global duplicate
counts.

Allowed only for future runs:

```text
protected duplicate response hash count must remain 0
within-block duplicate response hash count must remain 0
accepted-output duplicate count must remain 0
global duplicate count reported separately
control-only duplicate count reported separately
no retroactive rescue of 868348
```

Advantages:

```text
matches the observed failure: duplicates are task_only only
avoids expensive rerun solely for harmless control duplicate rows
keeps protected quality gate strict
```

Risks:

```text
weaker than the previously precommitted global duplicate gate
may look like post-hoc gate narrowing if not reviewed before the next run
harder to defend than Option A for a paper-facing natural-output claim
```

## Recommended Default

Use Option A as the default next implementation path unless expert review
explicitly chooses Option B. Option A preserves the stricter quality semantics
and avoids reinterpreting the failed run.

## Next Allowed Action

```text
artifact-only allocation feasibility audit for globally unique 32-block
prompt/prefix rows, plus a reviewed no-submit route decision
```
