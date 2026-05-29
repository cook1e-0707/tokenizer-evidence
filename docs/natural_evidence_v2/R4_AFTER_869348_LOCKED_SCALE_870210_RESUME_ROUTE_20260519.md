# R4 After-869348 Locked-Scale 870210 Resume Route - 2026-05-19

## Decision

Resume only the failed or missing shards from job `870210`.

This is a runtime/storage recovery route, not a new protocol, model, decoder,
controller, payload, or gate. Job `870210` completed 79/96 shards and failed
because remote scratch was full. The route remains Qwen-only, same-contract
`a55e`, first-token event locked-scale generation.

## Source Run

```text
source_job_id: 870210
source_job_name: nat-ev-v2-r4lGen
source_array: 0-95%6
completed_shards: 0-74, 77, 79, 80, 81
failed_or_missing_shards: 75, 76, 78, 82-95
failure_class: runtime_storage_failure
hard_failure_evidence: shard_78 OSError [Errno 28] No space left on device
```

The run is not a locked-scale result until all 96 shards are complete and
aggregated.

## Cleanup Prerequisite

Remote scratch was cleaned before this resume route.

```text
cleanup_record:
  results/natural_evidence_v2/status/hpcstor_cleanup_20260519/
post_cleanup_free_space:
  166G available on /hpcstor6/scratch01/g/guanjie.lin001
```

## Resume Scope

```text
rerun_shards:
  75, 76, 78, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
array:
  75,76,78,82-95%6
output_dir:
  /hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_869348_locked_scale_generation_870210_resume_20260519
```

Completed shards from `870210` must not be rerun.

## User Override

The resume route uses the user's scheduling/logging override:

```text
do_not_set_TQDM_DISABLE: true
do_not_set_TRANSFORMERS_VERBOSITY: true
array_concurrency: 6
```

## Command

```bash
PLAN_ONLY=0 \
VALIDATE_PLAN_ONLY=0 \
OUTPUT_DIR=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_869348_locked_scale_generation_870210_resume_20260519 \
sbatch --array=75,76,78,82-95%6 \
  scripts/natural_evidence_v2/slurm/r4_after_869348_locked_scale_generation_h200.sbatch
```

## Aggregation

After the resume job completes, aggregate both roots:

```bash
python3 scripts/natural_evidence_v2/aggregate_r4_after_869348_locked_scale_generation.py \
  --output-dir results/natural_evidence_v2/status/r4_after_869348_locked_scale_generation_870210_plus_resume_aggregate_20260519 \
  --shard-roots \
    /hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_869348_locked_scale_generation_870210/shards \
    /hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_869348_locked_scale_generation_870210_resume_20260519/shards
```

The aggregation script must refuse duplicate complete shard artifacts and must
not report a locked-scale pass unless 96/96 shards are complete.

## Claim Control

This route does not unlock:

```text
training
Llama
same-family null
sanitizer
FAR aggregation
payload diversity
text-only phrase decoder success claim
paper-facing positive claim
```

