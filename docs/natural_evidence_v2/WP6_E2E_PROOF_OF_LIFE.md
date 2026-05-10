# WP6: Qwen V2 E2E Proof-of-Life

## Status
AUTHORIZED_TO_RUN_AFTER_WRAPPER_REVIEW

## Prerequisites (all met)
- ✅ WP4: prompt-local payload contract (16 slots, 2-way bucket)
- ✅ WP5-R2: protected adapter trained (job 851481, rank1=98.2%, mass=0.735)
- ✅ WP5-R2: task-only adapter trained (same job)
- ✅ Gate: `teacher_forced_gate_pass = true`
- ✅ User authorization: 2026-05-09 approval to enter WP6

## Contract Alignment Note

WP6 must use the WP4 contract that WP5-R2 actually trained against:

```text
results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/wp4_prompt_local_payload_contract.json
```

That contract encodes the prompt-local `a55e` payload+checksum target used by
the WP5 launch plan and WP5-R2 retry. The older
`contracts/wp4_prompt_local_contract_20260509_0610/` P00/P01 seeded oracle
contracts remain useful artifacts, but they were not the target schedule learned
by the WP5-R2 adapters and are not the controlling WP6 proof-of-life contract.

## What WP6 Does

Free-generation E2E evaluation across 5 arms:

1. **Generate responses** from eval prompts using:
   - `protected` (WP5-R2 adapter on Qwen base)
   - `raw` (base Qwen, no adapter)
   - `task_only_lora` (WP5-R2 task-only adapter)
   - `wrong_key` (protected adapter, wrong audit key)
   - `wrong_payload` (protected adapter, wrong payload)

2. **Detect micro-slots** in generated responses using WP3 slot policy

3. **Map observed tokens** at slot positions to bucket IDs (0 or 1)

4. **Decode payload** from accumulated bucket observations per frame

5. **Check proof-of-life gates**:
   - `protected_payload_recovery_at_64_min >= 0.80`
   - `raw_accepts_max == 0`
   - `task_only_accepts_max == 0`
   - `wrong_key_accepts_max == 0`
   - `wrong_payload_accepts_max == 0`
   - `free_generation_slot_detection_min >= 0.70`
   - `target_bucket_hit_rate_min in [0.25, 0.40]`
   - `output_forbidden_surface_rate_max == 0`

## Key Artifacts

- WP5-R2 protected adapter: `wp5_r2_teacher_forced_train_and_score_851481/protected_train/adapter`
- WP5-R2 task-only adapter: `wp5_r2_teacher_forced_train_and_score_851481/task_only_train/adapter`
- WP4 contract: `status/wp4_prompt_local_payload_contract_20260509_0611/wp4_prompt_local_payload_contract.json`
- Eval prompts: `status/wp3_r1_strict_density_expansion_plan_20260509_0355/restricted_step_label_strict_density_audit_prompts.jsonl` (`wp3_r1_eval` split)
- Primary bank: `buckets/qwen_v2_primary_2way_bank.jsonl`

## Implementation Order

1. Build `scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py`
   - Load model + adapter for each arm
   - Generate free responses from eval prompts
   - Write outputs JSONL

2. Build `scripts/natural_evidence_v2/decode_wp6_payload.py`
   - Read generated outputs
   - Detect micro-slots using WP3 detector contract
   - Map observed tokens to bucket IDs
   - Decode payload per frame
   - Write decode trace and summary

3. Build `scripts/natural_evidence_v2/slurm/wp6_e2e_eval.sbatch`
   - Wrap both scripts
   - Submit to DGXA100

4. Review results against proof-of-life gates

## Forbidden Actions
- No paper-facing positive claims
- No FAR aggregation
- No Llama replication
- No same-family nulls (until WP7)
- No sanitizer benchmarks (until WP8)
