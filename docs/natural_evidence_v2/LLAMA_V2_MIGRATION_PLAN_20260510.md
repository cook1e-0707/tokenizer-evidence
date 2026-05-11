# Llama-3.1-8B V2 Migration Plan

Status: artifact-only plan, no Slurm submitted.

## Scope

Migrate the natural_evidence_v2 pipeline from Qwen-2.5-7B to Llama-3.1-8B-Instruct
to establish cross-family generality. Per protocol contract step 8, this is now
allowed because Qwen WP6-R2 Option B positive recovery and null rejection passed.

## What Changes

- **Model**: `meta-llama/Meta-Llama-3.1-8B-Instruct` (replaces `Qwen/Qwen2.5-7B-Instruct`)
- **Tokenizer**: Llama BPE (different token vocabulary and boundaries)
- **Bucket bank**: Must regenerate with Llama tokenizer token IDs
  - Same action-verb surfaces: Create/Develop (bucket 0) vs Choose/Make (bucket 1)
  - Token IDs differ per tokenizer
- **LoRA target modules**: May need adjustment for Llama architecture
  - Qwen targets: `q_proj`, `v_proj` (verified in train script)
  - Llama targets: same names but different layer structure
  - Auto-detection in PEFT should handle this

## What Stays Same

- **Prompt set**: Reuse WP5 training prompts (512 protected, 512 task-only, 8192 score)
- **Slot policy**: Same restricted Step-label micro-slot detector
- **Bucket policy**: Same 2-way Create/Develop vs Choose/Make
- **Training hyperparams**: margin_lambda=30, steps=256, lr=1e-4, lora_r=32, lora_alpha=64
- **Decoder**: Same prompt-local 16-bit payload (0xa55e)
- **Decode protocol**: Same robust-block coordinate-majority decoder (8 blocks, 64 queries)
- **Gate thresholds**: Same as Qwen pipeline

## Gate Order

1. **WP5 teacher-forced**: Train protected + task-only LoRA, score bucket mass
   - Gate: protected mass lift vs base >= +0.15, vs task-only >= +0.10, rank-1 >= 70%
2. **WP6 E2E**: Generate with protected adapter, decode payload
   - Gate: 6/8 blocks accept, margin >= 3, nulls = 0
3. **WP6 FAR**: Aggregate null rejection FAR
   - Gate: FAR < 0.05 with Wilson CI

## Artifacts Created

- `scripts/natural_evidence_v2/build_llama_v2_bucket_bank.py` — Build Llama tokenizer bank
- `scripts/natural_evidence_v2/slurm/llama_v2_wp5_train_and_score.sbatch` — WP5 wrapper
- `scripts/natural_evidence_v2/slurm/llama_v2_wp6_e2e_eval.sbatch` — WP6 wrapper
- `configs/natural_evidence_v2/llama_v2_migration_plan.json` — Machine-readable plan

## Risk: Multi-Token Surfaces

Llama BPE may tokenize action verbs differently than Qwen:
- Some surfaces may be multi-token (e.g., "Create" → [C, reate])
- The bucket bank uses first-token-only scoring
- If a surface is multi-token, the first token may not be unique
- Mitigation: `build_llama_v2_bucket_bank.py` checks and reports multi-token surfaces
- If all surfaces are single-token, proceed directly
- If multi-token, need prefix-based scoring adaptation

## Risk: LoRA Adapter Compatibility

- Qwen and Llama have different layer counts and hidden sizes
- LoRA adapters are NOT cross-compatible
- Must train fresh Llama adapters from scratch
- Training data (prompts, slot policy) IS reusable

## Submission Plan

1. Build Llama bucket bank (CPU, can run on Chimera login node or Slurm)
2. Submit WP5 training (GPU, ~8h estimated)
3. Review WP5 results
4. Submit WP6 E2E (GPU, ~4h estimated)
5. Review WP6 results
6. Aggregate FAR

## Validation

All artifacts created locally. No Slurm interaction performed.
