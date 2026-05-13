# R4 prefix-native surface repair candidate

This artifact-only package builds a repaired candidate bank and
teacher-forced probe rows after the `853815` surface-mass failure.
It does not tokenize with Qwen locally, score a model, submit Slurm,
train, generate, run Llama, aggregate FAR, or make paper claims.

## Static Validation

- status: `PASS_PROXY_STATIC_VALIDATION_TOKENIZER_PENDING`
- coordinates: `32`
- entries: `256`
- probe rows: `8192`
- prompts: `256`
- missing binary-side coordinates: `0`
- first-word proxy overlap coordinates: `0`
- forbidden surface hits: `0`
- measured span-start failures: `0`

## Design Rationale

The candidate stops using free-floating phrase targets such as long
verb-object clauses. Instead it uses short, prefix-native continuations
whose measured span begins immediately after the lead-in prefix. The
binary sides reuse the learned R3/WP5 action families (`set`/`plan`
versus `create`/`prepare`) in ordinary cover-natural phrases.

## Limitation

Qwen tokenizer validation is intentionally not run locally because this
environment does not provide `transformers`. The local check uses a
normalized first-word proxy only. A future tokenizer/model scoring step
must be a separate reviewed Slurm-only route.
