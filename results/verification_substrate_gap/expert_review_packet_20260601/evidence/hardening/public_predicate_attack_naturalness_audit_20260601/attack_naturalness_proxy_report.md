# Public-Predicate Attack Naturalness Proxy Audit

This is an artifact-only proxy audit over existing guided rewrite/graft
source-mismatch examples. It checks surface readability signals such as
length ratio, token overlap, punctuation, isolated fragments, and known
broken-graft markers. It is not a semantic naturalness evaluation.

Status: `PASS_PUBLIC_PREDICATE_ATTACK_NATURALNESS_PROXY_AUDIT_RECORDED_NO_CLAIMS`
Rows: `60`
Proxy pass rows: `0`
Proxy fail rows: `60`
Proxy pass rate: `0.0`

## By Group

| Source | Model | Arm | Rows | Proxy pass | Proxy fail | Pass rate | Mean token overlap | Broken markers |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama_locked_adopted_880121 | llama | raw | 20 | 0 | 20 | 0.0 | 0.403244 | 1.0 |
| qwen_locked_adopted_870210_plus_870987 | qwen | raw | 20 | 0 | 20 | 0.0 | 0.424762 | 1.35 |
| qwen_locked_adopted_870210_plus_870987 | qwen | task_only | 20 | 0 | 20 | 0.0 | 0.456823 | 1.05 |

## Interpretation

Passing this proxy audit would not prove naturalness. Failing it records
that the current public-predicate guided rewrite/graft examples still
carry visible surface-quality risks. In all cases, accepted rows remain
source-mismatch spoofing evidence only, not protected success and not
codeword recovery.
